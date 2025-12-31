"""Deepagents는 계획(planning), 파일시스템(filesystem), 하위 에이전트(subagents) 기능을 포함합니다."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."


def get_default_model() -> ChatAnthropic:
    """Deep Agent를 위한 기본 모델을 반환합니다.

    Returns:
        Claude Sonnet 4로 구성된 ChatAnthropic 인스턴스.
    """
    return ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=20000,
    )


def create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph:
    """Deep Agent를 생성합니다.

    이 에이전트는 기본적으로 할 일 목록 작성 도구(write_todos), 7가지 파일 및 실행 도구
    (ls, read_file, write_file, edit_file, glob, grep, execute), 그리고 하위 에이전트 호출 도구를 가집니다.

    execute 도구는 백엔드가 SandboxBackendProtocol을 구현한 경우 쉘 명령을 실행할 수 있습니다.
    샌드박스 백엔드가 아닌 경우 execute 도구는 오류 메시지를 반환합니다.

    Args:
        model: 사용할 모델. 기본값은 Claude Sonnet 4입니다.
        tools: 에이전트가 접근할 수 있는 도구들입니다.
        system_prompt: 에이전트에게 제공할 추가 지침입니다. 시스템 프롬프트에 포함됩니다.
        middleware: 표준 미들웨어 이후에 적용할 추가 미들웨어입니다.
        subagents: 사용할 하위 에이전트 목록입니다. 각 하위 에이전트는 다음 키를 가진 딕셔너리여야 합니다:
                - `name`
                - `description` (메인 에이전트가 하위 에이전트 호출 여부를 결정할 때 사용)
                - `prompt` (하위 에이전트의 시스템 프롬프트로 사용)
                - (선택사항) `tools`
                - (선택사항) `model` (LanguageModelLike 인스턴스 또는 dict 설정)
                - (선택사항) `middleware` (List[AgentMiddleware])
        response_format: 에이전트에 사용할 구조화된 출력 응답 형식입니다.
        context_schema: Deep Agent의 스키마입니다.
        checkpointer: 실행 간 에이전트 상태를 유지하기 위한 선택적 체크포인터입니다.
        store: 영구 저장을 위한 선택적 저장소 (백엔드가 StoreBackend를 사용하는 경우 필수).
        backend: 파일 저장 및 실행을 위한 선택적 백엔드. Backend 인스턴스 또는
            `lambda rt: StateBackend(rt)`와 같은 호출 가능한 팩토리를 전달합니다. 실행 지원을 위해서는
            SandboxBackendProtocol을 구현하는 백엔드를 사용하십시오.
        interrupt_on: 도구 이름을 인터럽트 설정에 매핑하는 선택적 Dict[str, bool | InterruptOnConfig]입니다.
        debug: 디버그 모드 활성화 여부. create_agent로 전달됩니다.
        name: 에이전트의 이름. create_agent로 전달됩니다.
        cache: 에이전트에 사용할 캐시. create_agent로 전달됩니다.

    Returns:
        구성된 Deep Agent.
    """
    if model is None:
        model = get_default_model()
    elif isinstance(model, str):
        model = init_chat_model(model)

    if (
        model.profile is not None
        and isinstance(model.profile, dict)
        and "max_input_tokens" in model.profile
        and isinstance(model.profile["max_input_tokens"], int)
    ):
        trigger = ("fraction", 0.85)
        keep = ("fraction", 0.10)
    else:
        trigger = ("tokens", 170000)
        keep = ("messages", 6)

    deepagent_middleware = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        SubAgentMiddleware(
            default_model=model,
            default_tools=tools,
            subagents=subagents if subagents is not None else [],
            default_middleware=[
                TodoListMiddleware(),
                FilesystemMiddleware(backend=backend),
                SummarizationMiddleware(
                    model=model,
                    trigger=trigger,
                    keep=keep,
                    trim_tokens_to_summarize=None,
                ),
                AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
                PatchToolCallsMiddleware(),
            ],
            default_interrupt_on=interrupt_on,
            general_purpose_agent=True,
        ),
        SummarizationMiddleware(
            model=model,
            trigger=trigger,
            keep=keep,
            trim_tokens_to_summarize=None,
        ),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
    ]
    if middleware:
        deepagent_middleware.extend(middleware)
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    return create_agent(
        model,
        system_prompt=system_prompt + "\n\n" + BASE_AGENT_PROMPT if system_prompt else BASE_AGENT_PROMPT,
        tools=tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})
