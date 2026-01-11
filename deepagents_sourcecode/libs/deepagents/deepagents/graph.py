"""Deepagents는 계획, 파일 시스템, 서브에이전트 기능을 제공합니다."""

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

from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."


def get_default_model() -> ChatAnthropic:
    """Deep agents의 기본 모델을 가져옵니다.

    Returns:
        Claude Sonnet 4.5로 구성된 `ChatAnthropic` 인스턴스.
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
    skills: list[str] | None = None,
    memory: list[str] | None = None,
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
    """DeepAgent를 생성합니다.

    이 에이전트는 기본적으로 아래 기능(도구/미들웨어)을 포함합니다.
    - todo 작성 도구: `write_todos`
    - 파일/실행 도구: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`
    - 서브에이전트 호출 도구

    `execute` 도구는 backend가 `SandboxBackendProtocol`을 구현할 때 셸 커맨드를 실행할 수 있습니다.
    샌드박스가 아닌 backend에서는 `execute`가 오류 메시지를 반환합니다.

    Args:
        model: 사용할 모델. 기본값은 `claude-sonnet-4-5-20250929`.
        tools: 에이전트에 추가로 제공할 도구 목록.
        system_prompt: 에이전트에 추가로 주입할 지침. system prompt에 포함됩니다.
        middleware: 표준 미들웨어 뒤에 추가로 적용할 미들웨어 목록.
        subagents: 사용할 서브에이전트 정의 목록.

            각 서브에이전트는 아래 키를 가진 `dict` 형태입니다.
            - `name`
            - `description` (메인 에이전트가 어떤 서브에이전트를 호출할지 결정할 때 사용)
            - `prompt` (서브에이전트의 system prompt로 사용)
            - (optional) `tools`
            - (optional) `model` (`LanguageModelLike` 인스턴스 또는 설정 `dict`)
            - (optional) `middleware` (`AgentMiddleware` 리스트)
        skills: 스킬 소스 경로 목록(예: `["/skills/user/", "/skills/project/"]`) (선택).

            경로는 POSIX 형식(슬래시 `/`)으로 지정하며 backend root 기준 상대 경로입니다.
            `StateBackend`(기본값)를 사용할 때는 `invoke(files={...})`로 파일을 제공해야 합니다.
            `FilesystemBackend`에서는 backend의 `root_dir` 기준으로 디스크에서 스킬을 로드합니다.
            같은 이름의 스킬이 중복될 경우 뒤에 오는 소스가 우선합니다(last one wins).
        memory: 로드할 메모리 파일(AGENTS.md) 경로 목록(예: `["/memory/AGENTS.md"]`) (선택).
            표시 이름은 경로에서 자동 유도되며, 에이전트 시작 시 로드되어 system prompt에 포함됩니다.
        response_format: 구조화 출력 응답 포맷(선택).
        context_schema: DeepAgent의 컨텍스트 스키마(선택).
        checkpointer: 실행 간 state를 저장하기 위한 `Checkpointer`(선택).
        store: 영구 저장을 위한 store(선택). backend가 `StoreBackend`를 사용할 경우 필요합니다.
        backend: 파일 저장/실행을 위한 backend(선택).

            `Backend` 인스턴스 또는 `lambda rt: StateBackend(rt)` 같은 팩토리 함수를 전달할 수 있습니다.
            실행 지원이 필요하면 `SandboxBackendProtocol`을 구현한 backend를 사용하세요.
        interrupt_on: 도구 이름 → interrupt 설정 매핑(선택).
        debug: debug 모드 활성화 여부. `create_agent`로 전달됩니다.
        name: 에이전트 이름. `create_agent`로 전달됩니다.
        cache: 캐시 인스턴스. `create_agent`로 전달됩니다.

    Returns:
        설정된(compiled) deep agent 그래프.
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

    # 서브에이전트용 미들웨어 스택 구성(skills가 있으면 포함)
    subagent_middleware: list[AgentMiddleware] = [
        TodoListMiddleware(),
    ]

    backend = backend if backend is not None else (lambda rt: StateBackend(rt))

    if skills is not None:
        subagent_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
    subagent_middleware.extend(
        [
            FilesystemMiddleware(backend=backend),
            SummarizationMiddleware(
                model=model,
                trigger=trigger,
                keep=keep,
                trim_tokens_to_summarize=None,
            ),
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
            PatchToolCallsMiddleware(),
        ]
    )

    # 메인 에이전트 미들웨어 스택 구성
    deepagent_middleware: list[AgentMiddleware] = [
        TodoListMiddleware(),
    ]
    if memory is not None:
        deepagent_middleware.append(MemoryMiddleware(backend=backend, sources=memory))
    if skills is not None:
        deepagent_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
    deepagent_middleware.extend(
        [
            FilesystemMiddleware(backend=backend),
            SubAgentMiddleware(
                default_model=model,
                default_tools=tools,
                subagents=subagents if subagents is not None else [],
                default_middleware=subagent_middleware,
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
    )
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
