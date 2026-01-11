"""AGENTS.md 파일로부터 에이전트 메모리/컨텍스트를 로드하는 미들웨어입니다.

이 모듈은 AGENTS.md 사양(https://agents.md/)을 지원하여, 설정된 소스에서 메모리/컨텍스트를
읽어들인 다음 system prompt에 주입(inject)합니다.

## 개요

AGENTS.md는 프로젝트별 맥락과 지침을 제공하여 AI 에이전트가 더 안정적으로 작업하도록 돕습니다.
스킬(skills)이 “필요할 때만(on-demand) 호출되는 워크플로”라면, 메모리(memory)는 항상 로드되어
지속적인(persistent) 컨텍스트를 제공합니다.

## 사용 예시

```python
from deepagents import MemoryMiddleware
from deepagents.backends.filesystem import FilesystemBackend

# 보안 주의: FilesystemBackend는 전체 파일시스템에 대한 읽기/쓰기 권한을 가질 수 있습니다.
# 에이전트가 샌드박스에서 실행되도록 하거나, 파일 작업에 human-in-the-loop(HIL) 승인을 붙이세요.
backend = FilesystemBackend(root_dir="/")

middleware = MemoryMiddleware(
    backend=backend,
    sources=[
        "~/.deepagents/AGENTS.md",
        "./.deepagents/AGENTS.md",
    ],
)

agent = create_deep_agent(middleware=[middleware])
```

## 메모리 소스(Memory Sources)

소스는 로드할 AGENTS.md 파일 경로 리스트입니다. 소스는 지정된 순서대로 읽혀 하나로 결합되며,
뒤에 오는 소스가 결합된 프롬프트의 뒤쪽에 붙습니다.

## 파일 형식

AGENTS.md는 일반적인 Markdown이며 필수 구조는 없습니다. 관례적으로는 아래 섹션이 자주 포함됩니다.
- 프로젝트 개요
- 빌드/테스트 명령
- 코드 스타일 가이드
- 아키텍처 노트
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, NotRequired, TypedDict

from langchain.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain.tools import ToolRuntime
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class MemoryState(AgentState):
    """MemoryMiddleware의 state 스키마입니다.

    Attributes:
        memory_contents: 소스 경로 → 로드된 콘텐츠 매핑.
            최종 agent state에 포함되지 않도록 private로 표시됩니다.
    """

    memory_contents: NotRequired[Annotated[dict[str, str], PrivateStateAttr]]


class MemoryStateUpdate(TypedDict):
    """MemoryMiddleware의 state 업데이트 타입입니다."""

    memory_contents: dict[str, str]


MEMORY_SYSTEM_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
    The above <agent_memory> was loaded in from files in your filesystem. As you learn from your interactions with the user, you can save new knowledge by calling the `edit_file` tool.

    **Learning from feedback:**
    - One of your MAIN PRIORITIES is to learn from your interactions with the user. These learnings can be implicit or explicit. This means that in the future, you will remember this important information.
    - When you need to remember something, updating memory must be your FIRST, IMMEDIATE action - before responding to the user, before calling other tools, before doing anything else. Just update memory immediately.
    - When user says something is better/worse, capture WHY and encode it as a pattern.
    - Each correction is a chance to improve permanently - don't just fix the immediate issue, update your instructions.
    - A great opportunity to update your memories is when the user interrupts a tool call and provides feedback. You should update your memories immediately before revising the tool call.
    - Look for the underlying principle behind corrections, not just the specific mistake.
    - The user might not explicitly ask you to remember something, but if they provide information that is useful for future use, you should update your memories immediately.

    **Asking for information:**
    - If you lack context to perform an action (e.g. send a Slack DM, requires a user ID/email) you should explicitly ask the user for this information.
    - It is preferred for you to ask for information, don't assume anything that you do not know!
    - When the user provides information that is useful for future use, you should update your memories immediately.

    **When to update memories:**
    - When the user explicitly asks you to remember something (e.g., "remember my email", "save this preference")
    - When the user describes your role or how you should behave (e.g., "you are a web researcher", "always do X")
    - When the user gives feedback on your work - capture what was wrong and how to improve
    - When the user provides information required for tool use (e.g., slack channel ID, email addresses)
    - When the user provides context useful for future tasks, such as how to use tools, or which actions to take in a particular situation
    - When you discover new patterns or preferences (coding styles, conventions, workflows)

    **When to NOT update memories:**
    - When the information is temporary or transient (e.g., "I'm running late", "I'm on my phone right now")
    - When the information is a one-time task request (e.g., "Find me a recipe", "What's 25 * 4?")
    - When the information is a simple question that doesn't reveal lasting preferences (e.g., "What day is it?", "Can you explain X?")
    - When the information is an acknowledgment or small talk (e.g., "Sounds good!", "Hello", "Thanks for that")
    - When the information is stale or irrelevant in future conversations
    - Never store API keys, access tokens, passwords, or any other credentials in any file, memory, or system prompt.
    - If the user asks where to put API keys or provides an API key, do NOT echo or save it.

    **Examples:**
    Example 1 (remembering user information):
    User: Can you connect to my google account?
    Agent: Sure, I'll connect to your google account, what's your google account email?
    User: john@example.com
    Agent: Let me save this to my memory.
    Tool Call: edit_file(...) -> remembers that the user's google account email is john@example.com

    Example 2 (remembering implicit user preferences):
    User: Can you write me an example for creating a deep agent in LangChain?
    Agent: Sure, I'll write you an example for creating a deep agent in LangChain <example code in Python>
    User: Can you do this in JavaScript
    Agent: Let me save this to my memory.
    Tool Call: edit_file(...) -> remembers that the user prefers to get LangChaincode examples in JavaScript
    Agent: Sure, here is the JavaScript example<example code in JavaScript>

    Example 3 (do not remember transient information):
    User: I'm going to play basketball tonight so I will be offline for a few hours.
    Agent: Okay I'll add a black to your calendar.
    Tool Call: create_calendar_event(...) -> just calls a tool, does not commit anything to memory, as it is transient information
</memory_guidelines>
"""


class MemoryMiddleware(AgentMiddleware):
    """AGENTS.md 파일에서 에이전트 메모리를 로드하는 미들웨어입니다.

    설정된 소스에서 메모리를 로드한 뒤 system prompt에 주입합니다.
    여러 소스를 결합하여 한 번에 주입하는 구성을 지원합니다.

    Args:
        backend: 파일 작업을 위한 backend 인스턴스 또는 팩토리 함수.
        sources: 로드할 AGENTS.md 파일 경로 리스트.
    """

    state_schema = MemoryState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES,
        sources: list[str],
    ) -> None:
        """메모리 미들웨어를 초기화합니다.

        Args:
            backend: backend 인스턴스 또는 (runtime을 받아 backend를 만드는) 팩토리 함수.
                `StateBackend`를 사용하려면 팩토리 형태로 전달해야 합니다.
            sources: 로드할 메모리 파일 경로 리스트(예: `["~/.deepagents/AGENTS.md", "./.deepagents/AGENTS.md"]`).
                표시 이름은 경로로부터 자동 유도됩니다. 소스는 지정 순서대로 로드됩니다.
        """
        self._backend = backend
        self.sources = sources

    def _get_backend(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> BackendProtocol:
        """Backend를 인스턴스 또는 팩토리로부터 해석(resolve)합니다."""
        if callable(self._backend):
            # backend 팩토리를 호출하기 위한 ToolRuntime을 구성합니다.
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            return self._backend(tool_runtime)
        return self._backend

    def _format_agent_memory(self, contents: dict[str, str]) -> str:
        """메모리 소스 경로와 콘텐츠를 짝지어 포맷팅합니다."""
        if not contents:
            return MEMORY_SYSTEM_PROMPT.format(agent_memory="(No memory loaded)")

        sections = []
        for path in self.sources:
            if contents.get(path):
                sections.append(f"{path}\n{contents[path]}")

        if not sections:
            return MEMORY_SYSTEM_PROMPT.format(agent_memory="(No memory loaded)")

        memory_body = "\n\n".join(sections)
        return MEMORY_SYSTEM_PROMPT.format(agent_memory=memory_body)

    async def _load_memory_from_backend(
        self,
        backend: BackendProtocol,
        path: str,
    ) -> str | None:
        """backend에서 특정 경로의 메모리(AGENTS.md) 콘텐츠를 로드합니다."""
        results = await backend.adownload_files([path])
        # 단일 path에 대해 단일 응답이 와야 합니다.
        if len(results) != 1:
            raise AssertionError(f"Expected 1 response for path {path}, got {len(results)}")
        response = results[0]

        if response.error is not None:
            # 현재는 메모리 파일을 optional로 취급합니다.
            # file_not_found는 정상적으로 발생할 수 있으므로 조용히 스킵하여 점진적 저하(graceful degradation)를 허용합니다.
            if response.error == "file_not_found":
                return None
            # 그 외 오류는 예외로 올립니다.
            raise ValueError(f"Failed to download {path}: {response.error}")

        if response.content is not None:
            return response.content.decode("utf-8")

        return None

    def _load_memory_from_backend_sync(
        self,
        backend: BackendProtocol,
        path: str,
    ) -> str | None:
        """backend에서 특정 경로의 메모리(AGENTS.md) 콘텐츠를 동기로 로드합니다.

        Args:
            backend: Backend to load from.
            path: Path to the AGENTS.md file.

        Returns:
            File content if found, None otherwise.
        """
        results = backend.download_files([path])
        # 단일 path에 대해 단일 응답이 와야 합니다.
        if len(results) != 1:
            raise AssertionError(f"Expected 1 response for path {path}, got {len(results)}")
        response = results[0]

        if response.error is not None:
            # 현재는 메모리 파일을 optional로 취급합니다.
            # file_not_found는 정상적으로 발생할 수 있으므로 조용히 스킵하여 점진적 저하(graceful degradation)를 허용합니다.
            if response.error == "file_not_found":
                return None
            # 그 외 오류는 예외로 올립니다.
            raise ValueError(f"Failed to download {path}: {response.error}")

        if response.content is not None:
            return response.content.decode("utf-8")

        return None

    def before_agent(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> MemoryStateUpdate | None:
        """에이전트 실행 전에 메모리 콘텐츠를 로드합니다(동기).

        Loads memory from all configured sources and stores in state.
        Only loads if not already present in state.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with memory_contents populated.
        """
        # 이미 로드되어 있으면 스킵
        if "memory_contents" in state:
            return None

        backend = self._get_backend(state, runtime, config)
        contents: dict[str, str] = {}

        for path in self.sources:
            content = self._load_memory_from_backend_sync(backend, path)
            if content:
                contents[path] = content
                logger.debug(f"Loaded memory from: {path}")

        return MemoryStateUpdate(memory_contents=contents)

    async def abefore_agent(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> MemoryStateUpdate | None:
        """에이전트 실행 전에 메모리 콘텐츠를 로드합니다(async).

        Loads memory from all configured sources and stores in state.
        Only loads if not already present in state.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with memory_contents populated.
        """
        # 이미 로드되어 있으면 스킵
        if "memory_contents" in state:
            return None

        backend = self._get_backend(state, runtime, config)
        contents: dict[str, str] = {}

        for path in self.sources:
            content = await self._load_memory_from_backend(backend, path)
            if content:
                contents[path] = content
                logger.debug(f"Loaded memory from: {path}")

        return MemoryStateUpdate(memory_contents=contents)

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """메모리 콘텐츠를 system prompt에 주입합니다.

        Args:
            request: Model request to modify.

        Returns:
            Modified request with memory injected into system prompt.
        """
        contents = request.state.get("memory_contents", {})
        agent_memory = self._format_agent_memory(contents)

        if request.system_prompt:
            system_prompt = agent_memory + "\n\n" + request.system_prompt
        else:
            system_prompt = agent_memory

        return request.override(system_message=SystemMessage(system_prompt))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """System prompt에 메모리를 주입한 뒤 model call을 수행하도록 감쌉니다.

        Args:
            request: Model request being processed.
            handler: Handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) system prompt에 메모리를 주입한 뒤 model call을 수행하도록 감쌉니다.

        Args:
            request: Model request being processed.
            handler: Async handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        modified_request = self.modify_request(request)
        return await handler(modified_request)
