"""CLI를 위한 에이전트 관리 및 생성."""

import os
import shutil
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.sandbox import SandboxBackendProtocol
from langchain.agents.middleware import (
    InterruptOnConfig,
)
from langchain.agents.middleware.types import AgentState
from langchain.messages import ToolCall
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.pregel import Pregel
from langgraph.runtime import Runtime

from deepagents_cli.agent_memory import AgentMemoryMiddleware
from deepagents_cli.config import COLORS, config, console, get_default_coding_instructions, settings
from deepagents_cli.integrations.sandbox_factory import get_default_working_dir
from deepagents_cli.shell import ShellMiddleware
from deepagents_cli.skills import SkillsMiddleware


def list_agents() -> None:
    """사용 가능한 모든 에이전트를 나열합니다."""
    agents_dir = settings.user_deepagents_dir

    if not agents_dir.exists() or not any(agents_dir.iterdir()):
        console.print("[yellow]에이전트를 찾을 수 없습니다.[/yellow]")
        console.print(
            "[dim]처음 사용할 때 ~/.deepagents/에 에이전트가 생성됩니다.[/dim]",
            style=COLORS["dim"],
        )
        return

    console.print("\n[bold]사용 가능한 에이전트:[/bold]\n", style=COLORS["primary"])

    for agent_path in sorted(agents_dir.iterdir()):
        if agent_path.is_dir():
            agent_name = agent_path.name
            agent_md = agent_path / "agent.md"

            if agent_md.exists():
                console.print(f"  • [bold]{agent_name}[/bold]", style=COLORS["primary"])
                console.print(f"    {agent_path}", style=COLORS["dim"])
            else:
                console.print(f"  • [bold]{agent_name}[/bold] [dim](미완성)[/dim]", style=COLORS["tool"])
                console.print(f"    {agent_path}", style=COLORS["dim"])

    console.print()


def reset_agent(agent_name: str, source_agent: str | None = None) -> None:
    """에이전트를 기본값으로 재설정하거나 다른 에이전트로부터 복사합니다."""
    agents_dir = settings.user_deepagents_dir
    agent_dir = agents_dir / agent_name

    if source_agent:
        source_dir = agents_dir / source_agent
        source_md = source_dir / "agent.md"

        if not source_md.exists():
            console.print(
                f"[bold red]오류:[/bold red] 소스 에이전트 '{source_agent}'를 찾을 수 없거나 agent.md가 없습니다"
            )
            return

        source_content = source_md.read_text()
        action_desc = f"contents of agent '{source_agent}'"
    else:
        source_content = get_default_coding_instructions()
        action_desc = "default"

    if agent_dir.exists():
        shutil.rmtree(agent_dir)
        console.print(f"기존 에이전트 디렉터리를 제거했습니다: {agent_dir}", style=COLORS["tool"])

    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "agent.md"
    agent_md.write_text(source_content)

    console.print(f"✓ 에이전트 '{agent_name}'가 {action_desc}(으)로 재설정되었습니다", style=COLORS["primary"])
    console.print(f"Location: {agent_dir}\n", style=COLORS["dim"])


def get_system_prompt(assistant_id: str, sandbox_type: str | None = None) -> str:
    """에이전트에 대한 기본 시스템 프롬프트를 가져옵니다.

    Args:
        assistant_id: 경로 참조를 위한 에이전트 식별자
        sandbox_type: 샌드박스 공급자 유형("modal", "runloop", "daytona").
                     None인 경우 에이전트는 로컬 모드에서 작동합니다.

    Returns:
        시스템 프롬프트 문자열 (agent.md 내용 제외)
    """
    agent_dir_path = f"~/.deepagents/{assistant_id}"

    if sandbox_type:
        # Get provider-specific working directory

        working_dir = get_default_working_dir(sandbox_type)

        working_dir_section = f"""### Current Working Directory

You are working in a **remote Linux sandbox** at `{working_dir}`.

All code execution and file operations happen in this sandbox environment.

**IMPORTANT:**
- The CLI runs locally on the user's machine, but executes code remotely.
- Use `{working_dir}` as your working directory for all operations.

"""
    else:
        cwd = Path.cwd()
        working_dir_section = f"""<env>
WORKING_DIRECTORY: {cwd}
</env>

### Current Working Directory

The filesystem backend is currently operating at: `{cwd}`

### File System and Paths

**IMPORTANT - Path Handling:**
- All file paths MUST be absolute (e.g. `{cwd}/file.txt`).
- Use the WORKING_DIRECTORY from <env> to construct absolute paths.
- Example: To create a file in the working directory, use `{cwd}/research_project/file.md`
- Do NOT use relative paths - always construct the full absolute path.

"""

    return (
        working_dir_section
        + f"""### Skills Directory

Your skills are stored at: `{agent_dir_path}/skills/`
Skills may contain scripts or support files. Use the physical filesystem path when running skill scripts with bash:
Example: `bash python {agent_dir_path}/skills/web-research/script.py`

### Human-in-the-Loop Tool Approvals

Some tool calls require user approval before execution. If a tool call is rejected by the user:
1. Accept the decision immediately - do NOT try the same command again.
2. Explain that you understand the user rejected the operation.
3. Propose an alternative or ask for clarification.
4. NEVER try to bypass a rejection by retrying the exact same command.

Respect user decisions and work collaboratively.

### Web Search Tool Usage

When using the web_search tool:
1. The tool returns search results with titles, URLs, and content snippets.
2. You MUST read and process these results, then respond to the user naturally.
3. Do NOT show raw JSON or tool results directly to the user.
4. Synthesize information from multiple sources into a coherent answer.
5. Cite sources by mentioning page titles or URLs when relevant.
6. If you don't find what you need in the search, explain what you found and ask clarifying questions.

The user ONLY sees your text response, not the tool results. Always provide a complete, natural language answer after using web_search.

### Todo List Management

When using the write_todos tool:
1. Keep the todo list minimal - aim for 3-6 items max.
2. Only create todos for complex, multi-step tasks that really need tracking.
3. Break down tasks into clear, actionable items without being overly granular.
4. For simple tasks (1-2 steps), just do them - don't create a todo.
5. When first creating a todo list for a task, ALWAYS ask the user if the plan looks good before starting work.
   - Create the todos so they render, then ask "Does this plan look good?" or similar.
   - Wait for the user's response before marking the first todo in_progress.
   - Adjust the plan if they want changes.
6. Update todo status promptly as you complete each item.

The todo list is a planning tool - use it judiciously to avoid overwhelming the user with excessive task tracking."""
    )


def _format_write_file_description(tool_call: ToolCall, _state: AgentState, _runtime: Runtime) -> str:
    """승인 프롬프트를 위한 write_file 도구 호출 포맷."""
    args = tool_call["args"]
    file_path = args.get("file_path", "unknown")
    content = args.get("content", "")

    action = "덮어쓰기(Overwrite)" if Path(file_path).exists() else "생성(Create)"
    line_count = len(content.splitlines())

    return f"파일: {file_path}\n작업: 파일 {action}\n줄 수: {line_count}"


def _format_edit_file_description(tool_call: ToolCall, _state: AgentState, _runtime: Runtime) -> str:
    """승인 프롬프트를 위한 edit_file 도구 호출 포맷."""
    args = tool_call["args"]
    file_path = args.get("file_path", "unknown")
    replace_all = bool(args.get("replace_all", False))

    return f"파일: {file_path}\n작업: 텍스트 교체 ({'모든 항목' if replace_all else '단일 항목'})"


def _format_web_search_description(tool_call: ToolCall, _state: AgentState, _runtime: Runtime) -> str:
    """Format web_search tool call for approval prompt."""
    args = tool_call["args"]
    query = args.get("query", "unknown")
    max_results = args.get("max_results", 5)

    return f"쿼리: {query}\n최대 결과: {max_results}\n\n⚠️  이 작업은 Tavily API 크레딧을 사용합니다"


def _format_fetch_url_description(tool_call: ToolCall, _state: AgentState, _runtime: Runtime) -> str:
    """Format fetch_url tool call for approval prompt."""
    args = tool_call["args"]
    url = args.get("url", "unknown")
    timeout = args.get("timeout", 30)

    return f"URL: {url}\n시간 제한: {timeout}초\n\n⚠️  웹 콘텐츠를 가져와 마크다운으로 변환합니다"


def _format_task_description(tool_call: ToolCall, _state: AgentState, _runtime: Runtime) -> str:
    """승인 프롬프트를 위한 task(서브 에이전트) 도구 호출 포맷.

    task 도구 서명은: task(description: str, subagent_type: str)
    description에는 서브 에이전트에게 전송될 모든 지침이 포함됩니다.
    """
    args = tool_call["args"]
    description = args.get("description", "unknown")
    subagent_type = args.get("subagent_type", "unknown")

    # Truncate description if too long for display
    description_preview = description
    if len(description) > 500:
        description_preview = description[:500] + "..."

    return (
        f"서브 에이전트 유형: {subagent_type}\n\n"
        f"작업 지침:\n"
        f"{'─' * 40}\n"
        f"{description_preview}\n"
        f"{'─' * 40}\n\n"
        f"⚠️  서브 에이전트는 파일 작업 및 셸 명령에 접근할 수 있습니다"
    )


def _format_shell_description(tool_call: ToolCall, _state: AgentState, _runtime: Runtime) -> str:
    """Format shell tool call for approval prompt."""
    args = tool_call["args"]
    command = args.get("command", "없음")
    return f"셸 명령: {command}\n작업 디렉터리: {Path.cwd()}"


def _format_execute_description(tool_call: ToolCall, _state: AgentState, _runtime: Runtime) -> str:
    """Format execute tool call for approval prompt."""
    args = tool_call["args"]
    command = args.get("command", "없음")
    return f"명령 실행: {command}\n위치: 원격 샌드박스"


def _add_interrupt_on() -> dict[str, InterruptOnConfig]:
    """파괴적인 도구에 대해 히먼-인-더-루프(human-in-the-loop) interrupt_on 설정을 구성합니다."""
    shell_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_shell_description,
    }

    execute_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_execute_description,
    }

    write_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_write_file_description,
    }

    edit_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_edit_file_description,
    }

    web_search_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_web_search_description,
    }

    fetch_url_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_fetch_url_description,
    }

    task_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_task_description,
    }
    return {
        "shell": shell_interrupt_config,
        "execute": execute_interrupt_config,
        "write_file": write_file_interrupt_config,
        "edit_file": edit_file_interrupt_config,
        "web_search": web_search_interrupt_config,
        "fetch_url": fetch_url_interrupt_config,
        "task": task_interrupt_config,
    }


def create_cli_agent(
    model: str | BaseChatModel,
    assistant_id: str,
    *,
    tools: list[BaseTool] | None = None,
    sandbox: SandboxBackendProtocol | None = None,
    sandbox_type: str | None = None,
    system_prompt: str | None = None,
    auto_approve: bool = False,
    enable_memory: bool = True,
    enable_skills: bool = True,
    enable_shell: bool = True,
) -> tuple[Pregel, CompositeBackend]:
    """유연한 옵션으로 CLI 구성 에이전트를 생성합니다.

    이것은 deepagents CLI 에이전트 생성을 위한 주요 진입점이며,
    내부적으로 사용되거나 외부 코드(예: 벤치마킹 프레임워크, Harbor)에서 사용할 수 있습니다.

    Args:
        model: 사용할 LLM 모델 (예: "anthropic:claude-sonnet-4-5-20250929")
        assistant_id: 메모리/상태 저장을 위한 에이전트 식별자
        tools: 에이전트에 제공할 추가 도구 (기본값: 빈 목록)
        sandbox: 원격 실행을 위한 선택적 샌드박스 백엔드 (예: ModalBackend).
                 None인 경우 로컬 파일시스템 + 셸을 사용합니다.
        sandbox_type: 샌드박스 공급자 유형("modal", "runloop", "daytona").
                     시스템 프롬프트 생성에 사용됩니다.
        system_prompt: 기본 시스템 프롬프트를 재정의합니다. None인 경우
                      sandbox_type 및 assistant_id를 기반으로 생성합니다.
        auto_approve: True인 경우 사람의 확인 없이 모든 도구 호출을 자동으로 승인합니다.
                     자동화된 워크플로에 유용합니다.
        enable_memory: 영구 메모리를 위한 AgentMemoryMiddleware 활성화
        enable_skills: 사용자 정의 에이전트 스킬을 위한 SkillsMiddleware 활성화
        enable_shell: 로컬 셸 실행을 위한 ShellMiddleware 활성화 (로컬 모드에서만)

    Returns:
        (agent_graph, composite_backend)의 2-튜플
        - agent_graph: 실행 준비된 구성된 LangGraph Pregel 인스턴스
        - composite_backend: 파일 작업을 위한 CompositeBackend
    """
    if tools is None:
        tools = []

    # Setup agent directory for persistent memory (if enabled)
    if enable_memory or enable_skills:
        agent_dir = settings.ensure_agent_dir(assistant_id)
        agent_md = agent_dir / "agent.md"
        if not agent_md.exists():
            source_content = get_default_coding_instructions()
            agent_md.write_text(source_content)

    # Skills directories (if enabled)
    skills_dir = None
    project_skills_dir = None
    if enable_skills:
        skills_dir = settings.ensure_user_skills_dir(assistant_id)
        project_skills_dir = settings.get_project_skills_dir()

    # Build middleware stack based on enabled features
    agent_middleware = []

    # CONDITIONAL SETUP: Local vs Remote Sandbox
    if sandbox is None:
        # ========== LOCAL MODE ==========
        composite_backend = CompositeBackend(
            default=FilesystemBackend(),  # Current working directory
            routes={},  # No virtualization - use real paths
        )

        # Add memory middleware
        if enable_memory:
            agent_middleware.append(AgentMemoryMiddleware(settings=settings, assistant_id=assistant_id))

        # Add skills middleware
        if enable_skills:
            agent_middleware.append(
                SkillsMiddleware(
                    skills_dir=skills_dir,
                    assistant_id=assistant_id,
                    project_skills_dir=project_skills_dir,
                )
            )

        # Add shell middleware (only in local mode)
        if enable_shell:
            # Create environment for shell commands
            # Restore user's original LANGSMITH_PROJECT so their code traces separately
            shell_env = os.environ.copy()
            if settings.user_langchain_project:
                shell_env["LANGSMITH_PROJECT"] = settings.user_langchain_project

            agent_middleware.append(
                ShellMiddleware(
                    workspace_root=str(Path.cwd()),
                    env=shell_env,
                )
            )
    else:
        # ========== REMOTE SANDBOX MODE ==========
        composite_backend = CompositeBackend(
            default=sandbox,  # Remote sandbox (ModalBackend, etc.)
            routes={},  # No virtualization
        )

        # Add memory middleware
        if enable_memory:
            agent_middleware.append(AgentMemoryMiddleware(settings=settings, assistant_id=assistant_id))

        # Add skills middleware
        if enable_skills:
            agent_middleware.append(
                SkillsMiddleware(
                    skills_dir=skills_dir,
                    assistant_id=assistant_id,
                    project_skills_dir=project_skills_dir,
                )
            )

        # Note: Shell middleware not used in sandbox mode
        # File operations and execute tool are provided by the sandbox backend

    # Get or use custom system prompt
    if system_prompt is None:
        system_prompt = get_system_prompt(assistant_id=assistant_id, sandbox_type=sandbox_type)

    # Configure interrupt_on based on auto_approve setting
    if auto_approve:
        # No interrupts - all tools run automatically
        interrupt_on = {}
    else:
        # Full HITL for destructive operations
        interrupt_on = _add_interrupt_on()

    # Create the agent
    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        backend=composite_backend,
        middleware=agent_middleware,
        interrupt_on=interrupt_on,
        checkpointer=InMemorySaver(),
    ).with_config(config)
    return agent, composite_backend
