"""에이전트별 장기 메모리를 시스템 프롬프트에 로드하기 위한 미들웨어."""

import contextlib
from collections.abc import Awaitable, Callable
from typing import NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime

from deepagents_cli.config import Settings


class AgentMemoryState(AgentState):
    """에이전트 메모리 미들웨어를 위한 상태."""

    user_memory: NotRequired[str]
    """~/.deepagents/{agent}/의 개인 설정 (모든 곳에 적용됨)."""

    project_memory: NotRequired[str]
    """프로젝트별 컨텍스트 (프로젝트 루트에서 로드됨)."""


class AgentMemoryStateUpdate(TypedDict):
    """에이전트 메모리 미들웨어에 대한 상태 업데이트."""

    user_memory: NotRequired[str]
    """~/.deepagents/{agent}/의 개인 설정 (모든 곳에 적용됨)."""

    project_memory: NotRequired[str]
    """프로젝트별 컨텍스트 (프로젝트 루트에서 로드됨)."""


# Long-term Memory Documentation
# Note: Claude Code loads CLAUDE.md files hierarchically and combines them (not precedence-based):
# - Loads recursively from cwd up to (but not including) root directory
# - Multiple files are combined hierarchically: enterprise → project → user
# - Both [project-root]/CLAUDE.md and [project-root]/.claude/CLAUDE.md are loaded if both exist
# - Files higher in hierarchy load first, providing foundation for more specific memories
# We will follow that pattern for deepagents-cli
LONGTERM_MEMORY_SYSTEM_PROMPT = """

## Long-term Memory

Long-term memory is stored in files on the filesystem and persists across sessions.

**User Memory Location**: `{agent_dir_absolute}` (display: `{agent_dir_display}`)
**Project Memory Location**: {project_memory_info}

The system prompt is loaded from two sources at startup:
1. **User agent.md**: `{agent_dir_absolute}/agent.md` - personal settings that apply everywhere
2. **Project agent.md**: loaded from the project root if available - project-specific instructions

Project-specific agent.md files are loaded from the following locations (combined if both exist):
- `[project-root]/.deepagents/agent.md` (preferred)
- `[project-root]/agent.md` (fallback, included if both exist)

**When you should check/read memory (IMPORTANT - do this first):**
- **At the start of every new session**: Check both user and project memory
  - User: `ls {agent_dir_absolute}`
  - Project: `ls {project_deepagents_dir}` (if inside a project)
- **Before answering a question**: If asked "What do you know about X?" or "How do I do Y?", check project memory first, then user.
- **When the user asks you to do a task**: Check for project-specific guides or examples.
- **When the user refers to past work**: Search project memory files for relevant context.

**Memory-First Response Pattern:**
1. User asks question -> Check project directory first: `ls {project_deepagents_dir}`
2. If relevant files exist -> Read them: `read_file '{project_deepagents_dir}/[filename]'`
3. If needed, check user memory -> `ls {agent_dir_absolute}`
4. Answer by supplementing general knowledge with stored knowledge.

**When you should update memory:**
- **Immediately when the user describes your role or how you should behave**
- **Immediately when the user gives you feedback** - record what went wrong and how to do better in memory.
- When the user explicitly asks you to remember something.
- When patterns or preferences emerge (coding style, conventions, workflow).
- After a significant task where the context would be helpful for future sessions.

**Learning from Feedback:**
- When the user tells you something is better or worse, figure out why and encode it as a pattern.
- Every correction is an opportunity to improve permanently - don't just fix the immediate issue, update your instructions.
- If the user says "You should remember X" or "Pay attention to Y", treat this as highest priority and update memory immediately.
- Look for the underlying principles behind corrections, not just the specific mistakes.

## Deciding Where to Store Memory

When writing or updating agent memory, decide where each fact, configuration, or behavior belongs:

### User Agent File: `{agent_dir_absolute}/agent.md`
-> Describes the agent's **personality, style, and universal behaviors** across all projects.

**Store here:**
- General tone and communication style
- Universal coding preferences (formatting, commenting style, etc.)
- General workflows and methodologies to follow
- Tool usage patterns that apply everywhere
- Personal preferences that don't change between projects

**Examples:**
- "Be concise and direct in your answers"
- "Always use type hints in Python"
- "Prefer functional programming patterns"

### Project Agent File: `{project_deepagents_dir}/agent.md`
-> Describes **how this specific project works** and **how the agent should behave here only**.

**Store here:**
- Project-specific architecture and design patterns
- Coding conventions specific to this codebase
- Project structure and organization
- Testing strategies for this project
- Deployment processes and workflows
- Team conventions and guidelines

**Examples:**
- "This project uses FastAPI with SQLAlchemy"
- "Tests are located in tests/ directory mirroring src structure"
- "All API changes require updating OpenAPI specs"

### Project Memory Files: `{project_deepagents_dir}/*.md`
-> Use for **project-specific reference information** and structured notes.

**Store here:**
- API design documentation
- Architecture decisions and reasoning
- Deployment procedures
- Common debugging patterns
- Onboarding information

**Examples:**
- `{project_deepagents_dir}/api-design.md` - REST API patterns used
- `{project_deepagents_dir}/architecture.md` - System architecture overview
- `{project_deepagents_dir}/deployment.md` - How to deploy this project

### File Operations:

**User Memory:**
```
ls {agent_dir_absolute}                              # List user memory files
read_file '{agent_dir_absolute}/agent.md'            # Read user preferences
edit_file '{agent_dir_absolute}/agent.md' ...        # Update user preferences
```

**Project Memory (Preferred for project-specific info):**
```
ls {project_deepagents_dir}                          # List project memory files
read_file '{project_deepagents_dir}/agent.md'        # Read project guidelines
edit_file '{project_deepagents_dir}/agent.md' ...    # Update project guidelines
write_file '{project_deepagents_dir}/agent.md' ...  # Create project memory file
```

**IMPORTANT**:
- Project memory files are stored in `.deepagents/` inside the project root.
- Always use absolute paths for file operations.
- Determine if info is project-specific (check user vs project memory) before answering."""


DEFAULT_MEMORY_SNIPPET = """<user_memory>
{user_memory}
</user_memory>

<project_memory>
{project_memory}
</project_memory>"""


class AgentMemoryMiddleware(AgentMiddleware):
    """에이전트별 장기 메모리를 로드하기 위한 미들웨어.

    이 미들웨어는 파일(agent.md)에서 에이전트의 장기 메모리를 로드하고
    시스템 프롬프트에 주입합니다. 메모리는 대화 시작 시 한 번 로드되어
    상태에 저장됩니다.
    """

    state_schema = AgentMemoryState

    def __init__(
        self,
        *,
        settings: Settings,
        assistant_id: str,
        system_prompt_template: str | None = None,
    ) -> None:
        """에이전트 메모리 미들웨어를 초기화합니다.

        Args:
            settings: 프로젝트 감지 및 경로가 포함된 전역 설정 인스턴스.
            assistant_id: 에이전트 식별자.
            system_prompt_template: 시스템 프롬프트에 에이전트 메모리를 주입하기 위한
                선택적 사용자 정의 템플릿.
        """
        self.settings = settings
        self.assistant_id = assistant_id

        # User paths
        self.agent_dir = settings.get_agent_dir(assistant_id)
        # Store both display path (with ~) and absolute path for file operations
        self.agent_dir_display = f"~/.deepagents/{assistant_id}"
        self.agent_dir_absolute = str(self.agent_dir)

        # Project paths (from settings)
        self.project_root = settings.project_root

        self.system_prompt_template = system_prompt_template or DEFAULT_MEMORY_SNIPPET

    def before_agent(
        self,
        state: AgentMemoryState,
        runtime: Runtime,
    ) -> AgentMemoryStateUpdate:
        """에이전트 실행 전에 파일에서 에이전트 메모리를 로드합니다.

        사용자 agent.md와 프로젝트별 agent.md가 있으면 로드합니다.
        상태에 아직 없는 경우에만 로드합니다.

        사용자 업데이트를 포착하기 위해 매 호출마다 파일 존재 여부를 동적으로 확인합니다.

        Args:
            state: 현재 에이전트 상태.
            runtime: 런타임 컨텍스트.

        Returns:
            user_memory 및 project_memory가 채워진 업데이트된 상태.
        """
        result: AgentMemoryStateUpdate = {}

        # Load user memory if not already in state
        if "user_memory" not in state:
            user_path = self.settings.get_user_agent_md_path(self.assistant_id)
            if user_path.exists():
                with contextlib.suppress(OSError, UnicodeDecodeError):
                    result["user_memory"] = user_path.read_text()

        # Load project memory if not already in state
        if "project_memory" not in state:
            project_path = self.settings.get_project_agent_md_path()
            if project_path and project_path.exists():
                with contextlib.suppress(OSError, UnicodeDecodeError):
                    result["project_memory"] = project_path.read_text()

        return result

    def _build_system_prompt(self, request: ModelRequest) -> str:
        """메모리 섹션이 포함된 전체 시스템 프롬프트를 작성합니다.

        Args:
            request: 상태 및 기본 시스템 프롬프트가 포함된 모델 요청.

        Returns:
            메모리 섹션이 주입된 전체 시스템 프롬프트.
        """
        # Extract memory from state
        state = cast("AgentMemoryState", request.state)
        user_memory = state.get("user_memory")
        project_memory = state.get("project_memory")
        base_system_prompt = request.system_prompt

        # Build project memory info for documentation
        if self.project_root and project_memory:
            project_memory_info = f"`{self.project_root}` (detected)"
        elif self.project_root:
            project_memory_info = f"`{self.project_root}` (no agent.md found)"
        else:
            project_memory_info = "None (not in a git project)"

        # Build project deepagents directory path
        if self.project_root:
            project_deepagents_dir = str(self.project_root / ".deepagents")
        else:
            project_deepagents_dir = "[project-root]/.deepagents (not in a project)"

        # Format memory section with both memories
        memory_section = self.system_prompt_template.format(
            user_memory=user_memory if user_memory else "(No user agent.md)",
            project_memory=project_memory if project_memory else "(No project agent.md)",
        )

        system_prompt = memory_section

        if base_system_prompt:
            system_prompt += "\n\n" + base_system_prompt

        system_prompt += "\n\n" + LONGTERM_MEMORY_SYSTEM_PROMPT.format(
            agent_dir_absolute=self.agent_dir_absolute,
            agent_dir_display=self.agent_dir_display,
            project_memory_info=project_memory_info,
            project_deepagents_dir=project_deepagents_dir,
        )

        return system_prompt

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """시스템 프롬프트에 에이전트 메모리를 주입합니다.

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 핸들러 함수.

        Returns:
            핸들러의 모델 응답.
        """
        system_prompt = self._build_system_prompt(request)
        return handler(request.override(system_prompt=system_prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(비동기) 시스템 프롬프트에 에이전트 메모리를 주입합니다.

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 핸들러 함수.

        Returns:
            핸들러의 모델 응답.
        """
        system_prompt = self._build_system_prompt(request)
        return await handler(request.override(system_prompt=system_prompt))
