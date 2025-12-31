"""Middleware for loading and exposing agent skills to the system prompt.

This middleware implements Anthropic's "Agent Skills" pattern via progressive disclosure:
1. At session start, parse YAML frontmatter from SKILL.md files
2. Inject skill metadata (name + description) into system prompt
3. Agent reads full SKILL.md content when the skill is relevant

Skills directory structure (project-level):
{PROJECT_ROOT}/skills/
├── web-research/
│   ├── SKILL.md        # Required: YAML frontmatter + instructions
│   └── helper.py       # Optional: supporting files
├── code-review/
│   ├── SKILL.md
│   └── checklist.md

Adapted from deepagents-cli for use in research_agent project.
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime

from research_agent.skills.load import SkillMetadata, list_skills


class SkillsState(AgentState):
    """State for skills middleware."""

    skills_metadata: NotRequired[list[SkillMetadata]]
    """List of loaded skill metadata (name, description, path)."""


class SkillsStateUpdate(TypedDict):
    """State update for skills middleware."""

    skills_metadata: list[SkillMetadata]
    """List of loaded skill metadata (name, description, path)."""


# Skills system documentation template
SKILLS_SYSTEM_PROMPT = """

## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

{skills_locations}

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a **progressive disclosure** pattern. You know that skills exist (name + description above), but you only read full instructions when needed:

1. **Identify when a skill applies**: Check if the user's task matches a skill's description.
2. **Read the skill's full instructions**: The skill list above shows exact paths for use with read_file.
3. **Follow the skill's instructions**: SKILL.md contains step-by-step workflows, recommendations, and examples.
4. **Access supporting files**: Skills may include Python scripts, configs, or reference docs. Use absolute paths.

**When to use skills:**
- When the user's request matches a skill's domain (e.g., "research X" → web-research skill)
- When specialized knowledge or structured workflows would help
- When the skill provides proven patterns for complex tasks

**Skills are self-documenting:**
- Each SKILL.md tells you exactly what the skill does and how to use it.
- The skill list above shows full paths to each skill's SKILL.md file.

**Running skill scripts:**
Skills may include Python scripts or other executables. Always use the absolute paths from the skill list.

**Workflow example:**

User: "Can you research the latest developments in quantum computing?"

1. Check available skills above → find "web-research" skill with full path
2. Read the skill using the path shown in the list
3. Follow the skill's research workflow (plan → save → delegate → synthesize)
4. Use helper scripts with absolute paths

Note: Skills are tools that make you more capable and consistent. When in doubt, check if there's a skill for the task!
"""


class SkillsMiddleware(AgentMiddleware):
    """Middleware for loading and exposing agent skills.

    This middleware implements Anthropic's Agent Skills pattern:
    - At session start: load skill metadata (name, description) from YAML frontmatter
    - Inject skill list into system prompt for discoverability
    - Agent reads full SKILL.md content when skill is relevant (progressive disclosure)

    Supports both user-level and project-level skills:
    - Project skills: {PROJECT_ROOT}/skills/
    - Project skills override user skills with the same name

    Args:
        skills_dir: Path to user-level skills directory (agent-specific).
        assistant_id: Agent identifier for path references in prompt.
        project_skills_dir: Optional path to project-level skills directory.
    """

    state_schema = SkillsState

    def __init__(
        self,
        *,
        skills_dir: str | Path,
        assistant_id: str,
        project_skills_dir: str | Path | None = None,
    ) -> None:
        """Initialize skills middleware.

        Args:
            skills_dir: Path to user-level skills directory.
            assistant_id: Agent identifier.
            project_skills_dir: Optional path to project-level skills directory.
        """
        self.skills_dir = Path(skills_dir).expanduser()
        self.assistant_id = assistant_id
        self.project_skills_dir = (
            Path(project_skills_dir).expanduser() if project_skills_dir else None
        )
        # Store paths for prompt display
        self.user_skills_display = f"~/.deepagents/{assistant_id}/skills"
        self.system_prompt_template = SKILLS_SYSTEM_PROMPT

    def _format_skills_locations(self) -> str:
        """Format skill locations for system prompt display."""
        locations = [f"**User Skills**: `{self.user_skills_display}`"]
        if self.project_skills_dir:
            locations.append(
                f"**Project Skills**: `{self.project_skills_dir}` (overrides user skills)"
            )
        return "\n".join(locations)

    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        """Format skill metadata for system prompt display."""
        if not skills:
            locations = [f"{self.user_skills_display}/"]
            if self.project_skills_dir:
                locations.append(f"{self.project_skills_dir}/")
            return f"(No skills available. You can create skills in {' or '.join(locations)})"

        # Group skills by source
        user_skills = [s for s in skills if s["source"] == "user"]
        project_skills = [s for s in skills if s["source"] == "project"]

        lines = []

        # Display user skills
        if user_skills:
            lines.append("**User Skills:**")
            for skill in user_skills:
                lines.append(f"- **{skill['name']}**: {skill['description']}")
                lines.append(f"  → To read full instructions: `{skill['path']}`")
            lines.append("")

        # Display project skills
        if project_skills:
            lines.append("**Project Skills:**")
            for skill in project_skills:
                lines.append(f"- **{skill['name']}**: {skill['description']}")
                lines.append(f"  → To read full instructions: `{skill['path']}`")

        return "\n".join(lines)

    def before_agent(
        self, state: SkillsState, runtime: Runtime
    ) -> SkillsStateUpdate | None:
        """Load skill metadata before agent execution.

        This runs once at session start to discover available skills from
        both user-level and project-level directories.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with skills_metadata populated.
        """
        # Reload skills on each interaction to catch directory changes
        skills = list_skills(
            user_skills_dir=self.skills_dir,
            project_skills_dir=self.project_skills_dir,
        )
        return SkillsStateUpdate(skills_metadata=skills)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject skill documentation into system prompt.

        This runs before every model call to ensure skill information is available.

        Args:
            request: Model request being processed.
            handler: Handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        # Get skill metadata from state
        skills_metadata = request.state.get("skills_metadata", [])

        # Format skill locations and list
        skills_locations = self._format_skills_locations()
        skills_list = self._format_skills_list(skills_metadata)

        # Format skill documentation
        skills_section = self.system_prompt_template.format(
            skills_locations=skills_locations,
            skills_list=skills_list,
        )

        if request.system_prompt:
            system_prompt = request.system_prompt + "\n\n" + skills_section
        else:
            system_prompt = skills_section

        return handler(request.override(system_prompt=system_prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(Async) Inject skill documentation into system prompt.

        Args:
            request: Model request being processed.
            handler: Handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        # State is guaranteed to be SkillsState due to state_schema
        state = cast("SkillsState", request.state)
        skills_metadata = state.get("skills_metadata", [])

        # Format skill locations and list
        skills_locations = self._format_skills_locations()
        skills_list = self._format_skills_list(skills_metadata)

        # Format skill documentation
        skills_section = self.system_prompt_template.format(
            skills_locations=skills_locations,
            skills_list=skills_list,
        )

        # Inject into system prompt
        if request.system_prompt:
            system_prompt = request.system_prompt + "\n\n" + skills_section
        else:
            system_prompt = skills_section

        return await handler(request.override(system_prompt=system_prompt))
