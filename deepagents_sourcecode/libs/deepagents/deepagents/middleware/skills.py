"""에이전트 스킬(skills)을 로드하고 system prompt에 노출하는 미들웨어입니다.

이 모듈은 Anthropic의 agent skills 패턴(점진적 공개, progressive disclosure)을 구현하며,
backend 스토리지로부터 스킬을 로드하기 위해 “소스(sources)”를 설정할 수 있게 합니다.

## 아키텍처

스킬은 backend 내에서 스킬들이 정리되어 있는 경로(prefix)인 **sources**로부터 로드됩니다.
sources는 지정된 순서대로 로드되며, 동일한 스킬 이름이 충돌할 경우 뒤에 오는 source가 우선합니다
(last one wins). 이를 통해 `base -> user -> project -> team` 같은 레이어링이 가능합니다.

이 미들웨어는 backend API만 사용하며(직접 파일시스템 접근 없음), 따라서 filesystem/state/remote storage 등
다양한 backend 구현에 이식(portable) 가능합니다.

StateBackend(ephemeral/in-memory)를 사용할 때는 팩토리 함수를 전달하세요.
```python
SkillsMiddleware(backend=lambda rt: StateBackend(rt), ...)
```

## 스킬 디렉토리 구조

각 스킬은 YAML frontmatter가 포함된 `SKILL.md`를 가진 디렉토리입니다.

```
/skills/user/web-research/
├── SKILL.md          # 필수: YAML frontmatter + Markdown 지침
└── helper.py         # 선택: 보조 파일(스크립트/데이터 등)
```

`SKILL.md` 형식 예시:
```markdown
---
name: web-research
description: Structured approach to conducting thorough web research
license: MIT
---

# Web Research Skill

## When to Use
- User asks you to research a topic
...
```

## 스킬 메타데이터(SkillMetadata)

YAML frontmatter에서 Agent Skills 사양에 따라 파싱되는 필드:
- `name`: 스킬 식별자(최대 64자, 소문자 영숫자+하이픈)
- `description`: 스킬 설명(최대 1024자)
- `path`: backend 내 `SKILL.md`의 경로
- Optional: `license`, `compatibility`, `metadata`, `allowed_tools`

## 소스(Sources)

source는 backend 내 “스킬 디렉토리들의 루트 경로”입니다.
source의 표시 이름은 경로의 마지막 컴포넌트로부터 유도됩니다(예: `"/skills/user/" -> "user"`).

```python
[
    "/skills/user/",
    "/skills/project/",
]
```

## 경로 규칙

모든 경로는 `PurePosixPath`를 통해 POSIX 표기(슬래시 `/`)를 사용합니다.
- backend 경로 예: `"/skills/user/web-research/SKILL.md"`
- 플랫폼 독립적인 가상 경로(virtual path)
- 실제 플랫폼별 변환은 backend가 필요 시 처리합니다.

## 사용 예시

```python
from deepagents.backends.state import StateBackend
from deepagents.middleware.skills import SkillsMiddleware

middleware = SkillsMiddleware(
    backend=my_backend,
    sources=[
        "/skills/base/",
        "/skills/user/",
        "/skills/project/",
    ],
)
```
"""

from __future__ import annotations

import logging
import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Annotated

import yaml
from langchain.agents.middleware.types import PrivateStateAttr

if TYPE_CHECKING:
    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

from collections.abc import Awaitable, Callable
from typing import NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# 보안: DoS 공격을 방지하기 위한 SKILL.md 최대 크기(10MB)
MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024

# Agent Skills specification constraints (https://agentskills.io/specification)
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024


class SkillMetadata(TypedDict):
    """Agent Skills 사양(https://agentskills.io/specification)에 따른 스킬 메타데이터입니다."""

    name: str
    """스킬 식별자(최대 64자, 소문자 영숫자 및 하이픈)."""

    description: str
    """스킬 설명(최대 1024자)."""

    path: str
    """`SKILL.md` 파일의 경로."""

    license: str | None
    """라이선스 이름 또는 번들된 라이선스 파일에 대한 참조."""

    compatibility: str | None
    """환경 요구사항(최대 500자)."""

    metadata: dict[str, str]
    """추가 메타데이터를 위한 임의의 key-value 맵."""

    allowed_tools: list[str]
    """사전 승인된 도구 목록(공백 구분). (실험적)"""


class SkillsState(AgentState):
    """SkillsMiddleware의 state 스키마입니다."""

    skills_metadata: NotRequired[Annotated[list[SkillMetadata], PrivateStateAttr]]
    """설정된 모든 source에서 로드된 스킬 메타데이터 목록."""


class SkillsStateUpdate(TypedDict):
    """SkillsMiddleware의 state 업데이트 타입입니다."""

    skills_metadata: list[SkillMetadata]
    """state에 병합할 스킬 메타데이터 목록."""


def _validate_skill_name(name: str, directory_name: str) -> tuple[bool, str]:
    """Agent Skills 사양에 따라 스킬 이름을 검증합니다.

    Requirements per spec:
    - Max 64 characters
    - Lowercase alphanumeric and hyphens only (a-z, 0-9, -)
    - Cannot start or end with hyphen
    - No consecutive hyphens
    - Must match parent directory name

    Args:
        name: Skill name from YAML frontmatter
        directory_name: Parent directory name

    Returns:
        (is_valid, error_message) tuple. Error message is empty if valid.
    """
    if not name:
        return False, "name is required"
    if len(name) > MAX_SKILL_NAME_LENGTH:
        return False, "name exceeds 64 characters"
    # Pattern: lowercase alphanumeric, single hyphens between segments, no start/end hyphen
    if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", name):
        return False, "name must be lowercase alphanumeric with single hyphens only"
    if name != directory_name:
        return False, f"name '{name}' must match directory name '{directory_name}'"
    return True, ""


def _parse_skill_metadata(
    content: str,
    skill_path: str,
    directory_name: str,
) -> SkillMetadata | None:
    """SKILL.md에서 YAML frontmatter를 파싱합니다.

    Extracts metadata per Agent Skills specification from YAML frontmatter delimited
    by --- markers at the start of the content.

    Args:
        content: Content of the SKILL.md file
        skill_path: Path to the SKILL.md file (for error messages and metadata)
        directory_name: Name of the parent directory containing the skill

    Returns:
        SkillMetadata if parsing succeeds, None if parsing fails or validation errors occur
    """
    if len(content) > MAX_SKILL_FILE_SIZE:
        logger.warning("Skipping %s: content too large (%d bytes)", skill_path, len(content))
        return None

    # Match YAML frontmatter between --- delimiters
    frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(frontmatter_pattern, content, re.DOTALL)

    if not match:
        logger.warning("Skipping %s: no valid YAML frontmatter found", skill_path)
        return None

    frontmatter_str = match.group(1)

    # Parse YAML using safe_load for proper nested structure support
    try:
        frontmatter_data = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        logger.warning("Invalid YAML in %s: %s", skill_path, e)
        return None

    if not isinstance(frontmatter_data, dict):
        logger.warning("Skipping %s: frontmatter is not a mapping", skill_path)
        return None

    # Validate required fields
    name = frontmatter_data.get("name")
    description = frontmatter_data.get("description")

    if not name or not description:
        logger.warning("Skipping %s: missing required 'name' or 'description'", skill_path)
        return None

    # Validate name format per spec (warn but continue loading for backwards compatibility)
    is_valid, error = _validate_skill_name(str(name), directory_name)
    if not is_valid:
        logger.warning(
            "Skill '%s' in %s does not follow Agent Skills specification: %s. Consider renaming for spec compliance.",
            name,
            skill_path,
            error,
        )

    # Validate description length per spec (max 1024 chars)
    description_str = str(description).strip()
    if len(description_str) > MAX_SKILL_DESCRIPTION_LENGTH:
        logger.warning(
            "Description exceeds %d characters in %s, truncating",
            MAX_SKILL_DESCRIPTION_LENGTH,
            skill_path,
        )
        description_str = description_str[:MAX_SKILL_DESCRIPTION_LENGTH]

    if frontmatter_data.get("allowed-tools"):
        allowed_tools = frontmatter_data.get("allowed-tools").split(" ")
    else:
        allowed_tools = []

    return SkillMetadata(
        name=str(name),
        description=description_str,
        path=skill_path,
        metadata=frontmatter_data.get("metadata", {}),
        license=frontmatter_data.get("license", "").strip() or None,
        compatibility=frontmatter_data.get("compatibility", "").strip() or None,
        allowed_tools=allowed_tools,
    )


def _list_skills(backend: BackendProtocol, source_path: str) -> list[SkillMetadata]:
    """하나의 source(backend 경로)에서 모든 스킬을 나열합니다.

    Scans backend for subdirectories containing SKILL.md files, downloads their content,
    parses YAML frontmatter, and returns skill metadata.

    Expected structure:
        source_path/
        ├── skill-name/
        │   ├── SKILL.md        # Required
        │   └── helper.py       # Optional

    Args:
        backend: Backend instance to use for file operations
        source_path: Path to the skills directory in the backend

    Returns:
        List of skill metadata from successfully parsed SKILL.md files
    """
    base_path = source_path

    skills: list[SkillMetadata] = []
    items = backend.ls_info(base_path)
    # 스킬 디렉토리 목록(SKILL.md를 담고 있을 수 있는 하위 디렉토리)을 수집
    skill_dirs = []
    for item in items:
        if not item.get("is_dir"):
            continue
        skill_dirs.append(item["path"])

    if not skill_dirs:
        return []

    # 각 스킬 디렉토리마다 SKILL.md 존재 여부를 확인하고 다운로드합니다.
    skill_md_paths = []
    for skill_dir_path in skill_dirs:
        # 안전하고 표준화된 경로 연산을 위해 PurePosixPath로 SKILL.md 경로를 구성합니다.
        skill_dir = PurePosixPath(skill_dir_path)
        skill_md_path = str(skill_dir / "SKILL.md")
        skill_md_paths.append((skill_dir_path, skill_md_path))

    paths_to_download = [skill_md_path for _, skill_md_path in skill_md_paths]
    responses = backend.download_files(paths_to_download)

    # 다운로드된 각 SKILL.md를 파싱합니다.
    for (skill_dir_path, skill_md_path), response in zip(skill_md_paths, responses, strict=True):
        if response.error:
            # SKILL.md가 없는 디렉토리는 스킵
            continue

        if response.content is None:
            logger.warning("Downloaded skill file %s has no content", skill_md_path)
            continue

        try:
            content = response.content.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.warning("Error decoding %s: %s", skill_md_path, e)
            continue

        # PurePosixPath로 디렉토리 이름을 추출합니다.
        directory_name = PurePosixPath(skill_dir_path).name

        # 메타데이터 파싱
        skill_metadata = _parse_skill_metadata(
            content=content,
            skill_path=skill_md_path,
            directory_name=directory_name,
        )
        if skill_metadata:
            skills.append(skill_metadata)

    return skills


async def _alist_skills(backend: BackendProtocol, source_path: str) -> list[SkillMetadata]:
    """하나의 source(backend 경로)에서 모든 스킬을 나열합니다(async 버전).

    Scans backend for subdirectories containing SKILL.md files, downloads their content,
    parses YAML frontmatter, and returns skill metadata.

    Expected structure:
        source_path/
        ├── skill-name/
        │   ├── SKILL.md        # Required
        │   └── helper.py       # Optional

    Args:
        backend: Backend instance to use for file operations
        source_path: Path to the skills directory in the backend

    Returns:
        List of skill metadata from successfully parsed SKILL.md files
    """
    base_path = source_path

    skills: list[SkillMetadata] = []
    items = await backend.als_info(base_path)
    # 스킬 디렉토리 목록(SKILL.md를 담고 있을 수 있는 하위 디렉토리)을 수집
    skill_dirs = []
    for item in items:
        if not item.get("is_dir"):
            continue
        skill_dirs.append(item["path"])

    if not skill_dirs:
        return []

    # 각 스킬 디렉토리마다 SKILL.md 존재 여부를 확인하고 다운로드합니다.
    skill_md_paths = []
    for skill_dir_path in skill_dirs:
        # 안전하고 표준화된 경로 연산을 위해 PurePosixPath로 SKILL.md 경로를 구성합니다.
        skill_dir = PurePosixPath(skill_dir_path)
        skill_md_path = str(skill_dir / "SKILL.md")
        skill_md_paths.append((skill_dir_path, skill_md_path))

    paths_to_download = [skill_md_path for _, skill_md_path in skill_md_paths]
    responses = await backend.adownload_files(paths_to_download)

    # 다운로드된 각 SKILL.md를 파싱합니다.
    for (skill_dir_path, skill_md_path), response in zip(skill_md_paths, responses, strict=True):
        if response.error:
            # SKILL.md가 없는 디렉토리는 스킵
            continue

        if response.content is None:
            logger.warning("Downloaded skill file %s has no content", skill_md_path)
            continue

        try:
            content = response.content.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.warning("Error decoding %s: %s", skill_md_path, e)
            continue

        # PurePosixPath로 디렉토리 이름을 추출합니다.
        directory_name = PurePosixPath(skill_dir_path).name

        # 메타데이터 파싱
        skill_metadata = _parse_skill_metadata(
            content=content,
            skill_path=skill_md_path,
            directory_name=directory_name,
        )
        if skill_metadata:
            skills.append(skill_metadata)

    return skills


SKILLS_SYSTEM_PROMPT = """

## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

{skills_locations}

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a **progressive disclosure** pattern - you see their name and description above, but only read full instructions when needed:

1. **Recognize when a skill applies**: Check if the user's task matches a skill's description
2. **Read the skill's full instructions**: Use the path shown in the skill list above
3. **Follow the skill's instructions**: SKILL.md contains step-by-step workflows, best practices, and examples
4. **Access supporting files**: Skills may include helper scripts, configs, or reference docs - use absolute paths

**When to Use Skills:**
- User's request matches a skill's domain (e.g., "research X" -> web-research skill)
- You need specialized knowledge or structured workflows
- A skill provides proven patterns for complex tasks

**Executing Skill Scripts:**
Skills may contain Python scripts or other executable files. Always use absolute paths from the skill list.

**Example Workflow:**

User: "Can you research the latest developments in quantum computing?"

1. Check available skills -> See "web-research" skill with its path
2. Read the skill using the path shown
3. Follow the skill's research workflow (search -> organize -> synthesize)
4. Use any helper scripts with absolute paths

Remember: Skills make you more capable and consistent. When in doubt, check if a skill exists for the task!
"""


class SkillsMiddleware(AgentMiddleware):
    """에이전트 스킬을 로드하고 system prompt에 노출하는 미들웨어입니다.

    Loads skills from backend sources and injects them into the system prompt
    using progressive disclosure (metadata first, full content on demand).

    Skills are loaded in source order with later sources overriding earlier ones.

    Example:
        ```python
        from deepagents.backends.filesystem import FilesystemBackend

        backend = FilesystemBackend(root_dir="/path/to/skills")
        middleware = SkillsMiddleware(
            backend=backend,
            sources=[
                "/path/to/skills/user/",
                "/path/to/skills/project/",
            ],
        )
        ```

    Args:
        backend: Backend instance for file operations
        sources: List of skill source paths. Source names are derived from the last path component.
    """

    state_schema = SkillsState

    def __init__(self, *, backend: BACKEND_TYPES, sources: list[str]) -> None:
        """스킬 미들웨어를 초기화합니다.

        Args:
            backend: Backend instance or factory function that takes runtime and returns a backend.
                     Use a factory for StateBackend: `lambda rt: StateBackend(rt)`
            sources: List of skill source paths (e.g., ["/skills/user/", "/skills/project/"]).
        """
        self._backend = backend
        self.sources = sources
        self.system_prompt_template = SKILLS_SYSTEM_PROMPT

    def _get_backend(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> BackendProtocol:
        """백엔드 인스턴스/팩토리로부터 실제 백엔드를 해석(resolve)합니다.

        Args:
            state: Current agent state.
            runtime: Runtime context for factory functions.
            config: Runnable config to pass to backend factory.

        Returns:
            Resolved backend instance
        """
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
            backend = self._backend(tool_runtime)
            if backend is None:
                raise AssertionError("SkillsMiddleware requires a valid backend instance")
            return backend

        return self._backend

    def _format_skills_locations(self) -> str:
        """System prompt에 표시할 skills location 섹션을 포맷팅합니다."""
        locations = []
        for i, source_path in enumerate(self.sources):
            name = PurePosixPath(source_path.rstrip("/")).name.capitalize()
            suffix = " (higher priority)" if i == len(self.sources) - 1 else ""
            locations.append(f"**{name} Skills**: `{source_path}`{suffix}")
        return "\n".join(locations)

    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        """System prompt에 표시할 skills 목록을 포맷팅합니다."""
        if not skills:
            paths = [f"{source_path}" for source_path in self.sources]
            return f"(No skills available yet. You can create skills in {' or '.join(paths)})"

        lines = []
        for skill in skills:
            lines.append(f"- **{skill['name']}**: {skill['description']}")
            lines.append(f"  -> Read `{skill['path']}` for full instructions")

        return "\n".join(lines)

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """모델 요청의 system prompt에 skills 섹션을 주입합니다.

        Args:
            request: Model request to modify

        Returns:
            New model request with skills documentation injected into system prompt
        """
        skills_metadata = request.state.get("skills_metadata", [])
        skills_locations = self._format_skills_locations()
        skills_list = self._format_skills_list(skills_metadata)

        skills_section = self.system_prompt_template.format(
            skills_locations=skills_locations,
            skills_list=skills_list,
        )

        if request.system_prompt:
            system_prompt = request.system_prompt + "\n\n" + skills_section
        else:
            system_prompt = skills_section

        return request.override(system_prompt=system_prompt)

    def before_agent(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> SkillsStateUpdate | None:
        """에이전트 실행 전에 스킬 메타데이터를 로드합니다(동기).

        Runs before each agent interaction to discover available skills from all
        configured sources. Re-loads on every call to capture any changes.

        Skills are loaded in source order with later sources overriding
        earlier ones if they contain skills with the same name (last one wins).

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with skills_metadata populated, or None if already present
        """
        # state에 skills_metadata가 이미 있으면(비어 있어도) 스킵
        if "skills_metadata" in state:
            return None

        # backend 해석(인스턴스/팩토리 모두 지원)
        backend = self._get_backend(state, runtime, config)
        all_skills: dict[str, SkillMetadata] = {}

        # source를 순서대로 로드합니다.
        # 뒤에 오는 source가 앞의 source를 덮어씁니다(last one wins).
        for source_path in self.sources:
            source_skills = _list_skills(backend, source_path)
            for skill in source_skills:
                all_skills[skill["name"]] = skill

        skills = list(all_skills.values())
        return SkillsStateUpdate(skills_metadata=skills)

    async def abefore_agent(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> SkillsStateUpdate | None:
        """에이전트 실행 전에 스킬 메타데이터를 로드합니다(async).

        Runs before each agent interaction to discover available skills from all
        configured sources. Re-loads on every call to capture any changes.

        Skills are loaded in source order with later sources overriding
        earlier ones if they contain skills with the same name (last one wins).

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with skills_metadata populated, or None if already present
        """
        # state에 skills_metadata가 이미 있으면(비어 있어도) 스킵
        if "skills_metadata" in state:
            return None

        # backend 해석(인스턴스/팩토리 모두 지원)
        backend = self._get_backend(state, runtime, config)
        all_skills: dict[str, SkillMetadata] = {}

        # source를 순서대로 로드합니다.
        # 뒤에 오는 source가 앞의 source를 덮어씁니다(last one wins).
        for source_path in self.sources:
            source_skills = await _alist_skills(backend, source_path)
            for skill in source_skills:
                all_skills[skill["name"]] = skill

        skills = list(all_skills.values())
        return SkillsStateUpdate(skills_metadata=skills)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """System prompt에 skills 섹션을 주입한 뒤 model call을 수행하도록 감쌉니다.

        Args:
            request: Model request being processed
            handler: Handler function to call with modified request

        Returns:
            Model response from handler
        """
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) System prompt에 skills 섹션을 주입한 뒤 model call을 수행하도록 감쌉니다.

        Args:
            request: Model request being processed
            handler: Async handler function to call with modified request

        Returns:
            Model response from handler
        """
        modified_request = self.modify_request(request)
        return await handler(modified_request)


__all__ = ["SkillMetadata", "SkillsMiddleware"]
