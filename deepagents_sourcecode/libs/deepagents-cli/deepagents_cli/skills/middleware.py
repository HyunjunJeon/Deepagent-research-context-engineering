"""에이전트 기술을 시스템 프롬프트에 로드하고 노출하기 위한 미들웨어.

이 미들웨어는 점진적 노출(progressive disclosure)을 통해 Anthropic의 "Agent Skills" 패턴을 구현합니다:
1. 세션 시작 시 SKILL.md 파일에서 YAML frontmatter 파싱
2. 시스템 프롬프트에 기술 메타데이터(이름 + 설명) 주입
3. 에이전트는 작업과 관련이 있을 때 SKILL.md의 전체 내용을 읽음

기술 디렉토리 구조 (에이전트별 + 프로젝트):
사용자 수준: ~/.deepagents/{AGENT_NAME}/skills/
프로젝트 수준: {PROJECT_ROOT}/.deepagents/skills/

구조 예시:
~/.deepagents/{AGENT_NAME}/skills/
├── web-research/
│   ├── SKILL.md        # 필수: YAML frontmatter + 지침
│   └── helper.py       # 선택 사항: 지원 파일
├── code-review/
│   ├── SKILL.md
│   └── checklist.md

.deepagents/skills/
├── project-specific/
│   └── SKILL.md        # 프로젝트 전용 기술
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

from deepagents_cli.skills.load import SkillMetadata, list_skills


class SkillsState(AgentState):
    """기술 미들웨어를 위한 상태."""

    skills_metadata: NotRequired[list[SkillMetadata]]
    """로드된 기술 메타데이터 목록 (이름, 설명, 경로)."""


class SkillsStateUpdate(TypedDict):
    """기술 미들웨어를 위한 상태 업데이트."""

    skills_metadata: list[SkillMetadata]
    """로드된 기술 메타데이터 목록 (이름, 설명, 경로)."""


# 기술 시스템 문서
SKILLS_SYSTEM_PROMPT = """

## 기술 시스템 (Skills System)

당신은 전문적인 능력과 도메인 지식을 제공하는 기술 라이브러리에 접근할 수 있습니다.

{skills_locations}

**사용 가능한 기술:**

{skills_list}

**기술 사용 방법 (점진적 노출):**

기술은 **점진적 노출(progressive disclosure)** 패턴을 따릅니다. 당신은 기술이 존재한다는 것(위의 이름 + 설명)은 알고 있지만, 필요할 때만 전체 지침을 읽습니다:

1. **기술이 적용되는 시기 파악**: 사용자의 작업이 기술의 설명과 일치하는지 확인하십시오.
2. **기술의 전체 지침 읽기**: 위의 기술 목록은 read_file과 함께 사용할 정확한 경로를 보여줍니다.
3. **기술의 지침 따르기**: SKILL.md에는 단계별 워크플로우, 권장 사항 및 예시가 포함되어 있습니다.
4. **지원 파일 접근**: 기술에는 Python 스크립트, 설정 또는 참조 문서가 포함될 수 있습니다. 절대 경로를 사용하십시오.

**기술을 사용해야 하는 경우:**
- 사용자의 요청이 기술의 도메인과 일치할 때 (예: "X 조사해줘" → web-research 기술)
- 전문 지식이나 구조화된 워크플로우가 필요할 때
- 기술이 복잡한 작업에 대해 검증된 패턴을 제공할 때

**기술은 자체 문서화됨:**
- 각 SKILL.md는 기술이 수행하는 작업과 사용 방법을 정확하게 알려줍니다.
- 위의 기술 목록은 각 기술의 SKILL.md 파일에 대한 전체 경로를 보여줍니다.

**기술 스크립트 실행:**
기술에는 Python 스크립트나 기타 실행 파일이 포함될 수 있습니다. 항상 기술 목록의 절대 경로를 사용하십시오.

**워크플로우 예시:**

사용자: "양자 컴퓨팅의 최신 개발 동향을 조사해 줄 수 있어?"

1. 위에서 사용 가능한 기술 확인 → 전체 경로와 함께 "web-research" 기술 확인
2. 목록에 표시된 경로를 사용하여 기술 읽기
3. 기술의 조사 워크플로우 따르기 (조사 → 정리 → 합성)
4. 절대 경로와 함께 헬퍼 스크립트 사용

주의: 기술은 당신을 더 유능하고 일관성 있게 만드는 도구입니다. 의심스러울 때는 해당 작업에 대한 기술이 있는지 확인하십시오!
"""


class SkillsMiddleware(AgentMiddleware):
    """에이전트 기술을 로드하고 노출하기 위한 미들웨어.

    이 미들웨어는 Anthropic의 에이전트 기술 패턴을 구현합니다:
    - 세션 시작 시 YAML frontmatter에서 기술 메타데이터(이름, 설명)를 로드함
    - 발견 가능성을 위해 시스템 프롬프트에 기술 목록을 주입함
    - 기술이 관련 있을 때 에이전트가 전체 SKILL.md 내용을 읽음 (점진적 노출)

    사용자 수준 및 프로젝트 수준 기술을 모두 지원합니다:
    - 사용자 기술: ~/.deepagents/{AGENT_NAME}/skills/
    - 프로젝트 기술: {PROJECT_ROOT}/.deepagents/skills/
    - 프로젝트 기술은 이름이 같은 사용자 기술을 오버라이드함

    Args:
        skills_dir: 사용자 수준 기술 디렉토리 경로 (에이전트별).
        assistant_id: 프롬프트의 경로 참조를 위한 에이전트 식별자.
        project_skills_dir: 선택적인 프로젝트 수준 기술 디렉토리 경로.
    """

    state_schema = SkillsState

    def __init__(
        self,
        *,
        skills_dir: str | Path,
        assistant_id: str,
        project_skills_dir: str | Path | None = None,
    ) -> None:
        """기술 미들웨어를 초기화합니다.

        Args:
            skills_dir: 사용자 수준 기술 디렉토리 경로.
            assistant_id: 에이전트 식별자.
            project_skills_dir: 선택적인 프로젝트 수준 기술 디렉토리 경로.
        """
        self.skills_dir = Path(skills_dir).expanduser()
        self.assistant_id = assistant_id
        self.project_skills_dir = Path(project_skills_dir).expanduser() if project_skills_dir else None
        # 프롬프트 표시를 위한 경로 저장
        self.user_skills_display = f"~/.deepagents/{assistant_id}/skills"
        self.system_prompt_template = SKILLS_SYSTEM_PROMPT

    def _format_skills_locations(self) -> str:
        """시스템 프롬프트 표시를 위해 기술 위치 형식을 지정합니다."""
        locations = [f"**사용자 기술**: `{self.user_skills_display}`"]
        if self.project_skills_dir:
            locations.append(f"**프로젝트 기술**: `{self.project_skills_dir}` (사용자 기술을 오버라이드함)")
        return "\n".join(locations)

    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        """시스템 프롬프트 표시를 위해 기술 메타데이터 형식을 지정합니다."""
        if not skills:
            locations = [f"{self.user_skills_display}/"]
            if self.project_skills_dir:
                locations.append(f"{self.project_skills_dir}/")
            return f"(현재 사용 가능한 기술이 없습니다. {' 또는 '.join(locations)} 에 기술을 생성할 수 있습니다)"

        # 출처별로 기술 그룹화
        user_skills = [s for s in skills if s["source"] == "user"]
        project_skills = [s for s in skills if s["source"] == "project"]

        lines = []

        # 사용자 기술 표시
        if user_skills:
            lines.append("**사용자 기술:**")
            for skill in user_skills:
                lines.append(f"- **{skill['name']}**: {skill['description']}")
                lines.append(f"  → 전체 지침을 보려면 `{skill['path']}` 읽기")
            lines.append("")

        # 프로젝트 기술 표시
        if project_skills:
            lines.append("**프로젝트 기술:**")
            for skill in project_skills:
                lines.append(f"- **{skill['name']}**: {skill['description']}")
                lines.append(f"  → 전체 지침을 보려면 `{skill['path']}` 읽기")

        return "\n".join(lines)

    def before_agent(self, state: SkillsState, runtime: Runtime) -> SkillsStateUpdate | None:
        """에이전트 실행 전 기술 메타데이터를 로드합니다.

        이는 사용자 수준 및 프로젝트 수준 디렉토리 모두에서 사용 가능한 기술을 검색하기 위해
        세션 시작 시 한 번 실행됩니다.

        Args:
            state: 현재 에이전트 상태.
            runtime: 런타임 컨텍스트.

        Returns:
            skills_metadata가 채워진 업데이트된 상태.
        """
        # 기술 디렉토리의 변경 사항을 포착하기 위해
        # 에이전트와의 매 상호 작용마다 기술을 다시 로드합니다.
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
        """시스템 프롬프트에 기술 문서를 주입합니다.

        이것은 기술 정보가 항상 사용 가능하도록 매 모델 호출 시 실행됩니다.

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 핸들러 함수.

        Returns:
            핸들러의 모델 응답.
        """
        # 상태에서 기술 메타데이터 가져오기
        skills_metadata = request.state.get("skills_metadata", [])

        # 기술 위치 및 목록 형식 지정
        skills_locations = self._format_skills_locations()
        skills_list = self._format_skills_list(skills_metadata)

        # 기술 문서 형식 지정
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
        """(비동기) 시스템 프롬프트에 기술 문서를 주입합니다.

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 핸들러 함수.

        Returns:
            핸들러의 모델 응답.
        """
        # state_schema로 인해 상태는 SkillsState임이 보장됨
        state = cast("SkillsState", request.state)
        skills_metadata = state.get("skills_metadata", [])

        # 기술 위치 및 목록 형식 지정
        skills_locations = self._format_skills_locations()
        skills_list = self._format_skills_list(skills_metadata)

        # 기술 문서 형식 지정
        skills_section = self.system_prompt_template.format(
            skills_locations=skills_locations,
            skills_list=skills_list,
        )

        # 시스템 프롬프트에 주입
        if request.system_prompt:
            system_prompt = request.system_prompt + "\n\n" + skills_section
        else:
            system_prompt = skills_section

        return await handler(request.override(system_prompt=system_prompt))
