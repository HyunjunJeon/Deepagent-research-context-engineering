"""DeepAgent 기반의 리서치 에이전트.

Skills System Integration:
- SkillsMiddleware를 통해 Progressive Disclosure 패턴 구현
- PROJECT_ROOT/skills/ 에서 프로젝트 스킬 로드
- 세션 시작 시 스킬 메타데이터만 로드하여 컨텍스트 절약

Multi-SubAgent System (Claude Code 패턴 기반):
- researcher: 자율적 연구 DeepAgent (넓게 → 깊게 패턴) - CompiledSubAgent
- explorer: 빠른 읽기 전용 탐색 (filesystem tools)
- synthesizer: 연구 결과 통합 (read_file, write_file, think_tool)

SubAgent는 subagent_type 파라미터로 선택되며 각각 격리된 컨텍스트에서 실행됩니다.
"""

from datetime import datetime
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.tools import ToolRuntime
from langchain_openai import ChatOpenAI

from research_agent.prompts import (
    EXPLORER_INSTRUCTIONS,
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
    SYNTHESIZER_INSTRUCTIONS,
)
from research_agent.researcher import (
    get_researcher_subagent,  # NEW: Autonomous researcher
)
from research_agent.skills import SkillsMiddleware
from research_agent.tools import tavily_search, think_tool

# 한도 설정
max_concurrent_research_units = 3
max_researcher_iterations = 3

# 현재 날짜 계산
current_date = datetime.now().strftime("%Y-%m-%d")

# 오케스트레이터용 지침 결합
# NOTE: Researcher는 이제 자율적 DeepAgent로 전환됨 (researcher/ 모듈)
INSTRUCTIONS = (
    RESEARCH_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )
)

# =============================================================================
# SubAgent Definitions
# =============================================================================

# 1. Researcher SubAgent: 자율적 연구 DeepAgent (CompiledSubAgent)
# - researcher/ 모듈에서 get_researcher_subagent()로 동적 생성
# - "넓게 탐색 → 깊게 파기" 패턴의 자체 계획 능력 보유
# - agent 생성 시점에 backend_factory와 함께 호출

# 2. Explorer SubAgent: 빠른 읽기 전용 탐색
# - 용도: 코드베이스/문서의 빠른 탐색 및 패턴 검색
# - 도구: 파일시스템 도구만 사용 (read_file, glob, grep 등)
# - 참고: tools는 비워두고 SubAgentMiddleware가 기본 도구 제공
explorer_agent = {
    "name": "explorer",
    "description": "Fast read-only exploration of codebases and documents. Use for finding files, searching patterns, and quick information retrieval. Cannot modify files.",
    "system_prompt": EXPLORER_INSTRUCTIONS,
    "tools": [],  # SubAgentMiddleware가 기본 filesystem 도구 제공
}

# 3. Synthesizer SubAgent: 연구 결과 통합
# - 용도: 다중 연구 결과를 통합하여 보고서 작성
# - 도구: read_file, write_file, think_tool
synthesizer_agent = {
    "name": "synthesizer",
    "description": "Synthesize multiple research findings into coherent reports. Use for combining sub-agent results, creating summaries, and writing final reports.",
    "system_prompt": SYNTHESIZER_INSTRUCTIONS,
    "tools": [think_tool],  # think_tool
}

# Simple SubAgent 목록 (researcher는 동적으로 추가)
SIMPLE_SUBAGENTS = [explorer_agent, synthesizer_agent]

model = ChatOpenAI(model="gpt-4.1", temperature=0.0)

# Backend 설정

# 1. 로컬 파일 시스템 백엔드 생성 (현재의 부모 디렉터리를 루트로 설정)
BASE_DIR = Path(__file__).resolve().parent.parent
RESEARCH_WORKSPACE_DIR = BASE_DIR / "research_workspace"

fs_backend = FilesystemBackend(
    root_dir=RESEARCH_WORKSPACE_DIR,
    virtual_mode=True,
    max_file_size_mb=20,
)

# 3. Skills 디렉토리 설정 (프로젝트 레벨 스킬만 사용)
PROJECT_SKILLS_DIR = BASE_DIR / "skills"

# SkillsMiddleware 인스턴스 생성
# - Progressive Disclosure: 스킬 메타데이터만 시스템 프롬프트에 주입
# - 에이전트가 필요할 때만 전체 SKILL.md 읽기
skills_middleware = SkillsMiddleware(
    skills_dir=PROJECT_SKILLS_DIR,  # 프로젝트 스킬을 기본으로 사용
    assistant_id="research",
    project_skills_dir=PROJECT_SKILLS_DIR,
)


# 2. CompositeBackend를 factory 함수로 구성 (문서 권장 패턴)
def backend_factory(rt: ToolRuntime):
    """런타임을 받아 CompositeBackend를 생성하는 factory 함수."""
    return CompositeBackend(
        default=StateBackend(rt),  # 기본적으로는 인메모리 상태 사용
        routes={
            "/": fs_backend  # '/'로 시작하는 경로는 로컬 파일 시스템으로 라우팅
        },
    )


# =============================================================================
# Deep Agent 생성
# =============================================================================
#
# 통합 구성:
# 1. SkillsMiddleware: 시스템 프롬프트에 스킬 목록 자동 주입
# 2. Multi-SubAgent: researcher (CompiledSubAgent), explorer, synthesizer
# 3. FilesystemBackend: $PROJECT_ROOT/research_workspace/에 영구 저장

# Researcher는 자율적 DeepAgent (CompiledSubAgent)로 생성
# Backend를 공유하여 중간 결과 저장 가능
researcher_subagent = get_researcher_subagent(
    model=model,
    backend=backend_factory,  # Backend 공유
)

# 전체 SubAgent 목록 구성 (CompiledSubAgent + Simple SubAgents)
ALL_SUBAGENTS = [researcher_subagent, *SIMPLE_SUBAGENTS]

agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool],
    system_prompt=INSTRUCTIONS,
    backend=backend_factory,
    subagents=ALL_SUBAGENTS,  # 다중 전문화 SubAgent (researcher는 CompiledSubAgent)
    middleware=[skills_middleware],
)
