"""딥 리서치 에이전트 예제.

이 모듈은 deepagents 패키지를 사용하여
웹 검색과 전략적 사고를 위한 커스텀 도구를 포함한
리서치 에이전트를 구성하는 방법을 보여준다.
"""

from research_agent.prompts import (
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from research_agent.tools import tavily_search, think_tool

__all__ = [
    "tavily_search",
    "think_tool",
    "RESEARCHER_INSTRUCTIONS",
    "RESEARCH_WORKFLOW_INSTRUCTIONS",
    "SUBAGENT_DELEGATION_INSTRUCTIONS",
]
