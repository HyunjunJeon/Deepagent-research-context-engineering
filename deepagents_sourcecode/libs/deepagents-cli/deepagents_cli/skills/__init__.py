"""deepagents CLI를 위한 Skills 모듈.

공개 API:
- SkillsMiddleware: 기술을 에이전트 실행에 통합하기 위한 미들웨어
- execute_skills_command: 기술 하위 명령(list/create/info) 실행
- setup_skills_parser: 기술 명령을 위한 argparse 설정

기타 모든 구성 요소는 내부 구현 세부 사항입니다.
"""

from deepagents_cli.skills.commands import (
    execute_skills_command,
    setup_skills_parser,
)
from deepagents_cli.skills.middleware import SkillsMiddleware

__all__ = [
    "SkillsMiddleware",
    "execute_skills_command",
    "setup_skills_parser",
]
