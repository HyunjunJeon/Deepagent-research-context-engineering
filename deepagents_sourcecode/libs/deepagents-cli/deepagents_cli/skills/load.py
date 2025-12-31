"""SKILL.md 파일에서 에이전트 기술을 파싱하고 로드하기 위한 기술 로더.

이 모듈은 YAML frontmatter 파싱을 통해 Anthropic의 에이전트 기술 패턴을 구현합니다.
각 기술은 다음을 포함하는 SKILL.md 파일이 있는 디렉토리입니다:
- YAML frontmatter (이름, 설명 필수)
- 에이전트를 위한 마크다운 지침
- 선택적 지원 파일 (스크립트, 설정 등)

SKILL.md 구조 예시:
```markdown
---
name: web-research
description: 철저한 웹 조사를 수행하기 위한 구조화된 접근 방식
---

# 웹 조사 기술

## 사용 시기
- 사용자가 주제 조사를 요청할 때
...
```
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, NotRequired, TypedDict

import yaml

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# SKILL.md 파일의 최대 크기 (10MB)
MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024

# Agent Skills 사양 제약 조건 (https://agentskills.io/specification)
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024


class SkillMetadata(TypedDict):
    """Agent Skills 사양(https://agentskills.io/specification)에 따른 기술 메타데이터."""

    name: str
    """기술 이름 (최대 64자, 소문자 영숫자와 하이픈)."""

    description: str
    """기술이 수행하는 작업에 대한 설명 (최대 1024자)."""

    path: str
    """SKILL.md 파일 경로."""

    source: str
    """기술의 출처 ('user' 또는 'project')."""

    # Agent Skills 사양에 따른 선택적 필드
    license: NotRequired[str | None]
    """라이선스 이름 또는 번들로 제공되는 라이선스 파일에 대한 참조."""

    compatibility: NotRequired[str | None]
    """환경 요구 사항 (최대 500자)."""

    metadata: NotRequired[dict[str, str] | None]
    """추가 메타데이터를 위한 임의의 키-값 매핑."""

    allowed_tools: NotRequired[str | None]
    """사전 승인된 도구의 공백으로 구분된 목록."""


def _is_safe_path(path: Path, base_dir: Path) -> bool:
    """경로가 base_dir 내에 안전하게 포함되어 있는지 확인합니다.

    심볼릭 링크나 경로 조작을 통한 디렉토리 탐색 공격을 방지합니다.
    이 함수는 두 경로를 정식 형식(심볼릭 링크 따름)으로 해결하고,
    대상 경로가 기본 디렉토리 내에 있는지 확인합니다.

    Args:
        path: 검증할 경로
        base_dir: 경로를 포함해야 하는 기본 디렉토리

    Returns:
        경로가 base_dir 내에 안전하게 있으면 True, 그렇지 않으면 False

    예시:
        >>> base = Path("/home/user/.deepagents/skills")
        >>> safe = Path("/home/user/.deepagents/skills/web-research/SKILL.md")
        >>> unsafe = Path("/home/user/.deepagents/skills/../../.ssh/id_rsa")
        >>> _is_safe_path(safe, base)
        True
        >>> _is_safe_path(unsafe, base)
        False
    """
    try:
        # 두 경로를 정식 형식으로 해결 (심볼릭 링크 따름)
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()

        # 해결된 경로가 기본 디렉토리 내에 있는지 확인
        # 이는 기본 디렉토리 외부를 가리키는 심볼릭 링크를 포착함
        resolved_path.relative_to(resolved_base)
        return True
    except ValueError:
        # 경로가 base_dir의 하위가 아님 (디렉토리 외부)
        return False
    except (OSError, RuntimeError):
        # 경로 해결 중 오류 발생 (예: 순환 심볼릭 링크, 너무 많은 수준)
        return False


def _validate_skill_name(name: str, directory_name: str) -> tuple[bool, str]:
    """Agent Skills 사양에 따라 기술 이름을 검증합니다.

    요구 사항:
    - 최대 64자
    - 소문자 영숫자와 하이픈만 허용 (a-z, 0-9, -)
    - 하이픈으로 시작하거나 끝날 수 없음
    - 연속된 하이픈 허용 안 함
    - 상위 디렉토리 이름과 일치해야 함

    Args:
        name: YAML frontmatter의 기술 이름.
        directory_name: 상위 디렉토리 이름.

    Returns:
        (유효 여부, 오류 메시지) 튜플. 유효한 경우 오류 메시지는 비어 있습니다.
    """
    if not name:
        return False, "이름은 필수입니다"
    if len(name) > MAX_SKILL_NAME_LENGTH:
        return False, "이름이 64자를 초과합니다"
    # 패턴: 소문자 영숫자, 세그먼트 사이의 단일 하이픈, 시작/종료 하이픈 없음
    if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", name):
        return False, "이름은 소문자 영숫자와 단일 하이픈만 사용해야 합니다"
    if name != directory_name:
        return False, f"이름 '{name}'은 디렉토리 이름 '{directory_name}'과 일치해야 합니다"
    return True, ""


def _parse_skill_metadata(skill_md_path: Path, source: str) -> SkillMetadata | None:
    """Agent Skills 사양에 따라 SKILL.md 파일에서 YAML frontmatter를 파싱합니다.

    Args:
        skill_md_path: SKILL.md 파일 경로.
        source: 기술 출처 ('user' 또는 'project').

    Returns:
        모든 필드가 포함된 SkillMetadata, 파싱 실패 시 None.
    """
    try:
        # 보안: DoS 공격 방지를 위해 파일 크기 확인
        file_size = skill_md_path.stat().st_size
        if file_size > MAX_SKILL_FILE_SIZE:
            logger.warning("건너뛰는 중 %s: 파일이 너무 큼 (%d 바이트)", skill_md_path, file_size)
            return None

        content = skill_md_path.read_text(encoding="utf-8")

        # --- 구분 기호 사이의 YAML frontmatter 매칭
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if not match:
            logger.warning("건너뛰는 중 %s: 유효한 YAML frontmatter를 찾을 수 없음", skill_md_path)
            return None

        frontmatter_str = match.group(1)

        # 적절한 중첩 구조 지원을 위해 safe_load를 사용하여 YAML 파싱
        try:
            frontmatter_data = yaml.safe_load(frontmatter_str)
        except yaml.YAMLError as e:
            logger.warning("%s의 잘못된 YAML: %s", skill_md_path, e)
            return None

        if not isinstance(frontmatter_data, dict):
            logger.warning("건너뛰는 중 %s: frontmatter가 매핑이 아님", skill_md_path)
            return None

        # 필수 필드 검증
        name = frontmatter_data.get("name")
        description = frontmatter_data.get("description")

        if not name or not description:
            logger.warning("건너뛰는 중 %s: 필수 'name' 또는 'description'이 누락됨", skill_md_path)
            return None

        # 사양에 따라 이름 형식 검증 (경고하지만 하위 호환성을 위해 로드함)
        directory_name = skill_md_path.parent.name
        is_valid, error = _validate_skill_name(str(name), directory_name)
        if not is_valid:
            logger.warning(
                "%s의 '%s' 기술이 Agent Skills 사양을 따르지 않음: %s. "
                "사양을 준수하도록 이름을 변경하는 것을 고려하십시오.",
                skill_md_path,
                name,
                error,
            )

        # 설명 길이 검증 (사양: 최대 1024자)
        description_str = str(description)
        if len(description_str) > MAX_SKILL_DESCRIPTION_LENGTH:
            logger.warning(
                "%s의 설명이 %d자를 초과하여 잘림",
                skill_md_path,
                MAX_SKILL_DESCRIPTION_LENGTH,
            )
            description_str = description_str[:MAX_SKILL_DESCRIPTION_LENGTH]

        return SkillMetadata(
            name=str(name),
            description=description_str,
            path=str(skill_md_path),
            source=source,
            license=frontmatter_data.get("license"),
            compatibility=frontmatter_data.get("compatibility"),
            metadata=frontmatter_data.get("metadata"),
            allowed_tools=frontmatter_data.get("allowed-tools"),
        )

    except (OSError, UnicodeDecodeError) as e:
        logger.warning("%s 읽기 오류: %s", skill_md_path, e)
        return None


def _list_skills(skills_dir: Path, source: str) -> list[SkillMetadata]:
    """단일 기술 디렉토리에서 모든 기술을 나열합니다(내부 헬퍼).

    기술 디렉토리에서 SKILL.md 파일이 포함된 하위 디렉토리를 스캔하고,
    YAML frontmatter를 파싱하여 기술 메타데이터를 반환합니다.

    기술 조직 구성:
    skills/
    ├── skill-name/
    │   ├── SKILL.md        # 필수: YAML frontmatter가 있는 지침
    │   ├── script.py       # 선택 사항: 지원 파일
    │   └── config.json     # 선택 사항: 지원 파일

    Args:
        skills_dir: 기술 디렉토리 경로.
        source: 기술 출처 ('user' 또는 'project').

    Returns:
        이름, 설명, 경로 및 출처가 포함된 기술 메타데이터 딕셔너리 목록.
    """
    # 기술 디렉토리 존재 여부 확인
    skills_dir = skills_dir.expanduser()
    if not skills_dir.exists():
        return []

    # 보안 검사를 위해 기본 디렉토리를 정식 경로로 해결
    try:
        resolved_base = skills_dir.resolve()
    except (OSError, RuntimeError):
        # 기본 디렉토리를 해결할 수 없음, 안전하게 종료
        return []

    skills: list[SkillMetadata] = []

    # 하위 디렉토리 순회
    for skill_dir in skills_dir.iterdir():
        # 보안: 기술 디렉토리 외부를 가리키는 심볼릭 링크 포착
        if not _is_safe_path(skill_dir, resolved_base):
            continue

        if not skill_dir.is_dir():
            continue

        # SKILL.md 파일 찾기
        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            continue

        # 보안: 읽기 전에 SKILL.md 경로가 안전한지 검증
        # 이는 외부를 가리키는 심볼릭 링크인 SKILL.md 파일을 포착함
        if not _is_safe_path(skill_md_path, resolved_base):
            continue

        # 메타데이터 파싱
        metadata = _parse_skill_metadata(skill_md_path, source=source)
        if metadata:
            skills.append(metadata)

    return skills


def list_skills(*, user_skills_dir: Path | None = None, project_skills_dir: Path | None = None) -> list[SkillMetadata]:
    """사용자 및/또는 프로젝트 디렉토리에서 기술을 나열합니다.

    두 디렉토리가 모두 제공되면 사용자 기술과 이름이 동일한 프로젝트 기술이
    사용자 기술을 오버라이드합니다.

    Args:
        user_skills_dir: 사용자 수준 기술 디렉토리 경로.
        project_skills_dir: 프로젝트 수준 기술 디렉토리 경로.

    Returns:
        두 출처의 기술 메타데이터가 병합된 목록이며, 이름이 충돌할 경우
        프로젝트 기술이 사용자 기술보다 우선합니다.
    """
    all_skills: dict[str, SkillMetadata] = {}

    # 사용자 기술 먼저 로드 (기본)
    if user_skills_dir:
        user_skills = _list_skills(user_skills_dir, source="user")
        for skill in user_skills:
            all_skills[skill["name"]] = skill

    # 프로젝트 기술 두 번째로 로드 (오버라이드/확장)
    if project_skills_dir:
        project_skills = _list_skills(project_skills_dir, source="project")
        for skill in project_skills:
            # 프로젝트 기술은 이름이 같은 사용자 기술을 오버라이드함
            all_skills[skill["name"]] = skill

    return list(all_skills.values())
