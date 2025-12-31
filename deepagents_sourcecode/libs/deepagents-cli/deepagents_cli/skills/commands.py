"""기술 관리를 위한 CLI 명령.

이 명령들은 cli.py를 통해 CLI에 등록됩니다:
- deepagents skills list --agent <agent> [--project]
- deepagents skills create <name>
- deepagents skills info <name>
"""

import argparse
import re
from pathlib import Path
from typing import Any

from deepagents_cli.config import COLORS, Settings, console
from deepagents_cli.skills.load import MAX_SKILL_NAME_LENGTH, list_skills


def _validate_name(name: str) -> tuple[bool, str]:
    """Agent Skills 사양에 따라 이름을 검증합니다.

    요구 사항 (https://agentskills.io/specification):
    - 최대 64자
    - 소문자 영숫자와 하이픈만 허용 (a-z, 0-9, -)
    - 하이픈으로 시작하거나 끝날 수 없음
    - 연속된 하이픈 허용 안 함
    - 경로 탐색 시퀀스 허용 안 함

    Args:
        name: 검증할 이름

    Returns:
        (유효 여부, 오류 메시지) 튜플. 유효한 경우 오류 메시지는 비어 있습니다.
    """
    # 비어 있거나 공백만 있는 이름 확인
    if not name or not name.strip():
        return False, "비어 있을 수 없습니다"

    # 길이 확인 (사양: 최대 64자)
    if len(name) > MAX_SKILL_NAME_LENGTH:
        return False, "64자를 초과할 수 없습니다"

    # 경로 탐색 시퀀스 확인
    if ".." in name or "/" in name or "\\" in name:
        return False, "경로 요소를 포함할 수 없습니다"

    # 사양: 소문자 영숫자와 하이픈만 허용
    # 패턴 보장: 시작/종료 하이픈 없음, 연속 하이픈 없음
    if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", name):
        return (
            False,
            "소문자, 숫자, 하이픈만 사용해야 합니다 (대문자, 밑줄 불가능, 하이픈으로 시작하거나 끝날 수 없음)",
        )

    return True, ""


def _validate_skill_path(skill_dir: Path, base_dir: Path) -> tuple[bool, str]:
    """해결된 기술 디렉토리가 기본 디렉토리 내에 있는지 확인합니다.

    Args:
        skill_dir: 검증할 기술 디렉토리 경로
        base_dir: skill_dir을 포함해야 하는 기본 기술 디렉토리

    Returns:
        (유효 여부, 오류 메시지) 튜플. 유효한 경우 오류 메시지는 비어 있습니다.
    """
    try:
        # 두 경로를 정식 형식으로 해결
        resolved_skill = skill_dir.resolve()
        resolved_base = base_dir.resolve()

        # skill_dir이 base_dir 내에 있는지 확인
        # Python 3.9+인 경우 is_relative_to 사용, 그렇지 않으면 문자열 비교 사용
        if hasattr(resolved_skill, "is_relative_to"):
            if not resolved_skill.is_relative_to(resolved_base):
                return False, f"기술 디렉토리는 {base_dir} 내에 있어야 합니다"
        else:
            # 이전 Python 버전을 위한 폴백
            try:
                resolved_skill.relative_to(resolved_base)
            except ValueError:
                return False, f"기술 디렉토리는 {base_dir} 내에 있어야 합니다"

        return True, ""
    except (OSError, RuntimeError) as e:
        return False, f"잘못된 경로: {e}"


def _list(agent: str, *, project: bool = False) -> None:
    """지정된 에이전트에 대해 사용 가능한 모든 기술을 나열합니다.

    Args:
        agent: 기술을 위한 에이전트 식별자 (기본값: agent).
        project: True인 경우 프로젝트 기술만 표시합니다.
            False인 경우 모든 기술(사용자 + 프로젝트)을 표시합니다.
    """
    settings = Settings.from_environment()
    user_skills_dir = settings.get_user_skills_dir(agent)
    project_skills_dir = settings.get_project_skills_dir()

    # --project 플래그가 사용된 경우 프로젝트 기술만 표시
    if project:
        if not project_skills_dir:
            console.print("[yellow]프로젝트 디렉토리가 아닙니다.[/yellow]")
            console.print(
                "[dim]프로젝트 기술을 사용하려면 프로젝트 루트에 .git 디렉토리가 필요합니다.[/dim]",
                style=COLORS["dim"],
            )
            return

        if not project_skills_dir.exists() or not any(project_skills_dir.iterdir()):
            console.print("[yellow]프로젝트 기술을 찾을 수 없습니다.[/yellow]")
            console.print(
                f"[dim]프로젝트 기술을 추가하면 {project_skills_dir}/ 에 생성됩니다.[/dim]",
                style=COLORS["dim"],
            )
            console.print(
                "\n[dim]프로젝트 기술 생성:\n  deepagents skills create my-skill --project[/dim]",
                style=COLORS["dim"],
            )
            return

        skills = list_skills(user_skills_dir=None, project_skills_dir=project_skills_dir)
        console.print("\n[bold]프로젝트 기술:[/bold]\n", style=COLORS["primary"])
    else:
        # 사용자 및 프로젝트 기술 모두 로드
        skills = list_skills(user_skills_dir=user_skills_dir, project_skills_dir=project_skills_dir)

        if not skills:
            console.print("[yellow]기술을 찾을 수 없습니다.[/yellow]")
            console.print(
                "[dim]기술을 추가하면 ~/.deepagents/agent/skills/ 에 생성됩니다.[/dim]",
                style=COLORS["dim"],
            )
            console.print(
                "\n[dim]첫 번째 기술 생성:\n  deepagents skills create my-skill[/dim]",
                style=COLORS["dim"],
            )
            return

        console.print("\n[bold]사용 가능한 기술:[/bold]\n", style=COLORS["primary"])

    # 출처별로 기술 그룹화
    user_skills = [s for s in skills if s["source"] == "user"]
    project_skills_list = [s for s in skills if s["source"] == "project"]

    # 사용자 기술 표시
    if user_skills and not project:
        console.print("[bold cyan]사용자 기술:[/bold cyan]", style=COLORS["primary"])
        for skill in user_skills:
            skill_path = Path(skill["path"])
            console.print(f"  • [bold]{skill['name']}[/bold]", style=COLORS["primary"])
            console.print(f"    {skill['description']}", style=COLORS["dim"])
            console.print(f"    위치: {skill_path.parent}/", style=COLORS["dim"])
            console.print()

    # 프로젝트 기술 표시
    if project_skills_list:
        if not project and user_skills:
            console.print()
        console.print("[bold green]프로젝트 기술:[/bold green]", style=COLORS["primary"])
        for skill in project_skills_list:
            skill_path = Path(skill["path"])
            console.print(f"  • [bold]{skill['name']}[/bold]", style=COLORS["primary"])
            console.print(f"    {skill['description']}", style=COLORS["dim"])
            console.print(f"    위치: {skill_path.parent}/", style=COLORS["dim"])
            console.print()


def _create(skill_name: str, agent: str, project: bool = False) -> None:
    """템플릿 SKILL.md 파일을 사용하여 새 기술을 생성합니다.

    Args:
        skill_name: 생성할 기술의 이름.
        agent: 기술을 위한 에이전트 식별자
        project: True인 경우 프로젝트 기술 디렉토리에 생성합니다.
            False인 경우 사용자 기술 디렉토리에 생성합니다.
    """
    # 기술 이름 먼저 검증 (Agent Skills 사양에 따름)
    is_valid, error_msg = _validate_name(skill_name)
    if not is_valid:
        console.print(f"[bold red]오류:[/bold red] 잘못된 기술 이름: {error_msg}")
        console.print(
            "[dim]Agent Skills 사양에 따라: 이름은 소문자 영숫자와 하이픈만 사용해야 합니다.\n"
            "예시: web-research, code-review, data-analysis[/dim]",
            style=COLORS["dim"],
        )
        return

    # 대상 디렉토리 결정
    settings = Settings.from_environment()
    if project:
        if not settings.project_root:
            console.print("[bold red]오류:[/bold red] 프로젝트 디렉토리가 아닙니다.")
            console.print(
                "[dim]프로젝트 기술을 사용하려면 프로젝트 루트에 .git 디렉토리가 필요합니다.[/dim]",
                style=COLORS["dim"],
            )
            return
        skills_dir = settings.ensure_project_skills_dir()
    else:
        skills_dir = settings.ensure_user_skills_dir(agent)

    skill_dir = skills_dir / skill_name

    # 해결된 경로가 skills_dir 내에 있는지 확인
    is_valid_path, path_error = _validate_skill_path(skill_dir, skills_dir)
    if not is_valid_path:
        console.print(f"[bold red]오류:[/bold red] {path_error}")
        return

    if skill_dir.exists():
        console.print(f"[bold red]오류:[/bold red] '{skill_name}' 기술이 이미 {skill_dir} 에 존재합니다")
        return

    # 기술 디렉토리 생성
    skill_dir.mkdir(parents=True, exist_ok=True)

    # 템플릿 SKILL.md 생성 (사양: https://agentskills.io/specification)
    template = f"""---
name: {skill_name}
description: 이 기술이 수행하는 작업과 사용 시기에 대한 간략한 설명.
# Agent Skills 사양에 따른 선택적 필드:
# license: Apache-2.0
# compatibility: Designed for deepagents CLI
# metadata:
#   author: your-org
#   version: "1.0"
# allowed-tools: Bash(git:*) Read
---

# {skill_name.title().replace("-", " ")} 기술

## 설명

[이 기술이 수행하는 작업과 사용해야 하는 시기에 대한 자세한 설명을 제공하십시오]

## 사용 시기

- [시나리오 1: 사용자가 ...를 요청할 때]
- [시나리오 2: ...가 필요할 때]
- [시나리오 3: 태스크에 ...가 포함될 때]

## 사용 방법

### 1단계: [첫 번째 작업]
[먼저 수행할 작업을 설명하십시오]

### 2단계: [두 번째 작업]
[다음에 수행할 작업을 설명하십시오]

### 3단계: [최종 작업]
[태스크를 완료하는 방법을 설명하십시오]

## 권장 사항

- [권장 사항 1]
- [권장 사항 2]
- [권장 사항 3]

## 지원 파일

이 기술 디렉토리에는 지침에서 참조하는 지원 파일이 포함될 수 있습니다:
- `helper.py` - 자동화를 위한 Python 스크립트
- `config.json` - 설정 파일
- `reference.md` - 추가 참조 문서

## 예시

### 예시 1: [시나리오 이름]

**사용자 요청:** "[사용자 요청 예시]"

**접근 방식:**
1. [단계별 분석]
2. [도구 및 명령 사용]
3. [예상 결과]

### 예시 2: [다른 시나리오]

**사용자 요청:** "[다른 예시]"

**접근 방식:**
1. [다른 접근 방식]
2. [관련 명령]
3. [예상 결과]

## 참고 사항

- [추가 팁, 경고 또는 컨텍스트]
- [알려진 제한 사항 또는 예외 케이스]
- [도움이 되는 외부 리소스 링크]
"""

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(template)

    console.print(f"✓ '{skill_name}' 기술이 성공적으로 생성되었습니다!", style=COLORS["primary"])
    console.print(f"위치: {skill_dir}\n", style=COLORS["dim"])
    console.print(
        "[dim]SKILL.md 파일을 편집하여 사용자 정의하십시오:\n"
        "  1. YAML frontmatter에서 설명을 업데이트하십시오\n"
        "  2. 지침과 예시를 채우십시오\n"
        "  3. 지원 파일(스크립트, 설정 등)을 추가하십시오\n"
        "\n"
        f"  nano {skill_md}\n"
        "\n"
        "💡 기술 예시는 deepagents 저장소의 examples/skills/ 를 참조하십시오:\n"
        "   - web-research: 구조화된 연구 워크플로우\n"
        "   - langgraph-docs: LangGraph 문서 조회\n"
        "\n"
        "   예시 복사: cp -r examples/skills/web-research ~/.deepagents/agent/skills/\n",
        style=COLORS["dim"],
    )


def _info(skill_name: str, *, agent: str = "agent", project: bool = False) -> None:
    """특정 기술에 대한 자세한 정보를 표시합니다.

    Args:
        skill_name: 세부 정보를 표시할 기술의 이름.
        agent: 기술을 위한 에이전트 식별자 (기본값: agent).
        project: True인 경우 프로젝트 기술만 검색합니다. False인 경우 사용자 및 프로젝트 기술 모두에서 검색합니다.
    """
    settings = Settings.from_environment()
    user_skills_dir = settings.get_user_skills_dir(agent)
    project_skills_dir = settings.get_project_skills_dir()

    # --project 플래그에 따라 기술 로드
    if project:
        if not project_skills_dir:
            console.print("[bold red]오류:[/bold red] 프로젝트 디렉토리가 아닙니다.")
            return
        skills = list_skills(user_skills_dir=None, project_skills_dir=project_skills_dir)
    else:
        skills = list_skills(user_skills_dir=user_skills_dir, project_skills_dir=project_skills_dir)

    # 기술 찾기
    skill = next((s for s in skills if s["name"] == skill_name), None)

    if not skill:
        console.print(f"[bold red]오류:[/bold red] '{skill_name}' 기술을 찾을 수 없습니다.")
        console.print("\n[dim]사용 가능한 기술:[/dim]", style=COLORS["dim"])
        for s in skills:
            console.print(f"  - {s['name']}", style=COLORS["dim"])
        return

    # 전체 SKILL.md 파일 읽기
    skill_path = Path(skill["path"])
    skill_content = skill_path.read_text()

    # 출처 레이블 결정
    source_label = "프로젝트 기술" if skill["source"] == "project" else "사용자 기술"
    source_color = "green" if skill["source"] == "project" else "cyan"

    console.print(
        f"\n[bold]기술: {skill['name']}[/bold] [bold {source_color}]({source_label})[/bold {source_color}]\n",
        style=COLORS["primary"],
    )
    console.print(f"[bold]설명:[/bold] {skill['description']}\n", style=COLORS["dim"])
    console.print(f"[bold]위치:[/bold] {skill_path.parent}/\n", style=COLORS["dim"])

    # 지원 파일 나열
    skill_dir = skill_path.parent
    supporting_files = [f for f in skill_dir.iterdir() if f.name != "SKILL.md"]

    if supporting_files:
        console.print("[bold]지원 파일:[/bold]", style=COLORS["dim"])
        for file in supporting_files:
            console.print(f"  - {file.name}", style=COLORS["dim"])
        console.print()

    # 전체 SKILL.md 내용 표시
    console.print("[bold]전체 SKILL.md 내용:[/bold]\n", style=COLORS["primary"])
    console.print(skill_content, style=COLORS["dim"])
    console.print()


def setup_skills_parser(
    subparsers: Any,
) -> argparse.ArgumentParser:
    """모든 하위 명령과 함께 기술 하위 명령 파서를 설정합니다."""
    skills_parser = subparsers.add_parser(
        "skills",
        help="에이전트 기술 관리",
        description="에이전트 기술 관리 - 기술 정보 생성, 나열 및 보기",
    )
    skills_subparsers = skills_parser.add_subparsers(dest="skills_command", help="기술 명령")

    # 기술 목록
    list_parser = skills_subparsers.add_parser(
        "list", help="사용 가능한 모든 기술 나열", description="사용 가능한 모든 기술 나열"
    )
    list_parser.add_argument(
        "--agent",
        default="agent",
        help="기술을 위한 에이전트 식별자 (기본값: agent)",
    )
    list_parser.add_argument(
        "--project",
        action="store_true",
        help="프로젝트 수준 기술만 표시",
    )

    # 기술 생성
    create_parser = skills_subparsers.add_parser(
        "create",
        help="새 기술 생성",
        description="템플릿 SKILL.md 파일을 사용하여 새 기술 생성",
    )
    create_parser.add_argument("name", help="생성할 기술 이름 (예: web-research)")
    create_parser.add_argument(
        "--agent",
        default="agent",
        help="기술을 위한 에이전트 식별자 (기본값: agent)",
    )
    create_parser.add_argument(
        "--project",
        action="store_true",
        help="사용자 디렉토리 대신 프로젝트 디렉토리에 기술 생성",
    )

    # 기술 정보
    info_parser = skills_subparsers.add_parser(
        "info",
        help="기술에 대한 자세한 정보 표시",
        description="특정 기술에 대한 자세한 정보 표시",
    )
    info_parser.add_argument("name", help="정보를 표시할 기술 이름")
    info_parser.add_argument(
        "--agent",
        default="agent",
        help="기술을 위한 에이전트 식별자 (기본값: agent)",
    )
    info_parser.add_argument(
        "--project",
        action="store_true",
        help="프로젝트 기술만 검색",
    )
    return skills_parser


def execute_skills_command(args: argparse.Namespace) -> None:
    """파싱된 인수를 기반으로 기술 하위 명령을 실행합니다.

    Args:
        args: skills_command 속성이 있는 파싱된 명령줄 인수
    """
    # agent 인수 검증
    if args.agent:
        is_valid, error_msg = _validate_name(args.agent)
        if not is_valid:
            console.print(f"[bold red]오류:[/bold red] 잘못된 에이전트 이름: {error_msg}")
            console.print(
                "[dim]에이전트 이름은 영문자, 숫자, 하이픈 및 밑줄만 포함할 수 있습니다.[/dim]",
                style=COLORS["dim"],
            )
            return

    if args.skills_command == "list":
        _list(agent=args.agent, project=args.project)
    elif args.skills_command == "create":
        _create(args.name, agent=args.agent, project=args.project)
    elif args.skills_command == "info":
        _info(args.name, agent=args.agent, project=args.project)
    else:
        # 하위 명령이 제공되지 않은 경우 도움말 표시
        console.print("[yellow]기술 하위 명령을 지정하십시오: list, create, 또는 info[/yellow]")
        console.print("\n[bold]사용법:[/bold]", style=COLORS["primary"])
        console.print("  deepagents skills <command> [options]\n")
        console.print("[bold]사용 가능한 명령:[/bold]", style=COLORS["primary"])
        console.print("  list              사용 가능한 모든 기술 나열")
        console.print("  create <name>     새 기술 생성")
        console.print("  info <name>       기술에 대한 자세한 정보 표시")
        console.print("\n[bold]예시:[/bold]", style=COLORS["primary"])
        console.print("  deepagents skills list")
        console.print("  deepagents skills create web-research")
        console.print("  deepagents skills info web-research")
        console.print("\n[dim]특정 명령에 대한 추가 도움말:[/dim]", style=COLORS["dim"])
        console.print("  deepagents skills <command> --help", style=COLORS["dim"])


__all__ = [
    "execute_skills_command",
    "setup_skills_parser",
]
