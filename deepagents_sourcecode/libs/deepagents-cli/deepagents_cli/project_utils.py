"""프로젝트 루트 탐지 및 프로젝트별 설정을 위한 유틸리티입니다."""

from pathlib import Path


def find_project_root(start_path: Path | None = None) -> Path | None:
    """`.git` 디렉토리를 기준으로 프로젝트 루트를 찾습니다.

    Walks up the directory tree from start_path (or cwd) looking for a .git
    directory, which indicates the project root.

    Args:
        start_path: Directory to start searching from. Defaults to current working directory.

    Returns:
        Path to the project root if found, None otherwise.
    """
    current = Path(start_path or Path.cwd()).resolve()

    # 디렉토리 트리를 위로 올라가며 탐색
    for parent in [current, *list(current.parents)]:
        git_dir = parent / ".git"
        if git_dir.exists():
            return parent

    return None


def find_project_agent_md(project_root: Path) -> list[Path]:
    """프로젝트 전용 `agent.md` 파일을 찾습니다(복수 가능).

    Checks two locations and returns ALL that exist:
    1. project_root/.deepagents/agent.md
    2. project_root/agent.md

    Both files will be loaded and combined if both exist.

    Args:
        project_root: Path to the project root directory.

    Returns:
        List of paths to project agent.md files (may contain 0, 1, or 2 paths).
    """
    paths = []

    # .deepagents/agent.md 확인(우선)
    deepagents_md = project_root / ".deepagents" / "agent.md"
    if deepagents_md.exists():
        paths.append(deepagents_md)

    # 루트 agent.md 확인(폴백이지만 둘 다 있으면 함께 포함)
    root_md = project_root / "agent.md"
    if root_md.exists():
        paths.append(root_md)

    return paths
