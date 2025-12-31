"""CLI를 위한 구성, 상수 밎 모델 생성."""

import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import dotenv
from rich.console import Console

from deepagents_cli._version import __version__

dotenv.load_dotenv()

# CRITICAL: Override LANGSMITH_PROJECT to route agent traces to separate project
# LangSmith reads LANGSMITH_PROJECT at invocation time, so we override it here
# and preserve the user's original value for shell commands
_deepagents_project = os.environ.get("DEEPAGENTS_LANGSMITH_PROJECT")
_original_langsmith_project = os.environ.get("LANGSMITH_PROJECT")
if _deepagents_project:
    # Override LANGSMITH_PROJECT for agent traces
    os.environ["LANGSMITH_PROJECT"] = _deepagents_project

# Now safe to import LangChain modules
from langchain_core.language_models import BaseChatModel

# Color scheme
COLORS = {
    "primary": "#10b981",
    "dim": "#6b7280",
    "user": "#ffffff",
    "agent": "#10b981",
    "thinking": "#34d399",
    "tool": "#fbbf24",
}

# ASCII art banner

DEEP_AGENTS_ASCII = f"""
 ██████╗  ███████╗ ███████╗ ██████╗
 ██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗
 ██║  ██║ █████╗   █████╗   ██████╔╝
 ██║  ██║ ██╔══╝   ██╔══╝   ██╔═══╝
 ██████╔╝ ███████╗ ███████╗ ██║
 ╚═════╝  ╚══════╝ ╚══════╝ ╚═╝

  █████╗   ██████╗  ███████╗ ███╗   ██╗ ████████╗ ███████╗
 ██╔══██╗ ██╔════╝  ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██╔════╝
 ███████║ ██║  ███╗ █████╗   ██╔██╗ ██║    ██║    ███████╗
 ██╔══██║ ██║   ██║ ██╔══╝   ██║╚██╗██║    ██║    ╚════██║
 ██║  ██║ ╚██████╔╝ ███████╗ ██║ ╚████║    ██║    ███████║
 ╚═╝  ╚═╝  ╚═════╝  ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚══════╝
                                              v{__version__}
"""

# Interactive commands
# Interactive commands
COMMANDS = {
    "clear": "화면을 지우고 대화를 재설정합니다",
    "help": "도움말 정보를 표시합니다",
    "tokens": "현재 세션의 토큰 사용량을 표시합니다",
    "quit": "CLI를 종료합니다",
    "exit": "CLI를 종료합니다",
}


# Maximum argument length for display
MAX_ARG_LENGTH = 150

# Agent configuration
config = {"recursion_limit": 1000}

# Rich console instance
console = Console(highlight=False)


def _find_project_root(start_path: Path | None = None) -> Path | None:
    """git 디렉터리를 찾아 프로젝트 루트를 찾습니다.

    start_path(또는 cwd)에서 디렉터리 트리를 따라 올라가며 프로젝트 루트를 나타내는
    .git 디렉터리를 찾습니다.

    Args:
        start_path: 검색을 시작할 디렉터리. 기본값은 현재 작업 디렉터리입니다.

    Returns:
        찾은 경우 프로젝트 루트의 경로, 그렇지 않으면 None입니다.
    """
    current = Path(start_path or Path.cwd()).resolve()

    # Walk up the directory tree
    for parent in [current, *list(current.parents)]:
        git_dir = parent / ".git"
        if git_dir.exists():
            return parent

    return None


def _find_project_agent_md(project_root: Path) -> list[Path]:
    """프로젝트별 agent.md 파일(들)을 찾습니다.

    두 위치를 확인하고 존재하는 모든 위치를 반환합니다:
    1. project_root/.deepagents/agent.md
    2. project_root/agent.md

    두 파일이 모두 존재하면 둘 다 로드되어 결합됩니다.

    Args:
        project_root: 프로젝트 루트 디렉터리 경로.

    Returns:
        프로젝트 agent.md 파일 경로 목록 (0, 1 또는 2개의 경로를 포함할 수 있음).
    """
    paths = []

    # Check .deepagents/agent.md (preferred)
    deepagents_md = project_root / ".deepagents" / "agent.md"
    if deepagents_md.exists():
        paths.append(deepagents_md)

    # Check root agent.md (fallback, but also include if both exist)
    root_md = project_root / "agent.md"
    if root_md.exists():
        paths.append(root_md)

    return paths


@dataclass
class Settings:
    """DeepAgents-cli를 위한 전역 설정 및 환경 감지.

    이 클래스는 시작 시 한 번 초기화되며 다음 정보에 대한 액세스를 제공합니다:
    - 사용 가능한 모델 및 API 키
    - 현재 프로젝트 정보
    - 도구 가용성 (예: Tavily)
    - 파일 시스템 경로

    Attributes:
        project_root: 현재 프로젝트 루트 디렉터리 (git 프로젝트 내인 경우)

        openai_api_key: OpenAI API 키 (사용 가능한 경우)
        anthropic_api_key: Anthropic API 키 (사용 가능한 경우)
        tavily_api_key: Tavily API 키 (사용 가능한 경우)
        deepagents_langchain_project: DeepAgents 에이전트 추적을 위한 LangSmith 프로젝트 이름
        user_langchain_project: 환경의 원래 LANGSMITH_PROJECT (사용자 코드용)
    """

    # API keys
    openai_api_key: str | None
    anthropic_api_key: str | None
    google_api_key: str | None
    tavily_api_key: str | None

    # LangSmith configuration
    deepagents_langchain_project: str | None  # For deepagents agent tracing
    user_langchain_project: str | None  # Original LANGSMITH_PROJECT for user code

    # Model configuration
    model_name: str | None = None  # Currently active model name
    model_provider: str | None = None  # Provider (openai, anthropic, google)

    # Project information
    project_root: Path | None = None

    @classmethod
    def from_environment(cls, *, start_path: Path | None = None) -> "Settings":
        """현재 환경을 감지하여 설정을 생성합니다.

        Args:
            start_path: 프로젝트 감지를 시작할 디렉터리(기본값은 cwd)

        Returns:
            감지된 구성이 포함된 Settings 인스턴스
        """
        # Detect API keys
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        google_key = os.environ.get("GOOGLE_API_KEY")
        tavily_key = os.environ.get("TAVILY_API_KEY")

        # Detect LangSmith configuration
        # DEEPAGENTS_LANGSMITH_PROJECT: Project for deepagents agent tracing
        # user_langchain_project: User's ORIGINAL LANGSMITH_PROJECT (before override)
        # Note: LANGSMITH_PROJECT was already overridden at module import time (above)
        # so we use the saved original value, not the current os.environ value
        deepagents_langchain_project = os.environ.get("DEEPAGENTS_LANGSMITH_PROJECT")
        user_langchain_project = _original_langsmith_project  # Use saved original!

        # Detect project
        project_root = _find_project_root(start_path)

        return cls(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            google_api_key=google_key,
            tavily_api_key=tavily_key,
            deepagents_langchain_project=deepagents_langchain_project,
            user_langchain_project=user_langchain_project,
            project_root=project_root,
        )

    @property
    def has_openai(self) -> bool:
        """OpenAI API 키가 구성되어 있는지 확인합니다."""
        return self.openai_api_key is not None

    @property
    def has_anthropic(self) -> bool:
        """Anthropic API 키가 구성되어 있는지 확인합니다."""
        return self.anthropic_api_key is not None

    @property
    def has_google(self) -> bool:
        """Google API 키가 구성되어 있는지 확인합니다."""
        return self.google_api_key is not None

    @property
    def has_tavily(self) -> bool:
        """Tavily API 키가 구성되어 있는지 확인합니다."""
        return self.tavily_api_key is not None

    @property
    def has_deepagents_langchain_project(self) -> bool:
        """DeepAgents LangChain 프로젝트 이름이 구성되어 있는지 확인합니다."""
        return self.deepagents_langchain_project is not None

    @property
    def has_project(self) -> bool:
        """현재 git 프로젝트 내에 있는지 확인합니다."""
        return self.project_root is not None

    @property
    def user_deepagents_dir(self) -> Path:
        """기본 사용자 수준 .deepagents 디렉터리를 가져옵니다.

        Returns:
            ~/.deepagents 경로
        """
        return Path.home() / ".deepagents"

    def get_user_agent_md_path(self, agent_name: str) -> Path:
        """특정 에이전트에 대한 사용자 수준 agent.md 경로를 가져옵니다.

        파일 존재 여부와 상관없이 경로를 반환합니다.

        Args:
            agent_name: 에이전트 이름

        Returns:
            ~/.deepagents/{agent_name}/agent.md 경로
        """
        return Path.home() / ".deepagents" / agent_name / "agent.md"

    def get_project_agent_md_path(self) -> Path | None:
        """프로젝트 수준 agent.md 경로를 가져옵니다.

        파일 존재 여부와 상관없이 경로를 반환합니다.

        Returns:
            {project_root}/.deepagents/agent.md 경로, 프로젝트 내에 없는 경우 None
        """
        if not self.project_root:
            return None
        return self.project_root / ".deepagents" / "agent.md"

    @staticmethod
    def _is_valid_agent_name(agent_name: str) -> bool:
        """유효하지 않은 파일시스템 경로 및 보안 문제를 방지하기 위해 검증합니다."""
        if not agent_name or not agent_name.strip():
            return False
        # Allow only alphanumeric, hyphens, underscores, and whitespace
        return bool(re.match(r"^[a-zA-Z0-9_\-\s]+$", agent_name))

    def get_agent_dir(self, agent_name: str) -> Path:
        """전역 에이전트 디렉터리 경로를 가져옵니다.

        Args:
            agent_name: 에이전트 이름

        Returns:
            ~/.deepagents/{agent_name} 경로
        """
        if not self._is_valid_agent_name(agent_name):
            msg = (
                f"Invalid agent name: {agent_name!r}. "
                "Agent names can only contain letters, numbers, hyphens, underscores, and spaces."
            )
            raise ValueError(msg)
        return Path.home() / ".deepagents" / agent_name

    def ensure_agent_dir(self, agent_name: str) -> Path:
        """전역 에이전트 디렉터리가 존재하는지 확인하고 경로를 반환합니다.

        Args:
            agent_name: 에이전트 이름

        Returns:
            ~/.deepagents/{agent_name} 경로
        """
        if not self._is_valid_agent_name(agent_name):
            msg = (
                f"Invalid agent name: {agent_name!r}. "
                "Agent names can only contain letters, numbers, hyphens, underscores, and spaces."
            )
            raise ValueError(msg)
        agent_dir = self.get_agent_dir(agent_name)
        agent_dir.mkdir(parents=True, exist_ok=True)
        return agent_dir

    def ensure_project_deepagents_dir(self) -> Path | None:
        """프로젝트 .deepagents 디렉터리가 존재하는지 확인하고 경로를 반환합니다.

        Returns:
            프로젝트 .deepagents 디렉터리 경로, 프로젝트 내에 없는 경우 None
        """
        if not self.project_root:
            return None

        project_deepagents_dir = self.project_root / ".deepagents"
        project_deepagents_dir.mkdir(parents=True, exist_ok=True)
        return project_deepagents_dir

    def get_user_skills_dir(self, agent_name: str) -> Path:
        """특정 에이전트에 대한 사용자 수준 기술(skills) 디렉터리 경로를 가져옵니다.

        Args:
            agent_name: 에이전트 이름

        Returns:
            ~/.deepagents/{agent_name}/skills/ 경로
        """
        return self.get_agent_dir(agent_name) / "skills"

    def ensure_user_skills_dir(self, agent_name: str) -> Path:
        """사용자 수준 기술(skills) 디렉터리가 존재하는지 확인하고 경로를 반환합니다.

        Args:
            agent_name: 에이전트 이름

        Returns:
            ~/.deepagents/{agent_name}/skills/ 경로
        """
        skills_dir = self.get_user_skills_dir(agent_name)
        skills_dir.mkdir(parents=True, exist_ok=True)
        return skills_dir

    def get_project_skills_dir(self) -> Path | None:
        """프로젝트 수준 기술(skills) 디렉터리 경로를 가져옵니다.

        Returns:
            {project_root}/.deepagents/skills/ 경로, 프로젝트 내에 없는 경우 None
        """
        if not self.project_root:
            return None
        return self.project_root / ".deepagents" / "skills"

    def ensure_project_skills_dir(self) -> Path | None:
        """프로젝트 수준 기술(skills) 디렉터리가 존재하는지 확인하고 경로를 반환합니다.

        Returns:
            {project_root}/.deepagents/skills/ 경로, 프로젝트 내에 없는 경우 None
        """
        if not self.project_root:
            return None
        skills_dir = self.get_project_skills_dir()
        skills_dir.mkdir(parents=True, exist_ok=True)
        return skills_dir


# Global settings instance (initialized once)
settings = Settings.from_environment()


class SessionState:
    """변경 가능한 세션 상태를 유지합니다 (자동 승인 모드 등)."""

    def __init__(self, auto_approve: bool = False, no_splash: bool = False) -> None:
        self.auto_approve = auto_approve
        self.no_splash = no_splash
        self.exit_hint_until: float | None = None
        self.exit_hint_handle = None
        self.thread_id = str(uuid.uuid4())

    def toggle_auto_approve(self) -> bool:
        """자동 승인을 토글하고 새로운 상태를 반환합니다."""
        self.auto_approve = not self.auto_approve
        return self.auto_approve


def get_default_coding_instructions() -> str:
    """기본 코딩 에이전트 지침을 가져옵니다.

    이는 에이전트가 수정할 수 없는 불변의 기본 지침입니다.
    장기 메모리(agent.md)는 미들웨어에서 별도로 처리합니다.
    """
    default_prompt_path = Path(__file__).parent / "default_agent_prompt.md"
    return default_prompt_path.read_text()


def _detect_provider(model_name: str) -> str | None:
    """모델 이름에서 공급자를 자동 감지합니다.

    Args:
        model_name: 공급자를 감지할 모델 이름

    Returns:
        공급자 이름(openai, anthropic, google) 또는 감지할 수 없는 경우 None
    """
    model_lower = model_name.lower()
    if any(x in model_lower for x in ["gpt", "o1", "o3"]):
        return "openai"
    if "claude" in model_lower:
        return "anthropic"
    if "gemini" in model_lower:
        return "google"
    return None


def create_model(model_name_override: str | None = None) -> BaseChatModel:
    """사용 가능한 API 키를 기반으로 적절한 모델을 생성합니다.

    전역 설정 인스턴스를 사용하여 생성할 모델을 결정합니다.

    Args:
        model_name_override: 환경 변수 대신 사용할 선택적 모델 이름

    Returns:
        ChatModel 인스턴스 (OpenAI, Anthropic, 또는 Google)

    Raises:
        API 키가 구성되지 않았거나 모델 공급자를 결정할 수 없는 경우 SystemExit
    """
    # Determine provider and model
    if model_name_override:
        # Use provided model, auto-detect provider
        provider = _detect_provider(model_name_override)
        if not provider:
            console.print(
                f"[bold red]오류:[/bold red] 모델 이름에서 공급자를 감지할 수 없습니다: {model_name_override}"
            )
            console.print("\n지원되는 모델 이름 패턴:")
            console.print("  - OpenAI: gpt-*, o1-*, o3-*")
            console.print("  - Anthropic: claude-*")
            console.print("  - Google: gemini-*")
            sys.exit(1)

        # Check if API key for detected provider is available
        if provider == "openai" and not settings.has_openai:
            console.print(f"[bold red]오류:[/bold red] 모델 '{model_name_override}'은(는) OPENAI_API_KEY가 필요합니다")
            sys.exit(1)
        elif provider == "anthropic" and not settings.has_anthropic:
            console.print(
                f"[bold red]오류:[/bold red] 모델 '{model_name_override}'은(는) ANTHROPIC_API_KEY가 필요합니다"
            )
            sys.exit(1)
        elif provider == "google" and not settings.has_google:
            console.print(f"[bold red]오류:[/bold red] 모델 '{model_name_override}'은(는) GOOGLE_API_KEY가 필요합니다")
            sys.exit(1)

        model_name = model_name_override
    # Use environment variable defaults, detect provider by API key priority
    elif settings.has_openai:
        provider = "openai"
        model_name = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
    elif settings.has_anthropic:
        provider = "anthropic"
        model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    elif settings.has_google:
        provider = "google"
        model_name = os.environ.get("GOOGLE_MODEL", "gemini-3-pro-preview")
    else:
        console.print("[bold red]오류:[/bold red] API 키가 구성되지 않았습니다.")
        console.print("\n다음 환경 변수 중 하나를 설정하십시오:")
        console.print("  - OPENAI_API_KEY     (OpenAI 모델용, 예: gpt-5-mini)")
        console.print("  - ANTHROPIC_API_KEY  (Claude 모델용)")
        console.print("  - GOOGLE_API_KEY     (Google Gemini 모델용)")
        console.print("\n예시:")
        console.print("  export OPENAI_API_KEY=your_api_key_here")
        console.print("\n또는 .env 파일에 추가하십시오.")
        sys.exit(1)

    # Store model info in settings for display
    settings.model_name = model_name
    settings.model_provider = provider

    # Create and return the model
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model_name=model_name,
            max_tokens=20_000,  # type: ignore[arg-type]
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_tokens=None,
        )
