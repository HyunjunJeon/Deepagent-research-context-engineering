"""DeepAgents를 위한 메인 진입점 및 CLI 루프."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from deepagents.backends.protocol import SandboxBackendProtocol

# Now safe to import agent (which imports LangChain modules)
from deepagents_cli.agent import create_cli_agent, list_agents, reset_agent
from deepagents_cli.commands import execute_bash_command, handle_command

# CRITICAL: Import config FIRST to set LANGSMITH_PROJECT before LangChain loads
from deepagents_cli.config import (
    COLORS,
    DEEP_AGENTS_ASCII,
    SessionState,
    console,
    create_model,
    settings,
)
from deepagents_cli.execution import execute_task
from deepagents_cli.input import ImageTracker, create_prompt_session
from deepagents_cli.integrations.sandbox_factory import (
    create_sandbox,
    get_default_working_dir,
)
from deepagents_cli.skills import execute_skills_command, setup_skills_parser
from deepagents_cli.tools import fetch_url, http_request, web_search
from deepagents_cli.ui import TokenTracker, show_help


def check_cli_dependencies() -> None:
    """CLI 선택적 종속성이 설치되어 있는지 확인합니다."""
    missing = []

    try:
        import rich
    except ImportError:
        missing.append("rich")

    try:
        import requests
    except ImportError:
        missing.append("requests")

    try:
        import dotenv
    except ImportError:
        missing.append("python-dotenv")

    try:
        import tavily
    except ImportError:
        missing.append("tavily-python")

    try:
        import prompt_toolkit
    except ImportError:
        missing.append("prompt-toolkit")

    if missing:
        print("\n❌ 필수 CLI 종속성이 누락되었습니다!")
        print("\nDeepAgents CLI를 사용하려면 다음 패키지가 필요합니다:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\n다음 명령으로 설치하십시오:")
        print("  pip install deepagents[cli]")
        print("\n또는 모든 종속성을 설치하십시오:")
        print("  pip install 'deepagents[cli]'")
        sys.exit(1)


def parse_args():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="DeepAgents - AI 코딩 도우미",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )

    subparsers = parser.add_subparsers(dest="command", help="실행할 명령")

    # List command
    subparsers.add_parser("list", help="사용 가능한 모든 에이전트 나열")

    # Help command
    subparsers.add_parser("help", help="도움말 정보 표시")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="에이전트 초기화")
    reset_parser.add_argument("--agent", required=True, help="초기화할 에이전트 이름")
    reset_parser.add_argument("--target", dest="source_agent", help="다른 에이전트에서 프롬프트 복사")

    # Skills command - setup delegated to skills module
    setup_skills_parser(subparsers)

    # Default interactive mode
    parser.add_argument(
        "--agent",
        default="agent",
        help="별도의 메모리 저장소를 위한 에이전트 식별자 (기본값: agent).",
    )
    parser.add_argument(
        "--model",
        help="사용할 모델 (예: claude-sonnet-4-5-20250929, gpt-5-mini, gemini-3-pro-preview). 모델 이름에서 공급자가 자동 감지됩니다.",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="프롬프트 없이 도구 사용 자동 승인 (human-in-the-loop 비활성화)",
    )
    parser.add_argument(
        "--sandbox",
        choices=["none", "modal", "daytona", "runloop"],
        default="none",
        help="코드 실행을 위한 원격 샌드박스 (기본값: none - 로컬 전용)",
    )
    parser.add_argument(
        "--sandbox-id",
        help="재사용할 기존 샌드박스 ID (생성 및 정리 건너뜀)",
    )
    parser.add_argument(
        "--sandbox-setup",
        help="생성 후 샌드박스에서 실행할 설정 스크립트 경로",
    )
    parser.add_argument(
        "--no-splash",
        action="store_true",
        help="시작 스플래시 화면 비활성화",
    )

    return parser.parse_args()


async def simple_cli(
    agent,
    assistant_id: str | None,
    session_state,
    baseline_tokens: int = 0,
    backend=None,
    sandbox_type: str | None = None,
    setup_script_path: str | None = None,
    no_splash: bool = False,
) -> None:
    """메인 CLI 루프.

    Args:
        backend: 파일 작업을 위한 백엔드 (CompositeBackend)
        sandbox_type: 사용 중인 샌드박스 유형 (예: "modal", "runloop", "daytona").
                     None인 경우 로컬 모드에서 실행.
        sandbox_id: 활성 샌드박스의 ID
        setup_script_path: 실행된 설정 스크립트 경로 (있는 경우)
        no_splash: True인 경우 시작 스플래시 화면 표시 건너뜀
    """
    console.clear()
    if not no_splash:
        console.print(DEEP_AGENTS_ASCII, style=f"bold {COLORS['primary']}")
        console.print()

    # Extract sandbox ID from backend if using sandbox mode
    sandbox_id: str | None = None
    if backend:
        from deepagents.backends.composite import CompositeBackend

        # Check if it's a CompositeBackend with a sandbox default backend
        if isinstance(backend, CompositeBackend):
            if isinstance(backend.default, SandboxBackendProtocol):
                sandbox_id = backend.default.id
        elif isinstance(backend, SandboxBackendProtocol):
            sandbox_id = backend.id

    # Display sandbox info persistently (survives console.clear())
    if sandbox_type and sandbox_id:
        console.print(f"[yellow]⚡ {sandbox_type.capitalize()} 샌드박스: {sandbox_id}[/yellow]")
        if setup_script_path:
            console.print(f"[green]✓ 설정 스크립트 ({setup_script_path}) 완료됨[/green]")
        console.print()

    # Display model info
    if settings.model_name and settings.model_provider:
        provider_display = {
            "openai": "OpenAI",
            "anthropic": "Anthropic",
            "google": "Google",
        }.get(settings.model_provider, settings.model_provider)
        console.print(
            f"[green]✓ Model:[/green] {provider_display} → '{settings.model_name}'",
            style=COLORS["dim"],
        )
        console.print()

    if not settings.has_tavily:
        console.print(
            "[yellow]⚠ 웹 검색 비활성화됨:[/yellow] TAVILY_API_KEY를 찾을 수 없습니다.",
            style=COLORS["dim"],
        )
        console.print("  웹 검색을 활성화하려면 Tavily API 키를 설정하세요:", style=COLORS["dim"])
        console.print("    export TAVILY_API_KEY=your_api_key_here", style=COLORS["dim"])
        console.print(
            "  또는 .env 파일에 추가하세요. 키 발급: https://tavily.com",
            style=COLORS["dim"],
        )
        console.print()

    if settings.has_deepagents_langchain_project:
        console.print(
            f"[green]✓ LangSmith 추적 활성화됨:[/green] Deepagents → '{settings.deepagents_langchain_project}'",
            style=COLORS["dim"],
        )
        if settings.user_langchain_project:
            console.print(f"  [dim]사용자 코드 (shell) → '{settings.user_langchain_project}'[/dim]")
        console.print()

    console.print("... 코딩 준비 완료! 무엇을 만들고 싶으신가요?", style=COLORS["agent"])

    if sandbox_type:
        working_dir = get_default_working_dir(sandbox_type)
        console.print(f"  [dim]로컬 CLI 디렉터리: {Path.cwd()}[/dim]")
        console.print(f"  [dim]코드 실행: 원격 샌드박스 ({working_dir})[/dim]")
    else:
        console.print(f"  [dim]작업 디렉터리: {Path.cwd()}[/dim]")

    console.print()

    if session_state.auto_approve:
        console.print("  [yellow]⚡ 자동 승인: 켜짐[/yellow] [dim](확인 없이 도구 실행)[/dim]")
        console.print()

    # Localize modifier names and show key symbols (macOS vs others)
    if sys.platform == "darwin":
        tips = (
            "  팁: ⏎ Enter로 제출, ⌥ Option + ⏎ Enter로 줄바꿈 (또는 Esc+Enter), "
            "⌃E로 편집기 열기, ⌃T로 자동 승인 전환, ⌃C로 중단"
        )
    else:
        tips = (
            "  팁: Enter로 제출, Alt+Enter (또는 Esc+Enter)로 줄바꿈, "
            "Ctrl+E로 편집기 열기, Ctrl+T로 자동 승인 전환, Ctrl+C로 중단"
        )
    console.print(tips, style=f"dim {COLORS['dim']}")

    console.print()

    # Create prompt session, image tracker, and token tracker
    image_tracker = ImageTracker()
    session = create_prompt_session(assistant_id, session_state, image_tracker=image_tracker)
    token_tracker = TokenTracker()
    token_tracker.set_baseline(baseline_tokens)

    while True:
        try:
            user_input = await session.prompt_async()
            if session_state.exit_hint_handle:
                session_state.exit_hint_handle.cancel()
                session_state.exit_hint_handle = None
            session_state.exit_hint_until = None
            user_input = user_input.strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            console.print("\n안녕히 가세요!", style=COLORS["primary"])
            break

        if not user_input:
            continue

        # Check for slash commands first
        if user_input.startswith("/"):
            result = handle_command(user_input, agent, token_tracker)
            if result == "exit":
                console.print("\n안녕히 가세요!", style=COLORS["primary"])
                break
            if result:
                # Command was handled, continue to next input
                continue

        # Check for bash commands (!)
        if user_input.startswith("!"):
            execute_bash_command(user_input)
            continue

        # Handle regular quit keywords
        if user_input.lower() in ["quit", "exit", "q"]:
            console.print("\n안녕히 가세요!", style=COLORS["primary"])
            break

        await execute_task(
            user_input,
            agent,
            assistant_id,
            session_state,
            token_tracker,
            backend=backend,
            image_tracker=image_tracker,
        )


async def _run_agent_session(
    model,
    assistant_id: str,
    session_state,
    sandbox_backend=None,
    sandbox_type: str | None = None,
    setup_script_path: str | None = None,
) -> None:
    """에이전트를 생성하고 CLI 세션을 실행하는 도우미.

    샌드박스 모드와 로컬 모드 간의 중복을 피하기 위해 추출되었습니다.

    Args:
        model: 사용할 LLM 모델
        assistant_id: 메모리 저장을 위한 에이전트 식별자
        session_state: 자동 승인 설정이 포함된 세션 상태
        sandbox_backend: 원격 실행을 위한 선택적 샌드박스 백엔드
        sandbox_type: 사용 중인 샌드박스 유형
        setup_script_path: 실행된 설정 스크립트 경로 (있는 경우)
    """
    # Create agent with conditional tools
    tools = [http_request, fetch_url]
    if settings.has_tavily:
        tools.append(web_search)

    agent, composite_backend = create_cli_agent(
        model=model,
        assistant_id=assistant_id,
        tools=tools,
        sandbox=sandbox_backend,
        sandbox_type=sandbox_type,
        auto_approve=session_state.auto_approve,
    )

    # Calculate baseline token count for accurate token tracking
    from .agent import get_system_prompt
    from .token_utils import calculate_baseline_tokens

    agent_dir = settings.get_agent_dir(assistant_id)
    system_prompt = get_system_prompt(assistant_id=assistant_id, sandbox_type=sandbox_type)
    baseline_tokens = calculate_baseline_tokens(model, agent_dir, system_prompt, assistant_id)

    await simple_cli(
        agent,
        assistant_id,
        session_state,
        baseline_tokens,
        backend=composite_backend,
        sandbox_type=sandbox_type,
        setup_script_path=setup_script_path,
        no_splash=session_state.no_splash,
    )


async def main(
    assistant_id: str,
    session_state,
    sandbox_type: str = "none",
    sandbox_id: str | None = None,
    setup_script_path: str | None = None,
    model_name: str | None = None,
) -> None:
    """조건부 샌드박스 지원이 포함된 메인 진입점.

    Args:
        assistant_id: 메모리 저장을 위한 에이전트 식별자
        session_state: 자동 승인 설정이 포함된 세션 상태
        sandbox_type: 샌드박스 유형 ("none", "modal", "runloop", "daytona")
        sandbox_id: 재사용할 선택적 기존 샌드박스 ID
        setup_script_path: 샌드박스에서 실행할 선택적 설정 스크립트 경로
        model_name: 환경 변수 대신 사용할 선택적 모델 이름
    """
    model = create_model(model_name)

    # Branch 1: User wants a sandbox
    if sandbox_type != "none":
        # Try to create sandbox
        try:
            console.print()
            with create_sandbox(
                sandbox_type, sandbox_id=sandbox_id, setup_script_path=setup_script_path
            ) as sandbox_backend:
                console.print(f"[yellow]⚡ 원격 실행 활성화됨 ({sandbox_type})[/yellow]")
                console.print()

                await _run_agent_session(
                    model,
                    assistant_id,
                    session_state,
                    sandbox_backend,
                    sandbox_type=sandbox_type,
                    setup_script_path=setup_script_path,
                )
        except (ImportError, ValueError, RuntimeError, NotImplementedError) as e:
            # Sandbox creation failed - fail hard (no silent fallback)
            console.print()
            console.print("[red]❌ 샌드박스 생성 실패[/red]")
            console.print(f"[dim]{e}[/dim]")
            sys.exit(1)
        except KeyboardInterrupt:
            console.print("\n\n[yellow]중단됨[/yellow]")
            sys.exit(0)
        except Exception as e:
            console.print(f"\n[bold red]❌ 오류:[/bold red] {e}\n")
            console.print_exception()
            sys.exit(1)

    # Branch 2: User wants local mode (none or default)
    else:
        try:
            await _run_agent_session(model, assistant_id, session_state, sandbox_backend=None)
        except KeyboardInterrupt:
            console.print("\n\n[yellow]중단됨[/yellow]")
            sys.exit(0)
        except Exception as e:
            console.print(f"\n[bold red]❌ 오류:[/bold red] {e}\n")
            console.print_exception()
            sys.exit(1)


def cli_main() -> None:
    """콘솔 스크립트 진입점."""
    # Fix for gRPC fork issue on macOS
    # https://github.com/grpc/grpc/issues/37642
    if sys.platform == "darwin":
        os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

    # Note: LANGSMITH_PROJECT is already overridden in config.py (before LangChain imports)
    # This ensures agent traces → DEEPAGENTS_LANGSMITH_PROJECT
    # Shell commands → user's original LANGSMITH_PROJECT (via ShellMiddleware env)

    # Check dependencies first
    check_cli_dependencies()

    try:
        args = parse_args()

        if args.command == "help":
            show_help()
        elif args.command == "list":
            list_agents()
        elif args.command == "reset":
            reset_agent(args.agent, args.source_agent)
        elif args.command == "skills":
            execute_skills_command(args)
        else:
            # Create session state from args
            session_state = SessionState(auto_approve=args.auto_approve, no_splash=args.no_splash)

            # API key validation happens in create_model()
            asyncio.run(
                main(
                    args.agent,
                    session_state,
                    args.sandbox,
                    args.sandbox_id,
                    args.sandbox_setup,
                    getattr(args, "model", None),
                )
            )
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C - suppress ugly traceback
        console.print("\n\n[yellow]중단됨[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    cli_main()
