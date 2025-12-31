"""슬래시 명령 및 bash 실행을 위한 명령 처리기."""

import subprocess
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from .config import COLORS, DEEP_AGENTS_ASCII, console
from .ui import TokenTracker, show_interactive_help


def handle_command(command: str, agent, token_tracker: TokenTracker) -> str | bool:
    """슬래시 명령을 처리합니다. 종료하려면 'exit', 처리된 경우 True, 에이전트에게 전달하려면 False를 반환합니다."""
    cmd = command.lower().strip().lstrip("/")

    if cmd in ["quit", "exit", "q"]:
        return "exit"

    if cmd == "clear":
        # Reset agent conversation state
        agent.checkpointer = InMemorySaver()

        # Reset token tracking to baseline
        token_tracker.reset()

        # Clear screen and show fresh UI
        console.clear()
        console.print(DEEP_AGENTS_ASCII, style=f"bold {COLORS['primary']}")
        console.print()
        console.print("... 새로 시작! 화면이 지워지고 대화가 초기화되었습니다.", style=COLORS["agent"])
        console.print()
        return True

    if cmd == "help":
        show_interactive_help()
        return True

    if cmd == "tokens":
        token_tracker.display_session()
        return True

    console.print()
    console.print(f"[yellow]알 수 없는 명령: /{cmd}[/yellow]")
    console.print("[dim]사용 가능한 명령을 보려면 /help를 입력하세요.[/dim]")
    console.print()
    return True

    return False


def execute_bash_command(command: str) -> bool:
    """bash 명령을 실행하고 출력을 표시합니다. 처리된 경우 True를 반환합니다."""
    cmd = command.strip().lstrip("!")

    if not cmd:
        return True

    try:
        console.print()
        console.print(f"[dim]$ {cmd}[/dim]")

        # Execute the command
        result = subprocess.run(
            cmd, check=False, shell=True, capture_output=True, text=True, timeout=30, cwd=Path.cwd()
        )

        # Display output
        if result.stdout:
            console.print(result.stdout, style=COLORS["dim"], markup=False)
        if result.stderr:
            console.print(result.stderr, style="red", markup=False)

        # Show return code if non-zero
        if result.returncode != 0:
            console.print(f"[dim]Exit code: {result.returncode}[/dim]")

        console.print()
        return True

    except subprocess.TimeoutExpired:
        console.print("[red]30초 후 명령 시간 초과[/red]")
        console.print()
        return True
    except Exception as e:
        console.print(f"[red]명령 실행 오류: {e}[/red]")
        console.print()
        return True
