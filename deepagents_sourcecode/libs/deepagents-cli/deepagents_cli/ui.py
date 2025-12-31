"""CLI를 위한 UI 렌더링 및 디스플레이 유틸리티."""

import json
import re
import shutil
from pathlib import Path
from typing import Any

from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from .config import COLORS, COMMANDS, DEEP_AGENTS_ASCII, MAX_ARG_LENGTH, console
from .file_ops import FileOperationRecord


def truncate_value(value: str, max_length: int = MAX_ARG_LENGTH) -> str:
    """max_length를 초과하는 경우 문자열 값을 자릅니다."""
    if len(value) > max_length:
        return value[:max_length] + "..."
    return value


def format_tool_display(tool_name: str, tool_args: dict) -> str:
    """도구 호출을 도구별 스마트 포맷팅으로 표시합니다.

    모든 인수보다는 각 도구 유형에 가장 관련성 높은 정보를 표시합니다.

    Args:
        tool_name: 호출되는 도구의 이름
        tool_args: 도구 인수 딕셔너리

    Returns:
        표시용으로 포맷팅된 문자열 (예: "read_file(config.py)")

    Examples:
        read_file(path="/long/path/file.py") → "read_file(file.py)"
        web_search(query="how to code", max_results=5) → 'web_search("how to code")'
        shell(command="pip install foo") → 'shell("pip install foo")'
    """
    # Tool-specific formatting - show the most important argument(s)
    if tool_name in ("read_file", "write_file", "edit_file"):
        return _format_file_tool(tool_name, tool_args)

    if tool_name == "web_search":
        return _format_web_search_tool(tool_name, tool_args)

    if tool_name == "grep":
        return _format_grep_tool(tool_name, tool_args)

    if tool_name == "shell":
        return _format_shell_tool(tool_name, tool_args)

    if tool_name == "ls":
        return _format_ls_tool(tool_name, tool_args)

    if tool_name == "glob":
        return _format_glob_tool(tool_name, tool_args)

    if tool_name == "http_request":
        return _format_http_request_tool(tool_name, tool_args)

    if tool_name == "fetch_url":
        return _format_fetch_url_tool(tool_name, tool_args)

    if tool_name == "task":
        return _format_task_tool(tool_name, tool_args)

    if tool_name == "write_todos":
        return _format_write_todos_tool(tool_name, tool_args)

    # Fallback: generic formatting
    arg_str = ", ".join(f"{k}={truncate_value(str(v), 20)}" for k, v in tool_args.items())
    return f"{tool_name}({arg_str})"


def _abbreviate_path(path_str: str, max_length: int = 60) -> str:
    """파일 경로를 지능적으로 축약합니다 - 베이스네임 또는 상대 경로를 표시합니다."""
    try:
        path = Path(path_str)

        # If it's just a filename (no directory parts), return as-is
        if len(path.parts) == 1:
            return path_str

        # Try to get relative path from current working directory
        try:
            rel_path = path.relative_to(Path.cwd())
            rel_str = str(rel_path)
            # Use relative if it's shorter and not too long
            if len(rel_str) < len(path_str) and len(rel_str) <= max_length:
                return rel_str
        except (ValueError, Exception):
            pass

        # If absolute path is reasonable length, use it
        if len(path_str) <= max_length:
            return path_str

        # Otherwise, just show basename (filename only)
        return path.name
    except Exception:
        # Fallback to original string if any error
        return truncate_value(path_str, max_length)


def _format_file_tool(tool_name: str, tool_args: dict) -> str:
    path_value = tool_args.get("file_path")
    if path_value is None:
        path_value = tool_args.get("path")
    if path_value is not None:
        path = _abbreviate_path(str(path_value))
        return f"{tool_name}({path})"
    return f"{tool_name}(...)"


def _format_web_search_tool(tool_name: str, tool_args: dict) -> str:
    if "query" in tool_args:
        query = str(tool_args["query"])
        query = truncate_value(query, 100)
        return f'{tool_name}("{query}")'
    return f"{tool_name}()"


def _format_grep_tool(tool_name: str, tool_args: dict) -> str:
    if "pattern" in tool_args:
        pattern = str(tool_args["pattern"])
        pattern = truncate_value(pattern, 70)
        return f'{tool_name}("{pattern}")'
    return f"{tool_name}()"


def _format_shell_tool(tool_name: str, tool_args: dict) -> str:
    if "command" in tool_args:
        command = str(tool_args["command"])
        command = truncate_value(command, 120)
        return f'{tool_name}("{command}")'
    return f"{tool_name}()"


def _format_ls_tool(tool_name: str, tool_args: dict) -> str:
    if tool_args.get("path"):
        path = _abbreviate_path(str(tool_args["path"]))
        return f"{tool_name}({path})"
    return f"{tool_name}()"


def _format_glob_tool(tool_name: str, tool_args: dict) -> str:
    if "pattern" in tool_args:
        pattern = str(tool_args["pattern"])
        pattern = truncate_value(pattern, 80)
        return f'{tool_name}("{pattern}")'
    return f"{tool_name}()"


def _format_http_request_tool(tool_name: str, tool_args: dict) -> str:
    parts = []
    if "method" in tool_args:
        parts.append(str(tool_args["method"]).upper())
    if "url" in tool_args:
        url = str(tool_args["url"])
        url = truncate_value(url, 80)
        parts.append(url)
    if parts:
        return f"{tool_name}({' '.join(parts)})"
    return f"{tool_name}()"


def _format_fetch_url_tool(tool_name: str, tool_args: dict) -> str:
    if "url" in tool_args:
        url = str(tool_args["url"])
        url = truncate_value(url, 80)
        return f'{tool_name}("{url}")'
    return f"{tool_name}()"


def _format_task_tool(tool_name: str, tool_args: dict) -> str:
    if "description" in tool_args:
        desc = str(tool_args["description"])
        desc = truncate_value(desc, 100)
        return f'{tool_name}("{desc}")'
    return f"{tool_name}()"


def _format_write_todos_tool(tool_name: str, tool_args: dict) -> str:
    if "todos" in tool_args and isinstance(tool_args["todos"], list):
        count = len(tool_args["todos"])
        return f"{tool_name}({count} items)"
    return f"{tool_name}()"


def format_tool_message_content(content: Any) -> str:
    """ToolMessage 내용을 출력 가능한 문자열로 변환합니다."""
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            else:
                try:
                    parts.append(json.dumps(item))
                except Exception:
                    parts.append(str(item))
        return "\n".join(parts)
    return str(content)


class TokenTracker:
    """대화 전반에 걸친 토큰 사용량을 추적합니다."""

    def __init__(self) -> None:
        self.baseline_context = 0  # Baseline system context (system + agent.md + tools)
        self.current_context = 0  # Total context including messages
        self.last_output = 0

    def set_baseline(self, tokens: int) -> None:
        """기준 컨텍스트 토큰 수를 설정합니다.

        Args:
            tokens: 기준 토큰 수 (시스템 프롬프트 + agent.md + 도구)
        """
        self.baseline_context = tokens
        self.current_context = tokens

    def reset(self) -> None:
        """기준으로 재설정합니다 (/clear 명령용)."""
        self.current_context = self.baseline_context
        self.last_output = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        """응답에서 토큰을 추가합니다."""
        # input_tokens IS the current context size (what was sent to the model)
        self.current_context = input_tokens
        self.last_output = output_tokens

    def display_last(self) -> None:
        """이번 턴 이후의 현재 컨텍스트 크기를 표시합니다."""
        if self.last_output and self.last_output >= 1000:
            console.print(f"  생성됨: {self.last_output:,} 토큰", style="dim")
        if self.current_context:
            console.print(f"  현재 컨텍스트: {self.current_context:,} 토큰", style="dim")

    def display_session(self) -> None:
        """현재 컨텍스트 크기를 표시합니다."""
        console.print("\n[bold]토큰 사용량:[/bold]", style=COLORS["primary"])

        # Check if we've had any actual API calls yet (current > baseline means we have conversation)
        has_conversation = self.current_context > self.baseline_context

        if self.baseline_context > 0:
            console.print(
                f"  기준(Baseline): {self.baseline_context:,} 토큰 [dim](시스템 + agent.md)[/dim]",
                style=COLORS["dim"],
            )

            if not has_conversation:
                # Before first message - warn that tools aren't counted yet
                console.print("  [dim]참고: 도구 정의(~5k 토큰)는 첫 번째 메시지 이후에 포함됩니다[/dim]")

        if has_conversation:
            tools_and_conversation = self.current_context - self.baseline_context
            console.print(f"  도구 + 대화: {tools_and_conversation:,} 토큰", style=COLORS["dim"])

        console.print(f"  합계: {self.current_context:,} 토큰", style="bold " + COLORS["dim"])
        console.print()


def render_todo_list(todos: list[dict]) -> None:
    """작업 목록을 체크박스가 있는 rich 패널로 렌더링합니다."""
    if not todos:
        return

    lines = []
    for todo in todos:
        status = todo.get("status", "pending")
        content = todo.get("content", "")

        if status == "completed":
            icon = "☑"
            style = "green"
        elif status == "in_progress":
            icon = "⏳"
            style = "yellow"
        else:  # pending
            icon = "☐"
            style = "dim"

        lines.append(f"[{style}]{icon} {content}[/{style}]")

    panel = Panel(
        "\n".join(lines),
        title="[bold]작업 목록[/bold]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print(panel)


def _format_line_span(start: int | None, end: int | None) -> str:
    if start is None and end is None:
        return ""
    if start is not None and end is None:
        return f"({start}행부터)"
    if start is None and end is not None:
        return f"({end}행까지)"
    if start == end:
        return f"({start}행)"
    return f"({start}-{end}행)"


def render_file_operation(record: FileOperationRecord) -> None:
    """파일시스템 도구 호출에 대한 간략한 요약을 렌더링합니다."""
    label_lookup = {
        "read_file": "읽기",
        "write_file": "쓰기",
        "edit_file": "업데이트",
    }
    label = label_lookup.get(record.tool_name, record.tool_name)
    header = Text()
    header.append("⏺ ", style=COLORS["tool"])
    header.append(f"{label}({record.display_path})", style=f"bold {COLORS['tool']}")
    console.print(header)

    def _print_detail(message: str, *, style: str = COLORS["dim"]) -> None:
        detail = Text()
        detail.append("  ⎿  ", style=style)
        detail.append(message, style=style)
        console.print(detail)

    if record.status == "error":
        _print_detail(record.error or "파일 작업 실행 오류", style="red")
        return

    if record.tool_name == "read_file":
        lines = record.metrics.lines_read
        span = _format_line_span(record.metrics.start_line, record.metrics.end_line)
        detail = f"{lines}줄 읽음"
        if span:
            detail = f"{detail} {span}"
        _print_detail(detail)
    else:
        if record.tool_name == "write_file":
            added = record.metrics.lines_added
            removed = record.metrics.lines_removed
            lines = record.metrics.lines_written
            detail = f"{lines}줄 씀"
            if added or removed:
                detail = f"{detail} (+{added} / -{removed})"
        else:
            added = record.metrics.lines_added
            removed = record.metrics.lines_removed
            detail = f"총 {record.metrics.lines_written}줄 편집됨"
            if added or removed:
                detail = f"{detail} (+{added} / -{removed})"
        _print_detail(detail)

    # Skip diff display for HIL-approved operations that succeeded
    # (user already saw the diff during approval)
    if record.diff and not (record.hitl_approved and record.status == "success"):
        render_diff(record)


def render_diff(record: FileOperationRecord) -> None:
    """파일 작업에 대한 diff를 렌더링합니다."""
    if not record.diff:
        return
    render_diff_block(record.diff, f"{record.display_path} 차이(Diff)")


def _wrap_diff_line(
    code: str,
    marker: str,
    color: str,
    line_num: int | None,
    width: int,
    term_width: int,
) -> list[str]:
    """긴 diff 줄을 적절한 들여쓰기로 줄바꿈합니다.

    Args:
        code: 래핑할 코드 콘텐츠
        marker: Diff 마커 ('+', '-', ' ')
        color: 해당 줄의 색상
        line_num: 표시할 줄 번호 (연속 줄의 경우 None)
        width: 줄 번호 열의 너비
        term_width: 터미널 너비

    Returns:
        포맷팅된 줄 목록 (줄바꿈된 경우 여러 개일 수 있음)
    """
    # Escape Rich markup in code content
    code = escape(code)

    prefix_len = width + 4  # line_num + space + marker + 2 spaces
    available_width = term_width - prefix_len

    if len(code) <= available_width:
        if line_num is not None:
            return [f"[dim]{line_num:>{width}}[/dim] [{color}]{marker}  {code}[/{color}]"]
        return [f"{' ' * width} [{color}]{marker}  {code}[/{color}]"]

    lines = []
    remaining = code
    first = True

    while remaining:
        if len(remaining) <= available_width:
            chunk = remaining
            remaining = ""
        else:
            # Try to break at a good point (space, comma, etc.)
            chunk = remaining[:available_width]
            # Look for a good break point in the last 20 chars
            break_point = max(
                chunk.rfind(" "),
                chunk.rfind(","),
                chunk.rfind("("),
                chunk.rfind(")"),
            )
            if break_point > available_width - 20:
                # Found a good break point
                chunk = remaining[: break_point + 1]
                remaining = remaining[break_point + 1 :]
            else:
                # No good break point, just split
                chunk = remaining[:available_width]
                remaining = remaining[available_width:]

        if first and line_num is not None:
            lines.append(f"[dim]{line_num:>{width}}[/dim] [{color}]{marker}  {chunk}[/{color}]")
            first = False
        else:
            lines.append(f"{' ' * width} [{color}]{marker}  {chunk}[/{color}]")

    return lines


def format_diff_rich(diff_lines: list[str]) -> str:
    """줄 번호와 색상으로 diff 줄을 포맷팅합니다.

    Args:
        diff_lines: 통합 diff의 Diff 줄
    """
    if not diff_lines:
        return "[dim]감지된 변경 사항 없음[/dim]"

    # Get terminal width
    term_width = shutil.get_terminal_size().columns

    # Find max line number for width calculation
    max_line = max(
        (
            int(m.group(i))
            for line in diff_lines
            if (m := re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", line))
            for i in (1, 2)
        ),
        default=0,
    )
    width = max(3, len(str(max_line)))

    formatted_lines = []
    old_num = new_num = 0

    # Rich colors with backgrounds for better visibility
    # White text on dark backgrounds for additions/deletions
    addition_color = "white on dark_green"
    deletion_color = "white on dark_red"
    context_color = "dim"

    for line in diff_lines:
        if line.strip() == "...":
            formatted_lines.append(f"[{context_color}]...[/{context_color}]")
        elif line.startswith(("---", "+++")):
            continue
        elif m := re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", line):
            old_num, new_num = int(m.group(1)), int(m.group(2))
        elif line.startswith("-"):
            formatted_lines.extend(_wrap_diff_line(line[1:], "-", deletion_color, old_num, width, term_width))
            old_num += 1
        elif line.startswith("+"):
            formatted_lines.extend(_wrap_diff_line(line[1:], "+", addition_color, new_num, width, term_width))
            new_num += 1
        elif line.startswith(" "):
            formatted_lines.extend(_wrap_diff_line(line[1:], " ", context_color, old_num, width, term_width))
            old_num += 1
            new_num += 1

    return "\n".join(formatted_lines)


def render_diff_block(diff: str, title: str) -> None:
    """diff 문자열을 줄 번호와 색상으로 렌더링합니다."""
    try:
        # Parse diff into lines and format with line numbers
        diff_lines = diff.splitlines()
        formatted_diff = format_diff_rich(diff_lines)

        # Print with a simple header
        console.print()
        console.print(f"[bold {COLORS['primary']}]═══ {title} ═══[/bold {COLORS['primary']}]")
        console.print(formatted_diff)
        console.print()
    except (ValueError, AttributeError, IndexError, OSError):
        # Fallback to simple rendering if formatting fails
        console.print()
        console.print(f"[bold {COLORS['primary']}]{title}[/bold {COLORS['primary']}]")
        console.print(diff)
        console.print()


def show_interactive_help() -> None:
    """대화형 세션 중 사용할 수 있는 명령을 표시합니다."""
    console.print()
    console.print()
    console.print("[bold]대화형 명령:[/bold]", style=COLORS["primary"])
    console.print()

    for cmd, desc in COMMANDS.items():
        console.print(f"  /{cmd:<12} {desc}", style=COLORS["dim"])

    console.print()
    console.print("[bold]편집 기능:[/bold]", style=COLORS["primary"])
    console.print("  Enter           메시지 제출", style=COLORS["dim"])
    console.print(
        "  Alt+Enter       줄바꿈 삽입 (Mac의 경우 Option+Enter, 또는 ESC 후 Enter)",
        style=COLORS["dim"],
    )
    console.print("  Ctrl+E          외부 편집기에서 열기 (기본값 nano)", style=COLORS["dim"])
    console.print("  Ctrl+T          자동 승인 모드 전환", style=COLORS["dim"])
    console.print("  방향키          입력 탐색", style=COLORS["dim"])
    console.print("  Ctrl+C          입력 취소 또는 작업 중인 에이전트 중단", style=COLORS["dim"])
    console.print()
    console.print("[bold]특수 기능:[/bold]", style=COLORS["primary"])
    console.print("  @filename       @를 입력하여 파일 자동 완성 및 콘텐츠 주입", style=COLORS["dim"])
    console.print("  /command        /를 입력하여 사용 가능한 명령 확인", style=COLORS["dim"])
    console.print(
        "  !command        !를 입력하여 bash 명령 실행 (예: !ls, !git status)",
        style=COLORS["dim"],
    )
    console.print("                  입력하면 완성이 자동으로 나타납니다", style=COLORS["dim"])
    console.print()
    console.print("[bold]자동 승인 모드:[/bold]", style=COLORS["primary"])
    console.print("  Ctrl+T          자동 승인 모드 전환", style=COLORS["dim"])
    console.print(
        "  --auto-approve  자동 승인이 활성화된 상태로 CLI 시작 (명령줄을 통해)",
        style=COLORS["dim"],
    )
    console.print("  활성화되면 도구 작업이 확인 프롬프트 없이 실행됩니다", style=COLORS["dim"])
    console.print()


def show_help() -> None:
    """도움말 정보를 표시합니다."""
    console.print()
    console.print(DEEP_AGENTS_ASCII, style=f"bold {COLORS['primary']}")
    console.print()

    console.print("[bold]사용법:[/bold]", style=COLORS["primary"])
    console.print("  deepagents [OPTIONS]                           대화형 세션 시작")
    console.print("  deepagents list                                사용 가능한 모든 에이전트 나열")
    console.print("  deepagents reset --agent AGENT                 에이전트를 기본 프롬프트로 초기화")
    console.print("  deepagents reset --agent AGENT --target SOURCE 에이전트를 다른 에이전트의 복사본으로 초기화")
    console.print("  deepagents help                                이 도움말 메시지 표시")
    console.print()

    console.print("[bold]옵션:[/bold]", style=COLORS["primary"])
    console.print("  --agent NAME                  에이전트 식별자 (기본값: agent)")
    console.print("  --model MODEL                 사용할 모델 (예: claude-sonnet-4-5-20250929, gpt-4o)")
    console.print("  --auto-approve                프롬프트 없이 도구 사용 자동 승인")
    console.print("  --sandbox TYPE                실행을 위한 원격 샌드박스 (modal, runloop, daytona)")
    console.print("  --sandbox-id ID               기존 샌드박스 재사용 (생성/정리 건너뜀)")
    console.print()

    console.print("[bold]예시:[/bold]", style=COLORS["primary"])
    console.print("  deepagents                              # 기본 에이전트로 시작", style=COLORS["dim"])
    console.print(
        "  deepagents --agent mybot                # 'mybot'이라는 이름의 에이전트로 시작",
        style=COLORS["dim"],
    )
    console.print(
        "  deepagents --model gpt-4o               # 특정 모델 사용 (공급자 자동 감지)",
        style=COLORS["dim"],
    )
    console.print(
        "  deepagents --auto-approve               # 자동 승인이 활성화된 상태로 시작",
        style=COLORS["dim"],
    )
    console.print(
        "  deepagents --sandbox runloop            # Runloop 샌드박스에서 코드 실행",
        style=COLORS["dim"],
    )
    console.print(
        "  deepagents --sandbox modal              # Modal 샌드박스에서 코드 실행",
        style=COLORS["dim"],
    )
    console.print(
        "  deepagents --sandbox runloop --sandbox-id dbx_123  # 기존 샌드박스 재사용",
        style=COLORS["dim"],
    )
    console.print("  deepagents list                         # 모든 에이전트 나열", style=COLORS["dim"])
    console.print("  deepagents reset --agent mybot          # mybot을 기본값으로 초기화", style=COLORS["dim"])
    console.print(
        "  deepagents reset --agent mybot --target other # mybot을 'other' 에이전트의 복사본으로 초기화",
        style=COLORS["dim"],
    )
    console.print()

    console.print("[bold]장기 기억(Long-term Memory):[/bold]", style=COLORS["primary"])
    console.print("  기본적으로 장기 기억은 'agent'라는 에이전트 이름을 사용하여 활성화됩니다.", style=COLORS["dim"])
    console.print("  기억에는 다음이 포함됩니다:", style=COLORS["dim"])
    console.print("  - 지침이 포함된 영구 agent.md 파일", style=COLORS["dim"])
    console.print("  - 세션 간 컨텍스트 저장을 위한 /memories/ 폴더", style=COLORS["dim"])
    console.print()

    console.print("[bold]에이전트 저장소:[/bold]", style=COLORS["primary"])
    console.print("  에이전트는 다음 경로에 저장됩니다: ~/.deepagents/AGENT_NAME/", style=COLORS["dim"])
    console.print("  각 에이전트에는 프롬프트가 포함된 agent.md 파일이 있습니다", style=COLORS["dim"])
    console.print()

    console.print("[bold]대화형 기능:[/bold]", style=COLORS["primary"])
    console.print("  Enter           메시지 제출", style=COLORS["dim"])
    console.print(
        "  Alt+Enter       여러 줄 입력을 위한 줄바꿈 (Option+Enter 또는 ESC 후 Enter)",
        style=COLORS["dim"],
    )
    console.print("  Ctrl+J          줄바꿈 삽입 (대안)", style=COLORS["dim"])
    console.print("  Ctrl+T          자동 승인 모드 전환", style=COLORS["dim"])
    console.print("  방향키          입력 탐색", style=COLORS["dim"])
    console.print("  @filename       @를 입력하여 파일 자동 완성 및 콘텐츠 주입", style=COLORS["dim"])
    console.print("  /command        /를 입력하여 사용 가능한 명령 확인 (자동 완성)", style=COLORS["dim"])
    console.print()

    console.print("[bold]대화형 명령:[/bold]", style=COLORS["primary"])
    console.print("  /help           사용 가능한 명령 및 기능 표시", style=COLORS["dim"])
    console.print("  /clear          화면 지우기 및 대화 초기화", style=COLORS["dim"])
    console.print("  /tokens         현재 세션의 토큰 사용량 표시", style=COLORS["dim"])
    console.print("  /quit, /exit    세션 종료", style=COLORS["dim"])
    console.print("  quit, exit, q   세션 종료 (입력하고 Enter 누름)", style=COLORS["dim"])
    console.print()
