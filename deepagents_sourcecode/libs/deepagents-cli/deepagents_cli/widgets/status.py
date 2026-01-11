"""deepagents-cli용 상태 표시줄(Status bar) 위젯입니다."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

TOKENS_K_THRESHOLD = 1000


class StatusBar(Horizontal):
    """모드/자동 승인/작업 디렉토리 등을 표시하는 상태 표시줄입니다."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        dock: bottom;
        background: $surface;
        padding: 0 1;
    }

    StatusBar .status-mode {
        width: auto;
        padding: 0 1;
    }

    StatusBar .status-mode.normal {
        display: none;
    }

    StatusBar .status-mode.bash {
        background: #ff1493;
        color: white;
        text-style: bold;
    }

    StatusBar .status-mode.command {
        background: #8b5cf6;
        color: white;
    }

    StatusBar .status-auto-approve {
        width: auto;
        padding: 0 1;
    }

    StatusBar .status-auto-approve.on {
        background: #10b981;
        color: black;
    }

    StatusBar .status-auto-approve.off {
        background: #f59e0b;
        color: black;
    }

    StatusBar .status-message {
        width: auto;
        padding: 0 1;
        color: $text-muted;
    }

    StatusBar .status-message.thinking {
        color: $warning;
    }

    StatusBar .status-cwd {
        width: 1fr;
        text-align: right;
        color: $text-muted;
    }

    StatusBar .status-tokens {
        width: auto;
        padding: 0 1;
        color: $text-muted;
    }
    """

    mode: reactive[str] = reactive("normal", init=False)
    status_message: reactive[str] = reactive("", init=False)
    auto_approve: reactive[bool] = reactive(default=False, init=False)
    cwd: reactive[str] = reactive("", init=False)
    tokens: reactive[int] = reactive(0, init=False)

    def __init__(self, cwd: str | Path | None = None, **kwargs: Any) -> None:
        """상태 표시줄을 초기화합니다.

        Args:
            cwd: Current working directory to display
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        # 초기 cwd를 저장(compose()에서 사용)
        self._initial_cwd = str(cwd) if cwd else str(Path.cwd())

    def compose(self) -> ComposeResult:
        """상태 표시줄 레이아웃을 구성합니다."""
        yield Static("", classes="status-mode normal", id="mode-indicator")
        yield Static(
            "manual | shift+tab to cycle",
            classes="status-auto-approve off",
            id="auto-approve-indicator",
        )
        yield Static("", classes="status-message", id="status-message")
        yield Static("", classes="status-tokens", id="tokens-display")
        # CWD shown in welcome banner, not pinned in status bar

    def on_mount(self) -> None:
        """마운트(on_mount) 이후 reactive 값을 설정해 watcher가 안전하게 동작하도록 합니다."""
        self.cwd = self._initial_cwd

    def watch_mode(self, mode: str) -> None:
        """모드(mode) 변경 시 표시를 갱신합니다."""
        try:
            indicator = self.query_one("#mode-indicator", Static)
        except NoMatches:
            return
        indicator.remove_class("normal", "bash", "command")

        if mode == "bash":
            indicator.update("BASH")
            indicator.add_class("bash")
        elif mode == "command":
            indicator.update("CMD")
            indicator.add_class("command")
        else:
            indicator.update("")
            indicator.add_class("normal")

    def watch_auto_approve(self, new_value: bool) -> None:  # noqa: FBT001
        """auto-approve 상태 변경 시 표시를 갱신합니다."""
        try:
            indicator = self.query_one("#auto-approve-indicator", Static)
        except NoMatches:
            return
        indicator.remove_class("on", "off")

        if new_value:
            indicator.update("auto | shift+tab to cycle")
            indicator.add_class("on")
        else:
            indicator.update("manual | shift+tab to cycle")
            indicator.add_class("off")

    def watch_cwd(self, new_value: str) -> None:
        """작업 디렉토리(cwd) 변경 시 표시를 갱신합니다."""
        try:
            display = self.query_one("#cwd-display", Static)
        except NoMatches:
            return
        display.update(self._format_cwd(new_value))

    def watch_status_message(self, new_value: str) -> None:
        """상태 메시지(status message) 변경 시 표시를 갱신합니다."""
        try:
            msg_widget = self.query_one("#status-message", Static)
        except NoMatches:
            return

        msg_widget.remove_class("thinking")
        if new_value:
            msg_widget.update(new_value)
            if "thinking" in new_value.lower() or "executing" in new_value.lower():
                msg_widget.add_class("thinking")
        else:
            msg_widget.update("")

    def _format_cwd(self, cwd_path: str = "") -> str:
        """표시용으로 현재 작업 디렉토리를 포맷팅합니다."""
        path = Path(cwd_path or self.cwd or self._initial_cwd)
        try:
            # 홈 디렉토리는 ~로 표시 시도
            home = Path.home()
            if path.is_relative_to(home):
                return "~/" + str(path.relative_to(home))
        except (ValueError, RuntimeError):
            pass
        return str(path)

    def set_mode(self, mode: str) -> None:
        """현재 입력 모드를 설정합니다.

        Args:
            mode: One of "normal", "bash", or "command"
        """
        self.mode = mode

    def set_auto_approve(self, *, enabled: bool) -> None:
        """auto-approve 상태를 설정합니다.

        Args:
            enabled: Whether auto-approve is enabled
        """
        self.auto_approve = enabled

    def set_status_message(self, message: str) -> None:
        """상태 메시지를 설정합니다.

        Args:
            message: Status message to display (empty string to clear)
        """
        self.status_message = message

    def watch_tokens(self, new_value: int) -> None:
        """토큰 수 변경 시 표시를 갱신합니다."""
        try:
            display = self.query_one("#tokens-display", Static)
        except NoMatches:
            return

        if new_value > 0:
            # 천 단위는 K suffix로 표시
            if new_value >= TOKENS_K_THRESHOLD:
                display.update(f"{new_value / TOKENS_K_THRESHOLD:.1f}K tokens")
            else:
                display.update(f"{new_value} tokens")
        else:
            display.update("")

    def set_tokens(self, count: int) -> None:
        """토큰 수를 설정합니다.

        Args:
            count: Current context token count
        """
        self.tokens = count
