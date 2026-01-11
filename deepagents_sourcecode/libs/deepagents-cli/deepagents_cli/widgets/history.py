"""입력 히스토리를 파일로 지속(persist)하기 위한 커맨드 히스토리 관리자입니다."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - used at runtime in type hints


class HistoryManager:
    """파일 지속을 포함한 커맨드 히스토리를 관리합니다.

    Uses append-only writes for concurrent safety. Multiple agents can
    safely write to the same history file without corruption.
    """

    def __init__(self, history_file: Path, max_entries: int = 100) -> None:
        """히스토리 관리자를 초기화합니다.

        Args:
            history_file: Path to the JSON-lines history file
            max_entries: Maximum number of entries to keep
        """
        self.history_file = history_file
        self.max_entries = max_entries
        self._entries: list[str] = []
        self._current_index: int = -1
        self._temp_input: str = ""
        self._load_history()

    def _load_history(self) -> None:
        """파일에서 히스토리를 로드합니다."""
        if not self.history_file.exists():
            return

        try:
            with self.history_file.open("r", encoding="utf-8") as f:
                entries = []
                for raw_line in f:
                    line = raw_line.rstrip("\n\r")
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        entry = line
                    entries.append(entry if isinstance(entry, str) else str(entry))
                self._entries = entries[-self.max_entries :]
        except (OSError, UnicodeDecodeError):
            self._entries = []

    def _append_to_file(self, text: str) -> None:
        """히스토리 파일에 항목 하나를 append 합니다(concurrent-safe)."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with self.history_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(text) + "\n")
        except OSError:
            pass

    def _compact_history(self) -> None:
        """오래된 항목을 제거하기 위해 히스토리 파일을 재작성합니다.

        Only called when entries exceed 2x max_entries to minimize rewrites.
        """
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with self.history_file.open("w", encoding="utf-8") as f:
                for entry in self._entries:
                    f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def add(self, text: str) -> None:
        """커맨드를 히스토리에 추가합니다.

        Args:
            text: The command text to add
        """
        text = text.strip()
        # 빈 문자열 또는 slash 커맨드는 스킵
        if not text or text.startswith("/"):
            return

        # 직전 항목과 중복이면 스킵
        if self._entries and self._entries[-1] == text:
            return

        self._entries.append(text)

        # 파일에 append(빠르고 concurrent-safe)
        self._append_to_file(text)

        # 엔트리가 2배를 초과할 때만 compact(드문 작업)
        if len(self._entries) > self.max_entries * 2:
            self._entries = self._entries[-self.max_entries :]
            self._compact_history()

        self.reset_navigation()

    def get_previous(self, current_input: str, prefix: str = "") -> str | None:
        """이전 히스토리 항목을 가져옵니다.

        Args:
            current_input: Current input text (saved on first navigation)
            prefix: Optional prefix to filter entries

        Returns:
            Previous matching entry or None
        """
        if not self._entries:
            return None

        # 첫 네비게이션 시 현재 입력을 저장
        if self._current_index == -1:
            self._temp_input = current_input
            self._current_index = len(self._entries)

        # 뒤로 탐색하며 prefix에 매칭되는 항목을 찾음
        for i in range(self._current_index - 1, -1, -1):
            if self._entries[i].startswith(prefix):
                self._current_index = i
                return self._entries[i]

        return None

    def get_next(self, prefix: str = "") -> str | None:
        """다음 히스토리 항목을 가져옵니다.

        Args:
            prefix: Optional prefix to filter entries

        Returns:
            Next matching entry, original input at end, or None
        """
        if self._current_index == -1:
            return None

        # 앞으로 탐색하며 prefix에 매칭되는 항목을 찾음
        for i in range(self._current_index + 1, len(self._entries)):
            if self._entries[i].startswith(prefix):
                self._current_index = i
                return self._entries[i]

        # 끝까지 가면 원래 입력으로 복귀
        result = self._temp_input
        self.reset_navigation()
        return result

    def reset_navigation(self) -> None:
        """네비게이션 상태를 초기화합니다."""
        self._current_index = -1
        self._temp_input = ""
