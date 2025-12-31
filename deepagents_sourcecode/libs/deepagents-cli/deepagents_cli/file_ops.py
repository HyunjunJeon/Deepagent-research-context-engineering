"""CLI 표시를 위한 파일 작업 추적 및 diff 계산 도움말."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from deepagents.backends.utils import perform_string_replacement

from deepagents_cli.config import settings

if TYPE_CHECKING:
    from deepagents.backends.protocol import BACKEND_TYPES

FileOpStatus = Literal["pending", "success", "error"]


@dataclass
class ApprovalPreview:
    """HITL 미리보기를 렌더링하는 데 사용되는 데이터."""

    title: str
    details: list[str]
    diff: str | None = None
    diff_title: str | None = None
    error: str | None = None


def _safe_read(path: Path) -> str | None:
    """파일 내용을 읽고, 실패 시 None을 반환합니다."""
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError):
        return None


def _count_lines(text: str) -> int:
    """빈 문자열을 0줄로 취급하여 텍스트의 줄 수를 셉니다."""
    if not text:
        return 0
    return len(text.splitlines())


def compute_unified_diff(
    before: str,
    after: str,
    display_path: str,
    *,
    max_lines: int | None = 800,
    context_lines: int = 3,
) -> str | None:
    """이전 내용과 이후 내용 간의 통합 diff를 계산합니다.

    Args:
        before: 원본 내용
        after: 새로운 내용
        display_path: diff 헤더에 표시할 경로
        max_lines: 최대 diff 줄 수 (제한 없으면 None)
        context_lines: 변경 사항 주변의 컨텍스트 줄 수 (기본값 3)

    Returns:
        통합 diff 문자열 또는 변경 사항이 없는 경우 None
    """
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    diff_lines = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"{display_path} (before)",
            tofile=f"{display_path} (after)",
            lineterm="",
            n=context_lines,
        )
    )
    if not diff_lines:
        return None
    if max_lines is not None and len(diff_lines) > max_lines:
        truncated = diff_lines[: max_lines - 1]
        truncated.append("...")
        return "\n".join(truncated)
    return "\n".join(diff_lines)


@dataclass
class FileOpMetrics:
    """파일 작업에 대한 줄 및 바이트 수준 메트릭."""

    lines_read: int = 0
    start_line: int | None = None
    end_line: int | None = None
    lines_written: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    bytes_written: int = 0


@dataclass
class FileOperationRecord:
    """단일 파일시스템 도구 호출을 추적합니다."""

    tool_name: str
    display_path: str
    physical_path: Path | None
    tool_call_id: str | None
    args: dict[str, Any] = field(default_factory=dict)
    status: FileOpStatus = "pending"
    error: str | None = None
    metrics: FileOpMetrics = field(default_factory=FileOpMetrics)
    diff: str | None = None
    before_content: str | None = None
    after_content: str | None = None
    read_output: str | None = None
    hitl_approved: bool = False


def resolve_physical_path(path_str: str | None, assistant_id: str | None) -> Path | None:
    """가상/상대 경로를 실제 파일시스템 경로로 변환합니다."""
    if not path_str:
        return None
    try:
        if assistant_id and path_str.startswith("/memories/"):
            agent_dir = settings.get_agent_dir(assistant_id)
            suffix = path_str.removeprefix("/memories/").lstrip("/")
            return (agent_dir / suffix).resolve()
        path = Path(path_str)
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()
    except (OSError, ValueError):
        return None


def format_display_path(path_str: str | None) -> str:
    """표시용으로 경로를 포맷합니다."""
    if not path_str:
        return "(알 수 없음)"
    try:
        path = Path(path_str)
        if path.is_absolute():
            return path.name or str(path)
        return str(path)
    except (OSError, ValueError):
        return str(path_str)


def build_approval_preview(
    tool_name: str,
    args: dict[str, Any],
    assistant_id: str | None,
) -> ApprovalPreview | None:
    """HITL 승인을 위한 요약 정보 및 diff를 수집합니다."""
    path_str = str(args.get("file_path") or args.get("path") or "")
    display_path = format_display_path(path_str)
    physical_path = resolve_physical_path(path_str, assistant_id)

    if tool_name == "write_file":
        content = str(args.get("content", ""))
        before = _safe_read(physical_path) if physical_path and physical_path.exists() else ""
        after = content
        diff = compute_unified_diff(before or "", after, display_path, max_lines=100)
        additions = 0
        if diff:
            additions = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
        total_lines = _count_lines(after)
        details = [
            f"파일: {path_str}",
            "작업: 새 파일 생성" + (" (기존 내용 덮어씀)" if before else ""),
            f"작성할 줄 수: {additions or total_lines}",
        ]
        return ApprovalPreview(
            title=f"{display_path} 쓰기",
            details=details,
            diff=diff,
            diff_title=f"{display_path} 차이(Diff)",
        )

    if tool_name == "edit_file":
        if physical_path is None:
            return ApprovalPreview(
                title=f"{display_path} 업데이트",
                details=[f"파일: {path_str}", "작업: 텍스트 교체"],
                error="파일 경로를 확인할 수 없습니다.",
            )
        before = _safe_read(physical_path)
        if before is None:
            return ApprovalPreview(
                title=f"{display_path} 업데이트",
                details=[f"파일: {path_str}", "작업: 텍스트 교체"],
                error="현재 파일 내용을 읽을 수 없습니다.",
            )
        old_string = str(args.get("old_string", ""))
        new_string = str(args.get("new_string", ""))
        replace_all = bool(args.get("replace_all", False))
        replacement = perform_string_replacement(before, old_string, new_string, replace_all)
        if isinstance(replacement, str):
            return ApprovalPreview(
                title=f"{display_path} 업데이트",
                details=[f"파일: {path_str}", "작업: 텍스트 교체"],
                error=replacement,
            )
        after, occurrences = replacement
        diff = compute_unified_diff(before, after, display_path, max_lines=None)
        additions = 0
        deletions = 0
        if diff:
            additions = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
            deletions = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
        details = [
            f"파일: {path_str}",
            f"작업: 텍스트 교체 ({'모든 발생' if replace_all else '단일 발생'})",
            f"일치하는 발생: {occurrences}",
            f"변경된 줄: +{additions} / -{deletions}",
        ]
        return ApprovalPreview(
            title=f"{display_path} 업데이트",
            details=details,
            diff=diff,
            diff_title=f"{display_path} 차이(Diff)",
        )

    return None


class FileOpTracker:
    """CLI 상호작용 중 파일 작업 메트릭을 수집합니다."""

    def __init__(self, *, assistant_id: str | None, backend: BACKEND_TYPES | None = None) -> None:
        """추적기를 초기화합니다."""
        self.assistant_id = assistant_id
        self.backend = backend
        self.active: dict[str | None, FileOperationRecord] = {}
        self.completed: list[FileOperationRecord] = []

    def start_operation(self, tool_name: str, args: dict[str, Any], tool_call_id: str | None) -> None:
        if tool_name not in {"read_file", "write_file", "edit_file"}:
            return
        path_str = str(args.get("file_path") or args.get("path") or "")
        display_path = format_display_path(path_str)
        record = FileOperationRecord(
            tool_name=tool_name,
            display_path=display_path,
            physical_path=resolve_physical_path(path_str, self.assistant_id),
            tool_call_id=tool_call_id,
            args=args,
        )
        if tool_name in {"write_file", "edit_file"}:
            if self.backend and path_str:
                try:
                    responses = self.backend.download_files([path_str])
                    if responses and responses[0].content is not None and responses[0].error is None:
                        record.before_content = responses[0].content.decode("utf-8")
                    else:
                        record.before_content = ""
                except Exception:
                    record.before_content = ""
            elif record.physical_path:
                record.before_content = _safe_read(record.physical_path) or ""
        self.active[tool_call_id] = record

    def update_args(self, tool_call_id: str, args: dict[str, Any]) -> None:
        """활성 작업의 인수를 업데이트하고 before_content 캡처를 다시 시도합니다."""
        record = self.active.get(tool_call_id)
        if not record:
            return

        record.args.update(args)

        # If we haven't captured before_content yet, try again now that we might have the path
        if record.before_content is None and record.tool_name in {"write_file", "edit_file"}:
            path_str = str(record.args.get("file_path") or record.args.get("path") or "")
            if path_str:
                record.display_path = format_display_path(path_str)
                record.physical_path = resolve_physical_path(path_str, self.assistant_id)
                if self.backend:
                    try:
                        responses = self.backend.download_files([path_str])
                        if responses and responses[0].content is not None and responses[0].error is None:
                            record.before_content = responses[0].content.decode("utf-8")
                        else:
                            record.before_content = ""
                    except Exception:
                        record.before_content = ""
                elif record.physical_path:
                    record.before_content = _safe_read(record.physical_path) or ""

    def complete_with_message(self, tool_message: Any) -> FileOperationRecord | None:
        tool_call_id = getattr(tool_message, "tool_call_id", None)
        record = self.active.get(tool_call_id)
        if record is None:
            return None

        content = tool_message.content
        if isinstance(content, list):
            # Some tool messages may return list segments; join them for analysis.
            joined = []
            for item in content:
                if isinstance(item, str):
                    joined.append(item)
                else:
                    joined.append(str(item))
            content_text = "\n".join(joined)
        else:
            content_text = str(content) if content is not None else ""

        if getattr(tool_message, "status", "success") != "success" or content_text.lower().startswith("error"):
            record.status = "error"
            record.error = content_text
            self._finalize(record)
            return record

        record.status = "success"

        if record.tool_name == "read_file":
            record.read_output = content_text
            lines = _count_lines(content_text)
            record.metrics.lines_read = lines
            offset = record.args.get("offset")
            limit = record.args.get("limit")
            if isinstance(offset, int):
                if offset > lines:
                    offset = 0
                record.metrics.start_line = offset + 1
                if lines:
                    record.metrics.end_line = offset + lines
            elif lines:
                record.metrics.start_line = 1
                record.metrics.end_line = lines
            if isinstance(limit, int) and lines > limit:
                record.metrics.end_line = (record.metrics.start_line or 1) + limit - 1
        else:
            # For write/edit operations, read back from backend (or local filesystem)
            self._populate_after_content(record)
            if record.after_content is None:
                record.status = "error"
                record.error = "업데이트된 파일 내용을 읽을 수 없습니다."
                self._finalize(record)
                return record
            record.metrics.lines_written = _count_lines(record.after_content)
            before_lines = _count_lines(record.before_content or "")
            diff = compute_unified_diff(
                record.before_content or "",
                record.after_content,
                record.display_path,
                max_lines=100,
            )
            record.diff = diff
            if diff:
                additions = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
                deletions = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
                record.metrics.lines_added = additions
                record.metrics.lines_removed = deletions
            elif record.tool_name == "write_file" and (record.before_content or "") == "":
                record.metrics.lines_added = record.metrics.lines_written
            record.metrics.bytes_written = len(record.after_content.encode("utf-8"))
            if record.diff is None and (record.before_content or "") != record.after_content:
                record.diff = compute_unified_diff(
                    record.before_content or "",
                    record.after_content,
                    record.display_path,
                    max_lines=100,
                )
            if record.diff is None and before_lines != record.metrics.lines_written:
                record.metrics.lines_added = max(record.metrics.lines_written - before_lines, 0)

        self._finalize(record)
        return record

    def mark_hitl_approved(self, tool_name: str, args: dict[str, Any]) -> None:
        """tool_name 및 file_path와 일치하는 작업을 HIL 승인됨으로 표시합니다."""
        file_path = args.get("file_path") or args.get("path")
        if not file_path:
            return

        # Mark all active records that match
        for record in self.active.values():
            if record.tool_name == tool_name:
                record_path = record.args.get("file_path") or record.args.get("path")
                if record_path == file_path:
                    record.hitl_approved = True

    def _populate_after_content(self, record: FileOperationRecord) -> None:
        # Use backend if available (works for any BackendProtocol implementation)
        if self.backend:
            try:
                file_path = record.args.get("file_path") or record.args.get("path")
                if file_path:
                    responses = self.backend.download_files([file_path])
                    if responses and responses[0].content is not None and responses[0].error is None:
                        record.after_content = responses[0].content.decode("utf-8")
                    else:
                        record.after_content = None
                else:
                    record.after_content = None
            except Exception:
                record.after_content = None
        else:
            # Fallback: direct filesystem read when no backend provided
            if record.physical_path is None:
                record.after_content = None
                return
            record.after_content = _safe_read(record.physical_path)

    def _finalize(self, record: FileOperationRecord) -> None:
        self.completed.append(record)
        self.active.pop(record.tool_call_id, None)
