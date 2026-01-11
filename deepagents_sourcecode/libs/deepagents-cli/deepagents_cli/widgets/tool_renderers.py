"""승인(approval) 위젯용 tool renderer들(레지스트리 패턴)입니다."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

from deepagents_cli.widgets.tool_widgets import (
    BashApprovalWidget,
    EditFileApprovalWidget,
    GenericApprovalWidget,
    WriteFileApprovalWidget,
)

if TYPE_CHECKING:
    from deepagents_cli.widgets.tool_widgets import ToolApprovalWidget

DIFF_HEADER_LINES = 2


class ToolRenderer:
    """tool 승인 위젯 렌더러의 베이스 클래스입니다."""

    def get_approval_widget(
        self, tool_args: dict[str, Any]
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        """이 tool에 대한 승인 위젯 클래스와 데이터를 반환합니다.

        Args:
            tool_args: The tool arguments from action_request

        Returns:
            Tuple of (widget_class, data_dict)
        """
        return GenericApprovalWidget, tool_args


class WriteFileRenderer(ToolRenderer):
    """`write_file` tool 렌더러(전체 파일 내용을 표시)."""

    def get_approval_widget(
        self, tool_args: dict[str, Any]
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        """`write_file` 요청을 표시할 승인 위젯과 데이터를 구성합니다."""
        # 문법 하이라이팅을 위해 확장자를 추출
        file_path = tool_args.get("file_path", "")
        content = tool_args.get("content", "")

        # 파일 확장자
        file_extension = "text"
        if "." in file_path:
            file_extension = file_path.rsplit(".", 1)[-1]

        data = {
            "file_path": file_path,
            "content": content,
            "file_extension": file_extension,
        }
        return WriteFileApprovalWidget, data


class EditFileRenderer(ToolRenderer):
    """`edit_file` tool 렌더러(unified diff 표시)."""

    def get_approval_widget(
        self, tool_args: dict[str, Any]
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        """`edit_file` 요청을 unified diff 형태로 표시할 승인 위젯/데이터를 구성합니다."""
        file_path = tool_args.get("file_path", "")
        old_string = tool_args.get("old_string", "")
        new_string = tool_args.get("new_string", "")

        # unified diff 생성
        diff_lines = self._generate_diff(old_string, new_string)

        data = {
            "file_path": file_path,
            "diff_lines": diff_lines,
            "old_string": old_string,
            "new_string": new_string,
        }
        return EditFileApprovalWidget, data

    def _generate_diff(self, old_string: str, new_string: str) -> list[str]:
        """old/new 문자열로부터 unified diff 라인을 생성합니다."""
        if not old_string and not new_string:
            return []

        old_lines = old_string.split("\n") if old_string else []
        new_lines = new_string.split("\n") if new_string else []

        # unified diff 생성
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="before",
            tofile="after",
            lineterm="",
            n=3,  # Context lines
        )

        # 헤더 라인(---, +++)은 제외
        diff_list = list(diff)
        return diff_list[DIFF_HEADER_LINES:] if len(diff_list) > DIFF_HEADER_LINES else diff_list


class BashRenderer(ToolRenderer):
    """`bash`/`shell` tool 렌더러(커맨드 표시)."""

    def get_approval_widget(
        self, tool_args: dict[str, Any]
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        """`bash`/`shell` 요청을 표시할 승인 위젯/데이터를 구성합니다."""
        data = {
            "command": tool_args.get("command", ""),
            "description": tool_args.get("description", ""),
        }
        return BashApprovalWidget, data


# tool 이름 → renderer 매핑 레지스트리
_RENDERER_REGISTRY: dict[str, type[ToolRenderer]] = {
    "write_file": WriteFileRenderer,
    "edit_file": EditFileRenderer,
    "bash": BashRenderer,
    "shell": BashRenderer,
}


def get_renderer(tool_name: str) -> ToolRenderer:
    """도구 이름에 맞는 renderer를 반환합니다.

    Args:
        tool_name: The name of the tool

    Returns:
        The appropriate ToolRenderer instance
    """
    renderer_class = _RENDERER_REGISTRY.get(tool_name, ToolRenderer)
    return renderer_class()
