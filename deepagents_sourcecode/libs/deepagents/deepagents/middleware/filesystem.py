"""에이전트에 파일 시스템 도구를 제공하는 미들웨어입니다."""
# ruff: noqa: E501

import os
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Literal, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command
from typing_extensions import TypedDict

from deepagents.backends import StateBackend

# Re-export type here for backwards compatibility
from deepagents.backends.protocol import BACKEND_TYPES as BACKEND_TYPES
from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.utils import (
    format_content_with_line_numbers,
    format_grep_matches,
    sanitize_tool_call_id,
    truncate_if_too_long,
)

EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
MAX_LINE_LENGTH = 2000
LINE_NUMBER_WIDTH = 6
DEFAULT_READ_OFFSET = 0
DEFAULT_READ_LIMIT = 500


class FileData(TypedDict):
    """파일 내용을 메타데이터와 함께 저장하기 위한 데이터 구조입니다."""

    content: list[str]
    """파일의 각 라인."""

    created_at: str
    """파일 생성 시각(ISO 8601)."""

    modified_at: str
    """파일 마지막 수정 시각(ISO 8601)."""


def _file_data_reducer(left: dict[str, FileData] | None, right: dict[str, FileData | None]) -> dict[str, FileData]:
    """파일 업데이트를 병합하며, 삭제를 지원합니다.

    오른쪽 딕셔너리의 값이 `None`인 엔트리를 “삭제 마커”로 취급해 삭제를 구현합니다.
    LangGraph의 state 관리에서 annotated reducer가 state 업데이트 병합 방식을 제어한다는
    전제에 맞춰 설계되었습니다.

    Args:
        left: Existing files dictionary. May be `None` during initialization.
        right: New files dictionary to merge. Files with `None` values are
            treated as deletion markers and removed from the result.

    Returns:
        Merged dictionary where right overwrites left for matching keys,
        and `None` values in right trigger deletions.

    Example:
        ```python
        existing = {"/file1.txt": FileData(...), "/file2.txt": FileData(...)}
        updates = {"/file2.txt": None, "/file3.txt": FileData(...)}
        result = file_data_reducer(existing, updates)
        # Result: {"/file1.txt": FileData(...), "/file3.txt": FileData(...)}
        ```
    """
    if left is None:
        return {k: v for k, v in right.items() if v is not None}

    result = {**left}
    for key, value in right.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def _validate_path(path: str, *, allowed_prefixes: Sequence[str] | None = None) -> str:
    r"""보안 관점에서 파일 경로를 검증하고 정규화합니다.

    디렉토리 트래버설 공격을 방지하고, 일관된 포맷을 강제하여 안전한 경로만 사용하도록 합니다.
    모든 경로는 `/`로 시작하며, 경로 구분자는 forward slash(`/`)로 정규화됩니다.

    이 함수는 “가상 파일시스템 경로(virtual paths)”를 대상으로 설계되었으며,
    경로 형식의 모호함을 피하기 위해 Windows 절대 경로(예: `C:/...`, `F:/...`)는 거부합니다.

    Args:
        path: The path to validate and normalize.
        allowed_prefixes: Optional list of allowed path prefixes. If provided,
            the normalized path must start with one of these prefixes.

    Returns:
        Normalized canonical path starting with `/` and using forward slashes.

    Raises:
        ValueError: If path contains traversal sequences (`..` or `~`), is a
            Windows absolute path (e.g., C:/...), or does not start with an
            allowed prefix when `allowed_prefixes` is specified.

    Example:
        ```python
        validate_path("foo/bar")  # Returns: "/foo/bar"
        validate_path("/./foo//bar")  # Returns: "/foo/bar"
        validate_path("../etc/passwd")  # Raises ValueError
        validate_path(r"C:\\Users\\file.txt")  # Raises ValueError
        validate_path("/data/file.txt", allowed_prefixes=["/data/"])  # OK
        validate_path("/etc/file.txt", allowed_prefixes=["/data/"])  # Raises ValueError
        ```
    """
    if ".." in path or path.startswith("~"):
        msg = f"Path traversal not allowed: {path}"
        raise ValueError(msg)

    # Windows 절대 경로(예: C:\..., D:/...)는 거부합니다.
    # 가상 파일시스템 경로 포맷의 일관성을 유지하기 위함입니다.
    if re.match(r"^[a-zA-Z]:", path):
        msg = f"Windows absolute paths are not supported: {path}. Please use virtual paths starting with / (e.g., /workspace/file.txt)"
        raise ValueError(msg)

    normalized = os.path.normpath(path)
    normalized = normalized.replace("\\", "/")

    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    if allowed_prefixes is not None and not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        msg = f"Path must start with one of {allowed_prefixes}: {path}"
        raise ValueError(msg)

    return normalized


class FilesystemState(AgentState):
    """FilesystemMiddleware의 state 스키마입니다."""

    files: Annotated[NotRequired[dict[str, FileData]], _file_data_reducer]
    """파일 시스템에 저장된 파일들."""


LIST_FILES_TOOL_DESCRIPTION = """Lists all files in the filesystem, filtering by directory.

Usage:
- The path parameter must be an absolute path, not a relative path
- The list_files tool will return a list of all files in the specified directory.
- This is very useful for exploring the file system and finding the right file to read or edit.
- You should almost ALWAYS use this tool before using the Read or Edit tools."""

READ_FILE_TOOL_DESCRIPTION = """Reads a file from the filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to 500 lines starting from the beginning of the file
- **IMPORTANT for large files and codebase exploration**: Use pagination with offset and limit parameters to avoid context overflow
  - First scan: read_file(path, limit=100) to see file structure
  - Read more sections: read_file(path, offset=100, limit=200) for next 200 lines
  - Only omit limit (read full file) when necessary for editing
- Specify offset and limit: read_file(path, offset=0, limit=100) reads first 100 lines
- Any lines longer than 2000 characters will be truncated
- Results are returned using cat -n format, with line numbers starting at 1
- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
- You should ALWAYS make sure a file has been read before editing it."""

EDIT_FILE_TOOL_DESCRIPTION = """Performs exact string replacements in files.

Usage:
- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: spaces + line number + tab. Everything after that tab is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
- ALWAYS prefer editing existing files. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance."""


WRITE_FILE_TOOL_DESCRIPTION = """Writes to a new file in the filesystem.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- The content parameter must be a string
- The write_file tool will create the a new file.
- Prefer to edit existing files over creating new ones when possible."""


GLOB_TOOL_DESCRIPTION = """Find files matching a glob pattern.

Usage:
- The glob tool finds files by matching patterns with wildcards
- Supports standard glob patterns: `*` (any characters), `**` (any directories), `?` (single character)
- Patterns can be absolute (starting with `/`) or relative
- Returns a list of absolute file paths that match the pattern

Examples:
- `**/*.py` - Find all Python files
- `*.txt` - Find all text files in root
- `/subdir/**/*.md` - Find all markdown files under /subdir"""

GREP_TOOL_DESCRIPTION = """Search for a pattern in files.

Usage:
- The grep tool searches for text patterns across files
- The pattern parameter is the text to search for (literal string, not regex)
- The path parameter filters which directory to search in (default is the current working directory)
- The glob parameter accepts a glob pattern to filter which files to search (e.g., `*.py`)
- The output_mode parameter controls the output format:
  - `files_with_matches`: List only file paths containing matches (default)
  - `content`: Show matching lines with file path and line numbers
  - `count`: Show count of matches per file

Examples:
- Search all files: `grep(pattern="TODO")`
- Search Python files only: `grep(pattern="import", glob="*.py")`
- Show matching lines: `grep(pattern="error", output_mode="content")`"""

EXECUTE_TOOL_DESCRIPTION = """Executes a given command in the sandbox environment with proper handling and security measures.

Before executing the command, please follow these steps:

1. Directory Verification:
   - If the command will create new directories or files, first use the ls tool to verify the parent directory exists and is the correct location
   - For example, before running "mkdir foo/bar", first use ls to check that "foo" exists and is the intended parent directory

2. Command Execution:
   - Always quote file paths that contain spaces with double quotes (e.g., cd "path with spaces/file.txt")
   - Examples of proper quoting:
     - cd "/Users/name/My Documents" (correct)
     - cd /Users/name/My Documents (incorrect - will fail)
     - python "/path/with spaces/script.py" (correct)
     - python /path/with spaces/script.py (incorrect - will fail)
   - After ensuring proper quoting, execute the command
   - Capture the output of the command

Usage notes:
  - The command parameter is required
  - Commands run in an isolated sandbox environment
  - Returns combined stdout/stderr output with exit code
  - If the output is very large, it may be truncated
  - VERY IMPORTANT: You MUST avoid using search commands like find and grep. Instead use the grep, glob tools to search. You MUST avoid read tools like cat, head, tail, and use read_file to read files.
  - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines (newlines are ok in quoted strings)
    - Use '&&' when commands depend on each other (e.g., "mkdir dir && cd dir")
    - Use ';' only when you need to run commands sequentially but don't care if earlier commands fail
  - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of cd

Examples:
  Good examples:
    - execute(command="pytest /foo/bar/tests")
    - execute(command="python /path/to/script.py")
    - execute(command="npm install && npm test")

  Bad examples (avoid these):
    - execute(command="cd /foo/bar && pytest tests")  # Use absolute path instead
    - execute(command="cat file.txt")  # Use read_file tool instead
    - execute(command="find . -name '*.py'")  # Use glob tool instead
    - execute(command="grep -r 'pattern' .")  # Use grep tool instead

Note: This tool is only available if the backend supports execution (SandboxBackendProtocol).
If execution is not supported, the tool will return an error message."""

FILESYSTEM_SYSTEM_PROMPT = """## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`

You have access to a filesystem which you can interact with using these tools.
All file paths must start with a /.

- ls: list files in a directory (requires absolute path)
- read_file: read a file from the filesystem
- write_file: write to a file in the filesystem
- edit_file: edit a file in the filesystem
- glob: find files matching a pattern (e.g., "**/*.py")
- grep: search for text within files"""

EXECUTION_SYSTEM_PROMPT = """## Execute Tool `execute`

You have access to an `execute` tool for running shell commands in a sandboxed environment.
Use this tool to run commands, scripts, tests, builds, and other shell operations.

- execute: run a shell command in the sandbox (returns output and exit code)"""


def _get_backend(backend: BACKEND_TYPES, runtime: ToolRuntime) -> BackendProtocol:
    """백엔드 또는 팩토리에서 해결된 백엔드 인스턴스를 가져옵니다.

    Args:
        backend: 백엔드 인스턴스 또는 팩토리 함수.
        runtime: 도구 런타임 컨텍스트.

    Returns:
        해결된 백엔드 인스턴스.
    """
    if callable(backend):
        return backend(runtime)
    return backend


def _ls_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """파일 목록(ls) 도구를 생성합니다.

    Args:
        backend: 파일 저장에 사용할 백엔드, 또는 런타임을 받아 백엔드를 반환하는 팩토리 함수.
        custom_description: 도구의 선택적 사용자 정의 설명.

    Returns:
        백엔드를 사용하여 파일을 나열하는 구성된 ls 도구.
    """
    tool_description = custom_description or LIST_FILES_TOOL_DESCRIPTION

    def sync_ls(runtime: ToolRuntime[None, FilesystemState], path: str) -> str:
        """파일 목록(ls) 도구의 동기 래퍼입니다."""
        resolved_backend = _get_backend(backend, runtime)
        validated_path = _validate_path(path)
        infos = resolved_backend.ls_info(validated_path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    async def async_ls(runtime: ToolRuntime[None, FilesystemState], path: str) -> str:
        """파일 목록(ls) 도구의 비동기 래퍼입니다."""
        resolved_backend = _get_backend(backend, runtime)
        validated_path = _validate_path(path)
        infos = await resolved_backend.als_info(validated_path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    return StructuredTool.from_function(
        name="ls",
        description=tool_description,
        func=sync_ls,
        coroutine=async_ls,
    )


def _read_file_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """`read_file` 도구를 생성합니다.

    Args:
        backend: 파일 저장에 사용할 백엔드 또는 (runtime을 받아 백엔드를 반환하는) 팩토리 함수.
        custom_description: 도구 설명을 커스텀할 때 사용(선택).

    Returns:
        backend를 통해 파일을 읽는 `read_file` 도구.
    """
    tool_description = custom_description or READ_FILE_TOOL_DESCRIPTION

    def sync_read_file(
        file_path: str,
        runtime: ToolRuntime[None, FilesystemState],
        offset: int = DEFAULT_READ_OFFSET,
        limit: int = DEFAULT_READ_LIMIT,
    ) -> str:
        """`read_file` 도구의 동기 래퍼입니다."""
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        return resolved_backend.read(file_path, offset=offset, limit=limit)

    async def async_read_file(
        file_path: str,
        runtime: ToolRuntime[None, FilesystemState],
        offset: int = DEFAULT_READ_OFFSET,
        limit: int = DEFAULT_READ_LIMIT,
    ) -> str:
        """`read_file` 도구의 비동기 래퍼입니다."""
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        return await resolved_backend.aread(file_path, offset=offset, limit=limit)

    return StructuredTool.from_function(
        name="read_file",
        description=tool_description,
        func=sync_read_file,
        coroutine=async_read_file,
    )


def _write_file_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """`write_file` 도구를 생성합니다.

    Args:
        backend: 파일 저장에 사용할 백엔드 또는 (runtime을 받아 백엔드를 반환하는) 팩토리 함수.
        custom_description: 도구 설명을 커스텀할 때 사용(선택).

    Returns:
        backend를 통해 새 파일을 생성하는 `write_file` 도구.
    """
    tool_description = custom_description or WRITE_FILE_TOOL_DESCRIPTION

    def sync_write_file(
        file_path: str,
        content: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> Command | str:
        """`write_file` 도구의 동기 래퍼입니다."""
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        res: WriteResult = resolved_backend.write(file_path, content)
        if res.error:
            return res.error
        # backend가 state 업데이트를 반환하면, ToolMessage와 함께 Command로 감쌉니다.
        if res.files_update is not None:
            return Command(
                update={
                    "files": res.files_update,
                    "messages": [
                        ToolMessage(
                            content=f"Updated file {res.path}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        return f"Updated file {res.path}"

    async def async_write_file(
        file_path: str,
        content: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> Command | str:
        """`write_file` 도구의 비동기 래퍼입니다."""
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        res: WriteResult = await resolved_backend.awrite(file_path, content)
        if res.error:
            return res.error
        # backend가 state 업데이트를 반환하면, ToolMessage와 함께 Command로 감쌉니다.
        if res.files_update is not None:
            return Command(
                update={
                    "files": res.files_update,
                    "messages": [
                        ToolMessage(
                            content=f"Updated file {res.path}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        return f"Updated file {res.path}"

    return StructuredTool.from_function(
        name="write_file",
        description=tool_description,
        func=sync_write_file,
        coroutine=async_write_file,
    )


def _edit_file_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """`edit_file` 도구를 생성합니다.

    Args:
        backend: 파일 저장에 사용할 백엔드 또는 (runtime을 받아 백엔드를 반환하는) 팩토리 함수.
        custom_description: 도구 설명을 커스텀할 때 사용(선택).

    Returns:
        backend를 통해 파일 내 문자열 치환을 수행하는 `edit_file` 도구.
    """
    tool_description = custom_description or EDIT_FILE_TOOL_DESCRIPTION

    def sync_edit_file(
        file_path: str,
        old_string: str,
        new_string: str,
        runtime: ToolRuntime[None, FilesystemState],
        *,
        replace_all: bool = False,
    ) -> Command | str:
        """`edit_file` 도구의 동기 래퍼입니다."""
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        res: EditResult = resolved_backend.edit(file_path, old_string, new_string, replace_all=replace_all)
        if res.error:
            return res.error
        if res.files_update is not None:
            return Command(
                update={
                    "files": res.files_update,
                    "messages": [
                        ToolMessage(
                            content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

    async def async_edit_file(
        file_path: str,
        old_string: str,
        new_string: str,
        runtime: ToolRuntime[None, FilesystemState],
        *,
        replace_all: bool = False,
    ) -> Command | str:
        """`edit_file` 도구의 비동기 래퍼입니다."""
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        res: EditResult = await resolved_backend.aedit(file_path, old_string, new_string, replace_all=replace_all)
        if res.error:
            return res.error
        if res.files_update is not None:
            return Command(
                update={
                    "files": res.files_update,
                    "messages": [
                        ToolMessage(
                            content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

    return StructuredTool.from_function(
        name="edit_file",
        description=tool_description,
        func=sync_edit_file,
        coroutine=async_edit_file,
    )


def _glob_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """`glob` 도구를 생성합니다.

    Args:
        backend: 파일 저장에 사용할 백엔드 또는 (runtime을 받아 백엔드를 반환하는) 팩토리 함수.
        custom_description: 도구 설명을 커스텀할 때 사용(선택).

    Returns:
        backend를 통해 패턴 매칭으로 파일을 찾는 `glob` 도구.
    """
    tool_description = custom_description or GLOB_TOOL_DESCRIPTION

    def sync_glob(pattern: str, runtime: ToolRuntime[None, FilesystemState], path: str = "/") -> str:
        """`glob` 도구의 동기 래퍼입니다."""
        resolved_backend = _get_backend(backend, runtime)
        infos = resolved_backend.glob_info(pattern, path=path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    async def async_glob(pattern: str, runtime: ToolRuntime[None, FilesystemState], path: str = "/") -> str:
        """`glob` 도구의 비동기 래퍼입니다."""
        resolved_backend = _get_backend(backend, runtime)
        infos = await resolved_backend.aglob_info(pattern, path=path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    return StructuredTool.from_function(
        name="glob",
        description=tool_description,
        func=sync_glob,
        coroutine=async_glob,
    )


def _grep_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """`grep` 도구를 생성합니다.

    Args:
        backend: 파일 저장에 사용할 백엔드 또는 (runtime을 받아 백엔드를 반환하는) 팩토리 함수.
        custom_description: 도구 설명을 커스텀할 때 사용(선택).

    Returns:
        backend를 통해 파일 내 패턴 검색을 수행하는 `grep` 도구.
    """
    tool_description = custom_description or GREP_TOOL_DESCRIPTION

    def sync_grep(
        pattern: str,
        runtime: ToolRuntime[None, FilesystemState],
        path: str | None = None,
        glob: str | None = None,
        output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
    ) -> str:
        """`grep` 도구의 동기 래퍼입니다."""
        resolved_backend = _get_backend(backend, runtime)
        raw = resolved_backend.grep_raw(pattern, path=path, glob=glob)
        if isinstance(raw, str):
            return raw
        formatted = format_grep_matches(raw, output_mode)
        return truncate_if_too_long(formatted)  # type: ignore[arg-type]

    async def async_grep(
        pattern: str,
        runtime: ToolRuntime[None, FilesystemState],
        path: str | None = None,
        glob: str | None = None,
        output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
    ) -> str:
        """`grep` 도구의 비동기 래퍼입니다."""
        resolved_backend = _get_backend(backend, runtime)
        raw = await resolved_backend.agrep_raw(pattern, path=path, glob=glob)
        if isinstance(raw, str):
            return raw
        formatted = format_grep_matches(raw, output_mode)
        return truncate_if_too_long(formatted)  # type: ignore[arg-type]

    return StructuredTool.from_function(
        name="grep",
        description=tool_description,
        func=sync_grep,
        coroutine=async_grep,
    )


def _supports_execution(backend: BackendProtocol) -> bool:
    """backend가 커맨드 실행을 지원하는지 확인합니다.

    - `CompositeBackend`인 경우: `default` backend가 실행을 지원하는지 확인합니다.
    - 그 외의 경우: `SandboxBackendProtocol` 구현 여부로 판단합니다.

    Args:
        backend: 확인할 backend.

    Returns:
        실행을 지원하면 `True`, 아니면 `False`.
    """
    # 순환 의존(circular dependency)을 피하기 위해 여기서 import 합니다.
    from deepagents.backends.composite import CompositeBackend

    # CompositeBackend는 default backend가 실행을 지원하는지 확인합니다.
    if isinstance(backend, CompositeBackend):
        return isinstance(backend.default, SandboxBackendProtocol)

    # 그 외 backend는 isinstance로 실행 지원 여부를 판단합니다.
    return isinstance(backend, SandboxBackendProtocol)


def _execute_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """샌드박스 커맨드 실행을 위한 `execute` 도구를 생성합니다.

    Args:
        backend: 실행에 사용할 backend 또는 (runtime을 받아 backend를 반환하는) 팩토리 함수.
        custom_description: 도구 설명을 커스텀할 때 사용(선택).

    Returns:
        backend가 `SandboxBackendProtocol`을 지원할 때 커맨드를 실행하는 `execute` 도구.
    """
    tool_description = custom_description or EXECUTE_TOOL_DESCRIPTION

    def sync_execute(
        command: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> str:
        """`execute` 도구의 동기 래퍼입니다."""
        resolved_backend = _get_backend(backend, runtime)

        # 런타임 체크: 지원하지 않으면 명시적인 오류 메시지로 종료
        if not _supports_execution(resolved_backend):
            return (
                "Error: Execution not available. This agent's backend "
                "does not support command execution (SandboxBackendProtocol). "
                "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
            )

        try:
            result = resolved_backend.execute(command)
        except NotImplementedError as e:
            # execute()가 존재하지만 NotImplementedError를 던지는 케이스 처리
            return f"Error: Execution not available. {e}"

        # (LLM 입력으로 쓰기 좋게) 출력 포맷팅
        parts = [result.output]

        if result.exit_code is not None:
            status = "succeeded" if result.exit_code == 0 else "failed"
            parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

        if result.truncated:
            parts.append("\n[Output was truncated due to size limits]")

        return "".join(parts)

    async def async_execute(
        command: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> str:
        """`execute` 도구의 비동기 래퍼입니다."""
        resolved_backend = _get_backend(backend, runtime)

        # 런타임 체크: 지원하지 않으면 명시적인 오류 메시지로 종료
        if not _supports_execution(resolved_backend):
            return (
                "Error: Execution not available. This agent's backend "
                "does not support command execution (SandboxBackendProtocol). "
                "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
            )

        try:
            result = await resolved_backend.aexecute(command)
        except NotImplementedError as e:
            # execute()가 존재하지만 NotImplementedError를 던지는 케이스 처리
            return f"Error: Execution not available. {e}"

        # (LLM 입력으로 쓰기 좋게) 출력 포맷팅
        parts = [result.output]

        if result.exit_code is not None:
            status = "succeeded" if result.exit_code == 0 else "failed"
            parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

        if result.truncated:
            parts.append("\n[Output was truncated due to size limits]")

        return "".join(parts)

    return StructuredTool.from_function(
        name="execute",
        description=tool_description,
        func=sync_execute,
        coroutine=async_execute,
    )


TOOL_GENERATORS = {
    "ls": _ls_tool_generator,
    "read_file": _read_file_tool_generator,
    "write_file": _write_file_tool_generator,
    "edit_file": _edit_file_tool_generator,
    "glob": _glob_tool_generator,
    "grep": _grep_tool_generator,
    "execute": _execute_tool_generator,
}


def _get_filesystem_tools(
    backend: BackendProtocol,
    custom_tool_descriptions: dict[str, str] | None = None,
) -> list[BaseTool]:
    """파일 시스템 도구(및 가능한 경우 실행 도구)를 구성해 반환합니다.

    Args:
        backend: 파일 저장(및 선택적 실행)에 사용할 backend.
        custom_tool_descriptions: 도구별 커스텀 설명(선택).

    Returns:
        구성된 도구 리스트: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`.
    """
    if custom_tool_descriptions is None:
        custom_tool_descriptions = {}
    tools = []

    for tool_name, tool_generator in TOOL_GENERATORS.items():
        tool = tool_generator(backend, custom_tool_descriptions.get(tool_name))
        tools.append(tool)
    return tools


TOO_LARGE_TOOL_MSG = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}
You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.
You can do this by specifying an offset and limit in the read_file tool call.
For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

Here are the first 10 lines of the result:
{content_sample}
"""


class FilesystemMiddleware(AgentMiddleware):
    """에이전트에 파일 시스템 도구(및 선택적 실행 도구)를 제공하는 미들웨어입니다.

    이 미들웨어는 에이전트에 아래 도구들을 추가합니다.
    - 파일 시스템 도구: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
    - (선택) 실행 도구: `execute` (backend가 `SandboxBackendProtocol`을 구현할 때)

    파일 저장은 `BackendProtocol`을 구현하는 어떤 backend든 사용할 수 있습니다.

    Args:
        backend: 파일 저장(및 선택적 실행)에 사용할 backend. 미지정 시 `StateBackend`를 기본값으로 사용합니다
            (에이전트 state에 저장되는 일시적(ephemeral) 스토리지).
            영구 저장이나 하이브리드 구성이 필요하면 route를 설정한 `CompositeBackend`를 사용하세요.
            커맨드 실행이 필요하면 `SandboxBackendProtocol`을 구현한 backend를 사용해야 합니다.
        system_prompt: 커스텀 system prompt 오버라이드(선택).
        custom_tool_descriptions: 도구 설명 오버라이드(선택).
        tool_token_limit_before_evict: tool 결과를 파일 시스템으로 축출(evict)하기 전 토큰 제한(선택).

    Example:
        ```python
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from deepagents.backends import StateBackend, StoreBackend, CompositeBackend
        from langchain.agents import create_agent

        # 일시적 저장만 사용(기본값, 실행 도구 없음)
        agent = create_agent(middleware=[FilesystemMiddleware()])

        # 하이브리드 저장(일시적 + /memories/ 영구 저장)
        backend = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend()})
        agent = create_agent(middleware=[FilesystemMiddleware(backend=backend)])

        # 샌드박스 backend(실행 도구 지원)
        from my_sandbox import DockerSandboxBackend

        sandbox = DockerSandboxBackend(container_id="my-container")
        agent = create_agent(middleware=[FilesystemMiddleware(backend=sandbox)])
        ```
    """

    state_schema = FilesystemState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES | None = None,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
        tool_token_limit_before_evict: int | None = 20000,
    ) -> None:
        """파일 시스템 미들웨어를 초기화합니다.

        Args:
            backend: 파일 저장/실행에 사용할 backend 또는 팩토리 callable.
                미지정 시 `StateBackend`를 기본값으로 사용합니다.
            system_prompt: 커스텀 system prompt 오버라이드(선택).
            custom_tool_descriptions: 도구 설명 오버라이드(선택).
            tool_token_limit_before_evict: tool 결과를 파일 시스템으로 축출(evict)하기 전 토큰 제한(선택).
        """
        self.tool_token_limit_before_evict = tool_token_limit_before_evict

        # backend가 주어지지 않으면 StateBackend 팩토리를 기본값으로 사용
        self.backend = backend if backend is not None else (lambda rt: StateBackend(rt))

        # system prompt 설정(완전 오버라이드 또는 None이면 동적 생성)
        self._custom_system_prompt = system_prompt

        self.tools = _get_filesystem_tools(self.backend, custom_tool_descriptions)

    def _get_backend(self, runtime: ToolRuntime) -> BackendProtocol:
        """백엔드 인스턴스/팩토리로부터 실제 백엔드를 해석(resolve)합니다.

        Args:
            runtime: tool runtime 컨텍스트.

        Returns:
            해석된 backend 인스턴스.
        """
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """백엔드 capability에 따라 system prompt/도구 목록을 갱신합니다.

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 핸들러 함수.

        Returns:
            핸들러가 반환한 모델 응답.
        """
        # execute 도구가 있는지, 그리고 backend가 실행을 지원하는지 확인
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # 실행 지원 여부를 확인하기 위해 backend를 해석
            backend = self._get_backend(request.runtime)
            backend_supports_execution = _supports_execution(backend)

            # execute 도구가 있지만 backend가 지원하지 않으면 tools에서 제거
            if not backend_supports_execution:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"]
                request = request.override(tools=filtered_tools)
                has_execute_tool = False

        # 커스텀 system prompt가 있으면 사용하고, 없으면 사용 가능한 도구 기준으로 동적 생성
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            # 사용 가능한 도구에 따라 동적 system prompt 구성
            prompt_parts = [FILESYSTEM_SYSTEM_PROMPT]

            # execute 도구가 가능하면 실행 관련 지침 추가
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts)

        if system_prompt:
            request = request.override(system_prompt=request.system_prompt + "\n\n" + system_prompt if request.system_prompt else system_prompt)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) backend capability에 따라 system prompt/도구 목록을 갱신합니다.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)
            backend_supports_execution = _supports_execution(backend)

            # If execute tool exists but backend doesn't support it, filter it out
            if not backend_supports_execution:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"]
                request = request.override(tools=filtered_tools)
                has_execute_tool = False

        # Use custom system prompt if provided, otherwise generate dynamically
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            # Build dynamic system prompt based on available tools
            prompt_parts = [FILESYSTEM_SYSTEM_PROMPT]

            # Add execution instructions if execute tool is available
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts)

        if system_prompt:
            request = request.override(system_prompt=request.system_prompt + "\n\n" + system_prompt if request.system_prompt else system_prompt)

        return await handler(request)

    def _process_large_message(
        self,
        message: ToolMessage,
        resolved_backend: BackendProtocol,
    ) -> tuple[ToolMessage, dict[str, FileData] | None]:
        """큰 ToolMessage를 처리하며, 콘텐츠를 파일 시스템으로 축출(evict)합니다.

        Args:
            message: The ToolMessage with large content to evict.
            resolved_backend: The filesystem backend to write the content to.

        Returns:
            A tuple of (processed_message, files_update):
            - processed_message: New ToolMessage with truncated content and file reference
            - files_update: Dict of file updates to apply to state, or None if eviction failed

        Note:
            The entire content is converted to string, written to /large_tool_results/{tool_call_id},
            and replaced with a truncated preview plus file reference. The replacement is always
            returned as a plain string for consistency, regardless of original content type.

            ToolMessage supports multimodal content blocks (images, audio, etc.), but these are
            uncommon in tool results. For simplicity, all content is stringified and evicted.
            The model can recover by reading the offloaded file from the backend.
        """
        # 축출 설정이 없으면 조기 종료
        if not self.tool_token_limit_before_evict:
            return message, None

        # 크기 체크와 축출을 위해 콘텐츠를 한 번만 문자열로 변환합니다.
        # 특수 케이스: 단일 텍스트 블록이면 가독성을 위해 텍스트만 추출합니다.
        if (
            isinstance(message.content, list)
            and len(message.content) == 1
            and isinstance(message.content[0], dict)
            and message.content[0].get("type") == "text"
            and "text" in message.content[0]
        ):
            content_str = str(message.content[0]["text"])
        elif isinstance(message.content, str):
            content_str = message.content
        else:
            # 여러 블록 또는 텍스트가 아닌 콘텐츠: 전체 구조를 문자열로 변환
            content_str = str(message.content)

        # 콘텐츠가 축출 임계치를 초과하는지 확인
        # token당 4 chars로 보수적으로 추정합니다(실제 비율은 콘텐츠에 따라 달라짐).
        # 실제로는 들어갈 수 있는 콘텐츠를 너무 일찍 축출하지 않도록 “높게” 잡는 쪽으로 동작합니다.
        if len(content_str) <= 4 * self.tool_token_limit_before_evict:
            return message, None

        # 콘텐츠를 파일 시스템에 기록
        sanitized_id = sanitize_tool_call_id(message.tool_call_id)
        file_path = f"/large_tool_results/{sanitized_id}"
        result = resolved_backend.write(file_path, content_str)
        if result.error:
            return message, None

        # 대체 메시지에 넣을 미리보기(트렁케이트) 생성
        content_sample = format_content_with_line_numbers([line[:1000] for line in content_str.splitlines()[:10]], start_line=1)
        replacement_text = TOO_LARGE_TOOL_MSG.format(
            tool_call_id=message.tool_call_id,
            file_path=file_path,
            content_sample=content_sample,
        )

        # 축출 후에는 항상 plain string ToolMessage로 반환
        processed_message = ToolMessage(
            content=replacement_text,
            tool_call_id=message.tool_call_id,
        )
        return processed_message, result.files_update

    def _intercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime: ToolRuntime) -> ToolMessage | Command:
        """state에 추가되기 전에 큰 tool result를 가로채서 처리합니다.

        Args:
            tool_result: The tool result to potentially evict (ToolMessage or Command).
            runtime: The tool runtime providing access to the filesystem backend.

        Returns:
            Either the original result (if small enough) or a Command with evicted
            content written to filesystem and truncated message.

        Note:
            Handles both single ToolMessage results and Command objects containing
            multiple messages. Large content is automatically offloaded to filesystem
            to prevent context window overflow.
        """
        if isinstance(tool_result, ToolMessage):
            resolved_backend = self._get_backend(runtime)
            processed_message, files_update = self._process_large_message(
                tool_result,
                resolved_backend,
            )
            return (
                Command(
                    update={
                        "files": files_update,
                        "messages": [processed_message],
                    }
                )
                if files_update is not None
                else processed_message
            )

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            accumulated_file_updates = dict(update.get("files", {}))
            resolved_backend = self._get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not isinstance(message, ToolMessage):
                    processed_messages.append(message)
                    continue

                processed_message, files_update = self._process_large_message(
                    message,
                    resolved_backend,
                )
                processed_messages.append(processed_message)
                if files_update is not None:
                    accumulated_file_updates.update(files_update)
            return Command(update={**update, "messages": processed_messages, "files": accumulated_file_updates})
        raise AssertionError(f"Unreachable code reached in _intercept_large_tool_result: for tool_result of type {type(tool_result)}")

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """도구 호출(tool call) 결과 크기를 확인하고, 너무 크면 파일 시스템으로 축출(evict)합니다.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The raw ToolMessage, or a pseudo tool message with the ToolResult in state.
        """
        if self.tool_token_limit_before_evict is None or request.tool_call["name"] in TOOL_GENERATORS:
            return handler(request)

        tool_result = handler(request)
        return self._intercept_large_tool_result(tool_result, request.runtime)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """(async) tool call 결과 크기를 확인하고, 너무 크면 파일 시스템으로 축출(evict)합니다.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The raw ToolMessage, or a pseudo tool message with the ToolResult in state.
        """
        if self.tool_token_limit_before_evict is None or request.tool_call["name"] in TOOL_GENERATORS:
            return await handler(request)

        tool_result = await handler(request)
        return self._intercept_large_tool_result(tool_result, request.runtime)
