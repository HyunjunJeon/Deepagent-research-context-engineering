"""메모리 백엔드 구현에서 공용으로 사용하는 유틸리티 함수 모음입니다.

이 모듈에는 (1) 사용자/에이전트에게 보여줄 문자열 포매터와 (2) 백엔드 및
Composite 라우터에서 사용하는 구조화된 헬퍼가 함께 들어 있습니다.
구조화 헬퍼를 사용하면 문자열 파싱에 의존하지 않고도 조합(composition)이
가능해집니다.
"""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import wcmatch.glob as wcglob

from deepagents.backends.protocol import FileInfo as _FileInfo
from deepagents.backends.protocol import GrepMatch as _GrepMatch

EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
MAX_LINE_LENGTH = 10000
LINE_NUMBER_WIDTH = 6
TOOL_RESULT_TOKEN_LIMIT = 20000  # Same threshold as eviction
TRUNCATION_GUIDANCE = "... [results truncated, try being more specific with your parameters]"

# 하위 호환성을 위해 protocol 타입을 재노출(re-export)합니다.
FileInfo = _FileInfo
GrepMatch = _GrepMatch


def sanitize_tool_call_id(tool_call_id: str) -> str:
    r"""`tool_call_id`를 안전하게 정규화합니다.

    경로 탐색(path traversal)이나 구분자 관련 문제를 피하기 위해 위험한 문자(`.`, `/`, `\`)를
    밑줄(`_`)로 치환합니다.
    """
    sanitized = tool_call_id.replace(".", "_").replace("/", "_").replace("\\", "_")
    return sanitized


def format_content_with_line_numbers(
    content: str | list[str],
    start_line: int = 1,
) -> str:
    """파일 내용을 라인 번호와 함께 포맷팅합니다(`cat -n` 스타일).

    `MAX_LINE_LENGTH`를 초과하는 라인은 연속 마커(예: `5.1`, `5.2`)를 붙여 여러 줄로 분할합니다.

    Args:
        content: 문자열 또는 라인 리스트 형태의 파일 내용
        start_line: 시작 라인 번호(기본값: 1)

    Returns:
        라인 번호와 연속 마커가 포함된 포맷 문자열
    """
    if isinstance(content, str):
        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
    else:
        lines = content

    result_lines = []
    for i, line in enumerate(lines):
        line_num = i + start_line

        if len(line) <= MAX_LINE_LENGTH:
            result_lines.append(f"{line_num:{LINE_NUMBER_WIDTH}d}\t{line}")
        else:
            # 긴 라인을 여러 조각으로 분할하고 연속 마커를 부여합니다.
            num_chunks = (len(line) + MAX_LINE_LENGTH - 1) // MAX_LINE_LENGTH
            for chunk_idx in range(num_chunks):
                start = chunk_idx * MAX_LINE_LENGTH
                end = min(start + MAX_LINE_LENGTH, len(line))
                chunk = line[start:end]
                if chunk_idx == 0:
                    # 첫 번째 조각: 일반 라인 번호 사용
                    result_lines.append(f"{line_num:{LINE_NUMBER_WIDTH}d}\t{chunk}")
                else:
                    # 후속 조각: 소수 표기(예: 5.1, 5.2)
                    continuation_marker = f"{line_num}.{chunk_idx}"
                    result_lines.append(f"{continuation_marker:>{LINE_NUMBER_WIDTH}}\t{chunk}")

    return "\n".join(result_lines)


def check_empty_content(content: str) -> str | None:
    """콘텐츠가 비어 있는지 확인하고, 비어 있으면 경고 메시지를 반환합니다.

    Args:
        content: 확인할 콘텐츠

    Returns:
        비어 있으면 경고 메시지, 아니면 `None`
    """
    if not content or content.strip() == "":
        return EMPTY_CONTENT_WARNING
    return None


def file_data_to_string(file_data: dict[str, Any]) -> str:
    """FileData 딕셔너리를 일반 문자열 콘텐츠로 변환합니다.

    Args:
        file_data: `'content'` 키를 포함한 FileData 딕셔너리

    Returns:
        줄바꿈으로 합쳐진 문자열 콘텐츠
    """
    return "\n".join(file_data["content"])


def create_file_data(content: str, created_at: str | None = None) -> dict[str, Any]:
    """타임스탬프를 포함한 FileData 딕셔너리를 생성합니다.

    Args:
        content: 파일 내용(문자열)
        created_at: 생성 시각(ISO 형식) 오버라이드(선택)

    Returns:
        content/created_at/modified_at를 포함한 FileData 딕셔너리
    """
    lines = content.split("\n") if isinstance(content, str) else content
    now = datetime.now(UTC).isoformat()

    return {
        "content": lines,
        "created_at": created_at or now,
        "modified_at": now,
    }


def update_file_data(file_data: dict[str, Any], content: str) -> dict[str, Any]:
    """기존 FileData의 생성 시각을 유지하면서 내용을 업데이트합니다.

    Args:
        file_data: 기존 FileData 딕셔너리
        content: 새 콘텐츠(문자열)

    Returns:
        업데이트된 FileData 딕셔너리
    """
    lines = content.split("\n") if isinstance(content, str) else content
    now = datetime.now(UTC).isoformat()

    return {
        "content": lines,
        "created_at": file_data["created_at"],
        "modified_at": now,
    }


def format_read_response(
    file_data: dict[str, Any],
    offset: int,
    limit: int,
) -> str:
    """`read` 응답을 라인 번호와 함께 포맷팅합니다.

    Args:
        file_data: FileData 딕셔너리
        offset: 라인 오프셋(0-index)
        limit: 최대 라인 수

    Returns:
        포맷된 콘텐츠 또는 오류 메시지
    """
    content = file_data_to_string(file_data)
    empty_msg = check_empty_content(content)
    if empty_msg:
        return empty_msg

    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    selected_lines = lines[start_idx:end_idx]
    return format_content_with_line_numbers(selected_lines, start_line=start_idx + 1)


def perform_string_replacement(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
) -> tuple[str, int] | str:
    """문자열 치환을 수행하고, 치환 대상 문자열의 출현 횟수를 검증합니다.

    Args:
        content: 원본 콘텐츠
        old_string: 치환할 문자열
        new_string: 대체 문자열
        replace_all: 모든 출현을 치환할지 여부

    Returns:
        성공 시 `(new_content, occurrences)` 튜플, 실패 시 오류 메시지 문자열
    """
    occurrences = content.count(old_string)

    if occurrences == 0:
        return f"Error: String not found in file: '{old_string}'"

    if occurrences > 1 and not replace_all:
        return f"Error: String '{old_string}' appears {occurrences} times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."

    new_content = content.replace(old_string, new_string)
    return new_content, occurrences


def truncate_if_too_long(result: list[str] | str) -> list[str] | str:
    """토큰 제한을 초과하는 결과를 잘라냅니다(대략 4 chars/token 기준)."""
    if isinstance(result, list):
        total_chars = sum(len(item) for item in result)
        if total_chars > TOOL_RESULT_TOKEN_LIMIT * 4:
            return result[: len(result) * TOOL_RESULT_TOKEN_LIMIT * 4 // total_chars] + [TRUNCATION_GUIDANCE]
        return result
    # string
    if len(result) > TOOL_RESULT_TOKEN_LIMIT * 4:
        return result[: TOOL_RESULT_TOKEN_LIMIT * 4] + "\n" + TRUNCATION_GUIDANCE
    return result


def _validate_path(path: str | None) -> str:
    """경로를 검증하고 정규화합니다.

    Args:
        path: 검증할 경로

    Returns:
        `/`로 시작하고 `/`로 끝나는 정규화된 경로

    Raises:
        ValueError: 경로가 유효하지 않은 경우
    """
    path = path or "/"
    if not path or path.strip() == "":
        raise ValueError("Path cannot be empty")

    normalized = path if path.startswith("/") else "/" + path

    if not normalized.endswith("/"):
        normalized += "/"

    return normalized


def _glob_search_files(
    files: dict[str, Any],
    pattern: str,
    path: str = "/",
) -> str:
    """in-memory 파일 맵에서 glob 패턴에 매칭되는 경로를 찾습니다.

    Args:
        files: Dictionary of file paths to FileData.
        pattern: Glob pattern (e.g., "*.py", "**/*.ts").
        path: Base path to search from.

    Returns:
        Newline-separated file paths, sorted by modification time (most recent first).
        Returns "No files found" if no matches.

    Example:
        ```python
        files = {"/src/main.py": FileData(...), "/test.py": FileData(...)}
        _glob_search_files(files, "*.py", "/")
        # Returns: "/test.py\n/src/main.py" (sorted by modified_at)
        ```
    """
    try:
        normalized_path = _validate_path(path)
    except ValueError:
        return "No files found"

    filtered = {fp: fd for fp, fd in files.items() if fp.startswith(normalized_path)}

    # 표준 glob semantics를 따릅니다.
    # - path separator가 없는 패턴(예: "*.py")은 `path` 기준 현재 디렉토리(비재귀)만 매칭합니다.
    # - 재귀 매칭이 필요하면 "**"를 명시적으로 사용해야 합니다.
    effective_pattern = pattern

    matches = []
    for file_path, file_data in filtered.items():
        relative = file_path[len(normalized_path) :].lstrip("/")
        if not relative:
            relative = file_path.split("/")[-1]

        if wcglob.globmatch(relative, effective_pattern, flags=wcglob.BRACE | wcglob.GLOBSTAR):
            matches.append((file_path, file_data["modified_at"]))

    matches.sort(key=lambda x: x[1], reverse=True)

    if not matches:
        return "No files found"

    return "\n".join(fp for fp, _ in matches)


def _format_grep_results(
    results: dict[str, list[tuple[int, str]]],
    output_mode: Literal["files_with_matches", "content", "count"],
) -> str:
    """Output mode에 따라 grep 검색 결과를 포맷팅합니다.

    Args:
        results: Dictionary mapping file paths to list of (line_num, line_content) tuples
        output_mode: Output format - "files_with_matches", "content", or "count"

    Returns:
        Formatted string output
    """
    if output_mode == "files_with_matches":
        return "\n".join(sorted(results.keys()))
    if output_mode == "count":
        lines = []
        for file_path in sorted(results.keys()):
            count = len(results[file_path])
            lines.append(f"{file_path}: {count}")
        return "\n".join(lines)
    lines = []
    for file_path in sorted(results.keys()):
        lines.append(f"{file_path}:")
        for line_num, line in results[file_path]:
            lines.append(f"  {line_num}: {line}")
    return "\n".join(lines)


def _grep_search_files(
    files: dict[str, Any],
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
) -> str:
    """파일 내용에서 정규식 패턴을 검색합니다.

    Args:
        files: Dictionary of file paths to FileData.
        pattern: Regex pattern to search for.
        path: Base path to search from.
        glob: Optional glob pattern to filter files (e.g., "*.py").
        output_mode: Output format - "files_with_matches", "content", or "count".

    Returns:
        Formatted search results. Returns "No matches found" if no results.

    Example:
        ```python
        files = {"/file.py": FileData(content=["import os", "print('hi')"], ...)}
        _grep_search_files(files, "import", "/")
        # Returns: "/file.py" (with output_mode="files_with_matches")
        ```
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    try:
        normalized_path = _validate_path(path)
    except ValueError:
        return "No matches found"

    filtered = {fp: fd for fp, fd in files.items() if fp.startswith(normalized_path)}

    if glob:
        filtered = {fp: fd for fp, fd in filtered.items() if wcglob.globmatch(Path(fp).name, glob, flags=wcglob.BRACE)}

    results: dict[str, list[tuple[int, str]]] = {}
    for file_path, file_data in filtered.items():
        for line_num, line in enumerate(file_data["content"], 1):
            if regex.search(line):
                if file_path not in results:
                    results[file_path] = []
                results[file_path].append((line_num, line))

    if not results:
        return "No matches found"
    return _format_grep_results(results, output_mode)


# -------- 조합(composition)을 위한 구조화 헬퍼 --------


def grep_matches_from_files(
    files: dict[str, Any],
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
) -> list[GrepMatch] | str:
    """in-memory 파일 맵에서 구조화된 grep 매칭을 반환합니다.

    성공 시 `list[GrepMatch]`를, 입력이 유효하지 않은 경우(예: 잘못된 정규식)는 오류 문자열을 반환합니다.
    백엔드가 도구(tool) 컨텍스트에서 예외를 던지지 않도록 하고, 사용자/에이전트에게 보여줄 오류 메시지를
    유지하기 위해 의도적으로 raise 하지 않습니다.
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    try:
        normalized_path = _validate_path(path)
    except ValueError:
        return []

    filtered = {fp: fd for fp, fd in files.items() if fp.startswith(normalized_path)}

    if glob:
        filtered = {fp: fd for fp, fd in filtered.items() if wcglob.globmatch(Path(fp).name, glob, flags=wcglob.BRACE)}

    matches: list[GrepMatch] = []
    for file_path, file_data in filtered.items():
        for line_num, line in enumerate(file_data["content"], 1):
            if regex.search(line):
                matches.append({"path": file_path, "line": int(line_num), "text": line})
    return matches


def build_grep_results_dict(matches: list[GrepMatch]) -> dict[str, list[tuple[int, str]]]:
    """구조화 매칭을 기존(formatter) 호환 dict 형태로 그룹화합니다."""
    grouped: dict[str, list[tuple[int, str]]] = {}
    for m in matches:
        grouped.setdefault(m["path"], []).append((m["line"], m["text"]))
    return grouped


def format_grep_matches(
    matches: list[GrepMatch],
    output_mode: Literal["files_with_matches", "content", "count"],
) -> str:
    """기존 포맷팅 로직을 이용해 구조화 grep 매칭을 문자열로 포맷팅합니다."""
    if not matches:
        return "No matches found"
    return _format_grep_results(build_grep_results_dict(matches), output_mode)
