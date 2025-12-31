"""메모리 백엔드 구현을 위한 공유 유틸리티 함수들.

이 모듈은 백엔드와 복합 라우터(composite router)에서 사용하는
사용자 대면 문자열 포맷터와 구조적 헬퍼 함수를 포함합니다.
구조적 헬퍼는 깨지기 쉬운 문자열 파싱 없이 구성을 가능하게 합니다.
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

# Re-export protocol types for backwards compatibility
FileInfo = _FileInfo
GrepMatch = _GrepMatch


def sanitize_tool_call_id(tool_call_id: str) -> str:
    r"""경로 탐색(path traversal) 및 구분자 문제를 방지하기 위해 tool_call_id를 정리(sanitize)합니다.

    위험한 문자(., /, \)를 밑줄(_)로 교체합니다.
    """
    sanitized = tool_call_id.replace(".", "_").replace("/", "_").replace("\\", "_")
    return sanitized


def format_content_with_line_numbers(
    content: str | list[str],
    start_line: int = 1,
) -> str:
    """파일 내용을 라인 번호와 함께 포맷팅합니다 (cat -n 스타일).

    MAX_LINE_LENGTH보다 긴 라인은 연속 마커(예: 5.1, 5.2)와 함께 청크로 나눕니다.

    Args:
        content: 문자열 또는 라인 리스트 형태의 파일 내용
        start_line: 시작 라인 번호 (기본값: 1)

    Returns:
        라인 번호와 연속 마커가 포함된 포맷팅된 내용
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
            # Split long line into chunks with continuation markers
            num_chunks = (len(line) + MAX_LINE_LENGTH - 1) // MAX_LINE_LENGTH
            for chunk_idx in range(num_chunks):
                start = chunk_idx * MAX_LINE_LENGTH
                end = min(start + MAX_LINE_LENGTH, len(line))
                chunk = line[start:end]
                if chunk_idx == 0:
                    # First chunk: use normal line number
                    result_lines.append(f"{line_num:{LINE_NUMBER_WIDTH}d}\t{chunk}")
                else:
                    # Continuation chunks: use decimal notation (e.g., 5.1, 5.2)
                    continuation_marker = f"{line_num}.{chunk_idx}"
                    result_lines.append(f"{continuation_marker:>{LINE_NUMBER_WIDTH}}\t{chunk}")

    return "\n".join(result_lines)


def check_empty_content(content: str) -> str | None:
    """내용이 비어 있는지 확인하고 경고 메시지를 반환합니다.

    Args:
        content: 확인할 내용

    Returns:
        비어 있는 경우 경고 메시지, 그렇지 않으면 None
    """
    if not content or content.strip() == "":
        return EMPTY_CONTENT_WARNING
    return None


def file_data_to_string(file_data: dict[str, Any]) -> str:
    """FileData를 일반 문자열 내용으로 변환합니다.

    Args:
        file_data: 'content' 키를 가진 FileData dict

    Returns:
        줄바꿈으로 연결된 문자열 형태의 내용
    """
    return "\n".join(file_data["content"])


def create_file_data(content: str, created_at: str | None = None) -> dict[str, Any]:
    """타임스탬프를 포함하는 FileData 객체를 생성합니다.

    Args:
        content: 문자열 형태의 파일 내용
        created_at: 선택적 생성 타임스탬프 (ISO 형식)

    Returns:
        내용과 타임스탬프를 포함하는 FileData dict
    """
    lines = content.split("\n") if isinstance(content, str) else content
    now = datetime.now(UTC).isoformat()

    return {
        "content": lines,
        "created_at": created_at or now,
        "modified_at": now,
    }


def update_file_data(file_data: dict[str, Any], content: str) -> dict[str, Any]:
    """생성 타임스탬프를 유지하면서 새로운 내용으로 FileData를 업데이트합니다.

    Args:
        file_data: 기존 FileData dict
        content: 문자열 형태의 새로운 내용

    Returns:
        업데이트된 FileData dict
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
    """읽기 응답을 위해 파일 데이터를 라인 번호와 함께 포맷팅합니다.

    Args:
        file_data: FileData dict
        offset: 라인 오프셋 (0부터 시작)
        limit: 최대 라인 수

    Returns:
        포맷팅된 내용 또는 에러 메시지
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
    """발생(occurrence) 검증과 함께 문자열 교체를 수행합니다.

    Args:
        content: 원본 내용
        old_string: 교체할 문자열
        new_string: 새로운 문자열
        replace_all: 모든 발생을 교체할지 여부

    Returns:
        성공 시 (new_content, occurrences) 튜플, 또는 에러 메시지 문자열
    """
    occurrences = content.count(old_string)

    if occurrences == 0:
        return f"Error: String not found in file: '{old_string}'"

    if occurrences > 1 and not replace_all:
        return f"Error: String '{old_string}' appears {occurrences} times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."

    new_content = content.replace(old_string, new_string)
    return new_content, occurrences


def truncate_if_too_long(result: list[str] | str) -> list[str] | str:
    """토큰 제한을 초과하는 경우 리스트 또는 문자열 결과를 잘라냅니다 (대략적 추정: 4자/토큰)."""
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
        /로 시작하는 정규화된 경로

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
    """glob 패턴과 일치하는 경로를 찾기 위해 파일 dict를 검색합니다.

    Args:
        files: 파일 경로에서 FileData로의 딕셔너리.
        pattern: Glob 패턴 (예: "*.py", "**/*.ts").
        path: 검색을 시작할 기본 경로.

    Returns:
        수정 시간순(최신순)으로 정렬된, 줄바꿈으로 구분된 파일 경로들.
        일치하는 항목이 없으면 "No files found"를 반환합니다.

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

    # Respect standard glob semantics:
    # - Patterns without path separators (e.g., "*.py") match only in the current
    #   directory (non-recursive) relative to `path`.
    # - Use "**" explicitly for recursive matching.
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
    """출력 모드에 따라 grep 검색 결과를 포맷팅합니다.

    Args:
        results: 파일 경로에서 (line_num, line_content) 튜플 리스트로의 딕셔너리
        output_mode: 출력 형식 - "files_with_matches", "content", 또는 "count"

    Returns:
        포맷팅된 문자열 출력
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
    """정규식 패턴에 대해 파일 내용을 검색합니다.

    Args:
        files: 파일 경로에서 FileData로의 딕셔너리.
        pattern: 검색할 정규식 패턴.
        path: 검색을 시작할 기본 경로.
        glob: 파일을 필터링할 선택적 glob 패턴 (예: "*.py").
        output_mode: 출력 형식 - "files_with_matches", "content", 또는 "count".

    Returns:
        포맷팅된 검색 결과. 결과가 없으면 "No matches found"를 반환합니다.

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


# -------- Structured helpers for composition --------


def grep_matches_from_files(
    files: dict[str, Any],
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
) -> list[GrepMatch] | str:
    """인메모리 파일 매핑에서 구조화된 grep 일치 항목을 반환합니다.

    성공 시 GrepMatch 리스트를 반환하며, 잘못된 입력(예: 잘못된 정규식)의 경우 문자열을 반환합니다.
    도구 컨텍스트에서 백엔드가 예외를 발생시키지 않고 사용자 대면 에러 메시지를 보존하기 위해,
    여기서는 의도적으로 예외를 발생시키지 않습니다.
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
    """구조화된 일치 항목을 포맷터가 사용하는 레거시 dict 형태로 그룹화합니다."""
    grouped: dict[str, list[tuple[int, str]]] = {}
    for m in matches:
        grouped.setdefault(m["path"], []).append((m["line"], m["text"]))
    return grouped


def format_grep_matches(
    matches: list[GrepMatch],
    output_mode: Literal["files_with_matches", "content", "count"],
) -> str:
    """기존 포맷팅 로직을 사용하여 구조화된 grep 일치 항목을 포맷팅합니다."""
    if not matches:
        return "No matches found"
    return _format_grep_results(build_grep_results_dict(matches), output_mode)
