"""에이전트에게 파일 시스템 도구를 제공하기 위한 미들웨어."""
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
    """메타데이터와 함께 파일 내용을 저장하기 위한 데이터 구조입니다."""

    content: list[str]
    """파일의 라인들."""

    created_at: str
    """파일 생성의 ISO 8601 타임스탬프."""

    modified_at: str
    """마지막 수정의 ISO 8601 타임스탬프."""


def _file_data_reducer(left: dict[str, FileData] | None, right: dict[str, FileData | None]) -> dict[str, FileData]:
    """삭제를 지원하며 파일 업데이트를 병합합니다.

    이 리듀서(reducer)는 오른쪽 딕셔너리의 `None` 값을 삭제 마커로 처리하여 파일 삭제를 가능하게 합니다.
    주석이 달린 리듀서가 상태 업데이트 병합 방식을 제어하는 LangGraph의 상태 관리와 함께 작동하도록 설계되었습니다.

    Args:
        left: 기존 파일 딕셔너리. 초기화 중에는 `None`일 수 있습니다.
        right: 병합할 새 파일 딕셔너리. `None` 값을 가진 파일은 삭제 마커로 처리되어 결과에서 제거됩니다.

    Returns:
        일치하는 키에 대해 오른쪽(right)이 왼쪽(left)을 덮어쓰고, 오른쪽의 `None` 값이 삭제를 트리거하는 병합된 딕셔너리.

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
    r"""보안을 위해 파일 경로를 검증하고 정규화합니다.

    디렉토리 탐색(directory traversal) 공격을 방지하고 일관된 포맷팅을 강제하여 경로를 안전하게 사용할 수 있도록 보장합니다.
    모든 경로는 포워드 슬래시(/)를 사용하고 선행 슬래시로 시작하도록 정규화됩니다.

    이 함수는 가상 파일 시스템 경로를 위해 설계되었으며, 일관성을 유지하고 경로 형식의 모호성을 방지하기 위해 Windows 절대 경로(예: C:/..., F:/...)를 거부합니다.

    Args:
        path: 검증하고 정규화할 경로.
        allowed_prefixes: 허용된 경로 접두사의 선택적 목록. 제공된 경우, 정규화된 경로는 이 접두사 중 하나로 시작해야 합니다.

    Returns:
        `/`로 시작하고 포워드 슬래시를 사용하는 정규화된 표준 경로.

    Raises:
        ValueError: 경로에 탐색 시퀀스(`..` 또는 `~`)가 포함되어 있거나, Windows 절대 경로(예: C:/...)이거나, `allowed_prefixes`가 지정되었을 때 허용된 접두사로 시작하지 않는 경우.

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

    # Reject Windows absolute paths (e.g., C:\..., D:/...)
    # This maintains consistency in virtual filesystem paths
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
    """파일 시스템 미들웨어를 위한 상태."""

    files: Annotated[NotRequired[dict[str, FileData]], _file_data_reducer]
    """파일 시스템 내의 파일들."""


LIST_FILES_TOOL_DESCRIPTION = """파일 시스템의 모든 파일을 나열하며, 디렉토리별로 필터링합니다.

사용법:
- path 매개변수는 상대 경로가 아닌 절대 경로여야 합니다
- list_files 도구는 지정된 디렉토리의 모든 파일 목록을 반환합니다.
- 파일 시스템을 탐색하고 읽거나 편집할 올바른 파일을 찾는 데 매우 유용합니다.
- Read 또는 Edit 도구를 사용하기 전에 거의 항상 이 도구를 먼저 사용해야 합니다."""

READ_FILE_TOOL_DESCRIPTION = """파일 시스템에서 파일을 읽습니다. 이 도구를 사용하여 모든 파일에 직접 접근할 수 있습니다.
이 도구가 머신의 모든 파일을 읽을 수 있다고 가정하십시오. 사용자가 파일 경로를 제공하면 해당 경로가 유효하다고 가정하십시오. 존재하지 않는 파일을 읽어도 괜찮으며, 에러가 반환될 것입니다.

사용법:
- file_path 매개변수는 상대 경로가 아닌 절대 경로여야 합니다
- 기본적으로 파일의 시작 부분부터 최대 500줄을 읽습니다
- **대용량 파일 및 코드베이스 탐색 시 중요**: 맥락(context) 오버플로를 방지하기 위해 offset 및 limit 매개변수와 함께 페이지네이션을 사용하십시오
  - 첫 번째 스캔: read_file(path, limit=100)으로 파일 구조 확인
  - 추가 섹션 읽기: read_file(path, offset=100, limit=200)으로 다음 200줄 읽기
  - 편집을 위해 필요한 경우에만 limit 생략 (전체 파일 읽기)
- offset 및 limit 지정: read_file(path, offset=0, limit=100)은 처음 100줄을 읽습니다
- 2000자를 초과하는 라인은 잘립니다(truncated)
- 결과는 cat -n 형식으로 반환되며, 줄 번호는 1부터 시작합니다
- 단일 응답에서 여러 도구를 호출할 수 있는 기능이 있습니다. 잠재적으로 유용한 여러 파일을 배치(batch)로 추측하여 읽는 것이 항상 더 좋습니다.
- 존재하지만 내용이 비어 있는 파일을 읽으면 파일 내용 대신 시스템 알림 경고를 받게 됩니다.
- 파일을 편집하기 전에 항상 파일이 읽혔는지 확인해야 합니다."""

EDIT_FILE_TOOL_DESCRIPTION = """파일에서 정확한 문자열 교체를 수행합니다.

사용법:
- 편집하기 전에 대화에서 `Read` 도구를 최소 한 번 이상 사용해야 합니다. 파일을 읽지 않고 편집을 시도하면 이 도구는 에러를 발생시킵니다.
- Read 도구 출력에서 텍스트를 편집할 때, 줄 번호 접두사 뒤에 나타나는 정확한 들여쓰기(탭/공백)를 보존해야 합니다. 줄 번호 접두사 형식은: 공백 + 줄 번호 + 탭입니다. 그 탭 이후의 모든 것이 일치해야 할 실제 파일 내용입니다. old_string이나 new_string에 줄 번호 접두사의 어떤 부분도 포함하지 마십시오.
- 항상 새 파일을 생성하는 것보다 기존 파일을 편집하는 것을 선호하십시오. 명시적으로 요구되지 않는 한 새 파일을 작성하지 마십시오.
- 사용자가 명시적으로 요청한 경우에만 이모지를 사용하십시오. 요청하지 않으면 파일에 이모지를 추가하지 마십시오.
- `old_string`이 파일 내에서 고유하지 않으면 편집이 실패합니다. 더 많은 주변 맥락을 포함하여 더 큰 문자열을 제공하거나 `replace_all`을 사용하여 `old_string`의 모든 인스턴스를 변경하십시오.
- 파일 전체에서 문자열을 교체하고 이름을 변경하려면 `replace_all`을 사용하십시오. 이 매개변수는 예를 들어 변수 이름을 바꾸고 싶을 때 유용합니다."""


WRITE_FILE_TOOL_DESCRIPTION = """파일 시스템에 새 파일을 씁니다.

사용법:
- file_path 매개변수는 상대 경로가 아닌 절대 경로여야 합니다
- content 매개변수는 문자열이어야 합니다
- write_file 도구는 새 파일을 생성합니다.
- 가능한 경우 새 파일을 생성하는 것보다 기존 파일을 편집하는 것을 선호하십시오."""


GLOB_TOOL_DESCRIPTION = """glob 패턴과 일치하는 파일을 찾습니다.

사용법:
- glob 도구는 와일드카드가 포함된 패턴을 일치시켜 파일을 찾습니다
- 표준 glob 패턴 지원: `*` (모든 문자), `**` (모든 디렉토리), `?` (단일 문자)
- 패턴은 절대 경로(`/`로 시작)이거나 상대 경로일 수 있습니다
- 패턴과 일치하는 절대 파일 경로 목록을 반환합니다

예시:
- `**/*.py` - 모든 Python 파일 찾기
- `*.txt` - 루트의 모든 텍스트 파일 찾기
- `/subdir/**/*.md` - /subdir 아래의 모든 마크다운 파일 찾기"""

GREP_TOOL_DESCRIPTION = """파일 내에서 패턴을 검색합니다.

사용법:
- grep 도구는 파일 전체에서 텍스트 패턴을 검색합니다
- pattern 매개변수는 검색할 텍스트입니다 (정규식이 아닌 리터럴 문자열)
- path 매개변수는 검색할 디렉토리를 필터링합니다 (기본값은 현재 작업 디렉토리)
- glob 매개변수는 검색할 파일을 필터링하는 glob 패턴을 허용합니다 (예: `*.py`)
- output_mode 매개변수는 출력 형식을 제어합니다:
  - `files_with_matches`: 일치하는 항목이 있는 파일 경로만 나열 (기본값)
  - `content`: 파일 경로 및 줄 번호와 함께 일치하는 라인 표시
  - `count`: 파일당 일치 횟수 표시

예시:
- 모든 파일 검색: `grep(pattern="TODO")`
- Python 파일만 검색: `grep(pattern="import", glob="*.py")`
- 일치하는 라인 표시: `grep(pattern="error", output_mode="content")`"""

EXECUTE_TOOL_DESCRIPTION = """적절한 처리 및 보안 조치를 갖춘 샌드박스 환경에서 주어진 명령을 실행합니다.

명령을 실행하기 전에 다음 단계를 따르십시오:

1. 디렉토리 확인:
   - 명령이 새 디렉토리나 파일을 생성할 경우, 먼저 ls 도구를 사용하여 상위 디렉토리가 존재하고 올바른 위치인지 확인하십시오
   - 예를 들어, "mkdir foo/bar"를 실행하기 전에 먼저 ls를 사용하여 "foo"가 존재하고 의도한 상위 디렉토리인지 확인하십시오

2. 명령 실행:
   - 공백이 포함된 파일 경로는 항상 큰따옴표로 묶으십시오 (예: cd "path with spaces/file.txt")
   - 올바른 인용 예시:
     - cd "/Users/name/My Documents" (올바름)
     - cd /Users/name/My Documents (틀림 - 실패함)
     - python "/path/with spaces/script.py" (올바름)
     - python /path/with spaces/script.py (틀림 - 실패함)
   - 적절한 인용을 확인한 후 명령을 실행하십시오
   - 명령의 출력을 캡처하십시오

사용 참고 사항:
  - command 매개변수는 필수입니다
  - 명령은 격리된 샌드박스 환경에서 실행됩니다
  - 종료 코드와 함께 결합된 stdout/stderr 출력을 반환합니다
  - 출력이 매우 큰 경우 잘릴(truncated) 수 있습니다
  - 매우 중요: find 및 grep과 같은 검색 명령 사용을 반드시 피해야 합니다. 대신 grep, glob 도구를 사용하여 검색하십시오. cat, head, tail과 같은 읽기 도구를 피하고 read_file을 사용하여 파일을 읽어야 합니다.
  - 여러 명령을 실행할 때는 ';' 또는 '&&' 연산자를 사용하여 분리하십시오. 개행 문자를 사용하지 마십시오 (인용된 문자열 내의 개행은 괜찮습니다)
    - 명령이 서로 의존할 때는 '&&'를 사용하십시오 (예: "mkdir dir && cd dir")
    - 명령을 순차적으로 실행해야 하지만 이전 명령이 실패해도 상관없을 때만 ';'를 사용하십시오
  - 절대 경로를 사용하고 cd 사용을 피하여 세션 전체에서 현재 작업 디렉토리를 유지하려고 노력하십시오

예시:
  좋은 예:
    - execute(command="pytest /foo/bar/tests")
    - execute(command="python /path/to/script.py")
    - execute(command="npm install && npm test")

  나쁜 예 (피해야 함):
    - execute(command="cd /foo/bar && pytest tests")  # 대신 절대 경로 사용
    - execute(command="cat file.txt")  # 대신 read_file 도구 사용
    - execute(command="find . -name '*.py'")  # 대신 glob 도구 사용
    - execute(command="grep -r 'pattern' .")  # 대신 grep 도구 사용

참고: 이 도구는 백엔드가 실행(SandboxBackendProtocol)을 지원하는 경우에만 사용할 수 있습니다.
실행이 지원되지 않으면 도구는 에러 메시지를 반환합니다."""

FILESYSTEM_SYSTEM_PROMPT = """## 파일 시스템 도구 `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`

이 도구들을 사용하여 상호 작용할 수 있는 파일 시스템에 접근할 수 있습니다.
모든 파일 경로는 /로 시작해야 합니다.

- ls: 디렉토리 내 파일 나열 (절대 경로 필요)
- read_file: 파일 시스템에서 파일 읽기
- write_file: 파일 시스템에 파일 쓰기
- edit_file: 파일 시스템의 파일 편집
- glob: 패턴과 일치하는 파일 찾기 (예: "**/*.py")
- grep: 파일 내 텍스트 검색"""

EXECUTION_SYSTEM_PROMPT = """## 실행 도구 `execute`

샌드박스 환경에서 쉘 명령을 실행하기 위한 `execute` 도구에 접근할 수 있습니다.
이 도구를 사용하여 명령, 스크립트, 테스트, 빌드 및 기타 쉘 작업을 실행하십시오.

- execute: 샌드박스에서 쉘 명령 실행 (출력 및 종료 코드 반환)"""


def _get_backend(backend: BACKEND_TYPES, runtime: ToolRuntime) -> BackendProtocol:
    """백엔드 인스턴스 또는 팩토리에서 해결된(resolved) 백엔드 인스턴스를 가져옵니다.

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
    """ls (파일 목록) 도구를 생성합니다.

    Args:
        backend: 파일 저장소에 사용할 백엔드, 또는 런타임을 받아 백엔드를 반환하는 팩토리 함수.
        custom_description: 도구에 대한 선택적 사용자 정의 설명.

    Returns:
        백엔드를 사용하여 파일을 나열하는 구성된 ls 도구.
    """
    tool_description = custom_description or LIST_FILES_TOOL_DESCRIPTION

    def sync_ls(runtime: ToolRuntime[None, FilesystemState], path: str) -> str:
        """ls 도구의 동기 래퍼."""
        resolved_backend = _get_backend(backend, runtime)
        validated_path = _validate_path(path)
        infos = resolved_backend.ls_info(validated_path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    async def async_ls(runtime: ToolRuntime[None, FilesystemState], path: str) -> str:
        """ls 도구의 비동기 래퍼."""
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
    """read_file 도구를 생성합니다.

    Args:
        backend: 파일 저장소에 사용할 백엔드, 또는 런타임을 받아 백엔드를 반환하는 팩토리 함수.
        custom_description: 도구에 대한 선택적 사용자 정의 설명.

    Returns:
        백엔드를 사용하여 파일을 읽는 구성된 read_file 도구.
    """
    tool_description = custom_description or READ_FILE_TOOL_DESCRIPTION

    def sync_read_file(
        file_path: str,
        runtime: ToolRuntime[None, FilesystemState],
        offset: int = DEFAULT_READ_OFFSET,
        limit: int = DEFAULT_READ_LIMIT,
    ) -> str:
        """read_file 도구의 동기 래퍼."""
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        return resolved_backend.read(file_path, offset=offset, limit=limit)

    async def async_read_file(
        file_path: str,
        runtime: ToolRuntime[None, FilesystemState],
        offset: int = DEFAULT_READ_OFFSET,
        limit: int = DEFAULT_READ_LIMIT,
    ) -> str:
        """read_file 도구의 비동기 래퍼."""
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
    """write_file 도구를 생성합니다.

    Args:
        backend: 파일 저장소에 사용할 백엔드, 또는 런타임을 받아 백엔드를 반환하는 팩토리 함수.
        custom_description: 도구에 대한 선택적 사용자 정의 설명.

    Returns:
        백엔드를 사용하여 새 파일을 생성하는 구성된 write_file 도구.
    """
    tool_description = custom_description or WRITE_FILE_TOOL_DESCRIPTION

    def sync_write_file(
        file_path: str,
        content: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> Command | str:
        """write_file 도구의 동기 래퍼."""
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        res: WriteResult = resolved_backend.write(file_path, content)
        if res.error:
            return res.error
        # If backend returns state update, wrap into Command with ToolMessage
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
        """write_file 도구의 비동기 래퍼."""
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        res: WriteResult = await resolved_backend.awrite(file_path, content)
        if res.error:
            return res.error
        # If backend returns state update, wrap into Command with ToolMessage
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
    """edit_file 도구를 생성합니다.

    Args:
        backend: 파일 저장소에 사용할 백엔드, 또는 런타임을 받아 백엔드를 반환하는 팩토리 함수.
        custom_description: 도구에 대한 선택적 사용자 정의 설명.

    Returns:
        백엔드를 사용하여 파일에서 문자열 교체를 수행하는 구성된 edit_file 도구.
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
        """edit_file 도구의 동기 래퍼."""
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
        """edit_file 도구의 비동기 래퍼."""
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
    """glob 도구를 생성합니다.

    Args:
        backend: 파일 저장소에 사용할 백엔드, 또는 런타임을 받아 백엔드를 반환하는 팩토리 함수.
        custom_description: 도구에 대한 선택적 사용자 정의 설명.

    Returns:
        백엔드를 사용하여 패턴별로 파일을 찾는 구성된 glob 도구.
    """
    tool_description = custom_description or GLOB_TOOL_DESCRIPTION

    def sync_glob(pattern: str, runtime: ToolRuntime[None, FilesystemState], path: str = "/") -> str:
        """glob 도구의 동기 래퍼."""
        resolved_backend = _get_backend(backend, runtime)
        infos = resolved_backend.glob_info(pattern, path=path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    async def async_glob(pattern: str, runtime: ToolRuntime[None, FilesystemState], path: str = "/") -> str:
        """glob 도구의 비동기 래퍼."""
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
    """grep 도구를 생성합니다.

    Args:
        backend: 파일 저장소에 사용할 백엔드, 또는 런타임을 받아 백엔드를 반환하는 팩토리 함수.
        custom_description: 도구에 대한 선택적 사용자 정의 설명.

    Returns:
        백엔드를 사용하여 파일에서 패턴을 검색하는 구성된 grep 도구.
    """
    tool_description = custom_description or GREP_TOOL_DESCRIPTION

    def sync_grep(
        pattern: str,
        runtime: ToolRuntime[None, FilesystemState],
        path: str | None = None,
        glob: str | None = None,
        output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
    ) -> str:
        """grep 도구의 동기 래퍼."""
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
        """grep 도구의 비동기 래퍼."""
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
    """백엔드가 명령 실행을 지원하는지 확인합니다.

    CompositeBackend의 경우, 기본(default) 백엔드가 실행을 지원하는지 확인합니다.
    다른 백엔드의 경우, SandboxBackendProtocol을 구현하는지 확인합니다.

    Args:
        backend: 확인할 백엔드.

    Returns:
        백엔드가 실행을 지원하면 True, 그렇지 않으면 False.
    """
    # Import here to avoid circular dependency
    from deepagents.backends.composite import CompositeBackend

    # For CompositeBackend, check the default backend
    if isinstance(backend, CompositeBackend):
        return isinstance(backend.default, SandboxBackendProtocol)

    # For other backends, use isinstance check
    return isinstance(backend, SandboxBackendProtocol)


def _execute_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """샌드박스 명령 실행을 위한 execute 도구를 생성합니다.

    Args:
        backend: 실행에 사용할 백엔드, 또는 런타임을 받아 백엔드를 반환하는 팩토리 함수.
        custom_description: 도구에 대한 선택적 사용자 정의 설명.

    Returns:
        백엔드가 SandboxBackendProtocol을 지원하는 경우 명령을 실행하는 구성된 execute 도구.
    """
    tool_description = custom_description or EXECUTE_TOOL_DESCRIPTION

    def sync_execute(
        command: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> str:
        """execute 도구의 동기 래퍼."""
        resolved_backend = _get_backend(backend, runtime)

        # Runtime check - fail gracefully if not supported
        if not _supports_execution(resolved_backend):
            return (
                "에러: 실행을 사용할 수 없습니다. 이 에이전트의 백엔드는 명령 실행(SandboxBackendProtocol)을 지원하지 않습니다. "
                "execute 도구를 사용하려면 SandboxBackendProtocol을 구현하는 백엔드를 제공하십시오."
            )

        try:
            result = resolved_backend.execute(command)
        except NotImplementedError as e:
            # Handle case where execute() exists but raises NotImplementedError
            return f"에러: 실행을 사용할 수 없습니다. {e}"

        # Format output for LLM consumption
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
        """execute 도구의 비동기 래퍼."""
        resolved_backend = _get_backend(backend, runtime)

        # Runtime check - fail gracefully if not supported
        if not _supports_execution(resolved_backend):
            return (
                "에러: 실행을 사용할 수 없습니다. 이 에이전트의 백엔드는 명령 실행(SandboxBackendProtocol)을 지원하지 않습니다. "
                "execute 도구를 사용하려면 SandboxBackendProtocol을 구현하는 백엔드를 제공하십시오."
            )

        try:
            result = await resolved_backend.aexecute(command)
        except NotImplementedError as e:
            # Handle case where execute() exists but raises NotImplementedError
            return f"에러: 실행을 사용할 수 없습니다. {e}"

        # Format output for LLM consumption
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
    """파일 시스템 및 실행 도구를 가져옵니다.

    Args:
        backend: 파일 저장 및 선택적 실행에 사용할 백엔드, 또는 런타임을 받아 백엔드를 반환하는 팩토리 함수.
        custom_tool_descriptions: 도구에 대한 선택적 사용자 정의 설명.

    Returns:
        구성된 도구 목록: ls, read_file, write_file, edit_file, glob, grep, execute.
    """
    if custom_tool_descriptions is None:
        custom_tool_descriptions = {}
    tools = []

    for tool_name, tool_generator in TOOL_GENERATORS.items():
        tool = tool_generator(backend, custom_tool_descriptions.get(tool_name))
        tools.append(tool)
    return tools


TOO_LARGE_TOOL_MSG = """도구 결과가 너무 큽니다. 이 도구 호출 {tool_call_id}의 결과가 파일 시스템의 다음 경로에 저장되었습니다: {file_path}
read_file 도구를 사용하여 파일 시스템에서 결과를 읽을 수 있지만, 한 번에 결과의 일부만 읽어야 합니다.
read_file 도구 호출에서 offset과 limit을 지정하여 이를 수행할 수 있습니다.
예를 들어, 처음 100줄을 읽으려면 offset=0 및 limit=100으로 read_file 도구를 사용할 수 있습니다.

다음은 결과의 처음 10줄입니다:
{content_sample}
"""


class FilesystemMiddleware(AgentMiddleware):
    """에이전트에게 파일 시스템 및 선택적 실행 도구를 제공하기 위한 미들웨어.

    이 미들웨어는 에이전트에게 ls, read_file, write_file, edit_file, glob, grep과 같은 파일 시스템 도구를 추가합니다.
    파일은 BackendProtocol을 구현하는 모든 백엔드를 사용하여 저장할 수 있습니다.

    백엔드가 SandboxBackendProtocol을 구현하는 경우, 쉘 명령 실행을 위한 execute 도구도 추가됩니다.

    Args:
        backend: 파일 저장 및 선택적 실행을 위한 백엔드. 제공되지 않은 경우 기본값은 StateBackend입니다
            (에이전트 상태의 임시 저장소). 영구 저장소 또는 하이브리드 설정의 경우,
            사용자 정의 라우트가 있는 CompositeBackend를 사용하십시오. 실행 지원을 위해서는
            SandboxBackendProtocol을 구현하는 백엔드를 사용하십시오.
        system_prompt: 선택적 사용자 정의 시스템 프롬프트 재정의(override).
        custom_tool_descriptions: 선택적 사용자 정의 도구 설명 재정의.
        tool_token_limit_before_evict: 도구 결과를 파일 시스템으로 축출(evict)하기 전의 선택적 토큰 제한.

    Example:
        ```python
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from deepagents.backends import StateBackend, StoreBackend, CompositeBackend
        from langchain.agents import create_agent

        # Ephemeral storage only (default, no execution)
        agent = create_agent(middleware=[FilesystemMiddleware()])

        # With hybrid storage (ephemeral + persistent /memories/)
        backend = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend()})
        agent = create_agent(middleware=[FilesystemMiddleware(backend=backend)])

        # With sandbox backend (supports execution)
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
            backend: 파일 저장 및 선택적 실행을 위한 백엔드, 또는 팩토리 콜러블.
                제공되지 않은 경우 기본값은 StateBackend입니다.
            system_prompt: 선택적 사용자 정의 시스템 프롬프트 재정의.
            custom_tool_descriptions: 선택적 사용자 정의 도구 설명 재정의.
            tool_token_limit_before_evict: 도구 결과를 파일 시스템으로 축출하기 전의 선택적 토큰 제한.
        """
        self.tool_token_limit_before_evict = tool_token_limit_before_evict

        # Use provided backend or default to StateBackend factory
        self.backend = backend if backend is not None else (lambda rt: StateBackend(rt))

        # Set system prompt (allow full override or None to generate dynamically)
        self._custom_system_prompt = system_prompt

        self.tools = _get_filesystem_tools(self.backend, custom_tool_descriptions)

    def _get_backend(self, runtime: ToolRuntime) -> BackendProtocol:
        """백엔드 인스턴스 또는 팩토리에서 해결된 백엔드 인스턴스를 가져옵니다.

        Args:
            runtime: 도구 런타임 컨텍스트.

        Returns:
            해결된 백엔드 인스턴스.
        """
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """시스템 프롬프트를 업데이트하고 백엔드 기능을 기반으로 도구를 필터링합니다.

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 핸들러 함수.

        Returns:
            핸들러로부터의 모델 응답.
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any(
            (tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools
        )

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)
            backend_supports_execution = _supports_execution(backend)

            # If execute tool exists but backend doesn't support it, filter it out
            if not backend_supports_execution:
                filtered_tools = [
                    tool
                    for tool in request.tools
                    if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"
                ]
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
            request = request.override(
                system_prompt=request.system_prompt + "\n\n" + system_prompt if request.system_prompt else system_prompt
            )

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) 시스템 프롬프트를 업데이트하고 백엔드 기능을 기반으로 도구를 필터링합니다.

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 핸들러 함수.

        Returns:
            핸들러로부터의 모델 응답.
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any(
            (tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools
        )

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)
            backend_supports_execution = _supports_execution(backend)

            # If execute tool exists but backend doesn't support it, filter it out
            if not backend_supports_execution:
                filtered_tools = [
                    tool
                    for tool in request.tools
                    if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"
                ]
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
            request = request.override(
                system_prompt=request.system_prompt + "\n\n" + system_prompt if request.system_prompt else system_prompt
            )

        return await handler(request)

    def _process_large_message(
        self,
        message: ToolMessage,
        resolved_backend: BackendProtocol,
    ) -> tuple[ToolMessage, dict[str, FileData] | None]:
        content = message.content
        if not isinstance(content, str) or len(content) <= 4 * self.tool_token_limit_before_evict:
            return message, None

        sanitized_id = sanitize_tool_call_id(message.tool_call_id)
        file_path = f"/large_tool_results/{sanitized_id}"
        result = resolved_backend.write(file_path, content)
        if result.error:
            return message, None
        content_sample = format_content_with_line_numbers(
            [line[:1000] for line in content.splitlines()[:10]], start_line=1
        )
        processed_message = ToolMessage(
            TOO_LARGE_TOOL_MSG.format(
                tool_call_id=message.tool_call_id,
                file_path=file_path,
                content_sample=content_sample,
            ),
            tool_call_id=message.tool_call_id,
        )
        return processed_message, result.files_update

    def _intercept_large_tool_result(
        self, tool_result: ToolMessage | Command, runtime: ToolRuntime
    ) -> ToolMessage | Command:
        if isinstance(tool_result, ToolMessage) and isinstance(tool_result.content, str):
            if not (
                self.tool_token_limit_before_evict and len(tool_result.content) > 4 * self.tool_token_limit_before_evict
            ):
                return tool_result
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
                if not (
                    self.tool_token_limit_before_evict
                    and isinstance(message, ToolMessage)
                    and isinstance(message.content, str)
                    and len(message.content) > 4 * self.tool_token_limit_before_evict
                ):
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

        return tool_result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """도구 호출 결과의 크기를 확인하고 너무 큰 경우 파일 시스템으로 축출합니다.

        Args:
            request: 처리 중인 도구 호출 요청.
            handler: 수정된 요청으로 호출할 핸들러 함수.

        Returns:
            원시 ToolMessage, 또는 상태 내 ToolResult를 포함하는 유사(pseudo) 도구 메시지.
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
        """(async) 도구 호출 결과의 크기를 확인하고 너무 큰 경우 파일 시스템으로 축출합니다.

        Args:
            request: 처리 중인 도구 호출 요청.
            handler: 수정된 요청으로 호출할 핸들러 함수.

        Returns:
            원시 ToolMessage, 또는 상태 내 ToolResult를 포함하는 유사(pseudo) 도구 메시지.
        """
        if self.tool_token_limit_before_evict is None or request.tool_call["name"] in TOOL_GENERATORS:
            return await handler(request)

        tool_result = await handler(request)
        return self._intercept_large_tool_result(tool_result, request.runtime)
