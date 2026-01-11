"""플러그인 가능한 메모리 백엔드를 위한 프로토콜 정의입니다.

이 모듈은 모든 백엔드 구현이 따라야 하는 BackendProtocol을 정의합니다.
백엔드는 다양한 위치(상태, 파일 시스템, 데이터베이스 등)에 파일을 저장할 수 있으며
파일 작업을 위한 균일한 인터페이스를 제공합니다.
"""

import abc
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypeAlias

from langchain.tools import ToolRuntime
from typing_extensions import TypedDict

FileOperationError = Literal[
    "file_not_found",  # Download: 파일이 존재하지 않음
    "permission_denied",  # Both: 접근 거부됨
    "is_directory",  # Download: 디렉토리를 파일로 다운로드하려고 시도
    "invalid_path",  # Both: 경로 문법이 잘못되었습니다(상위 디렉토리 누락, 잘못된 문자)
]
"""파일 업로드/다운로드 작업을 위한 표준화된 오류 코드입니다.

이는 LLM이 이해하고 잠재적으로 수정할 수 있는 일반적이고 복구 가능한 오류를 나타냅니다:
- file_not_found: 요청한 파일이 존재하지 않음(다운로드)
- parent_not_found: 상위 디렉토리가 존재하지 않음(업로드)
- permission_denied: 작업에 대한 접근 거부됨
- is_directory: 디렉토리를 파일로 다운로드하려고 시도
- invalid_path: 경로 문법이 잘못되었거나 잘못된 문자가 포함됨
"""


@dataclass
class FileDownloadResponse:
    """단일 파일 다운로드 작업의 결과입니다.

    이 응답은 배치 작업에서 부분적 성공을 허용하도록 설계되었습니다.
    오류는 LLM이 파일 작업을 수행하는 사용 사례에서 복구 가능한
    특정 조건에 대해 FileOperationError 리터럴을 사용하여 표준화됩니다.

    Attributes:
        path: 요청된 파일 경로입니다. 배치 결과 처리 시 상관관계에 유용하며,
            특히 오류 메시지에 유용합니다.
        content: 성공 시 파일 내용(바이트), 실패 시 None.
        error: 실패 시 표준화된 오류 코드, 성공 시 None.
            구조화되고 LLM이 조치 가능한 오류 보고를 위해 FileOperationError 리터럴을 사용합니다.

    Examples:
        >>> # 성공
        >>> FileDownloadResponse(path="/app/config.json", content=b"{...}", error=None)
        >>> # 실패
        >>> FileDownloadResponse(path="/wrong/path.txt", content=None, error="file_not_found")
    """

    path: str
    content: bytes | None = None
    error: FileOperationError | None = None


@dataclass
class FileUploadResponse:
    """단일 파일 업로드 작업의 결과입니다.

    이 응답은 배치 작업에서 부분적 성공을 허용하도록 설계되었습니다.
    오류는 LLM이 파일 작업을 수행하는 사용 사례에서 복구 가능한
    특정 조건에 대해 FileOperationError 리터럴을 사용하여 표준화됩니다.

    Attributes:
        path: 요청된 파일 경로입니다. 배치 결과 처리 시 상관관계에 유용하며,
            명확한 오류 메시지에 도움이 됩니다.
        error: 실패 시 표준화된 오류 코드, 성공 시 None.
            구조화되고 LLM이 조치 가능한 오류 보고를 위해 FileOperationError 리터럴을 사용합니다.

    Examples:
        >>> # 성공
        >>> FileUploadResponse(path="/app/data.txt", error=None)
        >>> # 실패
        >>> FileUploadResponse(path="/readonly/file.txt", error="permission_denied")
    """

    path: str
    error: FileOperationError | None = None


class FileInfo(TypedDict):
    """파일 목록 조회 시 사용하는 구조화된 항목 정보입니다.

    백엔드 간 최소 계약(minimal contract)이며, `"path"`만 필수입니다.
    나머지 필드는 best-effort로 제공되며, 백엔드에 따라 누락될 수 있습니다.
    """

    path: str
    is_dir: NotRequired[bool]
    size: NotRequired[int]  # bytes (approx)
    modified_at: NotRequired[str]  # ISO timestamp if known


class GrepMatch(TypedDict):
    """grep 매칭 결과(구조화) 엔트리입니다."""

    path: str
    line: int
    text: str


@dataclass
class WriteResult:
    """백엔드 write 작업의 결과입니다.

    Attributes:
        error: 실패 시 오류 메시지, 성공 시 `None`.
        path: 성공 시 작성된 파일의 절대 경로, 실패 시 `None`.
        files_update: checkpoint 기반 백엔드에서는 state 업데이트 딕셔너리,
            외부 스토리지 기반 백엔드에서는 `None`.
            checkpoint 백엔드는 LangGraph state 업데이트를 위해 `{file_path: file_data}`를 채웁니다.
            외부 백엔드는 `None`(이미 디스크/S3/DB 등에 영구 반영됨)을 사용합니다.

    Examples:
        >>> # Checkpoint storage
        >>> WriteResult(path="/f.txt", files_update={"/f.txt": {...}})
        >>> # External storage
        >>> WriteResult(path="/f.txt", files_update=None)
        >>> # Error
        >>> WriteResult(error="File exists")
    """

    error: str | None = None
    path: str | None = None
    files_update: dict[str, Any] | None = None


@dataclass
class EditResult:
    """백엔드 edit 작업의 결과입니다.

    Attributes:
        error: 실패 시 오류 메시지, 성공 시 `None`.
        path: 성공 시 수정된 파일의 절대 경로, 실패 시 `None`.
        files_update: checkpoint 기반 백엔드에서는 state 업데이트 딕셔너리,
            외부 스토리지 기반 백엔드에서는 `None`.
            checkpoint 백엔드는 LangGraph state 업데이트를 위해 `{file_path: file_data}`를 채웁니다.
            외부 백엔드는 `None`(이미 디스크/S3/DB 등에 영구 반영됨)을 사용합니다.
        occurrences: 치환 횟수. 실패 시 `None`.

    Examples:
        >>> # Checkpoint storage
        >>> EditResult(path="/f.txt", files_update={"/f.txt": {...}}, occurrences=1)
        >>> # External storage
        >>> EditResult(path="/f.txt", files_update=None, occurrences=2)
        >>> # Error
        >>> EditResult(error="File not found")
    """

    error: str | None = None
    path: str | None = None
    files_update: dict[str, Any] | None = None
    occurrences: int | None = None


class BackendProtocol(abc.ABC):
    """플러그인 가능한 메모리/파일 백엔드용 단일(unified) 프로토콜입니다.

    백엔드는 상태(state), 로컬 파일 시스템, 데이터베이스 등 다양한 위치에 파일을 저장할 수 있으며,
    파일 작업에 대해 일관된(uniform) 인터페이스를 제공합니다.

    모든 file data는 아래 구조의 딕셔너리로 표현합니다.

    ```python
    {
        "content": list[str],  # 텍스트 라인 목록
        "created_at": str,     # ISO 형식 타임스탬프
        "modified_at": str,    # ISO 형식 타임스탬프
    }
    ```
    """

    def ls_info(self, path: str) -> list["FileInfo"]:
        """디렉토리 내 파일/폴더를 메타데이터와 함께 나열합니다.

        Args:
            path: 나열할 디렉토리의 절대 경로. 반드시 `/`로 시작해야 합니다.

        Returns:
            FileInfo 딕셔너리 리스트:
            - `path` (필수): 절대 경로
            - `is_dir` (선택): 디렉토리면 `True`
            - `size` (선택): 바이트 단위 크기
            - `modified_at` (선택): ISO 8601 타임스탬프
        """

    async def als_info(self, path: str) -> list["FileInfo"]:
        """`ls_info`의 async 버전입니다."""
        return await asyncio.to_thread(self.ls_info, path)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """파일을 읽어 라인 번호가 포함된 문자열로 반환합니다.

        Args:
            file_path: 읽을 파일의 절대 경로. 반드시 `/`로 시작해야 합니다.
            offset: 읽기 시작 라인(0-index). 기본값: 0.
            limit: 최대 읽기 라인 수. 기본값: 2000.

        Returns:
            라인 번호(`cat -n`) 형식으로 포맷된 파일 내용 문자열(라인 번호는 1부터 시작).
            2000자를 초과하는 라인은 잘립니다.

            파일이 없거나 읽을 수 없으면 오류 문자열을 반환합니다.

        !!! note
            - 큰 파일은 pagination(offset/limit)을 사용해 컨텍스트 오버플로우를 방지하세요.
            - 첫 스캔: `read(path, limit=100)`으로 구조 파악
            - 추가 읽기: `read(path, offset=100, limit=200)`으로 다음 구간
            - 편집 전에는 반드시 파일을 먼저 읽어야 합니다.
            - 파일이 비어 있으면 system reminder 경고가 반환될 수 있습니다.
        """

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """`read`의 async 버전입니다."""
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list["GrepMatch"] | str:
        """파일에서 리터럴(비정규식) 텍스트 패턴을 검색합니다.

        Args:
            pattern: Literal string to search for (NOT regex).
                     Performs exact substring matching within file content.
                     Example: "TODO" matches any line containing "TODO"

            path: Optional directory path to search in.
                  If None, searches in current working directory.
                  Example: "/workspace/src"

            glob: Optional glob pattern to filter which FILES to search.
                  Filters by filename/path, not content.
                  Supports standard glob wildcards:
                  - `*` matches any characters in filename
                  - `**` matches any directories recursively
                  - `?` matches single character
                  - `[abc]` matches one character from set

        Examples:
                  - "*.py" - only search Python files
                  - "**/*.txt" - search all .txt files recursively
                  - "src/**/*.js" - search JS files under src/
                  - "test[0-9].txt" - search test0.txt, test1.txt, etc.

        Returns:
            성공 시: 아래 필드를 가진 구조화 결과 `list[GrepMatch]`
            - path: 절대 파일 경로
            - line: 라인 번호(1-index)
            - text: 매칭된 라인의 전체 텍스트

            실패 시: 오류 메시지 문자열(예: invalid path, permission denied)
        """

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list["GrepMatch"] | str:
        """`grep_raw`의 async 버전입니다."""
        return await asyncio.to_thread(self.grep_raw, pattern, path, glob)

    def glob_info(self, pattern: str, path: str = "/") -> list["FileInfo"]:
        """Glob 패턴에 매칭되는 파일을 찾습니다.

        Args:
            pattern: Glob pattern with wildcards to match file paths.
                     Supports standard glob syntax:
                     - `*` matches any characters within a filename/directory
                     - `**` matches any directories recursively
                     - `?` matches a single character
                     - `[abc]` matches one character from set

            path: Base directory to search from. Default: "/" (root).
                  The pattern is applied relative to this path.

        Returns:
            FileInfo 리스트
        """

    async def aglob_info(self, pattern: str, path: str = "/") -> list["FileInfo"]:
        """`glob_info`의 async 버전입니다."""
        return await asyncio.to_thread(self.glob_info, pattern, path)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """새 파일을 생성하고 내용을 씁니다(동일 경로 파일이 이미 있으면 오류).

        Args:
            file_path: Absolute path where the file should be created.
                       Must start with '/'.
            content: String content to write to the file.

        Returns:
            WriteResult
        """

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """`write`의 async 버전입니다."""
        return await asyncio.to_thread(self.write, file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """기존 파일에서 정확한 문자열 매칭 기반 치환을 수행합니다.

        Args:
            file_path: Absolute path to the file to edit. Must start with '/'.
            old_string: Exact string to search for and replace.
                       Must match exactly including whitespace and indentation.
            new_string: String to replace old_string with.
                       Must be different from old_string.
            replace_all: If True, replace all occurrences. If False (default),
                        old_string must be unique in the file or the edit fails.

        Returns:
            EditResult
        """

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """`edit`의 async 버전입니다."""
        return await asyncio.to_thread(self.edit, file_path, old_string, new_string, replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """여러 파일을 샌드박스로 업로드합니다.

        This API is designed to allow developers to use it either directly or
        by exposing it to LLMs via custom tools.

        Args:
            files: List of (path, content) tuples to upload.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order (response[i] for files[i]).
            Check the error field to determine success/failure per file.

        Examples:
            ```python
            responses = sandbox.upload_files(
                [
                    ("/app/config.json", b"{...}"),
                    ("/app/data.txt", b"content"),
                ]
            )
            ```
        """

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """`upload_files`의 async 버전입니다."""
        return await asyncio.to_thread(self.upload_files, files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """여러 파일을 샌드박스에서 다운로드합니다.

        This API is designed to allow developers to use it either directly or
        by exposing it to LLMs via custom tools.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
            Response order matches input order (response[i] for paths[i]).
            Check the error field to determine success/failure per file.
        """

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """`download_files`의 async 버전입니다."""
        return await asyncio.to_thread(self.download_files, paths)


@dataclass
class ExecuteResponse:
    """코드 실행 결과입니다.

    LLM이 소비하기 좋도록 단순화한 스키마입니다.
    """

    output: str
    """실행된 커맨드의 stdout+stderr 합쳐진 출력."""

    exit_code: int | None = None
    """프로세스 종료 코드. 0은 성공, 0이 아니면 실패."""

    truncated: bool = False
    """백엔드 제한으로 출력이 잘렸는지 여부."""


class SandboxBackendProtocol(BackendProtocol):
    """격리된 런타임을 제공하는 샌드박스 백엔드용 프로토콜입니다.

    샌드박스 백엔드는 별도 프로세스/컨테이너 같은 격리된 환경에서 실행되며,
    정해진 인터페이스를 통해 통신합니다.
    """

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """샌드박스 프로세스에서 커맨드를 실행합니다.

        LLM 친화적으로 단순화된 인터페이스입니다.

        Args:
            command: Full shell command string to execute.

        Returns:
            ExecuteResponse with combined output, exit code, optional signal, and truncation flag.
        """

    async def aexecute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """`execute`의 async 버전입니다."""
        return await asyncio.to_thread(self.execute, command)

    @property
    def id(self) -> str:
        """샌드박스 백엔드 인스턴스의 고유 식별자."""


BackendFactory: TypeAlias = Callable[[ToolRuntime], BackendProtocol]
BACKEND_TYPES = BackendProtocol | BackendFactory
