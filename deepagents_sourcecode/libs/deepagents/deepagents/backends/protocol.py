"""플러그형 메모리 백엔드를 위한 프로토콜 정의.

이 모듈은 모든 백엔드 구현이 따라야 하는 BackendProtocol을 정의합니다.
백엔드는 파일들을 서로 다른 위치(state, filesystem, database 등)에 저장할 수 있으며,
파일 작업에 대해 통일된 인터페이스를 제공합니다.
"""

import abc
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypeAlias

from langchain.tools import ToolRuntime
from typing_extensions import TypedDict

FileOperationError = Literal[
    "file_not_found",  # Download: file doesn't exist
    "permission_denied",  # Both: access denied
    "is_directory",  # Download: tried to download directory as file
    "invalid_path",  # Both: path syntax malformed (parent dir missing, invalid chars)
]
"""파일 업로드/다운로드 작업을 위한 표준화된 에러 코드.

이 코드들은 LLM이 이해하고 잠재적으로 수정할 수 있는 일반적인 복구 가능 에러들을 나타냅니다:
- file_not_found: 요청한 파일이 존재하지 않음 (다운로드)
- parent_not_found: 부모 디렉토리가 존재하지 않음 (업로드)
- permission_denied: 작업에 대한 접근이 거부됨
- is_directory: 디렉토리를 파일로 다운로드하려고 시도함
- invalid_path: 경로 구문이 잘못되었거나 유효하지 않은 문자를 포함함
"""


@dataclass
class FileDownloadResponse:
    """단일 파일 다운로드 작업의 결과.

    이 응답은 일괄 작업에서 부분적인 성공을 허용하도록 설계되었습니다.
    에러는 LLM이 파일 작업을 수행하는 사용 사례에서 복구 가능한 특정 조건들을 위해
    FileOperationError 리터럴을 사용하여 표준화되었습니다.

    Attributes:
        path: 요청된 파일 경로. 일괄 결과를 처리할 때 쉬운 상호참조를 위해 포함됩니다.
             에러 메시지에 특히 유용합니다.
        content: 성공 시 파일 내용(bytes), 실패 시 None.
        error: 실패 시 표준화된 에러 코드, 성공 시 None.
             구조화되고 LLM이 조치 가능한 에러 보고를 위해 FileOperationError 리터럴을 사용합니다.

    Examples:
        >>> # Success
        >>> FileDownloadResponse(path="/app/config.json", content=b"{...}", error=None)
        >>> # Failure
        >>> FileDownloadResponse(path="/wrong/path.txt", content=None, error="file_not_found")
    """

    path: str
    content: bytes | None = None
    error: FileOperationError | None = None


@dataclass
class FileUploadResponse:
    """단일 파일 업로드 작업의 결과.

    이 응답은 일괄 작업에서 부분적인 성공을 허용하도록 설계되었습니다.
    에러는 LLM이 파일 작업을 수행하는 사용 사례에서 복구 가능한 특정 조건들을 위해
    FileOperationError 리터럴을 사용하여 표준화되었습니다.

    Attributes:
        path: 요청된 파일 경로. 일괄 결과를 처리할 때 쉬운 상호참조와 명확한 에러 메시지를 위해 포함됩니다.
        error: 실패 시 표준화된 에러 코드, 성공 시 None.
            구조화되고 LLM이 조치 가능한 에러 보고를 위해 FileOperationError 리터럴을 사용합니다.

    Examples:
        >>> # Success
        >>> FileUploadResponse(path="/app/data.txt", error=None)
        >>> # Failure
        >>> FileUploadResponse(path="/readonly/file.txt", error="permission_denied")
    """

    path: str
    error: FileOperationError | None = None


class FileInfo(TypedDict):
    """구조화된 파일 목록 정보.

    백엔드 전반에서 사용되는 최소한의 계약입니다. "path"만 필수입니다.
    다른 필드들은 최선의 노력(best-effort)으로 제공되며 백엔드에 따라 없을 수 있습니다.
    """

    path: str
    is_dir: NotRequired[bool]
    size: NotRequired[int]  # bytes (approx)
    modified_at: NotRequired[str]  # ISO timestamp if known


class GrepMatch(TypedDict):
    """구조화된 grep 일치(match) 항목."""

    path: str
    line: int
    text: str


@dataclass
class WriteResult:
    """백엔드 쓰기(write) 작업의 결과.

    Attributes:
        error: 실패 시 에러 메시지, 성공 시 None.
        path: 작성된 파일의 절대 경로, 실패 시 None.
        files_update: 체크포인트 백엔드를 위한 상태 업데이트 dict, 외부 저장소인 경우 None.
            체크포인트 백엔드는 이를 LangGraph 상태를 위한 {file_path: file_data}로 채웁니다.
            외부 백엔드는 None으로 설정합니다 (이미 디스크/S3/데이터베이스 등에 영구 저장됨).

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
    """백엔드 편집(edit) 작업의 결과.

    Attributes:
        error: 실패 시 에러 메시지, 성공 시 None.
        path: 편집된 파일의 절대 경로, 실패 시 None.
        files_update: 체크포인트 백엔드를 위한 상태 업데이트 dict, 외부 저장소인 경우 None.
            체크포인트 백엔드는 이를 LangGraph 상태를 위한 {file_path: file_data}로 채웁니다.
            외부 백엔드는 None으로 설정합니다 (이미 디스크/S3/데이터베이스 등에 영구 저장됨).
        occurrences: 교체된 횟수, 실패 시 None.

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
    """플러그형 메모리 백엔드를 위한 프로토콜 (단일 통일 인터페이스).

    백엔드는 파일들을 다양한 위치(state, filesystem, database 등)에 저장할 수 있으며,
    파일 작업에 대해 통일된 인터페이스를 제공합니다.

    모든 파일 데이터는 다음 구조를 가진 딕셔너리로 표현됩니다:
    {
        "content": list[str], # 텍스트 내용의 라인 리스트
        "created_at": str, # ISO 형식 타임스탬프
        "modified_at": str, # ISO 형식 타임스탬프
    }
    """

    def ls_info(self, path: str) -> list["FileInfo"]:
        """디렉토리 내의 모든 파일과 메타데이터를 나열합니다.

        Args:
            path: 목록을 조회할 디렉토리의 절대 경로. '/'로 시작해야 합니다.

        Returns:
            파일 메타데이터를 포함하는 FileInfo 딕셔너리의 리스트:

            - `path` (필수): 절대 파일 경로
            - `is_dir` (선택): 디렉토리인 경우 True
            - `size` (선택): 바이트 단위 파일 크기
            - `modified_at` (선택): ISO 8601 타임스탬프
        """

    async def als_info(self, path: str) -> list["FileInfo"]:
        """Async version of ls_info."""
        return await asyncio.to_thread(self.ls_info, path)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """파일 내용을 라인 번호와 함께 읽습니다.

        Args:
            file_path: 읽을 파일의 절대 경로. '/'로 시작해야 합니다.
            offset: 읽기 시작할 라인 번호 (0부터 시작). 기본값: 0.
            limit: 읽을 최대 라인 수. 기본값: 2000.

        Returns:
            라인 번호가 포함된 파일 내용 문자열 (cat -n 형식), 1번 라인부터 시작합니다.
            2000자를 초과하는 라인은 잘립니다.

            파일이 존재하지 않거나 읽을 수 없는 경우 에러 문자열을 반환합니다.

        !!! note
            - 컨텍스트 오버플로우를 방지하기 위해 대용량 파일에는 페이지네이션(offset/limit)을 사용하세요.
            - 첫 스캔: `read(path, limit=100)`으로 파일 구조 확인
            - 추가 읽기: `read(path, offset=100, limit=200)`으로 다음 구간 읽기
            - ALWAYS read a file before editing it (편집 전 반드시 파일 읽기)
            - 파일이 존재하지만 비어있는 경우, 시스템 리마인더 경고를 받게 됩니다.
        """

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Async version of read."""
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list["GrepMatch"] | str:
        """파일들에서 리터럴 텍스트 패턴을 검색합니다.

        Args:
            pattern: 검색할 리터럴 문자열 (정규식 아님).
                     파일 내용 내에서 정확한 부분 문자열 매칭을 수행합니다.
                     예: "TODO"는 "TODO"를 포함하는 모든 라인과 일치합니다.

            path: 검색할 디렉토리 경로 (선택).
                  None인 경우 현재 작업 디렉토리에서 검색합니다.
                  예: "/workspace/src"

            glob: 검색할 파일을 필터링하기 위한 선택적 glob 패턴.
                  내용이 아닌 파일명/경로로 필터링합니다.
                  표준 glob 와일드카드를 지원합니다:
                  - `*`: 파일명의 모든 문자와 일치
                  - `**`: 모든 디렉토리를 재귀적으로 일치
                  - `?`: 단일 문자와 일치
                  - `[abc]`: 세트 내의 한 문자와 일치

        Examples:
                  - "*.py" - Python 파일만 검색
                  - "**/*.txt" - 모든 .txt 파일을 재귀적으로 검색
                  - "src/**/*.js" - src/ 하위의 JS 파일 검색
                  - "test[0-9].txt" - test0.txt, test1.txt 등을 검색

        Returns:
            성공 시: 다음을 포함하는 구조화된 결과 list[GrepMatch] 반환:
                - path: 절대 파일 경로
                - line: 라인 번호 (1부터 시작)
                - text: 매치를 포함하는 전체 라인 내용

            실패 시: 에러 메시지 문자열 (예: 잘못된 경로, 권한 거부)
        """

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list["GrepMatch"] | str:
        """Async version of grep_raw."""
        return await asyncio.to_thread(self.grep_raw, pattern, path, glob)

    def glob_info(self, pattern: str, path: str = "/") -> list["FileInfo"]:
        """glob 패턴과 일치하는 파일을 찾습니다.

        Args:
            pattern: 파일 경로와 일치시킬 와일드카드가 포함된 Glob 패턴.
                     표준 glob 문법을 지원합니다:
                     - `*` 파일명/디렉토리 내의 모든 문자와 일치
                     - `**` 모든 디렉토리를 재귀적으로 일치
                     - `?` 단일 문자와 일치
                     - `[abc]` 세트 내의 한 문자와 일치

            path: 검색을 시작할 기본 디렉토리. 기본값: "/" (루트).
                  패턴은 이 경로에 상대적으로 적용됩니다.

        Returns:
            list of FileInfo
        """

    async def aglob_info(self, pattern: str, path: str = "/") -> list["FileInfo"]:
        """Async version of glob_info."""
        return await asyncio.to_thread(self.glob_info, pattern, path)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """파일시스템 내 새 파일에 내용을 씁니다. 파일이 존재하면 에러가 발생합니다.

        Args:
            file_path: 파일이 생성될 절대 경로.
                       '/'로 시작해야 합니다.
            content: 파일에 쓸 문자열 내용.

        Returns:
            WriteResult
        """

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Async version of write."""
        return await asyncio.to_thread(self.write, file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """기존 파일에서 정확한 문자열 교체를 수행합니다.

        Args:
            file_path: 편집할 파일의 절대 경로. '/'로 시작해야 합니다.
            old_string: 검색 및 교체할 정확한 문자열.
                       공백과 들여쓰기를 포함하여 정확히 일치해야 합니다.
            new_string: old_string을 대체할 문자열.
                       old_string과 달라야 합니다.
            replace_all: True인 경우 모든 발생을 교체합니다. False(기본값)인 경우
                        old_string은 파일 내에서 유일해야 하며, 그렇지 않으면 편집이 실패합니다.

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
        """Async version of edit."""
        return await asyncio.to_thread(self.edit, file_path, old_string, new_string, replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """샌드박스에 여러 파일을 업로드합니다.

        이 API는 개발자가 직접 사용하거나 커스텀 도구를 통해
        LLM에게 노출할 수 있도록 설계되었습니다.

        Args:
            files: 업로드할 (path, content) 튜플의 리스트.

        Returns:
            FileUploadResponse 객체들의 리스트. 입력 파일마다 하나씩 반환됩니다.
            응답 순서는 입력 순서와 일치합니다 (files[i]에 대해 response[i]).
            파일별 성공/실패 여부는 error 필드를 확인하십시오.

        Examples:
            ```python
            responses = sandbox.upload_files([
                ("/app/config.json", b"{...}"),
                ("/app/data.txt", b"content"),
            ])
            ```
        """

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Async version of upload_files."""
        return await asyncio.to_thread(self.upload_files, files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """샌드박스에서 여러 파일을 다운로드합니다.

        이 API는 개발자가 직접 사용하거나 커스텀 도구를 통해
        LLM에게 노출할 수 있도록 설계되었습니다.

        Args:
            paths: 다운로드할 파일 경로의 리스트.

        Returns:
            FileDownloadResponse 객체들의 리스트. 입력 경로마다 하나씩 반환됩니다.
            응답 순서는 입력 순서와 일치합니다 (paths[i]에 대해 response[i]).
            파일별 성공/실패 여부는 error 필드를 확인하십시오.
        """

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async version of download_files."""
        return await asyncio.to_thread(self.download_files, paths)


@dataclass
class ExecuteResponse:
    """코드 실행 결과.

    LLM 소비에 최적화된 단순화된 스키마입니다.
    """

    output: str
    """실행된 명령의 결합된 표준 출력(stdout) 및 표준 에러(stderr)."""

    exit_code: int | None = None
    """프로세스 종료 코드. 0은 성공, 0이 아닌 값은 실패를 나타냅니다."""

    truncated: bool = False
    """백엔드 제한으로 인해 출력이 잘렸는지 여부."""


class SandboxBackendProtocol(BackendProtocol):
    """격리된 런타임을 가진 샌드박스 백엔드를 위한 프로토콜.

    샌드박스 백엔드는 격리된 환경(예: 별도 프로세스, 컨테이너)에서 실행되며
    정의된 인터페이스를 통해 통신합니다.
    """

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """프로세스에서 명령을 실행합니다.

        LLM 소비에 최적화된 단순화된 인터페이스.

        Args:
            command: 실행할 전체 쉘 명령 문자열.

        Returns:
            결합된 출력, 종료 코드, 선택적 시그널, 잘림(truncation) 플래그를 포함하는 ExecuteResponse.
        """

    async def aexecute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Async version of execute."""
        return await asyncio.to_thread(self.execute, command)

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend instance."""


BackendFactory: TypeAlias = Callable[[ToolRuntime], BackendProtocol]
BACKEND_TYPES = BackendProtocol | BackendFactory
