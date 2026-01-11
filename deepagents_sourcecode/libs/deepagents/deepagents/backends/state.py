"""StateBackend: LangGraph 에이전트 상태에 파일을 저장합니다(일시적)."""

from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)
from deepagents.backends.utils import (
    _glob_search_files,
    create_file_data,
    file_data_to_string,
    format_read_response,
    grep_matches_from_files,
    perform_string_replacement,
    update_file_data,
)

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime


class StateBackend(BackendProtocol):
    """에이전트 상태(state)에 파일을 저장하는 백엔드입니다(일시적).

    LangGraph의 state/checkpoint 메커니즘을 사용합니다. 파일은 동일한 스레드(대화)
    내에서는 유지되지만, 스레드를 넘어 영구 저장되지는 않습니다. 또한 state는
    각 에이전트 step 이후 자동으로 checkpoint 됩니다.

    주의: LangGraph state는 직접 mutation이 아니라 `Command` 객체를 통해 업데이트해야 합니다.
    따라서 일부 작업은 `None` 대신 `Command`에 적용될 업데이트 정보를 담아 반환합니다.
    (코드에서는 `uses_state=True` 플래그로 표현)
    """

    def __init__(self, runtime: "ToolRuntime"):
        """`ToolRuntime`으로 StateBackend를 초기화합니다."""
        self.runtime = runtime

    def ls_info(self, path: str) -> list[FileInfo]:
        """지정한 디렉토리 바로 아래의 파일/폴더를 나열합니다(비재귀).

        Args:
            path: 디렉토리 절대 경로.

        Returns:
            해당 디렉토리의 직계 자식 항목에 대한 FileInfo 딕셔너리 리스트.
            디렉토리는 경로가 `/`로 끝나며 `is_dir=True`입니다.
        """
        files = self.runtime.state.get("files", {})
        infos: list[FileInfo] = []
        subdirs: set[str] = set()

        # prefix 매칭을 위해 trailing slash를 갖는 형태로 정규화
        normalized_path = path if path.endswith("/") else path + "/"

        for k, fd in files.items():
            # 지정 디렉토리(또는 하위 디렉토리)에 속하는지 확인
            if not k.startswith(normalized_path):
                continue

            # 디렉토리 이후의 상대 경로
            relative = k[len(normalized_path) :]

            # 상대 경로에 `/`가 있으면 하위 디렉토리에 있는 파일입니다.
            if "/" in relative:
                # 즉시 하위 디렉토리 이름만 추출
                subdir_name = relative.split("/")[0]
                subdirs.add(normalized_path + subdir_name + "/")
                continue

            # 현재 디렉토리 바로 아래에 있는 파일
            size = len("\n".join(fd.get("content", [])))
            infos.append(
                {
                    "path": k,
                    "is_dir": False,
                    "size": int(size),
                    "modified_at": fd.get("modified_at", ""),
                }
            )

        # 디렉토리 항목을 결과에 추가
        for subdir in sorted(subdirs):
            infos.append(
                {
                    "path": subdir,
                    "is_dir": True,
                    "size": 0,
                    "modified_at": "",
                }
            )

        infos.sort(key=lambda x: x.get("path", ""))
        return infos

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """파일을 읽어 라인 번호가 포함된 문자열로 반환합니다.

        Args:
            file_path: 절대 파일 경로.
            offset: 읽기 시작 라인 오프셋(0-index).
            limit: 최대 읽기 라인 수.

        Returns:
            라인 번호가 포함된 포맷 문자열 또는 오류 메시지.
        """
        files = self.runtime.state.get("files", {})
        file_data = files.get(file_path)

        if file_data is None:
            return f"Error: File '{file_path}' not found"

        return format_read_response(file_data, offset, limit)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """새 파일을 생성합니다.

        LangGraph state 업데이트를 위해 `files_update`가 포함된 `WriteResult`를 반환합니다.
        """
        files = self.runtime.state.get("files", {})

        if file_path in files:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        new_file_data = create_file_data(content)
        return WriteResult(path=file_path, files_update={file_path: new_file_data})

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """파일 내 문자열을 치환하여 편집합니다.

        `files_update` 및 치환 횟수(`occurrences`)가 포함된 `EditResult`를 반환합니다.
        """
        files = self.runtime.state.get("files", {})
        file_data = files.get(file_path)

        if file_data is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)
        return EditResult(path=file_path, files_update={file_path: new_file_data}, occurrences=int(occurrences))

    def grep_raw(
        self,
        pattern: str,
        path: str = "/",
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        files = self.runtime.state.get("files", {})
        return grep_matches_from_files(files, pattern, path, glob)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Glob 패턴에 매칭되는 파일에 대한 FileInfo를 반환합니다."""
        files = self.runtime.state.get("files", {})
        result = _glob_search_files(files, pattern, path)
        if result == "No files found":
            return []
        paths = result.split("\n")
        infos: list[FileInfo] = []
        for p in paths:
            fd = files.get(p)
            size = len("\n".join(fd.get("content", []))) if fd else 0
            infos.append(
                {
                    "path": p,
                    "is_dir": False,
                    "size": int(size),
                    "modified_at": fd.get("modified_at", "") if fd else "",
                }
            )
        return infos

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """여러 파일을 state로 업로드합니다."""
        raise NotImplementedError(
            "StateBackend does not support upload_files yet. You can upload files "
            "directly by passing them in invoke if you're storing files in the memory."
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """여러 파일을 state에서 다운로드합니다."""
        state_files = self.runtime.state.get("files", {})
        responses: list[FileDownloadResponse] = []

        for path in paths:
            file_data = state_files.get(path)

            if file_data is None:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
                continue

            # state의 FileData를 bytes로 변환
            content_str = file_data_to_string(file_data)
            content_bytes = content_str.encode("utf-8")

            responses.append(FileDownloadResponse(path=path, content=content_bytes, error=None))

        return responses
