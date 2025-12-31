"""StateBackend: 파일을 LangGraph 에이전트 상태(임시)에 저장되도록 합니다."""

from typing import TYPE_CHECKING

from deepagents.backends.protocol import BackendProtocol, EditResult, FileInfo, GrepMatch, WriteResult
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
    """에이전트 상태(임시)에 파일을 저장하는 백엔드.

    LangGraph의 상태 관리 및 체크포인팅을 사용합니다. 파일은 하나의 대화 스레드 내에서만 지속되며
    스레드 간에는 공유되지 않습니다. 상태는 각 에이전트 단계 후에 자동으로 체크포인트됩니다.

    특수 처리: LangGraph 상태는 (직접 변경이 아닌) Command 객체를 통해 업데이트되어야 하므로,
    작업은 None 대신 Command 객체를 반환할 수 있습니다. 이는 uses_state=True 플래그로 표시됩니다.
    """

    def __init__(self, runtime: "ToolRuntime"):
        """런타임으로 StateBackend를 초기화합니다."""
        self.runtime = runtime

    def ls_info(self, path: str) -> list[FileInfo]:
        """지정된 디렉토리의 파일과 디렉토리를 나열합니다 (비재귀적).

        Args:
            path: 디렉토리의 절대 경로.

        Returns:
            디렉토리 바로 아래에 있는 파일 및 디렉토리에 대한 FileInfo 유사 dict 목록.
            디렉토리는 경로 끝에 /가 붙으며 is_dir=True입니다.
        """
        files = self.runtime.state.get("files", {})
        infos: list[FileInfo] = []
        subdirs: set[str] = set()

        # Normalize path to have trailing slash for proper prefix matching
        normalized_path = path if path.endswith("/") else path + "/"

        for k, fd in files.items():
            # Check if file is in the specified directory or a subdirectory
            if not k.startswith(normalized_path):
                continue

            # Get the relative path after the directory
            relative = k[len(normalized_path) :]

            # If relative path contains '/', it's in a subdirectory
            if "/" in relative:
                # Extract the immediate subdirectory name
                subdir_name = relative.split("/")[0]
                subdirs.add(normalized_path + subdir_name + "/")
                continue

            # This is a file directly in the current directory
            size = len("\n".join(fd.get("content", [])))
            infos.append({
                "path": k,
                "is_dir": False,
                "size": int(size),
                "modified_at": fd.get("modified_at", ""),
            })

        # Add directories to the results
        for subdir in sorted(subdirs):
            infos.append({
                "path": subdir,
                "is_dir": True,
                "size": 0,
                "modified_at": "",
            })

        infos.sort(key=lambda x: x.get("path", ""))
        return infos

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """파일 내용을 라인 번호와 함께 읽습니다.

        Args:
            file_path: 파일 절대 경로.
            offset: 읽기 시작할 라인 오프셋 (0부터 시작).
            limit: 읽을 최대 라인 수.

        Returns:
            라인 번호가 포함된 형식화된 파일 내용, 또는 에러 메시지.
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
        """내용을 포함하는 새 파일을 생성합니다.
        LangGraph 상태 업데이트를 위한 files_update가 포함된 WriteResult를 반환합니다.
        """
        files = self.runtime.state.get("files", {})

        if file_path in files:
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path."
            )

        new_file_data = create_file_data(content)
        return WriteResult(path=file_path, files_update={file_path: new_file_data})

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """문자열 발생(occurrences)을 교체하여 파일을 편집합니다.
        files_update와 occurrences가 포함된 EditResult를 반환합니다.
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
        """glob 패턴과 일치하는 파일에 대한 FileInfo를 가져옵니다."""
        files = self.runtime.state.get("files", {})
        result = _glob_search_files(files, pattern, path)
        if result == "No files found":
            return []
        paths = result.split("\n")
        infos: list[FileInfo] = []
        for p in paths:
            fd = files.get(p)
            size = len("\n".join(fd.get("content", []))) if fd else 0
            infos.append({
                "path": p,
                "is_dir": False,
                "size": int(size),
                "modified_at": fd.get("modified_at", "") if fd else "",
            })
        return infos
