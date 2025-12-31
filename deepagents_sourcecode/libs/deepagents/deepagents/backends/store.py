"""StoreBackend: LangGraph의 BaseStore(영구적, 스레드 간 공유)를 위한 어댑터."""

from typing import Any

from langgraph.config import get_config
from langgraph.store.base import BaseStore, Item

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


class StoreBackend(BackendProtocol):
    """파일을 LangGraph의 BaseStore(영구적)에 저장하는 백엔드.

    LangGraph의 Store를 사용하여 영구적이고 대화 간 공유되는 저장소를 사용합니다.
    파일은 네임스페이스를 통해 조직화되며 모든 스레드에서 지속됩니다.

    네임스페이스는 다중 에이전트 격리를 위해 선택적 assistant_id를 포함할 수 있습니다.
    """

    def __init__(self, runtime: "ToolRuntime"):
        """런타임으로 StoreBackend를 초기화합니다.

        Args:
            runtime: 저장소 접근 및 구성을 제공하는 ToolRuntime 인스턴스.
        """
        self.runtime = runtime

    def _get_store(self) -> BaseStore:
        """저장소(store) 인스턴스를 가져옵니다.

        Returns:
            런타임의 BaseStore 인스턴스.

        Raises:
            ValueError: 런타임에서 저장소를 사용할 수 없는 경우.
        """
        store = self.runtime.store
        if store is None:
            msg = "Store is required but not available in runtime"
            raise ValueError(msg)
        return store

    def _get_namespace(self) -> tuple[str, ...]:
        """저장소 작업을 위한 네임스페이스를 가져옵니다.

        우선순위:
        1) 존재하는 경우 `self.runtime.config` 사용 (테스트에서 명시적으로 전달).
        2) 가능한 경우 `langgraph.config.get_config()`로 폴백(fallback).
        3) ("filesystem",)으로 기본 설정.

        config 메타데이터에 assistant_id가 있는 경우,
        에이전트별 격리를 제공하기 위해 (assistant_id, "filesystem")을 반환합니다.
        """
        namespace = "filesystem"

        # Prefer the runtime-provided config when present
        runtime_cfg = getattr(self.runtime, "config", None)
        if isinstance(runtime_cfg, dict):
            assistant_id = runtime_cfg.get("metadata", {}).get("assistant_id")
            if assistant_id:
                return (assistant_id, namespace)
            return (namespace,)

        # Fallback to langgraph's context, but guard against errors when
        # called outside of a runnable context
        try:
            cfg = get_config()
        except Exception:
            return (namespace,)

        try:
            assistant_id = cfg.get("metadata", {}).get("assistant_id")  # type: ignore[assignment]
        except Exception:
            assistant_id = None

        if assistant_id:
            return (assistant_id, namespace)
        return (namespace,)

    def _convert_store_item_to_file_data(self, store_item: Item) -> dict[str, Any]:
        """저장소 Item을 FileData 형식으로 변환합니다.

        Args:
            store_item: 파일 데이터를 포함하는 저장소 Item.

        Returns:
            content, created_at, modified_at 필드를 포함하는 FileData dict.

        Raises:
            ValueError: 필수 필드가 누락되었거나 올바르지 않은 타입인 경우.
        """
        if "content" not in store_item.value or not isinstance(store_item.value["content"], list):
            msg = f"Store item does not contain valid content field. Got: {store_item.value.keys()}"
            raise ValueError(msg)
        if "created_at" not in store_item.value or not isinstance(store_item.value["created_at"], str):
            msg = f"Store item does not contain valid created_at field. Got: {store_item.value.keys()}"
            raise ValueError(msg)
        if "modified_at" not in store_item.value or not isinstance(store_item.value["modified_at"], str):
            msg = f"Store item does not contain valid modified_at field. Got: {store_item.value.keys()}"
            raise ValueError(msg)
        return {
            "content": store_item.value["content"],
            "created_at": store_item.value["created_at"],
            "modified_at": store_item.value["modified_at"],
        }

    def _convert_file_data_to_store_value(self, file_data: dict[str, Any]) -> dict[str, Any]:
        """FileData를 store.put()에 적합한 dict로 변환합니다.

        Args:
            file_data: 변환할 FileData.

        Returns:
            content, created_at, modified_at 필드를 포함하는 딕셔너리.
        """
        return {
            "content": file_data["content"],
            "created_at": file_data["created_at"],
            "modified_at": file_data["modified_at"],
        }

    def _search_store_paginated(
        self,
        store: BaseStore,
        namespace: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        page_size: int = 100,
    ) -> list[Item]:
        """자동 페이지네이션으로 저장소를 검색하여 모든 결과를 가져옵니다.

        Args:
            store: 검색할 저장소.
            namespace: 검색할 계층적 경로 접두사(prefix).
            query: 자연어 검색을 위한 선택적 쿼리.
            filter: 결과 필터링을 위한 키-값 쌍.
            page_size: 페이지당 가져올 아이템 수 (기본값: 100).

        Returns:
            검색 조건과 일치하는 모든 아이템 목록.

        Example:
            ```python
            store = _get_store(runtime)
            namespace = _get_namespace()
            all_items = _search_store_paginated(store, namespace)
            ```
        """
        all_items: list[Item] = []
        offset = 0
        while True:
            page_items = store.search(
                namespace,
                query=query,
                filter=filter,
                limit=page_size,
                offset=offset,
            )
            if not page_items:
                break
            all_items.extend(page_items)
            if len(page_items) < page_size:
                break
            offset += page_size

        return all_items

    def ls_info(self, path: str) -> list[FileInfo]:
        """지정된 디렉토리의 파일과 디렉토리를 나열합니다 (비재귀적).

        Args:
            path: 디렉토리의 절대 경로.

        Returns:
            디렉토리 바로 아래에 있는 파일 및 디렉토리에 대한 FileInfo 유사 dict 목록.
            디렉토리는 경로 끝에 /가 붙으며 is_dir=True입니다.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        # Retrieve all items and filter by path prefix locally to avoid
        # coupling to store-specific filter semantics
        items = self._search_store_paginated(store, namespace)
        infos: list[FileInfo] = []
        subdirs: set[str] = set()

        # Normalize path to have trailing slash for proper prefix matching
        normalized_path = path if path.endswith("/") else path + "/"

        for item in items:
            # Check if file is in the specified directory or a subdirectory
            if not str(item.key).startswith(normalized_path):
                continue

            # Get the relative path after the directory
            relative = str(item.key)[len(normalized_path) :]

            # If relative path contains '/', it's in a subdirectory
            if "/" in relative:
                # Extract the immediate subdirectory name
                subdir_name = relative.split("/")[0]
                subdirs.add(normalized_path + subdir_name + "/")
                continue

            # This is a file directly in the current directory
            try:
                fd = self._convert_store_item_to_file_data(item)
            except ValueError:
                continue
            size = len("\n".join(fd.get("content", [])))
            infos.append({
                "path": item.key,
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
        store = self._get_store()
        namespace = self._get_namespace()
        item: Item | None = store.get(namespace, file_path)

        if item is None:
            return f"Error: File '{file_path}' not found"

        try:
            file_data = self._convert_store_item_to_file_data(item)
        except ValueError as e:
            return f"Error: {e}"

        return format_read_response(file_data, offset, limit)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """내용을 포함하는 새 파일을 생성합니다.
        WriteResult를 반환합니다. 외부 저장소는 files_update=None을 설정합니다.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        # Check if file exists
        existing = store.get(namespace, file_path)
        if existing is not None:
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path."
            )

        # Create new file
        file_data = create_file_data(content)
        store_value = self._convert_file_data_to_store_value(file_data)
        store.put(namespace, file_path, store_value)
        return WriteResult(path=file_path, files_update=None)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """문자열 발생(occurrences)을 교체하여 파일을 편집합니다.
        EditResult를 반환합니다. 외부 저장소는 files_update=None을 설정합니다.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        # Get existing file
        item = store.get(namespace, file_path)
        if item is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        try:
            file_data = self._convert_store_item_to_file_data(item)
        except ValueError as e:
            return EditResult(error=f"Error: {e}")

        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)

        # Update file in store
        store_value = self._convert_file_data_to_store_value(new_file_data)
        store.put(namespace, file_path, store_value)
        return EditResult(path=file_path, files_update=None, occurrences=int(occurrences))

    # Removed legacy grep() convenience to keep lean surface

    def grep_raw(
        self,
        pattern: str,
        path: str = "/",
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        store = self._get_store()
        namespace = self._get_namespace()
        items = self._search_store_paginated(store, namespace)
        files: dict[str, Any] = {}
        for item in items:
            try:
                files[item.key] = self._convert_store_item_to_file_data(item)
            except ValueError:
                continue
        return grep_matches_from_files(files, pattern, path, glob)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        store = self._get_store()
        namespace = self._get_namespace()
        items = self._search_store_paginated(store, namespace)
        files: dict[str, Any] = {}
        for item in items:
            try:
                files[item.key] = self._convert_store_item_to_file_data(item)
            except ValueError:
                continue
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

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """저장소에 여러 파일을 업로드합니다.

        Args:
            files: 내용이 bytes인 (path, content) 튜플의 리스트.

        Returns:
            FileUploadResponse 객체들의 리스트. 입력 파일마다 하나씩 반환됩니다.
            응답 순서는 입력 순서와 일치합니다.
        """
        store = self._get_store()
        namespace = self._get_namespace()
        responses: list[FileUploadResponse] = []

        for path, content in files:
            content_str = content.decode("utf-8")
            # Create file data
            file_data = create_file_data(content_str)
            store_value = self._convert_file_data_to_store_value(file_data)

            # Store the file
            store.put(namespace, path, store_value)
            responses.append(FileUploadResponse(path=path, error=None))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """저장소에서 여러 파일을 다운로드합니다.

        Args:
            paths: 다운로드할 파일 경로의 리스트.

        Returns:
            FileDownloadResponse 객체들의 리스트. 입력 경로마다 하나씩 반환됩니다.
            응답 순서는 입력 순서와 일치합니다.
        """
        store = self._get_store()
        namespace = self._get_namespace()
        responses: list[FileDownloadResponse] = []

        for path in paths:
            item = store.get(namespace, path)

            if item is None:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
                continue

            file_data = self._convert_store_item_to_file_data(item)
            # Convert file data to bytes
            content_str = file_data_to_string(file_data)
            content_bytes = content_str.encode("utf-8")

            responses.append(FileDownloadResponse(path=path, content=content_bytes, error=None))

        return responses
