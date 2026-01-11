"""StoreBackend: LangGraph `BaseStore` 어댑터(영구 저장, 스레드 간 공유)."""

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
    """LangGraph `BaseStore`에 파일을 저장하는 백엔드입니다(영구 저장).

    LangGraph Store를 이용해 대화 스레드를 넘어서는 영구 저장을 제공합니다.
    파일은 namespace로 조직되며 모든 스레드에서 지속됩니다.

    멀티 에이전트 격리를 위해 namespace에 `assistant_id`를 포함할 수도 있습니다.
    """

    def __init__(self, runtime: "ToolRuntime"):
        """`ToolRuntime`으로 StoreBackend를 초기화합니다."""
        self.runtime = runtime

    def _get_store(self) -> BaseStore:
        """runtime으로부터 store 인스턴스를 가져옵니다."""
        store = self.runtime.store
        if store is None:
            msg = "Store is required but not available in runtime"
            raise ValueError(msg)
        return store

    def _get_namespace(self) -> tuple[str, ...]:
        """Store 작업에 사용할 namespace를 구합니다.

        Preference order:
        1) Use `self.runtime.config` if present (tests pass this explicitly).
        2) Fallback to `langgraph.config.get_config()` if available.
        3) Default to ("filesystem",).

        If an assistant_id is available in the config metadata, return
        (assistant_id, "filesystem") to provide per-assistant isolation.
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
        """Store `Item`을 FileData 형식 딕셔너리로 변환합니다."""
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
        """FileData를 `store.put()`에 넣기 좋은 형태로 변환합니다."""
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
        """Store 검색을 pagination으로 반복하여 모든 결과를 수집합니다.

        Args:
            store: The store to search.
            namespace: Hierarchical path prefix to search within.
            query: Optional query for natural language search.
            filter: Key-value pairs to filter results.
            page_size: Number of items to fetch per page (default: 100).

        Returns:
            List of all items matching the search criteria.

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
        """지정한 디렉토리 바로 아래의 파일/폴더를 나열합니다(비재귀).

        Args:
            path: Absolute path to directory.

        Returns:
            List of FileInfo-like dicts for files and directories directly in the directory.
            Directories have a trailing / in their path and is_dir=True.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        # store별 filter semantics에 결합되지 않도록 전체 아이템을 가져온 뒤,
        # path prefix 필터링은 로컬에서 수행합니다.
        items = self._search_store_paginated(store, namespace)
        infos: list[FileInfo] = []
        subdirs: set[str] = set()

        # prefix 매칭을 위해 trailing slash를 갖는 형태로 정규화
        normalized_path = path if path.endswith("/") else path + "/"

        for item in items:
            # 지정 디렉토리(또는 하위 디렉토리)에 속하는지 확인
            if not str(item.key).startswith(normalized_path):
                continue

            # 디렉토리 이후의 상대 경로
            relative = str(item.key)[len(normalized_path) :]

            # 상대 경로에 `/`가 있으면 하위 디렉토리입니다.
            if "/" in relative:
                # 즉시 하위 디렉토리 이름만 추출
                subdir_name = relative.split("/")[0]
                subdirs.add(normalized_path + subdir_name + "/")
                continue

            # 현재 디렉토리 바로 아래의 파일
            try:
                fd = self._convert_store_item_to_file_data(item)
            except ValueError:
                continue
            size = len("\n".join(fd.get("content", [])))
            infos.append(
                {
                    "path": item.key,
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
            file_path: Absolute file path.
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            Formatted file content with line numbers, or error message.
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
        """새 파일을 생성하고 내용을 씁니다.

        `WriteResult`를 반환합니다. 외부 스토리지 백엔드는 `files_update=None`을 사용합니다.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        # 파일 존재 여부 확인
        existing = store.get(namespace, file_path)
        if existing is not None:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        # 새 파일 생성
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
        """파일 내 문자열을 치환하여 편집합니다.

        `EditResult`를 반환합니다. 외부 스토리지 백엔드는 `files_update=None`을 사용합니다.
        """
        store = self._get_store()
        namespace = self._get_namespace()

        # 기존 파일 조회
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

        # store에 업데이트 반영
        store_value = self._convert_file_data_to_store_value(new_file_data)
        store.put(namespace, file_path, store_value)
        return EditResult(path=file_path, files_update=None, occurrences=int(occurrences))

    # API 표면을 간결하게 유지하기 위해 legacy grep() convenience는 제거됨

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
        """여러 파일을 store에 업로드합니다.

        Args:
            files: List of (path, content) tuples where content is bytes.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order.
        """
        store = self._get_store()
        namespace = self._get_namespace()
        responses: list[FileUploadResponse] = []

        for path, content in files:
            content_str = content.decode("utf-8")
            # 파일 데이터 생성
            file_data = create_file_data(content_str)
            store_value = self._convert_file_data_to_store_value(file_data)

            # 파일 저장
            store.put(namespace, path, store_value)
            responses.append(FileUploadResponse(path=path, error=None))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """여러 파일을 store에서 다운로드합니다.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
            Response order matches input order.
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
            # FileData를 bytes로 변환
            content_str = file_data_to_string(file_data)
            content_bytes = content_str.encode("utf-8")

            responses.append(FileDownloadResponse(path=path, content=content_bytes, error=None))

        return responses
