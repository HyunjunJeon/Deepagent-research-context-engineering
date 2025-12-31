"""CompositeBackend: 경로 접두사(prefix)를 기반으로 작업을 다른 백엔드로 라우팅합니다."""

from collections import defaultdict

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.state import StateBackend


class CompositeBackend:
    def __init__(
        self,
        default: BackendProtocol | StateBackend,
        routes: dict[str, BackendProtocol],
    ) -> None:
        # Default backend
        self.default = default

        # Virtual routes
        self.routes = routes

        # Sort routes by length (longest first) for correct prefix matching
        self.sorted_routes = sorted(routes.items(), key=lambda x: len(x[0]), reverse=True)

    def _get_backend_and_key(self, key: str) -> tuple[BackendProtocol, str]:
        """어떤 백엔드가 이 키를 처리하는지 결정하고 접두사를 제거합니다.

        Args:
            key: 원본 파일 경로

        Returns:
            (backend, stripped_key) 튜플. stripped_key는 라우트 접두사가
            제거된 상태입니다 (하지만 선행 슬래시는 유지됨).
        """
        # Check routes in order of length (longest first)
        for prefix, backend in self.sorted_routes:
            if key.startswith(prefix):
                # Strip full prefix and ensure a leading slash remains
                # e.g., "/memories/notes.txt" → "/notes.txt"; "/memories/" → "/"
                suffix = key[len(prefix) :]
                stripped_key = f"/{suffix}" if suffix else "/"
                return backend, stripped_key

        return self.default, key

    def ls_info(self, path: str) -> list[FileInfo]:
        """지정된 디렉토리의 파일과 디렉토리를 나열합니다 (비재귀적).

        Args:
            path: 디렉토리의 절대 경로.

        Returns:
            디렉토리 바로 아래에 있는 파일 및 디렉토리에 대한 FileInfo 유사 dict 목록 (라우트 접두사 추가됨).
            디렉토리는 경로 끝에 /가 붙으며 is_dir=True입니다.
        """
        # Check if path matches a specific route
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                # Query only the matching routed backend
                suffix = path[len(route_prefix) :]
                search_path = f"/{suffix}" if suffix else "/"
                infos = backend.ls_info(search_path)
                prefixed: list[FileInfo] = []
                for fi in infos:
                    fi = dict(fi)
                    fi["path"] = f"{route_prefix[:-1]}{fi['path']}"
                    prefixed.append(fi)
                return prefixed

        # At root, aggregate default and all routed backends
        if path == "/":
            results: list[FileInfo] = []
            results.extend(self.default.ls_info(path))
            for route_prefix, backend in self.sorted_routes:
                # Add the route itself as a directory (e.g., /memories/)
                results.append({
                    "path": route_prefix,
                    "is_dir": True,
                    "size": 0,
                    "modified_at": "",
                })

            results.sort(key=lambda x: x.get("path", ""))
            return results

        # Path doesn't match a route: query only default backend
        return self.default.ls_info(path)

    async def als_info(self, path: str) -> list[FileInfo]:
        """ls_info의 비동기 버전입니다."""
        # Check if path matches a specific route
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                # Query only the matching routed backend
                suffix = path[len(route_prefix) :]
                search_path = f"/{suffix}" if suffix else "/"
                infos = await backend.als_info(search_path)
                prefixed: list[FileInfo] = []
                for fi in infos:
                    fi = dict(fi)
                    fi["path"] = f"{route_prefix[:-1]}{fi['path']}"
                    prefixed.append(fi)
                return prefixed

        # At root, aggregate default and all routed backends
        if path == "/":
            results: list[FileInfo] = []
            results.extend(await self.default.als_info(path))
            for route_prefix, backend in self.sorted_routes:
                # Add the route itself as a directory (e.g., /memories/)
                results.append({
                    "path": route_prefix,
                    "is_dir": True,
                    "size": 0,
                    "modified_at": "",
                })

            results.sort(key=lambda x: x.get("path", ""))
            return results

        # Path doesn't match a route: query only default backend
        return await self.default.als_info(path)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """적절한 백엔드로 라우팅하여 파일 내용을 읽습니다.

        Args:
            file_path: 파일 절대 경로.
            offset: 읽기 시작할 라인 오프셋 (0부터 시작).
            limit: 읽을 최대 라인 수.

        Returns:
            라인 번호가 포함된 형식화된 파일 내용, 또는 에러 메시지.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.read(stripped_key, offset=offset, limit=limit)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """read의 비동기 버전입니다."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        return await backend.aread(stripped_key, offset=offset, limit=limit)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        # If path targets a specific route, search only that backend
        for route_prefix, backend in self.sorted_routes:
            if path is not None and path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1 :]
                raw = backend.grep_raw(pattern, search_path if search_path else "/", glob)
                if isinstance(raw, str):
                    return raw
                return [{**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw]

        # Otherwise, search default and all routed backends and merge
        all_matches: list[GrepMatch] = []
        raw_default = self.default.grep_raw(pattern, path, glob)  # type: ignore[attr-defined]
        if isinstance(raw_default, str):
            # This happens if error occurs
            return raw_default
        all_matches.extend(raw_default)

        for route_prefix, backend in self.routes.items():
            raw = backend.grep_raw(pattern, "/", glob)
            if isinstance(raw, str):
                # This happens if error occurs
                return raw
            all_matches.extend({**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw)

        return all_matches

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """grep_raw의 비동기 버전입니다."""
        # If path targets a specific route, search only that backend
        for route_prefix, backend in self.sorted_routes:
            if path is not None and path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1 :]
                raw = await backend.agrep_raw(pattern, search_path if search_path else "/", glob)
                if isinstance(raw, str):
                    return raw
                return [{**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw]

        # Otherwise, search default and all routed backends and merge
        all_matches: list[GrepMatch] = []
        raw_default = await self.default.agrep_raw(pattern, path, glob)  # type: ignore[attr-defined]
        if isinstance(raw_default, str):
            # This happens if error occurs
            return raw_default
        all_matches.extend(raw_default)

        for route_prefix, backend in self.routes.items():
            raw = await backend.agrep_raw(pattern, "/", glob)
            if isinstance(raw, str):
                # This happens if error occurs
                return raw
            all_matches.extend({**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw)

        return all_matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        results: list[FileInfo] = []

        # Route based on path, not pattern
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1 :]
                infos = backend.glob_info(pattern, search_path if search_path else "/")
                return [{**fi, "path": f"{route_prefix[:-1]}{fi['path']}"} for fi in infos]

        # Path doesn't match any specific route - search default backend AND all routed backends
        results.extend(self.default.glob_info(pattern, path))

        for route_prefix, backend in self.routes.items():
            infos = backend.glob_info(pattern, "/")
            results.extend({**fi, "path": f"{route_prefix[:-1]}{fi['path']}"} for fi in infos)

        # Deterministic ordering
        results.sort(key=lambda x: x.get("path", ""))
        return results

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """glob_info의 비동기 버전입니다."""
        results: list[FileInfo] = []

        # Route based on path, not pattern
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1 :]
                infos = await backend.aglob_info(pattern, search_path if search_path else "/")
                return [{**fi, "path": f"{route_prefix[:-1]}{fi['path']}"} for fi in infos]

        # Path doesn't match any specific route - search default backend AND all routed backends
        results.extend(await self.default.aglob_info(pattern, path))

        for route_prefix, backend in self.routes.items():
            infos = await backend.aglob_info(pattern, "/")
            results.extend({**fi, "path": f"{route_prefix[:-1]}{fi['path']}"} for fi in infos)

        # Deterministic ordering
        results.sort(key=lambda x: x.get("path", ""))
        return results

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """적절한 백엔드로 라우팅하여 새 파일을 생성합니다.

        Args:
            file_path: 파일 절대 경로.
            content: 문자열 형태의 파일 내용.

        Returns:
            성공 메시지 또는 Command 객체, 또는 파일이 이미 존재하는 경우 에러.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = backend.write(stripped_key, content)
        # If this is a state-backed update and default has state, merge so listings reflect changes
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:
                pass
        return res

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """write의 비동기 버전입니다."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = await backend.awrite(stripped_key, content)
        # If this is a state-backed update and default has state, merge so listings reflect changes
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:
                pass
        return res

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """적절한 백엔드로 라우팅하여 파일을 편집합니다.

        Args:
            file_path: 파일 절대 경로.
            old_string: 찾아서 교체할 문자열.
            new_string: 교체할 문자열.
            replace_all: True인 경우 모든 발생을 교체.

        Returns:
            성공 메시지 또는 Command 객체, 또는 실패 시 에러 메시지.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = backend.edit(stripped_key, old_string, new_string, replace_all=replace_all)
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:
                pass
        return res

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """edit의 비동기 버전입니다."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = await backend.aedit(stripped_key, old_string, new_string, replace_all=replace_all)
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:
                pass
        return res

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """기본(default) 백엔드를 통해 명령을 실행합니다.

        실행은 경로에 특정되지 않으므로, 항상 기본 백엔드로 위임됩니다.
        이 기능이 작동하려면 기본 백엔드가 SandboxBackendProtocol을 구현해야 합니다.

        Args:
            command: 실행할 전체 쉘 명령 문자열.

        Returns:
            결합된 출력, 종료 코드, 잘림(truncation) 플래그를 포함하는 ExecuteResponse.

        Raises:
            NotImplementedError: 기본 백엔드가 실행을 지원하지 않는 경우.
        """
        if isinstance(self.default, SandboxBackendProtocol):
            return self.default.execute(command)

        # This shouldn't be reached if the runtime check in the execute tool works correctly,
        # but we include it as a safety fallback.
        raise NotImplementedError(
            "Default backend doesn't support command execution (SandboxBackendProtocol). "
            "To enable execution, provide a default backend that implements SandboxBackendProtocol."
        )

    async def aexecute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """execute의 비동기 버전입니다."""
        if isinstance(self.default, SandboxBackendProtocol):
            return await self.default.aexecute(command)

        # This shouldn't be reached if the runtime check in the execute tool works correctly,
        # but we include it as a safety fallback.
        raise NotImplementedError(
            "Default backend doesn't support command execution (SandboxBackendProtocol). "
            "To enable execution, provide a default backend that implements SandboxBackendProtocol."
        )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """효율성을 위해 백엔드별로 배치 처리하여 여러 파일을 업로드합니다.

        파일을 대상 백엔드별로 그룹화하고, 각 백엔드의 upload_files를
        해당 백엔드의 모든 파일과 함께 한 번 호출한 다음, 결과를 원래 순서대로 병합합니다.

        Args:
            files: 업로드할 (path, content) 튜플의 리스트.

        Returns:
            FileUploadResponse 객체들의 리스트. 입력 파일마다 하나씩 반환됩니다.
            응답 순서는 입력 순서와 일치합니다.
        """
        # Pre-allocate result list
        results: list[FileUploadResponse | None] = [None] * len(files)

        # Group files by backend, tracking original indices
        from collections import defaultdict

        backend_batches: dict[BackendProtocol, list[tuple[int, str, bytes]]] = defaultdict(list)

        for idx, (path, content) in enumerate(files):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path, content))

        # Process each backend's batch
        for backend, batch in backend_batches.items():
            # Extract data for backend call
            indices, stripped_paths, contents = zip(*batch, strict=False)
            batch_files = list(zip(stripped_paths, contents, strict=False))

            # Call backend once with all its files
            batch_responses = backend.upload_files(batch_files)

            # Place responses at original indices with original paths
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileUploadResponse(
                    path=files[orig_idx][0],  # Original path
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return results  # type: ignore[return-value]

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """upload_files의 비동기 버전입니다."""
        # Pre-allocate result list
        results: list[FileUploadResponse | None] = [None] * len(files)

        # Group files by backend, tracking original indices
        backend_batches: dict[BackendProtocol, list[tuple[int, str, bytes]]] = defaultdict(list)

        for idx, (path, content) in enumerate(files):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path, content))

        # Process each backend's batch
        for backend, batch in backend_batches.items():
            # Extract data for backend call
            indices, stripped_paths, contents = zip(*batch, strict=False)
            batch_files = list(zip(stripped_paths, contents, strict=False))

            # Call backend once with all its files
            batch_responses = await backend.aupload_files(batch_files)

            # Place responses at original indices with original paths
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileUploadResponse(
                    path=files[orig_idx][0],  # Original path
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return results  # type: ignore[return-value]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """효율성을 위해 백엔드별로 배치 처리하여 여러 파일을 다운로드합니다.

        경로를 대상 백엔드별로 그룹화하고, 각 백엔드의 download_files를
        해당 백엔드의 모든 경로와 함께 한 번 호출한 다음, 결과를 원래 순서대로 병합합니다.

        Args:
            paths: 다운로드할 파일 경로의 리스트.

        Returns:
            FileDownloadResponse 객체들의 리스트. 입력 경로마다 하나씩 반환됩니다.
            응답 순서는 입력 순서와 일치합니다.
        """
        # Pre-allocate result list
        results: list[FileDownloadResponse | None] = [None] * len(paths)

        backend_batches: dict[BackendProtocol, list[tuple[int, str]]] = defaultdict(list)

        for idx, path in enumerate(paths):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path))

        # Process each backend's batch
        for backend, batch in backend_batches.items():
            # Extract data for backend call
            indices, stripped_paths = zip(*batch, strict=False)

            # Call backend once with all its paths
            batch_responses = backend.download_files(list(stripped_paths))

            # Place responses at original indices with original paths
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileDownloadResponse(
                    path=paths[orig_idx],  # Original path
                    content=batch_responses[i].content if i < len(batch_responses) else None,
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return results  # type: ignore[return-value]

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """download_files의 비동기 버전입니다."""
        # Pre-allocate result list
        results: list[FileDownloadResponse | None] = [None] * len(paths)

        backend_batches: dict[BackendProtocol, list[tuple[int, str]]] = defaultdict(list)

        for idx, path in enumerate(paths):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path))

        # Process each backend's batch
        for backend, batch in backend_batches.items():
            # Extract data for backend call
            indices, stripped_paths = zip(*batch, strict=False)

            # Call backend once with all its paths
            batch_responses = await backend.adownload_files(list(stripped_paths))

            # Place responses at original indices with original paths
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileDownloadResponse(
                    path=paths[orig_idx],  # Original path
                    content=batch_responses[i].content if i < len(batch_responses) else None,
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return results  # type: ignore[return-value]
