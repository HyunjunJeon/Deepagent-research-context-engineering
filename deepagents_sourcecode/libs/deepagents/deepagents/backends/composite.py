"""경로 prefix에 따라 파일 작업을 라우팅하는 합성(Composite) 백엔드입니다.

경로(prefix) 규칙에 따라 서로 다른 백엔드로 파일 작업을 위임합니다. 예를 들어,
임시 파일은 `StateBackend`에, 장기 메모리는 `StoreBackend`에 저장하는 식으로
경로별 저장 전략을 분리하고 싶을 때 사용합니다.

예시:
    ```python
    from deepagents.backends.composite import CompositeBackend
    from deepagents.backends.state import StateBackend
    from deepagents.backends.store import StoreBackend

    runtime = make_runtime()
    composite = CompositeBackend(
        default=StateBackend(runtime),
        routes={"/memories/": StoreBackend(runtime)},
    )

    composite.write("/temp.txt", "ephemeral")
    composite.write("/memories/note.md", "persistent")
    ```
"""

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


class CompositeBackend(BackendProtocol):
    """경로 prefix에 따라 파일 작업을 서로 다른 백엔드로 위임합니다.

    경로를 route prefix(길이가 긴 것부터)와 매칭한 후 해당 백엔드로 위임합니다.
    어떤 prefix에도 매칭되지 않으면 `default` 백엔드를 사용합니다.

    Attributes:
        default: 어떤 route에도 매칭되지 않을 때 사용할 백엔드.
        routes: 경로 prefix → 백엔드 매핑(예: `{"/memories/": store_backend}`).
        sorted_routes: 올바른 매칭을 위해 길이 기준(내림차순)으로 정렬한 routes.

    예시:
        ```python
        composite = CompositeBackend(
            default=StateBackend(runtime),
            routes={"/memories/": StoreBackend(runtime), "/cache/": StoreBackend(runtime)},
        )

        composite.write("/temp.txt", "data")
        composite.write("/memories/note.txt", "data")
        ```
    """

    def __init__(
        self,
        default: BackendProtocol | StateBackend,
        routes: dict[str, BackendProtocol],
    ) -> None:
        """합성 백엔드를 초기화합니다.

        Args:
            default: 어떤 route에도 매칭되지 않을 때 사용할 백엔드.
            routes: 경로 prefix → 백엔드 매핑. prefix는 반드시 `/`로 시작해야 하며,
                일반적으로 `/`로 끝나도록 지정하는 것을 권장합니다(예: `/memories/`).
        """
        # 기본(default) 백엔드
        self.default = default

        # 가상(virtual) route 설정
        self.routes = routes

        # prefix 매칭이 올바르게 동작하도록 길이 기준(내림차순)으로 정렬
        self.sorted_routes = sorted(routes.items(), key=lambda x: len(x[0]), reverse=True)

    def _get_backend_and_key(self, key: str) -> tuple[BackendProtocol, str]:
        """경로에 맞는 백엔드를 찾고 route prefix를 제거한 경로를 반환합니다.

        Args:
            key: 라우팅할 파일 경로.

        Returns:
            `(backend, stripped_path)` 튜플. `stripped_path`는 route prefix를 제거하되
            선행 `/`는 유지합니다.
        """
        # route prefix 길이가 긴 것부터 확인(가장 구체적인 route 우선)
        for prefix, backend in self.sorted_routes:
            if key.startswith(prefix):
                # prefix를 제거하되, 선행 슬래시를 유지
                # 예: "/memories/notes.txt" → "/notes.txt", "/memories/" → "/"
                suffix = key[len(prefix) :]
                stripped_key = f"/{suffix}" if suffix else "/"
                return backend, stripped_key

        return self.default, key

    def ls_info(self, path: str) -> list[FileInfo]:
        """디렉토리 내용을 나열합니다(비재귀).

        - `path`가 특정 route에 매칭되면 해당 백엔드만 나열합니다.
        - `path == "/"`이면 default 백엔드 + 가상 route 디렉토리를 함께 반환합니다.
        - 그 외에는 default 백엔드만 나열합니다.

        Args:
            path: `/`로 시작하는 절대 디렉토리 경로.

        Returns:
            FileInfo 딕셔너리 리스트. 디렉토리는 `path`가 `/`로 끝나며 `is_dir=True`입니다.
            route를 통해 조회된 경우 반환 경로에는 원래의 route prefix가 복원됩니다.

        예시:
            ```python
            infos = composite.ls_info("/")
            infos = composite.ls_info("/memories/")
            ```
        """
        # path가 특정 route에 매칭되는지 확인
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                # 매칭된 routed backend만 조회
                suffix = path[len(route_prefix) :]
                search_path = f"/{suffix}" if suffix else "/"
                infos = backend.ls_info(search_path)
                prefixed: list[FileInfo] = []
                for fi in infos:
                    fi = dict(fi)
                    fi["path"] = f"{route_prefix[:-1]}{fi['path']}"
                    prefixed.append(fi)
                return prefixed

        # 루트에서는 default + 모든 route 디렉토리를 합산
        if path == "/":
            results: list[FileInfo] = []
            results.extend(self.default.ls_info(path))
            for route_prefix, backend in self.sorted_routes:
                # route 자체를 디렉토리로 추가(예: /memories/)
                results.append(
                    {
                        "path": route_prefix,
                        "is_dir": True,
                        "size": 0,
                        "modified_at": "",
                    }
                )

            results.sort(key=lambda x: x.get("path", ""))
            return results

        # 어떤 route에도 매칭되지 않으면 default backend만 조회
        return self.default.ls_info(path)

    async def als_info(self, path: str) -> list[FileInfo]:
        """`ls_info`의 async 버전입니다."""
        # path가 특정 route에 매칭되는지 확인
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                # 매칭된 routed backend만 조회
                suffix = path[len(route_prefix) :]
                search_path = f"/{suffix}" if suffix else "/"
                infos = await backend.als_info(search_path)
                prefixed: list[FileInfo] = []
                for fi in infos:
                    fi = dict(fi)
                    fi["path"] = f"{route_prefix[:-1]}{fi['path']}"
                    prefixed.append(fi)
                return prefixed

        # 루트에서는 default + 모든 route 디렉토리를 합산
        if path == "/":
            results: list[FileInfo] = []
            results.extend(await self.default.als_info(path))
            for route_prefix, backend in self.sorted_routes:
                # route 자체를 디렉토리로 추가(예: /memories/)
                results.append(
                    {
                        "path": route_prefix,
                        "is_dir": True,
                        "size": 0,
                        "modified_at": "",
                    }
                )

            results.sort(key=lambda x: x.get("path", ""))
            return results

        # 어떤 route에도 매칭되지 않으면 default backend만 조회
        return await self.default.als_info(path)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """파일 내용을 읽습니다(경로에 맞는 백엔드로 라우팅).

        Args:
            file_path: 절대 파일 경로.
            offset: 읽기 시작 라인 오프셋(0-index).
            limit: 최대 읽기 라인 수.

        Returns:
            라인 번호가 포함된 포맷 문자열 또는 오류 메시지.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.read(stripped_key, offset=offset, limit=limit)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """`read`의 async 버전입니다."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        return await backend.aread(stripped_key, offset=offset, limit=limit)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """파일에서 정규식 패턴을 검색합니다.

        `path`에 따라 검색 대상 백엔드를 라우팅합니다.
        - 특정 route에 매칭되면 해당 백엔드만 검색
        - `"/"` 또는 `None`이면 default + 모든 route 백엔드를 검색하여 병합
        - 그 외는 default 백엔드만 검색

        Args:
            pattern: Regex pattern to search for.
            path: Directory to search. None searches all backends.
            glob: Glob pattern to filter files (e.g., "*.py", "**/*.txt").
                Filters by filename, not content.

        Returns:
            List of GrepMatch dicts with path (route prefix restored), line
            (1-indexed), and text. Returns error string on failure.

        Examples:
            ```python
            matches = composite.grep_raw("TODO", path="/memories/")
            matches = composite.grep_raw("error", path="/")
            matches = composite.grep_raw("import", path="/", glob="*.py")
            ```
        """
        # path가 특정 route를 가리키면 해당 백엔드만 검색
        for route_prefix, backend in self.sorted_routes:
            if path is not None and path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1 :]
                raw = backend.grep_raw(pattern, search_path if search_path else "/", glob)
                if isinstance(raw, str):
                    return raw
                return [{**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw]

        # path가 None 또는 "/"이면 default + 모든 route 백엔드를 검색하여 병합
        # 그 외에는 default 백엔드만 검색
        if path is None or path == "/":
            all_matches: list[GrepMatch] = []
            raw_default = self.default.grep_raw(pattern, path, glob)  # type: ignore[attr-defined]
            if isinstance(raw_default, str):
                # 에러가 발생하면 문자열 오류 메시지가 반환됩니다.
                return raw_default
            all_matches.extend(raw_default)

            for route_prefix, backend in self.routes.items():
                raw = backend.grep_raw(pattern, "/", glob)
                if isinstance(raw, str):
                    # 에러가 발생하면 문자열 오류 메시지가 반환됩니다.
                    return raw
                all_matches.extend({**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw)

            return all_matches
        # Path specified but doesn't match a route - search only default
        return self.default.grep_raw(pattern, path, glob)  # type: ignore[attr-defined]

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """`grep_raw`의 async 버전입니다.

        라우팅 동작과 파라미터에 대한 자세한 설명은 `grep_raw()`를 참고하세요.
        """
        # path가 특정 route를 가리키면 해당 백엔드만 검색
        for route_prefix, backend in self.sorted_routes:
            if path is not None and path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1 :]
                raw = await backend.agrep_raw(pattern, search_path if search_path else "/", glob)
                if isinstance(raw, str):
                    return raw
                return [{**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw]

        # path가 None 또는 "/"이면 default + 모든 route 백엔드를 검색하여 병합
        # 그 외에는 default 백엔드만 검색
        if path is None or path == "/":
            all_matches: list[GrepMatch] = []
            raw_default = await self.default.agrep_raw(pattern, path, glob)  # type: ignore[attr-defined]
            if isinstance(raw_default, str):
                # 에러가 발생하면 문자열 오류 메시지가 반환됩니다.
                return raw_default
            all_matches.extend(raw_default)

            for route_prefix, backend in self.routes.items():
                raw = await backend.agrep_raw(pattern, "/", glob)
                if isinstance(raw, str):
                    # 에러가 발생하면 문자열 오류 메시지가 반환됩니다.
                    return raw
                all_matches.extend({**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw)

            return all_matches
        # Path specified but doesn't match a route - search only default
        return await self.default.agrep_raw(pattern, path, glob)  # type: ignore[attr-defined]

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
        """`glob_info`의 async 버전입니다."""
        results: list[FileInfo] = []

        # pattern이 아니라 path 기준으로 라우팅
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1 :]
                infos = await backend.aglob_info(pattern, search_path if search_path else "/")
                return [{**fi, "path": f"{route_prefix[:-1]}{fi['path']}"} for fi in infos]

        # 어떤 route에도 매칭되지 않으면 default + 모든 route 백엔드를 검색
        results.extend(await self.default.aglob_info(pattern, path))

        for route_prefix, backend in self.routes.items():
            infos = await backend.aglob_info(pattern, "/")
            results.extend({**fi, "path": f"{route_prefix[:-1]}{fi['path']}"} for fi in infos)

        # deterministic ordering
        results.sort(key=lambda x: x.get("path", ""))
        return results

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """새 파일을 생성합니다(경로에 맞는 백엔드로 라우팅).

        Args:
            file_path: Absolute file path.
            content: File content as a string.

        Returns:
            Success message or Command object, or error if file already exists.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = backend.write(stripped_key, content)
        # state-backed 업데이트(그리고 default가 state를 갖는 경우)면,
        # listing이 변경을 반영하도록 default state에도 병합합니다.
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
        """`write`의 async 버전입니다."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = await backend.awrite(stripped_key, content)
        # state-backed 업데이트(그리고 default가 state를 갖는 경우)면,
        # listing이 변경을 반영하도록 default state에도 병합합니다.
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
        """파일을 편집합니다(경로에 맞는 백엔드로 라우팅).

        Args:
            file_path: Absolute file path.
            old_string: String to find and replace.
            new_string: Replacement string.
            replace_all: If True, replace all occurrences.

        Returns:
            Success message or Command object, or error message on failure.
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
        """`edit`의 async 버전입니다."""
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
        """Default 백엔드를 통해 셸 커맨드를 실행합니다.

        Args:
            command: Shell command to execute.

        Returns:
            ExecuteResponse with output, exit code, and truncation flag.

        Raises:
            NotImplementedError: If default backend doesn't implement SandboxBackendProtocol.

        Examples:
            ```python
            composite = CompositeBackend(default=FilesystemBackend(root_dir="/tmp"), routes={"/memories/": StoreBackend(runtime)})

            result = composite.execute("ls -la")
            ```
        """
        if isinstance(self.default, SandboxBackendProtocol):
            return self.default.execute(command)

        # execute 도구의 런타임 체크가 제대로 동작한다면 여기에 도달하지 않아야 하지만,
        # 안전장치(fallback)로 예외를 둡니다.
        raise NotImplementedError(
            "Default backend doesn't support command execution (SandboxBackendProtocol). "
            "To enable execution, provide a default backend that implements SandboxBackendProtocol."
        )

    async def aexecute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """`execute`의 async 버전입니다."""
        if isinstance(self.default, SandboxBackendProtocol):
            return await self.default.aexecute(command)

        # execute 도구의 런타임 체크가 제대로 동작한다면 여기에 도달하지 않아야 하지만,
        # 안전장치(fallback)로 예외를 둡니다.
        raise NotImplementedError(
            "Default backend doesn't support command execution (SandboxBackendProtocol). "
            "To enable execution, provide a default backend that implements SandboxBackendProtocol."
        )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """여러 파일을 업로드합니다(백엔드별로 배치 처리하여 효율화).

        Groups files by their target backend, calls each backend's upload_files
        once with all files for that backend, then merges results in original order.

        Args:
            files: List of (path, content) tuples to upload.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order.
        """
        # 결과 리스트를 미리 할당
        results: list[FileUploadResponse | None] = [None] * len(files)

        # 원래 인덱스를 유지하면서 백엔드별로 파일을 그룹화
        from collections import defaultdict

        backend_batches: dict[BackendProtocol, list[tuple[int, str, bytes]]] = defaultdict(list)

        for idx, (path, content) in enumerate(files):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path, content))

        # 백엔드별 배치를 처리
        for backend, batch in backend_batches.items():
            # 백엔드 호출에 필요한 데이터로 분해
            indices, stripped_paths, contents = zip(*batch, strict=False)
            batch_files = list(zip(stripped_paths, contents, strict=False))

            # 해당 백엔드로 1회 호출(배치)
            batch_responses = backend.upload_files(batch_files)

            # 원래 경로/인덱스 위치에 응답을 채웁니다.
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileUploadResponse(
                    path=files[orig_idx][0],  # Original path
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return results  # type: ignore[return-value]

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """`upload_files`의 async 버전입니다."""
        # 결과 리스트를 미리 할당
        results: list[FileUploadResponse | None] = [None] * len(files)

        # 원래 인덱스를 유지하면서 백엔드별로 파일을 그룹화
        backend_batches: dict[BackendProtocol, list[tuple[int, str, bytes]]] = defaultdict(list)

        for idx, (path, content) in enumerate(files):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path, content))

        # 백엔드별 배치를 처리
        for backend, batch in backend_batches.items():
            # 백엔드 호출에 필요한 데이터로 분해
            indices, stripped_paths, contents = zip(*batch, strict=False)
            batch_files = list(zip(stripped_paths, contents, strict=False))

            # 해당 백엔드로 1회 호출(배치)
            batch_responses = await backend.aupload_files(batch_files)

            # 원래 경로/인덱스 위치에 응답을 채웁니다.
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileUploadResponse(
                    path=files[orig_idx][0],  # Original path
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return results  # type: ignore[return-value]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """여러 파일을 다운로드합니다(백엔드별로 배치 처리하여 효율화).

        Groups paths by their target backend, calls each backend's download_files
        once with all paths for that backend, then merges results in original order.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
            Response order matches input order.
        """
        # 결과 리스트를 미리 할당
        results: list[FileDownloadResponse | None] = [None] * len(paths)

        backend_batches: dict[BackendProtocol, list[tuple[int, str]]] = defaultdict(list)

        for idx, path in enumerate(paths):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path))

        # 백엔드별 배치를 처리
        for backend, batch in backend_batches.items():
            # 백엔드 호출에 필요한 데이터로 분해
            indices, stripped_paths = zip(*batch, strict=False)

            # 해당 백엔드로 1회 호출(배치)
            batch_responses = backend.download_files(list(stripped_paths))

            # 원래 경로/인덱스 위치에 응답을 채웁니다.
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileDownloadResponse(
                    path=paths[orig_idx],  # Original path
                    content=batch_responses[i].content if i < len(batch_responses) else None,
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return results  # type: ignore[return-value]

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """`download_files`의 async 버전입니다."""
        # 결과 리스트를 미리 할당
        results: list[FileDownloadResponse | None] = [None] * len(paths)

        backend_batches: dict[BackendProtocol, list[tuple[int, str]]] = defaultdict(list)

        for idx, path in enumerate(paths):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path))

        # 백엔드별 배치를 처리
        for backend, batch in backend_batches.items():
            # 백엔드 호출에 필요한 데이터로 분해
            indices, stripped_paths = zip(*batch, strict=False)

            # 해당 백엔드로 1회 호출(배치)
            batch_responses = await backend.adownload_files(list(stripped_paths))

            # 원래 경로/인덱스 위치에 응답을 채웁니다.
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileDownloadResponse(
                    path=paths[orig_idx],  # Original path
                    content=batch_responses[i].content if i < len(batch_responses) else None,
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return results  # type: ignore[return-value]
