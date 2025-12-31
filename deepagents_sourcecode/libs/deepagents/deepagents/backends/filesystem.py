"""FilesystemBackend: 파일시스템에서 직접 파일을 읽고 씁니다.

보안 및 검색 업그레이드:
- virtual_mode일 때 루트 포함(root containment)을 통한 보안 경로 확인 (cwd로 샌드박싱됨)
- 가능한 경우 O_NOFOLLOW를 사용하여 파일 I/O 시 심볼릭 링크 따라가기 방지
- JSON 파싱을 포함한 Ripgrep 기반 검색과, 가상 경로 동작을 보존하면서
  정규식 및 선택적 glob 포함 필터링을 지원하는 Python 폴백(fallback) 기능
"""

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import wcmatch.glob as wcglob

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
    check_empty_content,
    format_content_with_line_numbers,
    perform_string_replacement,
)


class FilesystemBackend(BackendProtocol):
    """파일시스템에서 직접 파일을 읽고 쓰는 백엔드.

    파일은 실제 파일시스템 경로를 사용하여 접근합니다. 상대 경로는
    현재 작업 디렉토리에 상대적으로 해결(resolve)됩니다. 내용은 일반 텍스트로
    읽고 쓰이며, 메타데이터(타임스탬프)는 파일시스템 상태(stat)에서 파생됩니다.
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        virtual_mode: bool = False,
        max_file_size_mb: int = 10,
    ) -> None:
        """파일시스템 백엔드를 초기화합니다.

        Args:
            root_dir: 파일 작업을 위한 선택적 루트 디렉토리. 제공된 경우,
                     모든 파일 경로는 이 디렉토리에 상대적으로 해결됩니다.
                     제공되지 않은 경우, 현재 작업 디렉토리를 사용합니다.
        """
        self.cwd = Path(root_dir).resolve() if root_dir else Path.cwd()
        self.virtual_mode = virtual_mode
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def _resolve_path(self, key: str) -> Path:
        """보안 검사를 포함하여 파일 경로를 해결(resolve)합니다.

        virtual_mode=True일 때, 들어오는 경로를 self.cwd 하위의 가상 절대 경로로 취급하며,
        상위 경로 탐색(.., ~)을 허용하지 않고 해결된 경로가 루트 내에 머물도록 보장합니다.
        virtual_mode=False일 때, 레거시 동작을 유지합니다: 절대 경로는 그대료 허용되고,
        상대 경로는 cwd 하위로 해결됩니다.

        Args:
            key: 파일 경로 (절대, 상대, 또는 virtual_mode=True일 때 가상 경로)

        Returns:
            해결된 절대 Path 객체
        """
        if self.virtual_mode:
            vpath = key if key.startswith("/") else "/" + key
            if ".." in vpath or vpath.startswith("~"):
                raise ValueError("Path traversal not allowed")
            full = (self.cwd / vpath.lstrip("/")).resolve()
            try:
                full.relative_to(self.cwd)
            except ValueError:
                raise ValueError(f"Path:{full} outside root directory: {self.cwd}") from None
            return full

        path = Path(key)
        if path.is_absolute():
            return path
        return (self.cwd / path).resolve()

    def ls_info(self, path: str) -> list[FileInfo]:
        """지정된 디렉토리의 파일과 디렉토리를 나열합니다 (비재귀적).

        Args:
            path: 파일 목록을 가져올 절대 디렉토리 경로.

        Returns:
            디렉토리 바로 아래에 있는 파일 및 디렉토리에 대한 FileInfo 유사 dict 목록.
            디렉토리는 경로 끝에 /가 붙으며 is_dir=True입니다.
        """
        dir_path = self._resolve_path(path)
        if not dir_path.exists() or not dir_path.is_dir():
            return []

        results: list[FileInfo] = []

        # Convert cwd to string for comparison
        cwd_str = str(self.cwd)
        if not cwd_str.endswith("/"):
            cwd_str += "/"

        # List only direct children (non-recursive)
        try:
            for child_path in dir_path.iterdir():
                try:
                    is_file = child_path.is_file()
                    is_dir = child_path.is_dir()
                except OSError:
                    continue

                abs_path = str(child_path)

                if not self.virtual_mode:
                    # Non-virtual mode: use absolute paths
                    if is_file:
                        try:
                            st = child_path.stat()
                            results.append({
                                "path": abs_path,
                                "is_dir": False,
                                "size": int(st.st_size),
                                "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                            })
                        except OSError:
                            results.append({"path": abs_path, "is_dir": False})
                    elif is_dir:
                        try:
                            st = child_path.stat()
                            results.append({
                                "path": abs_path + "/",
                                "is_dir": True,
                                "size": 0,
                                "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                            })
                        except OSError:
                            results.append({"path": abs_path + "/", "is_dir": True})
                else:
                    # Virtual mode: strip cwd prefix
                    if abs_path.startswith(cwd_str):
                        relative_path = abs_path[len(cwd_str) :]
                    elif abs_path.startswith(str(self.cwd)):
                        # Handle case where cwd doesn't end with /
                        relative_path = abs_path[len(str(self.cwd)) :].lstrip("/")
                    else:
                        # Path is outside cwd, return as-is or skip
                        relative_path = abs_path

                    virt_path = "/" + relative_path

                    if is_file:
                        try:
                            st = child_path.stat()
                            results.append({
                                "path": virt_path,
                                "is_dir": False,
                                "size": int(st.st_size),
                                "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                            })
                        except OSError:
                            results.append({"path": virt_path, "is_dir": False})
                    elif is_dir:
                        try:
                            st = child_path.stat()
                            results.append({
                                "path": virt_path + "/",
                                "is_dir": True,
                                "size": 0,
                                "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                            })
                        except OSError:
                            results.append({"path": virt_path + "/", "is_dir": True})
        except (OSError, PermissionError):
            pass

        # Keep deterministic order by path
        results.sort(key=lambda x: x.get("path", ""))
        return results

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """파일 내용을 라인 번호와 함께 읽습니다.

        Args:
            file_path: 절대 또는 상대 파일 경로.
            offset: 읽기 시작할 라인 오프셋 (0부터 시작).
            limit: 읽을 최대 라인 수.

        Returns:
            라인 번호가 포함된 형식화된 파일 내용, 또는 에러 메시지.
        """
        resolved_path = self._resolve_path(file_path)

        if not resolved_path.exists() or not resolved_path.is_file():
            return f"Error: File '{file_path}' not found"

        try:
            # Open with O_NOFOLLOW where available to avoid symlink traversal
            fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()

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
        except (OSError, UnicodeDecodeError) as e:
            return f"Error reading file '{file_path}': {e}"

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """내용을 포함하는 새 파일을 생성합니다.
        WriteResult를 반환합니다. 외부 저장소는 files_update=None을 설정합니다.
        """
        resolved_path = self._resolve_path(file_path)

        if resolved_path.exists():
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path."
            )

        try:
            # Create parent directories if needed
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            # Prefer O_NOFOLLOW to avoid writing through symlinks
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved_path, flags, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)

            return WriteResult(path=file_path, files_update=None)
        except (OSError, UnicodeEncodeError) as e:
            return WriteResult(error=f"Error writing file '{file_path}': {e}")

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
        resolved_path = self._resolve_path(file_path)

        if not resolved_path.exists() or not resolved_path.is_file():
            return EditResult(error=f"Error: File '{file_path}' not found")

        try:
            # Read securely
            fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()

            result = perform_string_replacement(content, old_string, new_string, replace_all)

            if isinstance(result, str):
                return EditResult(error=result)

            new_content, occurrences = result

            # Write securely
            flags = os.O_WRONLY | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved_path, flags)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)

            return EditResult(path=file_path, files_update=None, occurrences=int(occurrences))
        except (OSError, UnicodeDecodeError, UnicodeEncodeError) as e:
            return EditResult(error=f"Error editing file '{file_path}': {e}")

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        # Validate regex
        try:
            re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        # Resolve base path
        try:
            base_full = self._resolve_path(path or ".")
        except ValueError:
            return []

        if not base_full.exists():
            return []

        # Try ripgrep first
        results = self._ripgrep_search(pattern, base_full, glob)
        if results is None:
            results = self._python_search(pattern, base_full, glob)

        matches: list[GrepMatch] = []
        for fpath, items in results.items():
            for line_num, line_text in items:
                matches.append({"path": fpath, "line": int(line_num), "text": line_text})
        return matches

    def _ripgrep_search(
        self, pattern: str, base_full: Path, include_glob: str | None
    ) -> dict[str, list[tuple[int, str]]] | None:
        cmd = ["rg", "--json"]
        if include_glob:
            cmd.extend(["--glob", include_glob])
        cmd.extend(["--", pattern, str(base_full)])

        try:
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        results: dict[str, list[tuple[int, str]]] = {}
        for line in proc.stdout.splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") != "match":
                continue
            pdata = data.get("data", {})
            ftext = pdata.get("path", {}).get("text")
            if not ftext:
                continue
            p = Path(ftext)
            if self.virtual_mode:
                try:
                    virt = "/" + str(p.resolve().relative_to(self.cwd))
                except Exception:
                    continue
            else:
                virt = str(p)
            ln = pdata.get("line_number")
            lt = pdata.get("lines", {}).get("text", "").rstrip("\n")
            if ln is None:
                continue
            results.setdefault(virt, []).append((int(ln), lt))

        return results

    def _python_search(
        self, pattern: str, base_full: Path, include_glob: str | None
    ) -> dict[str, list[tuple[int, str]]]:
        try:
            regex = re.compile(pattern)
        except re.error:
            return {}

        results: dict[str, list[tuple[int, str]]] = {}
        root = base_full if base_full.is_dir() else base_full.parent

        for fp in root.rglob("*"):
            if not fp.is_file():
                continue
            if include_glob and not wcglob.globmatch(fp.name, include_glob, flags=wcglob.BRACE):
                continue
            try:
                if fp.stat().st_size > self.max_file_size_bytes:
                    continue
            except OSError:
                continue
            try:
                content = fp.read_text()
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
            for line_num, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    if self.virtual_mode:
                        try:
                            virt_path = "/" + str(fp.resolve().relative_to(self.cwd))
                        except Exception:
                            continue
                    else:
                        virt_path = str(fp)
                    results.setdefault(virt_path, []).append((line_num, line))

        return results

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        if pattern.startswith("/"):
            pattern = pattern.lstrip("/")

        search_path = self.cwd if path == "/" else self._resolve_path(path)
        if not search_path.exists() or not search_path.is_dir():
            return []

        results: list[FileInfo] = []
        try:
            # Use recursive globbing to match files in subdirectories as tests expect
            for matched_path in search_path.rglob(pattern):
                try:
                    is_file = matched_path.is_file()
                except OSError:
                    continue
                if not is_file:
                    continue
                abs_path = str(matched_path)
                if not self.virtual_mode:
                    try:
                        st = matched_path.stat()
                        results.append({
                            "path": abs_path,
                            "is_dir": False,
                            "size": int(st.st_size),
                            "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                        })
                    except OSError:
                        results.append({"path": abs_path, "is_dir": False})
                else:
                    cwd_str = str(self.cwd)
                    if not cwd_str.endswith("/"):
                        cwd_str += "/"
                    if abs_path.startswith(cwd_str):
                        relative_path = abs_path[len(cwd_str) :]
                    elif abs_path.startswith(str(self.cwd)):
                        relative_path = abs_path[len(str(self.cwd)) :].lstrip("/")
                    else:
                        relative_path = abs_path
                    virt = "/" + relative_path
                    try:
                        st = matched_path.stat()
                        results.append({
                            "path": virt,
                            "is_dir": False,
                            "size": int(st.st_size),
                            "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                        })
                    except OSError:
                        results.append({"path": virt, "is_dir": False})
        except (OSError, ValueError):
            pass

        results.sort(key=lambda x: x.get("path", ""))
        return results

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """파일시스템에 여러 파일을 업로드합니다.

        Args:
            files: 내용이 bytes인 (path, content) 튜플의 리스트.

        Returns:
            FileUploadResponse 객체들의 리스트. 입력 파일마다 하나씩 반환됩니다.
            응답 순서는 입력 순서와 일치합니다.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                resolved_path = self._resolve_path(path)

                # Create parent directories if needed
                resolved_path.parent.mkdir(parents=True, exist_ok=True)

                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(resolved_path, flags, 0o644)
                with os.fdopen(fd, "wb") as f:
                    f.write(content)

                responses.append(FileUploadResponse(path=path, error=None))
            except FileNotFoundError:
                responses.append(FileUploadResponse(path=path, error="file_not_found"))
            except PermissionError:
                responses.append(FileUploadResponse(path=path, error="permission_denied"))
            except (ValueError, OSError) as e:
                # ValueError from _resolve_path for path traversal, OSError for other file errors
                if isinstance(e, ValueError) or "invalid" in str(e).lower():
                    responses.append(FileUploadResponse(path=path, error="invalid_path"))
                else:
                    # Generic error fallback
                    responses.append(FileUploadResponse(path=path, error="invalid_path"))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """파일시스템에서 여러 파일을 다운로드합니다.

        Args:
            paths: 다운로드할 파일 경로의 리스트.

        Returns:
            FileDownloadResponse 객체들의 리스트. 입력 경로마다 하나씩 반환됩니다.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                resolved_path = self._resolve_path(path)
                # Use flags to optionally prevent symlink following if
                # supported by the OS
                fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "rb") as f:
                    content = f.read()
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except FileNotFoundError:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
            except PermissionError:
                responses.append(FileDownloadResponse(path=path, content=None, error="permission_denied"))
            except IsADirectoryError:
                responses.append(FileDownloadResponse(path=path, content=None, error="is_directory"))
            except ValueError:
                responses.append(FileDownloadResponse(path=path, content=None, error="invalid_path"))
            # Let other errors propagate
        return responses
