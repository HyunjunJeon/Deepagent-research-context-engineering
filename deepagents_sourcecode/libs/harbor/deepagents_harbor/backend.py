"""Implement harbor backend."""

import base64
import shlex

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileInfo,
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)
from harbor.environments.base import BaseEnvironment


class HarborSandbox(SandboxBackendProtocol):
    """Python3가 사용 가능하다고 가정하지 않는 샌드박스 구현체입니다."""

    def __init__(self, environment: BaseEnvironment) -> None:
        """제공된 환경으로 HarborSandbox를 초기화합니다."""
        self.environment = environment

    async def aexecute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """작업 환경에서 bash 명령을 실행합니다."""
        result = await self.environment.exec(command)

        # These errors appear in harbor environments when running bash commands
        # in non-interactive/non-TTY contexts. They're harmless artifacts.
        # Filter them from both stdout and stderr, then collect them to show in stderr.
        error_messages = [
            "bash: cannot set terminal process group (-1): Inappropriate ioctl for device",
            "bash: no job control in this shell",
            "bash: initialize_job_control: no job control in background: Bad file descriptor",
        ]

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # Collect the bash messages if they appear (to move to stderr)
        bash_messages = []
        for error_msg in error_messages:
            if error_msg in stdout:
                bash_messages.append(error_msg)
                stdout = stdout.replace(error_msg, "")
            if error_msg in stderr:
                stderr = stderr.replace(error_msg, "")

        stdout = stdout.strip()
        stderr = stderr.strip()

        # Add bash messages to stderr
        if bash_messages:
            bash_msg_text = "\n".join(bash_messages)
            stderr = f"{bash_msg_text}\n{stderr}".strip() if stderr else bash_msg_text

        # Only append stderr label if there's actual stderr content
        if stderr:
            output = stdout + "\n\n stderr: " + stderr if stdout else "\n stderr: " + stderr
        else:
            output = stdout
        return ExecuteResponse(
            output=output,
            exit_code=result.return_code,
        )

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """작업 환경에서 bash 명령을 실행합니다."""
        raise NotImplementedError("이 백엔드는 비동기 실행만 지원합니다")

    @property
    def id(self) -> str:
        """샌드박스 백엔드의 고유 식별자입니다."""
        return self.environment.session_id

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """셸 명령을 사용하여 줄 번호가 있는 파일 내용을 읽습니다."""
        # Escape file path for shell
        safe_path = shlex.quote(file_path)

        # Check if file exists and handle empty files
        cmd = f"""
if [ ! -f {safe_path} ]; then
    echo "오류: 파일을 찾을 수 없습니다"
    exit 1
fi
if [ ! -s {safe_path} ]; then
    echo "시스템 알림: 파일이 존재하지만 내용이 비어 있습니다"
    exit 0
fi
# Use awk to add line numbers and handle offset/limit
awk -v offset={offset} -v limit={limit} '
    NR > offset && NR <= offset + limit {{
        printf "%6d\\t%s\\n", NR, $0
    }}
    NR > offset + limit {{ exit }}
' {safe_path}
"""
        result = await self.aexecute(cmd)

        if result.exit_code != 0 or "오류: 파일을 찾을 수 없습니다" in result.output:
            return f"오류: 파일 '{file_path}'을(를) 찾을 수 없습니다"

        return result.output.rstrip()

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """셸 명령을 사용하여 줄 번호가 있는 파일 내용을 읽습니다."""
        raise NotImplementedError("aread를 사용하십시오")

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """셸 명령을 사용하여 새 파일을 생성합니다."""
        # Encode content as base64 to avoid escaping issues
        content_b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        safe_path = shlex.quote(file_path)

        cmd = f"""
if [ -e {safe_path} ]; then
    echo "오류: 파일 '{file_path}'이(가) 이미 존재합니다" >&2
    exit 1
fi
parent_dir=$(dirname {safe_path})
mkdir -p "$parent_dir" 2>/dev/null
echo '{content_b64}' | base64 -d > {safe_path}
"""
        result = await self.aexecute(cmd)

        if result.exit_code != 0 or "Error:" in result.output:
            error_msg = result.output.strip() or f"파일 '{file_path}' 쓰기 실패"
            return WriteResult(error=error_msg)

        return WriteResult(path=file_path, files_update=None)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """셸 명령을 사용하여 새 파일을 생성합니다."""
        raise NotImplementedError("awrite를 사용하십시오")

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """셸 명령을 사용하여 문자열 조회를 대체하여 파일을 편집합니다."""
        # Encode strings as base64 to avoid escaping issues
        old_b64 = base64.b64encode(old_string.encode("utf-8")).decode("ascii")
        new_b64 = base64.b64encode(new_string.encode("utf-8")).decode("ascii")
        safe_path = shlex.quote(file_path)
        replace_all_str = "true" if replace_all else "false"

        # Use a shell script with perl for reliable string replacement
        cmd = f"""
if [ ! -f {safe_path} ]; then
    exit 3
fi

old=$(echo '{old_b64}' | base64 -d)
new=$(echo '{new_b64}' | base64 -d)

# Count occurrences using grep -F (fixed strings)
count=$(grep -o -F "$old" {safe_path} | wc -l)

if [ "$count" -eq 0 ]; then
    exit 1
elif [ "$count" -gt 1 ] && [ "{replace_all_str}" = "false" ]; then
    exit 2
fi

# Use perl for reliable string replacement (handles special chars)
if [ "{replace_all_str}" = "true" ]; then
    perl -i -pe 's/\\Q'"$old"'\\E/'"$new"'/g' {safe_path}
else
    perl -i -pe 's/\\Q'"$old"'\\E/'"$new"'/' {safe_path}
fi

echo "$count"
"""
        result = await self.aexecute(cmd)

        exit_code = result.exit_code
        output = result.output.strip()

        if exit_code == 1:
            return EditResult(error=f"오류: 파일에서 문자열을 찾을 수 없습니다: '{old_string}'")
        if exit_code == 2:
            return EditResult(
                error=f"오류: 문자열 '{old_string}'이(가) 여러 번 나옵니다. 모든 항목을 바꾸려면 replace_all=True를 사용하십시오."
            )
        if exit_code == 3:
            return EditResult(error=f"오류: 파일 '{file_path}'을(를) 찾을 수 없습니다")
        if exit_code != 0:
            return EditResult(error=f"파일 편집 오류: {output}")

        try:
            count = int(output.split("\n")[0])
        except (ValueError, IndexError):
            count = 1

        return EditResult(path=file_path, files_update=None, occurrences=count)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """셸 명령을 사용하여 문자열 조회를 대체하여 파일을 편집합니다."""
        raise NotImplementedError("aedit를 사용하십시오")

    async def als_info(self, path: str) -> list[FileInfo]:
        """셸 명령을 사용하여 디렉터리 내용과 메타데이터를 나열합니다."""
        safe_path = shlex.quote(path)

        cmd = f"""
if [ ! -d {safe_path} ]; then
    exit 1
fi
for entry in {safe_path}/*; do
    if [ -e "$entry" ]; then
        name=$(basename "$entry")
        if [ -d "$entry" ]; then
            printf '%s|true\\n' "$name"
        else
            printf '%s|false\\n' "$name"
        fi
    fi
done
"""
        result = await self.aexecute(cmd)

        if result.exit_code != 0:
            return []

        file_infos: list[FileInfo] = []
        for line in result.output.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 2:
                file_infos.append({"path": parts[0], "is_dir": parts[1] == "true"})

        return file_infos

    def ls_info(self, path: str) -> list[FileInfo]:
        """셸 명령을 사용하여 디렉터리 내용과 메타데이터를 나열합니다."""
        raise NotImplementedError("als_info를 사용하십시오")

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """grep을 사용하여 파일에서 패턴을 검색합니다."""
        search_path = shlex.quote(path or ".")

        # Build grep command
        grep_opts = "-rHn"  # recursive, with filename, with line number

        # Add glob pattern if specified
        glob_pattern = ""
        if glob:
            glob_pattern = f"--include={shlex.quote(glob)}"

        # Escape pattern for grep
        safe_pattern = shlex.quote(pattern)

        cmd = f"grep {grep_opts} {glob_pattern} -e {safe_pattern} {search_path} 2>/dev/null || true"
        result = await self.aexecute(cmd)

        output = result.output.rstrip()
        if not output:
            return []

        # Parse grep output into GrepMatch objects
        matches: list[GrepMatch] = []
        for line in output.split("\n"):
            # Format is: path:line_number:text
            parts = line.split(":", 2)
            if len(parts) >= 3:
                try:
                    matches.append({
                        "path": parts[0],
                        "line": int(parts[1]),
                        "text": parts[2],
                    })
                except ValueError:
                    continue

        return matches

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """grep을 사용하여 파일에서 패턴을 검색합니다."""
        raise NotImplementedError("agrep_raw를 사용하십시오")

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """셸 명령을 사용하여 glob 패턴과 일치하는 파일을 찾습니다.

        이 구현은 현재 모든 glob 패턴을 지원하지는 않습니다.
        """
        safe_path = shlex.quote(path)
        safe_pattern = shlex.quote(pattern)

        cmd = f"""
cd {safe_path} 2>/dev/null || exit 1
# Use find with shell globbing
for file in {safe_pattern}; do
    if [ -e "$file" ]; then
        if [ -d "$file" ]; then
            printf '%s|true\\n' "$file"
        else
            printf '%s|false\\n' "$file"
        fi
    fi
done
"""
        result = await self.aexecute(cmd)

        if result.exit_code != 0:
            return []

        output = result.output.strip()
        if not output:
            return []

        # Parse output into FileInfo dicts
        file_infos: list[FileInfo] = []
        for line in output.split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 2:
                file_infos.append({
                    "path": parts[0],
                    "is_dir": parts[1] == "true",
                })

        return file_infos

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """셸 명령을 사용하여 glob 패턴과 일치하는 파일을 찾습니다."""
        raise NotImplementedError("aglob_info를 사용하십시오")
