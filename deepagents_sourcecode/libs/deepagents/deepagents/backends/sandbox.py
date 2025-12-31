"""execute()만을 추상 메서드로 가지는 기본 샌드박스 구현.

이 모듈은 execute()를 통해 쉘 명령을 실행하여 모든 SandboxBackendProtocol
메서드를 구현하는 기본 클래스를 제공합니다. 구체적인 구현체는
오직 execute() 메서드만 구현하면 됩니다.
"""

from __future__ import annotations

import base64
import json
import shlex
from abc import ABC, abstractmethod

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)

_GLOB_COMMAND_TEMPLATE = """python3 -c "
import glob
import os
import json
import base64

# base64 인코딩된 파라미터 디코딩
path = base64.b64decode('{path_b64}').decode('utf-8')
pattern = base64.b64decode('{pattern_b64}').decode('utf-8')

os.chdir(path)
matches = sorted(glob.glob(pattern, recursive=True))
for m in matches:
    stat = os.stat(m)
    result = {{
        'path': m,
        'size': stat.st_size,
        'mtime': stat.st_mtime,
        'is_dir': os.path.isdir(m)
    }}
    print(json.dumps(result))
" 2>/dev/null"""

_WRITE_COMMAND_TEMPLATE = """python3 -c "
import os
import sys
import base64

file_path = '{file_path}'

# 파일이 이미 존재하는지 확인 (쓰기와 원자적)
if os.path.exists(file_path):
    print(f'Error: File \\'{file_path}\\' already exists', file=sys.stderr)
    sys.exit(1)

# 필요시 부모 디렉토리 생성
parent_dir = os.path.dirname(file_path) or '.'
os.makedirs(parent_dir, exist_ok=True)

# 내용 디코딩 및 쓰기
content = base64.b64decode('{content_b64}').decode('utf-8')
with open(file_path, 'w') as f:
    f.write(content)
" 2>&1"""

_EDIT_COMMAND_TEMPLATE = """python3 -c "
import sys
import base64

# 파일 내용 읽기
with open('{file_path}', 'r') as f:
    text = f.read()

# base64 인코딩된 문자열 디코딩
old = base64.b64decode('{old_b64}').decode('utf-8')
new = base64.b64decode('{new_b64}').decode('utf-8')

# 발생 횟수 계산
count = text.count(old)

# 문제가 발견되면 에러 코드와 함께 종료
if count == 0:
    sys.exit(1)  # 문자열을 찾을 수 없음
elif count > 1 and not {replace_all}:
    sys.exit(2)  # replace_all 없이 여러 번 발생

# 교체 수행
if {replace_all}:
    result = text.replace(old, new)
else:
    result = text.replace(old, new, 1)

# 파일에 다시 쓰기
with open('{file_path}', 'w') as f:
    f.write(result)

print(count)
" 2>&1"""

_READ_COMMAND_TEMPLATE = """python3 -c "
import os
import sys

file_path = '{file_path}'
offset = {offset}
limit = {limit}

# 파일이 존재하는지 확인
if not os.path.isfile(file_path):
    print('Error: File not found')
    sys.exit(1)

# 파일이 비어있는지 확인
if os.path.getsize(file_path) == 0:
    print('System reminder: File exists but has empty contents')
    sys.exit(0)

# offset과 limit으로 파일 읽기
with open(file_path, 'r') as f:
    lines = f.readlines()

# offset과 limit 적용
start_idx = offset
end_idx = offset + limit
selected_lines = lines[start_idx:end_idx]

# 라인 번호로 포맷팅 (1부터 시작, offset + 1부터 시작)
for i, line in enumerate(selected_lines):
    line_num = offset + i + 1
    # 포맷팅을 위해 끝의 개행 문자 제거 후 다시 추가
    line_content = line.rstrip('\\n')
    print(f'{{line_num:6d}}\\t{{line_content}}')
" 2>&1"""


class BaseSandbox(SandboxBackendProtocol, ABC):
    """execute()를 추상 메서드로 가지는 기본 샌드박스 구현.

    이 클래스는 쉘 명령을 사용하여 모든 프로토콜 메서드에 대한 기본 구현을
    제공합니다. 하위 클래스는 오직 execute()만 구현하면 됩니다.
    """

    @abstractmethod
    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """샌드박스에서 명령을 실행하고 ExecuteResponse를 반환합니다.

        Args:
            command: 실행할 전체 쉘 명령 문자열.

        Returns:
            결합된 출력, 종료 코드, 선택적 시그널, 잘림(truncation) 플래그를 포함하는 ExecuteResponse.
        """
        ...

    def ls_info(self, path: str) -> list[FileInfo]:
        """os.scandir을 사용하여 파일 메타데이터가 포함된 구조화된 목록을 반환합니다."""
        cmd = f"""python3 -c "
import os
import json

path = '{path}'

try:
    with os.scandir(path) as it:
        for entry in it:
            result = {{
                'path': entry.name,
                'is_dir': entry.is_dir(follow_symlinks=False)
            }}
            print(json.dumps(result))
except FileNotFoundError:
    pass
except PermissionError:
    pass
" 2>/dev/null"""

        result = self.execute(cmd)

        file_infos: list[FileInfo] = []
        for line in result.output.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                file_infos.append({"path": data["path"], "is_dir": data["is_dir"]})
            except json.JSONDecodeError:
                continue

        return file_infos

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """단일 쉘 명령을 사용하여 라인 번호와 함께 파일 내용을 읽습니다."""
        # offset과 limit으로 파일을 읽기 위해 템플릿 사용
        cmd = _READ_COMMAND_TEMPLATE.format(file_path=file_path, offset=offset, limit=limit)
        result = self.execute(cmd)

        output = result.output.rstrip()
        exit_code = result.exit_code

        if exit_code != 0 or "Error: File not found" in output:
            return f"Error: File '{file_path}' not found"

        return output

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """새 파일을 생성합니다. WriteResult를 반환하며, 실패 시 에러가 채워집니다."""
        # 이스케이프 문제를 피하기 위해 내용을 base64로 인코딩
        content_b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")

        # 단일 원자적 확인 + 쓰기 명령
        cmd = _WRITE_COMMAND_TEMPLATE.format(file_path=file_path, content_b64=content_b64)
        result = self.execute(cmd)

        # 에러 확인 (종료 코드 또는 출력 내 에러 메시지)
        if result.exit_code != 0 or "Error:" in result.output:
            error_msg = result.output.strip() or f"Failed to write file '{file_path}'"
            return WriteResult(error=error_msg)

        # 외부 저장소 - files_update 필요 없음
        return WriteResult(path=file_path, files_update=None)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """문자열 발생(occurrences)을 교체하여 파일을 편집합니다. EditResult를 반환합니다."""
        # 이스케이프 문제를 피하기 위해 문자열을 base64로 인코딩
        old_b64 = base64.b64encode(old_string.encode("utf-8")).decode("ascii")
        new_b64 = base64.b64encode(new_string.encode("utf-8")).decode("ascii")

        # 문자열 교체를 위해 템플릿 사용
        cmd = _EDIT_COMMAND_TEMPLATE.format(
            file_path=file_path, old_b64=old_b64, new_b64=new_b64, replace_all=replace_all
        )
        result = self.execute(cmd)

        exit_code = result.exit_code
        output = result.output.strip()

        if exit_code == 1:
            return EditResult(error=f"Error: String not found in file: '{old_string}'")
        if exit_code == 2:
            return EditResult(
                error=f"Error: String '{old_string}' appears multiple times. Use replace_all=True to replace all occurrences."
            )
        if exit_code != 0:
            return EditResult(error=f"Error: File '{file_path}' not found")

        count = int(output)
        # 외부 저장소 - files_update 필요 없음
        return EditResult(path=file_path, files_update=None, occurrences=count)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """구조화된 검색 결과 또는 잘못된 입력에 대한 에러 문자열을 반환합니다."""
        search_path = shlex.quote(path or ".")

        # 구조화된 출력을 얻기 위해 grep 명령 생성
        grep_opts = "-rHnF"  # 재귀적, 파일명 포함, 라인 번호 포함, 고정 문자열 (리터럴)

        # 지정된 경우 glob 패턴 추가
        glob_pattern = ""
        if glob:
            glob_pattern = f"--include='{glob}'"

        # 쉘을 위해 패턴 이스케이프
        pattern_escaped = shlex.quote(pattern)

        cmd = f"grep {grep_opts} {glob_pattern} -e {pattern_escaped} {search_path} 2>/dev/null || true"
        result = self.execute(cmd)

        output = result.output.rstrip()
        if not output:
            return []

        # grep 출력을 GrepMatch 객체로 파싱
        matches: list[GrepMatch] = []
        for line in output.split("\n"):
            # 형식: 경로:라인번호:텍스트
            parts = line.split(":", 2)
            if len(parts) >= 3:
                matches.append({
                    "path": parts[0],
                    "line": int(parts[1]),
                    "text": parts[2],
                })

        return matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """FileInfo dict를 반환하는 구조화된 glob 매칭입니다."""
        # 이스케이프 문제를 피하기 위해 패턴과 경로를 base64로 인코딩
        pattern_b64 = base64.b64encode(pattern.encode("utf-8")).decode("ascii")
        path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")

        cmd = _GLOB_COMMAND_TEMPLATE.format(path_b64=path_b64, pattern_b64=pattern_b64)
        result = self.execute(cmd)

        output = result.output.strip()
        if not output:
            return []

        # JSON 출력을 FileInfo dict로 파싱
        file_infos: list[FileInfo] = []
        for line in output.split("\n"):
            try:
                data = json.loads(line)
                file_infos.append({
                    "path": data["path"],
                    "is_dir": data["is_dir"],
                })
            except json.JSONDecodeError:
                continue

        return file_infos

    @property
    @abstractmethod
    def id(self) -> str:
        """샌드박스 백엔드의 고유 식별자입니다."""

    @abstractmethod
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """샌드박스에 여러 파일을 업로드합니다.

        구현체는 부분적 성공을 지원해야 합니다 - 파일별로 예외를 catch하고
        예외를 발생시키는 대신 FileUploadResponse 객체에 에러를 반환해야 합니다.
        """

    @abstractmethod
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """샌드박스에서 여러 파일을 다운로드합니다.

        구현체는 부분적 성공을 지원해야 합니다 - 파일별로 예외를 catch하고
        예외를 발생시키는 대신 FileDownloadResponse 객체에 에러를 반환해야 합니다.
        """
