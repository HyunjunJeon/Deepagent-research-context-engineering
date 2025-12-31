"""Modal 샌드박스 백엔드 구현."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    import modal


class ModalBackend(BaseSandbox):
    """SandboxBackendProtocol을 준수하는 Modal 백엔드 구현.

    이 구현은 BaseSandbox로부터 모든 파일 작업 메서드를 상속받으며,
    Modal의 API를 사용하여 execute() 메서드만 구현합니다.
    """

    def __init__(self, sandbox: modal.Sandbox) -> None:
        """Modal 샌드박스 인스턴스로 ModalBackend를 초기화합니다.

        Args:
            sandbox: 활성 Modal 샌드박스 인스턴스
        """
        self._sandbox = sandbox
        self._timeout = 30 * 60

    @property
    def id(self) -> str:
        """샌드박스 백엔드의 고유 식별자."""
        return self._sandbox.object_id

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """샌드박스에서 명령을 실행하고 ExecuteResponse를 반환합니다.

        Args:
            command: 실행할 전체 셸 명령 문자열.

        Returns:
            결합된 출력, 종료 코드 및 잘림 플래그가 포함된 ExecuteResponse.
        """
        # Modal의 exec API를 사용하여 명령 실행
        process = self._sandbox.exec("bash", "-c", command, timeout=self._timeout)

        # 프로세스가 완료될 때까지 대기
        process.wait()

        # stdout 및 stderr 읽기
        stdout = process.stdout.read()
        stderr = process.stderr.read()

        # stdout과 stderr 결합 (Runloop의 방식과 일치)
        output = stdout or ""
        if stderr:
            output += "\n" + stderr if output else stderr

        return ExecuteResponse(
            output=output,
            exit_code=process.returncode,
            truncated=False,  # Modal은 잘림 정보를 제공하지 않음
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Modal 샌드박스에서 여러 파일을 다운로드합니다.

        부분적인 성공을 지원하므로 개별 다운로드가 다른 다운로드에 영향을 주지 않고 실패할 수 있습니다.

        Args:
            paths: 다운로드할 파일 경로 목록.

        Returns:
            입력 경로당 하나씩 FileDownloadResponse 객체 목록.
            응답 순서는 입력 순서와 일치합니다.

        TODO: 표준화된 FileOperationError 코드를 사용하여 적절한 오류 처리를 구현해야 합니다.
        Modal의 sandbox.open()이 실제로 어떤 예외를 발생시키는지 확인이 필요합니다.
        현재는 정상적인 동작(happy path)만 구현되어 있습니다.
        """
        # 이 구현은 Modal 샌드박스 파일 API에 의존합니다.
        # https://modal.com/doc/guide/sandbox-files
        # 이 API는 현재 알파 단계이며 프로덕션 용도로는 권장되지 않습니다.
        # CLI 애플리케이션을 대상으로 하므로 여기에서 사용하는 것은 괜찮습니다.
        responses = []
        for path in paths:
            with self._sandbox.open(path, "rb") as f:
                content = f.read()
            responses.append(FileDownloadResponse(path=path, content=content, error=None))
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Modal 샌드박스에 여러 파일을 업로드합니다.

        부분적인 성공을 지원하므로 개별 업로드가 다른 업로드에 영향을 주지 않고 실패할 수 있습니다.

        Args:
            files: 업로드할 (경로, 내용) 튜플 목록.

        Returns:
            입력 파일당 하나씩 FileUploadResponse 객체 목록.
            응답 순서는 입력 순서와 일치합니다.

        TODO: 표준화된 FileOperationError 코드를 사용하여 적절한 오류 처리를 구현해야 합니다.
        Modal의 sandbox.open()이 실제로 어떤 예외를 발생시키는지 확인이 필요합니다.
        현재는 정상적인 동작(happy path)만 구현되어 있습니다.
        """
        # 이 구현은 Modal 샌드박스 파일 API에 의존합니다.
        # https://modal.com/doc/guide/sandbox-files
        # 이 API는 현재 알파 단계이며 프로덕션 용도로는 권장되지 않습니다.
        # CLI 애플리케이션을 대상으로 하므로 여기에서 사용하는 것은 괜찮습니다.
        responses = []
        for path, content in files:
            with self._sandbox.open(path, "wb") as f:
                f.write(content)
            responses.append(FileUploadResponse(path=path, error=None))
        return responses
