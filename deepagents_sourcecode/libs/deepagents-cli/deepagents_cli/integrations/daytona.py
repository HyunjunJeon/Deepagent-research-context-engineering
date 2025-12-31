"""Daytona 샌드박스 백엔드 구현."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from daytona import Sandbox


class DaytonaBackend(BaseSandbox):
    """SandboxBackendProtocol을 준수하는 Daytona 백엔드 구현.

    이 구현은 BaseSandbox로부터 모든 파일 작업 메서드를 상속받으며,
    Daytona의 API를 사용하여 execute() 메서드만 구현합니다.
    """

    def __init__(self, sandbox: Sandbox) -> None:
        """Daytona 샌드박스 클라이언트로 DaytonaBackend를 초기화합니다.

        Args:
            sandbox: Daytona 샌드박스 인스턴스
        """
        self._sandbox = sandbox
        self._timeout: int = 30 * 60  # 30분

    @property
    def id(self) -> str:
        """샌드박스 백엔드의 고유 식별자."""
        return self._sandbox.id

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """샌드박스에서 명령을 실행하고 ExecuteResponse를 반환합니다.

        Args:
            command: 실행할 전체 셸 명령 문자열.

        Returns:
            결합된 출력, 종료 코드, 선택적 시그널 및 잘림 플래그가 포함된 ExecuteResponse.
        """
        result = self._sandbox.process.exec(command, timeout=self._timeout)

        return ExecuteResponse(
            output=result.result,  # Daytona는 stdout/stderr를 결합함
            exit_code=result.exit_code,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Daytona 샌드박스에서 여러 파일을 다운로드합니다.

        효율성을 위해 Daytona의 네이티브 일괄 다운로드 API를 활용합니다.
        부분적인 성공을 지원하므로 개별 다운로드가 다른 다운로드에 영향을 주지 않고 실패할 수 있습니다.

        Args:
            paths: 다운로드할 파일 경로 목록.

        Returns:
            입력 경로당 하나씩 FileDownloadResponse 객체 목록.
            응답 순서는 입력 순서와 일치합니다.

        TODO: Daytona API 오류 문자열을 표준화된 FileOperationError 코드로 매핑해야 합니다.
        현재는 정상적인 동작(happy path)만 구현되어 있습니다.
        """
        from daytona import FileDownloadRequest

        # Daytona의 네이티브 일괄 API를 사용하여 일괄 다운로드 요청 생성
        download_requests = [FileDownloadRequest(source=path) for path in paths]
        daytona_responses = self._sandbox.fs.download_files(download_requests)

        # Daytona 결과를 당사의 응답 형식으로 변환
        # TODO: 사용 가능한 경우 resp.error를 표준화된 오류 코드로 매핑
        return [
            FileDownloadResponse(
                path=resp.source,
                content=resp.result,
                error=None,  # TODO: resp.error를 FileOperationError로 매핑
            )
            for resp in daytona_responses
        ]

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Daytona 샌드박스에 여러 파일을 업로드합니다.

        효율성을 위해 Daytona의 네이티브 일괄 업로드 API를 활용합니다.
        부분적인 성공을 지원하므로 개별 업로드가 다른 업로드에 영향을 주지 않고 실패할 수 있습니다.

        Args:
            files: 업로드할 (경로, 내용) 튜플 목록.

        Returns:
            입력 파일당 하나씩 FileUploadResponse 객체 목록.
            응답 순서는 입력 순서와 일치합니다.

        TODO: Daytona API 오류 문자열을 표준화된 FileOperationError 코드로 매핑해야 합니다.
        현재는 정상적인 동작(happy path)만 구현되어 있습니다.
        """
        from daytona import FileUpload

        # Daytona의 네이티브 일괄 API를 사용하여 일괄 업로드 요청 생성
        upload_requests = [FileUpload(source=content, destination=path) for path, content in files]
        self._sandbox.fs.upload_files(upload_requests)

        # TODO: Daytona가 오류 정보를 반환하는지 확인하고 FileOperationError 코드로 매핑
        return [FileUploadResponse(path=path, error=None) for path, _ in files]
