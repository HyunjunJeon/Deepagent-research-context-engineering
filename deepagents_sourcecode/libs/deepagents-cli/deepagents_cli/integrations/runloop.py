"""Runloop을 위한 BackendProtocol 구현."""

try:
    import runloop_api_client
except ImportError:
    msg = (
        "RunloopBackend를 위해서는 runloop_api_client 패키지가 필요합니다. "
        "`pip install runloop_api_client`로 설치하십시오."
    )
    raise ImportError(msg)

import os

from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox
from runloop_api_client import Runloop


class RunloopBackend(BaseSandbox):
    """Runloop devbox의 파일에서 작동하는 백엔드.

    이 구현은 Runloop API 클라이언트를 사용하여 명령을 실행하고
    원격 devbox 환경 내에서 파일을 조작합니다.
    """

    def __init__(
        self,
        devbox_id: str,
        client: Runloop | None = None,
        api_key: str | None = None,
    ) -> None:
        """Runloop 프로토콜을 초기화합니다.

        Args:
            devbox_id: 작업할 Runloop devbox의 ID.
            client: 선택적인 기존 Runloop 클라이언트 인스턴스
            api_key: 새 클라이언트를 생성하기 위한 선택적 API 키
                         (기본값은 RUNLOOP_API_KEY 환경 변수)
        """
        if client and api_key:
            msg = "client 또는 bearer_token 중 하나만 제공해야 하며, 둘 다 제공할 수는 없습니다."
            raise ValueError(msg)

        if client is None:
            api_key = api_key or os.environ.get("RUNLOOP_API_KEY", None)
            if api_key is None:
                msg = "client 또는 bearer_token 중 하나는 제공되어야 합니다."
                raise ValueError(msg)
            client = Runloop(bearer_token=api_key)

        self._client = client
        self._devbox_id = devbox_id
        self._timeout = 30 * 60

    @property
    def id(self) -> str:
        """샌드박스 백엔드의 고유 식별자."""
        return self._devbox_id

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """devbox에서 명령을 실행하고 ExecuteResponse를 반환합니다.

        Args:
            command: 실행할 전체 셸 명령 문자열.

        Returns:
            결합된 출력, 종료 코드, 선택적 시그널 및 잘림 플래그가 포함된 ExecuteResponse.
        """
        result = self._client.devboxes.execute_and_await_completion(
            devbox_id=self._devbox_id,
            command=command,
            timeout=self._timeout,
        )
        # stdout과 stderr 결합
        output = result.stdout or ""
        if result.stderr:
            output += "\n" + result.stderr if output else result.stderr

        return ExecuteResponse(
            output=output,
            exit_code=result.exit_status,
            truncated=False,  # Runloop는 잘림 정보를 제공하지 않음
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Runloop devbox에서 여러 파일을 다운로드합니다.

        Runloop API를 사용하여 파일을 개별적으로 다운로드합니다. 순서를 유지하고
        예외를 발생시키는 대신 파일별 오류를 보고하는 FileDownloadResponse 객체 목록을 반환합니다.

        TODO: 표준화된 FileOperationError 코드를 사용하여 적절한 오류 처리를 구현해야 합니다.
        현재는 정상적인 동작(happy path)만 구현되어 있습니다.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            # devboxes.download_file은 .read()를 노출하는 BinaryAPIResponse를 반환함
            resp = self._client.devboxes.download_file(self._devbox_id, path=path)
            content = resp.read()
            responses.append(FileDownloadResponse(path=path, content=content, error=None))

        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Runloop devbox에 여러 파일을 업로드합니다.

        Runloop API를 사용하여 파일을 개별적으로 업로드합니다. 순서를 유지하고
        예외를 발생시키는 대신 파일별 오류를 보고하는 FileUploadResponse 객체 목록을 반환합니다.

        TODO: 표준화된 FileOperationError 코드를 사용하여 적절한 오류 처리를 구현해야 합니다.
        현재는 정상적인 동작(happy path)만 구현되어 있습니다.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            # Runloop 클라이언트는 'file'을 바이트 또는 파일류 객체로 기대함
            self._client.devboxes.upload_file(self._devbox_id, path=path, file=content)
            responses.append(FileUploadResponse(path=path, error=None))

        return responses
