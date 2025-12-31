"""컨텍스트 매니저를 통한 샌드박스 수명 주기 관리."""

import os
import shlex
import string
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from deepagents.backends.protocol import SandboxBackendProtocol

from deepagents_cli.config import console


def _run_sandbox_setup(backend: SandboxBackendProtocol, setup_script_path: str) -> None:
    """환경 변수 확장을 포함하여 샌드박스에서 사용자 설정 스크립트를 실행합니다.

    Args:
        backend: 샌드박스 백엔드 인스턴스
        setup_script_path: 설정 스크립트 파일 경로
    """
    script_path = Path(setup_script_path)
    if not script_path.exists():
        msg = f"설정 스크립트를 찾을 수 없습니다: {setup_script_path}"
        raise FileNotFoundError(msg)

    console.print(f"[dim]설정 스크립트 실행 중: {setup_script_path}...[/dim]")

    # 스크립트 내용 읽기
    script_content = script_path.read_text()

    # 로컬 환경을 사용하여 ${VAR} 구문 확장
    template = string.Template(script_content)
    expanded_script = template.safe_substitute(os.environ)

    # 5분 타임아웃으로 샌드박스에서 실행
    result = backend.execute(f"bash -c {shlex.quote(expanded_script)}")

    if result.exit_code != 0:
        console.print(f"[red]❌ 설정 스크립트 실패 (종료 코드 {result.exit_code}):[/red]")
        console.print(f"[dim]{result.output}[/dim]")
        msg = "설정 실패 - 중단됨"
        raise RuntimeError(msg)

    console.print("[green]✓ 설정 완료[/green]")


@contextmanager
def create_modal_sandbox(
    *, sandbox_id: str | None = None, setup_script_path: str | None = None
) -> Generator[SandboxBackendProtocol, None, None]:
    """Modal 샌드박스를 생성하거나 연결합니다.

    Args:
        sandbox_id: 재사용할 기존 샌드박스 ID (선택 사항)
        setup_script_path: 샌드박스 시작 후 실행할 설정 스크립트 경로 (선택 사항)

    Yields:
        (ModalBackend, sandbox_id)

    Raises:
        ImportError: Modal SDK가 설치되지 않음
        Exception: 샌드박스 생성/연결 실패
        FileNotFoundError: 설정 스크립트를 찾을 수 없음
        RuntimeError: 설정 스크립트 실패
    """
    import modal

    from deepagents_cli.integrations.modal import ModalBackend

    console.print("[yellow]Modal 샌드박스 시작 중...[/yellow]")

    # 임시 앱 생성 (종료 시 자동 정리)
    app = modal.App("deepagents-sandbox")

    with app.run():
        if sandbox_id:
            sandbox = modal.Sandbox.from_id(sandbox_id=sandbox_id, app=app)
            should_cleanup = False
        else:
            sandbox = modal.Sandbox.create(app=app, workdir="/workspace")
            should_cleanup = True

            # 실행될 때까지 폴링 (Modal에서 필요)
            for _ in range(90):  # 180초 타임아웃 (90 * 2초)
                if sandbox.poll() is not None:  # 샌드박스가 예기치 않게 종료됨
                    msg = "시작 중 Modal 샌드박스가 예기치 않게 종료되었습니다"
                    raise RuntimeError(msg)
                # 간단한 명령을 시도하여 샌드박스가 준비되었는지 확인
                try:
                    process = sandbox.exec("echo", "ready", timeout=5)
                    process.wait()
                    if process.returncode == 0:
                        break
                except Exception:
                    pass
                time.sleep(2)
            else:
                # 타임아웃 - 정리 및 실패 처리
                sandbox.terminate()
                msg = "180초 이내에 Modal 샌드박스를 시작하지 못했습니다"
                raise RuntimeError(msg)

        backend = ModalBackend(sandbox)
        console.print(f"[green]✓ Modal 샌드박스 준비 완료: {backend.id}[/green]")

        # 설정 스크립트가 제공된 경우 실행
        if setup_script_path:
            _run_sandbox_setup(backend, setup_script_path)
        try:
            yield backend
        finally:
            if should_cleanup:
                try:
                    console.print(f"[dim]Modal 샌드박스 {sandbox_id} 종료 중...[/dim]")
                    sandbox.terminate()
                    console.print(f"[dim]✓ Modal 샌드박스 {sandbox_id} 종료됨[/dim]")
                except Exception as e:
                    console.print(f"[yellow]⚠ 정리 실패: {e}[/yellow]")


@contextmanager
def create_runloop_sandbox(
    *, sandbox_id: str | None = None, setup_script_path: str | None = None
) -> Generator[SandboxBackendProtocol, None, None]:
    """Runloop devbox를 생성하거나 연결합니다.

    Args:
        sandbox_id: 재사용할 기존 devbox ID (선택 사항)
        setup_script_path: 샌드박스 시작 후 실행할 설정 스크립트 경로 (선택 사항)

    Yields:
        (RunloopBackend, devbox_id)

    Raises:
        ImportError: Runloop SDK가 설치되지 않음
        ValueError: RUNLOOP_API_KEY가 설정되지 않음
        RuntimeError: 타임아웃 내에 devbox를 시작하지 못함
        FileNotFoundError: 설정 스크립트를 찾을 수 없음
        RuntimeError: 설정 스크립트 실패
    """
    from runloop_api_client import Runloop

    from deepagents_cli.integrations.runloop import RunloopBackend

    bearer_token = os.environ.get("RUNLOOP_API_KEY")
    if not bearer_token:
        msg = "RUNLOOP_API_KEY 환경 변수가 설정되지 않았습니다"
        raise ValueError(msg)

    client = Runloop(bearer_token=bearer_token)

    console.print("[yellow]Runloop devbox 시작 중...[/yellow]")

    if sandbox_id:
        devbox = client.devboxes.retrieve(id=sandbox_id)
        should_cleanup = False
    else:
        devbox = client.devboxes.create()
        sandbox_id = devbox.id
        should_cleanup = True

        # 실행될 때까지 폴링 (Runloop에서 필요)
        for _ in range(90):  # 180초 타임아웃 (90 * 2초)
            status = client.devboxes.retrieve(id=devbox.id)
            if status.status == "running":
                break
            time.sleep(2)
        else:
            # 타임아웃 - 정리 및 실패 처리
            client.devboxes.shutdown(id=devbox.id)
            msg = "180초 이내에 devbox를 시작하지 못했습니다"
            raise RuntimeError(msg)

    console.print(f"[green]✓ Runloop devbox 준비 완료: {sandbox_id}[/green]")

    backend = RunloopBackend(devbox_id=devbox.id, client=client)

    # 설정 스크립트가 제공된 경우 실행
    if setup_script_path:
        _run_sandbox_setup(backend, setup_script_path)
    try:
        yield backend
    finally:
        if should_cleanup:
            try:
                console.print(f"[dim]Runloop devbox {sandbox_id} 종료 중...[/dim]")
                client.devboxes.shutdown(id=devbox.id)
                console.print(f"[dim]✓ Runloop devbox {sandbox_id} 종료됨[/dim]")
            except Exception as e:
                console.print(f"[yellow]⚠ 정리 실패: {e}[/yellow]")


@contextmanager
def create_daytona_sandbox(
    *, sandbox_id: str | None = None, setup_script_path: str | None = None
) -> Generator[SandboxBackendProtocol, None, None]:
    """Daytona 샌드박스를 생성합니다.

    Args:
        sandbox_id: 재사용할 기존 샌드박스 ID (선택 사항)
        setup_script_path: 샌드박스 시작 후 실행할 설정 스크립트 경로 (선택 사항)

    Yields:
        (DaytonaBackend, sandbox_id)

    Note:
        ID로 기존 Daytona 샌드박스에 연결하는 기능은 아직 지원되지 않을 수 있습니다.
        sandbox_id가 제공되면 NotImplementedError가 발생합니다.
    """
    from daytona import Daytona, DaytonaConfig

    from deepagents_cli.integrations.daytona import DaytonaBackend

    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        msg = "DAYTONA_API_KEY 환경 변수가 설정되지 않았습니다"
        raise ValueError(msg)

    if sandbox_id:
        msg = (
            "ID로 기존 Daytona 샌드박스에 연결하는 기능은 아직 지원되지 않습니다. "
            "--sandbox-id를 생략하여 새 샌드박스를 생성하십시오."
        )
        raise NotImplementedError(msg)

    console.print("[yellow]Daytona 샌드박스 시작 중...[/yellow]")

    daytona = Daytona(DaytonaConfig(api_key=api_key))
    sandbox = daytona.create()
    sandbox_id = sandbox.id

    # 실행될 때까지 폴링 (Daytona에서 필요)
    for _ in range(90):  # 180초 타임아웃 (90 * 2초)
        # 간단한 명령을 시도하여 샌드박스가 준비되었는지 확인
        try:
            result = sandbox.process.exec("echo ready", timeout=5)
            if result.exit_code == 0:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        try:
            # 가능한 경우 정리
            sandbox.delete()
        finally:
            msg = "180초 이내에 Daytona 샌드박스를 시작하지 못했습니다"
            raise RuntimeError(msg)

    backend = DaytonaBackend(sandbox)
    console.print(f"[green]✓ Daytona 샌드박스 준비 완료: {backend.id}[/green]")

    # 설정 스크립트가 제공된 경우 실행
    if setup_script_path:
        _run_sandbox_setup(backend, setup_script_path)
    try:
        yield backend
    finally:
        console.print(f"[dim]Daytona 샌드박스 {sandbox_id} 삭제 중...[/dim]")
        try:
            sandbox.delete()
            console.print(f"[dim]✓ Daytona 샌드박스 {sandbox_id} 종료됨[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠ 정리 실패: {e}[/yellow]")


# 공급자별 작업 디렉토리 매핑
_PROVIDER_TO_WORKING_DIR = {
    "modal": "/workspace",
    "runloop": "/home/user",
    "daytona": "/home/daytona",
}


# 샌드박스 유형과 해당 컨텍스트 매니저 팩토리 매핑
_SANDBOX_PROVIDERS = {
    "modal": create_modal_sandbox,
    "runloop": create_runloop_sandbox,
    "daytona": create_daytona_sandbox,
}


@contextmanager
def create_sandbox(
    provider: str,
    *,
    sandbox_id: str | None = None,
    setup_script_path: str | None = None,
) -> Generator[SandboxBackendProtocol, None, None]:
    """지정된 공급자의 샌드박스를 생성하거나 연결합니다.

    이것은 적절한 공급자별 컨텍스트 매니저에 위임하는 샌드박스 생성을 위한 통합 인터페이스입니다.

    Args:
        provider: 샌드박스 공급자 ("modal", "runloop", "daytona")
        sandbox_id: 재사용할 기존 샌드박스 ID (선택 사항)
        setup_script_path: 샌드박스 시작 후 실행할 설정 스크립트 경로 (선택 사항)

    Yields:
        (SandboxBackend, sandbox_id)
    """
    if provider not in _SANDBOX_PROVIDERS:
        msg = f"알 수 없는 샌드박스 공급자: {provider}. 사용 가능한 공급자: {', '.join(get_available_sandbox_types())}"
        raise ValueError(msg)

    sandbox_provider = _SANDBOX_PROVIDERS[provider]

    with sandbox_provider(sandbox_id=sandbox_id, setup_script_path=setup_script_path) as backend:
        yield backend


def get_available_sandbox_types() -> list[str]:
    """사용 가능한 샌드박스 공급자 유형 목록을 가져옵니다.

    Returns:
        샌드박스 유형 이름 목록 (예: ["modal", "runloop", "daytona"])
    """
    return list(_SANDBOX_PROVIDERS.keys())


def get_default_working_dir(provider: str) -> str:
    """주어진 샌드박스 공급자의 기본 작업 디렉토리를 가져옵니다.

    Args:
        provider: 샌드박스 공급자 이름 ("modal", "runloop", "daytona")

    Returns:
        기본 작업 디렉토리 경로 (문자열)

    Raises:
        ValueError: 공급자를 알 수 없는 경우
    """
    if provider in _PROVIDER_TO_WORKING_DIR:
        return _PROVIDER_TO_WORKING_DIR[provider]
    msg = f"알 수 없는 샌드박스 공급자: {provider}"
    raise ValueError(msg)


__all__ = [
    "create_sandbox",
    "get_available_sandbox_types",
    "get_default_working_dir",
]
