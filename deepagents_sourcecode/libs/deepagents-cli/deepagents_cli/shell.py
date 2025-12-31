"""에이전트에 기본 셸 도구를 노출하는 단순화된 미들웨어."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import ToolException


class ShellMiddleware(AgentMiddleware[AgentState, Any]):
    """shell을 통해 에이전트에게 기본 셸 액세스 권한을 부여합니다.

    이 셸은 로컬 머신에서 실행되며 CLI 자체에서 제공하는 human-in-the-loop 안전장치 외에는
    어떠한 안전장치도 없습니다.
    """

    def __init__(
        self,
        *,
        workspace_root: str,
        timeout: float = 120.0,
        max_output_bytes: int = 100_000,
        env: dict[str, str] | None = None,
    ) -> None:
        """`ShellMiddleware`의 인스턴스를 초기화합니다.

        Args:
            workspace_root: 셸 명령을 위한 작업 디렉터리.
            timeout: 명령 완료를 기다리는 최대 시간(초).
                기본값은 120초입니다.
            max_output_bytes: 명령 출력에서 캡처할 최대 바이트 수.
                기본값은 100,000바이트입니다.
            env: 하위 프로세스에 전달할 환경 변수. None이면
                현재 프로세스의 환경을 사용합니다. 기본값은 None입니다.
        """
        super().__init__()
        self._timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._tool_name = "shell"
        self._env = env if env is not None else os.environ.copy()
        self._workspace_root = workspace_root

        # Build description with workspace info
        description = (
            f"Execute shell commands directly on the host. Commands run in this working directory: "
            f"{workspace_root}. Each command runs in a fresh shell environment with the "
            f"current process's environment variables. Commands may be truncated if they exceed "
            f"configured timeout or output limits."
        )

        @tool(self._tool_name, description=description)
        def shell_tool(
            command: str,
            runtime: ToolRuntime[None, AgentState],
        ) -> ToolMessage | str:
            """Execute a shell command.

            Args:
                command: The shell command to execute.
                runtime: The tool runtime context.
            """
            return self._run_shell_command(command, tool_call_id=runtime.tool_call_id)

        self._shell_tool = shell_tool
        self.tools = [self._shell_tool]

    def _run_shell_command(
        self,
        command: str,
        *,
        tool_call_id: str | None,
    ) -> ToolMessage | str:
        """셸 명령을 실행하고 결과를 반환합니다.

        Args:
            command: 실행할 셸 명령.
            tool_call_id: ToolMessage 생성을 위한 도구 호출 ID.

        Returns:
            명령 출력 또는 오류 메시지가 포함된 ToolMessage.
        """
        if not command or not isinstance(command, str):
            msg = "Shell 도구는 비어 있지 않은 명령 문자열을 필요로 합니다."
            raise ToolException(msg)

        try:
            result = subprocess.run(
                command,
                check=False,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=self._env,
                cwd=self._workspace_root,
            )

            # Combine stdout and stderr
            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                stderr_lines = result.stderr.strip().split("\n")
                for line in stderr_lines:
                    output_parts.append(f"[stderr] {line}")

            output = "\n".join(output_parts) if output_parts else "<no output>"

            # 필요한 경우 출력 자르기
            if len(output) > self._max_output_bytes:
                output = output[: self._max_output_bytes]
                output += f"\n\n... 출력이 {self._max_output_bytes}바이트에서 잘렸습니다."

            # 0이 아닌 경우 종료 코드 정보 추가
            if result.returncode != 0:
                output = f"{output.rstrip()}\n\n종료 코드: {result.returncode}"
                status = "error"
            else:
                status = "success"

        except subprocess.TimeoutExpired:
            output = f"오류: 명령이 {self._timeout:.1f}초 후에 시간 초과되었습니다."
            status = "error"

        return ToolMessage(
            content=output,
            tool_call_id=tool_call_id,
            name=self._tool_name,
            status=status,
        )


__all__ = ["ShellMiddleware"]
