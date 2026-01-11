"""`python -m deepagents.cli` 형태로 CLI를 실행할 수 있게 하는 엔트리포인트입니다.

Allow running the CLI as: python -m deepagents.cli.
"""

from deepagents_cli.main import cli_main

if __name__ == "__main__":
    cli_main()
