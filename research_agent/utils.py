"""Jupyter 노트북에서 메시지와 프롬프트를 표시하기 위한 유틸리티 함수 모음."""

import json

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def _decode_escaped_unicode(s: str) -> str:
    """JSON/JS 스타일 유니코드 이스케이프를 사람이 읽을 수 있는 문자열로 변환한다.

    안전을 위해 문자열이 아닐 경우 그대로 반환한다.
    """
    if not isinstance(s, str):
        return s
    # 빠른 감지: '\\u' 또는 '\\U' 시퀀스가 있으면 디코드 시도
    if "\\u" in s or "\\U" in s or "\\\\u" in s or "\\\\U" in s:
        try:
            # 일부 JSON 덤프에서는 이스케이프가 이중화('\\\\u') 될 수 있으므로 먼저 정리
            s2 = s.replace("\\\\u", "\\u").replace("\\\\U", "\\U")
            return s2.encode("utf-8").decode("unicode_escape")
        except Exception:
            return s
    return s


def format_message_content(message):
    """메시지 내용을 화면에 표시 가능한 문자열로 변환한다."""
    parts = []
    tool_calls_processed = False

    # 기본 텍스트 콘텐츠 처리
    if isinstance(message.content, str):
        parts.append(_decode_escaped_unicode(message.content))
    elif isinstance(message.content, list):
        # 도구 호출 등이 포함된 복합 콘텐츠 처리 (Anthropic 포맷)
        for item in message.content:
            if item.get("type") == "text":
                parts.append(_decode_escaped_unicode(item["text"]))
            elif item.get("type") == "tool_use":
                parts.append(f"\nTool Call: {_decode_escaped_unicode(item['name'])}")
                parts.append(
                    _decode_escaped_unicode(
                        f"   Args: {json.dumps(item['input'], indent=2, ensure_ascii=False)}"
                    )
                )
                parts.append(f"   ID: {item.get('id', 'N/A')}")
                tool_calls_processed = True
    else:
        parts.append(str(message.content))

    # 메시지에 연결된 도구 호출 처리(OpenAI 포맷) - 아직 처리되지 않은 경우에만
    if (
        not tool_calls_processed
        and hasattr(message, "tool_calls")
        and message.tool_calls
    ):
        for tool_call in message.tool_calls:
            parts.append(f"\nTool Call: {_decode_escaped_unicode(tool_call['name'])}")
            parts.append(
                _decode_escaped_unicode(
                    f"   Args: {json.dumps(tool_call['args'], indent=2, ensure_ascii=False)}"
                )
            )
            parts.append(_decode_escaped_unicode(f"   ID: {tool_call['id']}"))

    return "\n".join(parts)


def format_messages(messages):
    """Rich 포매팅을 사용해 메시지 목록을 출력한다."""
    for m in messages:
        msg_type = m.__class__.__name__.replace("Message", "")
        content = format_message_content(m)

        if msg_type == "Human":
            console.print(Panel(content, title="Human", border_style="blue"))
        elif msg_type == "Ai":
            console.print(Panel(content, title="Assistant", border_style="green"))
        elif msg_type == "Tool":
            console.print(Panel(content, title="Tool Output", border_style="yellow"))
        else:
            console.print(Panel(content, title=f"{msg_type}", border_style="white"))


def format_message(messages):
    """하위 호환성을 위한 format_messages 별칭 함수."""
    return format_messages(messages)


def show_prompt(prompt_text: str, title: str = "Prompt", border_style: str = "blue"):
    """Rich 포매팅과 XML 태그 하이라이팅을 사용해 프롬프트를 표시한다.

    Args:
        prompt_text: 표시할 프롬프트 문자열
        title: 패널 제목 (기본값: "Prompt")
        border_style: 패널 테두리 색상 스타일 (기본값: "blue")
    """
    # 프롬프트를 보기 좋게 표시하기 위한 텍스트 객체 생성
    formatted_text = Text(prompt_text)
    formatted_text.highlight_regex(r"<[^>]+>", style="bold blue")  # XML 태그 하이라이트
    formatted_text.highlight_regex(
        r"##[^#\n]+", style="bold magenta"
    )  # 헤더 하이라이트
    formatted_text.highlight_regex(
        r"###[^#\n]+", style="bold cyan"
    )  # 서브 헤더 하이라이트

    # 패널 형태로 출력해 가독성을 높인다
    console.print(
        Panel(
            formatted_text,
            title=f"[bold green]{title}[/bold green]",
            border_style=border_style,
            padding=(1, 2),
        )
    )
