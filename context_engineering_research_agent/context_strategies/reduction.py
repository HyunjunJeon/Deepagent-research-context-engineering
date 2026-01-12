"""Context Reduction 전략 구현.

## 개요

Context Reduction은 컨텍스트 윈도우 사용량이 임계값을 초과할 때
자동으로 대화 내용을 압축하는 전략입니다.

## 두 가지 기법

### 1. Compaction (압축)
- 오래된 메시지에서 도구 호출(tool_calls)과 도구 결과(ToolMessage) 제거
- 텍스트 응답만 유지
- 세부 실행 이력은 제거하고 결론만 보존

### 2. Summarization (요약)
- 컨텍스트가 임계값(기본 85%) 초과 시 트리거
- LLM을 사용하여 대화 내용 요약
- 핵심 정보만 유지하고 세부사항 압축

## DeepAgents 구현

SummarizationMiddleware에서 구현:
- `context_threshold`: 요약 트리거 임계값 (기본 0.85 = 85%)
- `model_context_window`: 모델의 컨텍스트 윈도우 크기
- 자동으로 토큰 사용량 추적 및 요약 트리거

## 사용 예시

```python
from deepagents.middleware.summarization import SummarizationMiddleware

middleware = SummarizationMiddleware(
    context_threshold=0.85,  # 85% 사용 시 요약
    model_context_window=200000,  # Claude의 컨텍스트 윈도우
)
```
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.utils import AnyMessage


@dataclass
class ReductionConfig:
    """Context Reduction 설정."""

    context_threshold: float = 0.85
    """컨텍스트 사용률 임계값. 기본 0.85 (85%)."""

    model_context_window: int = 200000
    """모델의 전체 컨텍스트 윈도우 크기."""

    compaction_age_threshold: int = 10
    """Compaction 대상이 되는 메시지 나이 (메시지 수 기준)."""

    min_messages_to_keep: int = 5
    """요약 후 유지할 최소 메시지 수."""

    chars_per_token: int = 4
    """토큰당 문자 수 근사값."""


@dataclass
class ReductionResult:
    """Reduction 처리 결과."""

    was_reduced: bool
    """실제로 축소가 발생했는지 여부."""

    technique_used: str | None = None
    """사용된 기법 ('compaction' 또는 'summarization')."""

    original_message_count: int = 0
    """원본 메시지 수."""

    reduced_message_count: int = 0
    """축소 후 메시지 수."""

    estimated_tokens_saved: int = 0
    """절약된 추정 토큰 수."""


class ContextReductionStrategy(AgentMiddleware):
    """Context Reduction 전략 구현.

    컨텍스트 윈도우 사용량이 임계값을 초과할 때 자동으로
    대화 내용을 압축하는 전략입니다.

    ## 동작 원리

    1. before_model_call에서 현재 토큰 사용량 추정
    2. 임계값 초과 시 먼저 Compaction 시도
    3. 여전히 초과하면 Summarization 실행
    4. 축소된 메시지로 요청 수정

    ## Compaction vs Summarization

    - **Compaction**: 빠르고 저렴함. 도구 호출/결과만 제거.
    - **Summarization**: 느리고 비용 발생. LLM이 내용 요약.

    우선순위: Compaction → Summarization

    Args:
        config: Reduction 설정. None이면 기본값 사용.
        summarization_model: 요약에 사용할 LLM. None이면 요약 비활성화.
    """

    def __init__(
        self,
        config: ReductionConfig | None = None,
        summarization_model: BaseChatModel | None = None,
    ) -> None:
        self.config = config or ReductionConfig()
        self._summarization_model = summarization_model

    def _estimate_tokens(self, messages: list[BaseMessage]) -> int:
        """메시지 목록의 총 토큰 수를 추정합니다."""
        total_chars = sum(len(str(msg.content)) for msg in messages)
        return total_chars // self.config.chars_per_token

    def _get_context_usage_ratio(self, messages: list[BaseMessage]) -> float:
        """현재 컨텍스트 사용률을 계산합니다."""
        estimated_tokens = self._estimate_tokens(messages)
        return estimated_tokens / self.config.model_context_window

    def _should_reduce(self, messages: list[BaseMessage]) -> bool:
        """축소가 필요한지 판단합니다."""
        usage_ratio = self._get_context_usage_ratio(messages)
        return usage_ratio > self.config.context_threshold

    def apply_compaction(
        self,
        messages: list[BaseMessage],
    ) -> tuple[list[BaseMessage], ReductionResult]:
        """Compaction을 적용합니다.

        오래된 메시지에서 도구 호출과 도구 결과를 제거합니다.
        """
        original_count = len(messages)
        compacted: list[BaseMessage] = []

        for i, msg in enumerate(messages):
            age = len(messages) - i

            if age <= self.config.compaction_age_threshold:
                compacted.append(msg)
                continue

            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    text_content = (
                        msg.text if hasattr(msg, "text") else str(msg.content)
                    )
                    if text_content.strip():
                        compacted.append(AIMessage(content=text_content))
                else:
                    compacted.append(msg)
            elif isinstance(msg, (HumanMessage, SystemMessage)):
                compacted.append(msg)

        result = ReductionResult(
            was_reduced=len(compacted) < original_count,
            technique_used="compaction",
            original_message_count=original_count,
            reduced_message_count=len(compacted),
            estimated_tokens_saved=(
                self._estimate_tokens(messages) - self._estimate_tokens(compacted)
            ),
        )

        return compacted, result

    def apply_summarization(
        self,
        messages: list[BaseMessage],
    ) -> tuple[list[BaseMessage], ReductionResult]:
        """Summarization을 적용합니다.

        LLM을 사용하여 대화 내용을 요약합니다.
        """
        if self._summarization_model is None:
            return messages, ReductionResult(was_reduced=False)

        original_count = len(messages)

        keep_count = self.config.min_messages_to_keep
        messages_to_summarize = (
            messages[:-keep_count] if len(messages) > keep_count else []
        )
        recent_messages = (
            messages[-keep_count:] if len(messages) > keep_count else messages
        )

        if not messages_to_summarize:
            return messages, ReductionResult(was_reduced=False)

        summary_prompt = self._create_summary_prompt(messages_to_summarize)

        summary_response = self._summarization_model.invoke(
            [
                SystemMessage(
                    content="당신은 대화 요약 전문가입니다. 핵심 정보만 간결하게 요약하세요."
                ),
                HumanMessage(content=summary_prompt),
            ]
        )

        summary_message = SystemMessage(
            content=f"[이전 대화 요약]\n{summary_response.content}"
        )

        summarized = [summary_message] + list(recent_messages)

        result = ReductionResult(
            was_reduced=True,
            technique_used="summarization",
            original_message_count=original_count,
            reduced_message_count=len(summarized),
            estimated_tokens_saved=(
                self._estimate_tokens(messages) - self._estimate_tokens(summarized)
            ),
        )

        return summarized, result

    def _create_summary_prompt(self, messages: list[BaseMessage]) -> str:
        """요약을 위한 프롬프트를 생성합니다."""
        conversation_text = []
        for msg in messages:
            role = msg.__class__.__name__.replace("Message", "")
            content = str(msg.content)[:500]
            conversation_text.append(f"[{role}]: {content}")

        return f"""다음 대화를 요약해주세요. 핵심 정보, 결정사항, 중요한 컨텍스트만 포함하세요.

대화 내용:
{chr(10).join(conversation_text)}

요약 (한국어로, 500자 이내):"""

    def reduce_context(
        self,
        messages: list[BaseMessage],
    ) -> tuple[list[BaseMessage], ReductionResult]:
        """컨텍스트를 축소합니다.

        먼저 Compaction을 시도하고, 여전히 임계값을 초과하면
        Summarization을 적용합니다.
        """
        if not self._should_reduce(messages):
            return messages, ReductionResult(was_reduced=False)

        compacted, compaction_result = self.apply_compaction(messages)

        if not self._should_reduce(compacted):
            return compacted, compaction_result

        summarized, summarization_result = self.apply_summarization(compacted)

        return summarized, summarization_result

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """모델 호출을 래핑하여 필요시 컨텍스트를 축소합니다."""
        messages = cast(list[BaseMessage], request.messages)
        reduced_messages, result = self.reduce_context(messages)
        if result.was_reduced:
            request = request.override(
                messages=cast(list[AnyMessage], reduced_messages)
            )

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """비동기 모델 호출을 래핑합니다."""
        messages = cast(list[BaseMessage], request.messages)
        reduced_messages, result = self.reduce_context(messages)
        if result.was_reduced:
            request = request.override(
                messages=cast(list[AnyMessage], reduced_messages)
            )

        return await handler(request)


REDUCTION_SYSTEM_PROMPT = """## Context Reduction 안내

대화가 길어지면 자동으로 컨텍스트가 압축됩니다.

압축 방식:
1. **Compaction**: 오래된 도구 호출/결과 제거
2. **Summarization**: LLM이 이전 대화 요약

중요한 정보는 파일시스템에 저장하는 것을 권장합니다.
요약으로 인해 세부사항이 손실될 수 있습니다.
"""
