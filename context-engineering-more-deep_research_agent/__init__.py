"""Context Engineering 연구용 DeepAgent 모듈.

이 모듈은 DeepAgents 라이브러리의 Context Engineering 전략을
명시적으로 구현하고 문서화한 연구용 에이전트입니다.

## Context Engineering 5가지 핵심 전략

1. **Context Offloading (컨텍스트 오프로딩)**
   - 대용량 도구 결과를 파일시스템으로 축출
   - 메시지에는 파일 경로만 남기고 실제 데이터는 외부 저장
   - FilesystemMiddleware의 tool_token_limit_before_evict 파라미터로 제어
   - 기본값: 20,000 토큰 초과 시 자동 축출

2. **Context Reduction (컨텍스트 축소)**
   - Compaction: 오래된 메시지의 도구 호출/결과 제거
   - Summarization: 컨텍스트가 임계값 초과 시 대화 요약
   - SummarizationMiddleware: 85% 컨텍스트 사용 시 트리거
   - 핵심 정보만 유지하고 세부사항 압축

3. **Context Retrieval (컨텍스트 검색)**
   - grep/glob 기반의 단순하고 빠른 검색
   - 벡터 DB나 복잡한 인덱싱 없이 직접 파일 검색
   - 필요한 정보만 선택적으로 로드
   - FilesystemMiddleware의 read_file, grep, glob 도구

4. **Context Isolation (컨텍스트 격리)**
   - SubAgent를 통한 독립된 컨텍스트 윈도우
   - 메인 에이전트와 상태 비공유
   - 복잡한 하위 작업을 격리된 환경에서 처리
   - SubAgentMiddleware의 task() 도구

5. **Context Caching (컨텍스트 캐싱)**
   - Anthropic Prompt Caching으로 시스템 프롬프트 캐싱
   - KV Cache 효율화로 비용 절감
   - AnthropicPromptCachingMiddleware로 구현

## 모듈 구조

```
context-engineering-more-deep_research_agent/
├── __init__.py                 # 이 파일
├── agent.py                    # 메인 에이전트 (5가지 전략 통합)
├── prompts.py                  # 시스템 프롬프트
├── tools.py                    # 연구 도구
├── utils.py                    # 유틸리티
├── backends/                   # 백엔드 구현
│   ├── __init__.py
│   ├── pyodide_sandbox.py     # WASM 기반 안전한 Python 실행
│   └── docker_shared.py       # Docker 공유 작업공간
├── context_strategies/         # Context Engineering 전략
│   ├── __init__.py
│   ├── offloading.py          # 1. Context Offloading
│   ├── reduction.py           # 2. Context Reduction
│   ├── retrieval.py           # 3. Context Retrieval
│   ├── isolation.py           # 4. Context Isolation
│   └── caching.py             # 5. Context Caching
├── research/                   # 연구 에이전트
│   ├── __init__.py
│   ├── agent.py
│   └── prompts.py
└── skills/                     # 스킬 미들웨어
    └── middleware.py
```

## 사용 예시

```python
from context_engineering_research_agent import agent

# 에이전트 실행
result = agent.invoke({
    "messages": [{"role": "user", "content": "Context Engineering 전략 연구"}]
})
```

## 참고 자료

- DeepAgents 공식 문서: https://docs.langchain.com/oss/python/deepagents/overview
- Anthropic Prompt Caching: https://docs.anthropic.com/claude/docs/prompt-caching
- LangGraph: https://docs.langchain.com/oss/python/langgraph/overview
"""

# 버전 정보
__version__ = "0.1.0"
__author__ = "Context Engineering Research Team"

# 주요 컴포넌트 export
from context_engineering_more_deep_research_agent.agent import (
    agent,
    create_context_aware_agent,
)
from context_engineering_more_deep_research_agent.context_strategies import (
    ContextCachingStrategy,
    ContextIsolationStrategy,
    ContextOffloadingStrategy,
    ContextReductionStrategy,
    ContextRetrievalStrategy,
)

__all__ = [
    # 에이전트
    "agent",
    "create_context_aware_agent",
    # Context Engineering 전략
    "ContextOffloadingStrategy",
    "ContextReductionStrategy",
    "ContextRetrievalStrategy",
    "ContextIsolationStrategy",
    "ContextCachingStrategy",
]
