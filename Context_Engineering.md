# Context Engineering

**Context Engineering(컨텍스트 엔지니어링)**을 “에이전트를 실제로 잘 동작하게 만드는 문맥 설계/운영 기술”로 정리합니다.

- YouTube: https://www.youtube.com/watch?v=6_BcCthVvb8
- PDF: `Context Engineering LangChain.pdf`
- PDF: `Manus Context Engineering LangChain Webinar.pdf`

---

## 1) Context Engineering이란?

YouTube 발표에서는 Context Engineering을 다음처럼 정의합니다.

- **“다음 스텝에 필요한 ‘딱 그 정보’만을 컨텍스트 윈도우에 채워 넣는 섬세한 기술(art)과 과학(science)”**  
  (영상 약 203초 부근: “the delicate art and science of filling the context window…”)

이 정의는 “좋은 프롬프트 한 줄”의 문제가 아니라, **에이전트가 수십~수백 턴을 거치며(tool call 포함) 컨텍스트가 폭발적으로 커지는 상황에서**:

- 무엇을 **남기고**
- 무엇을 **버리고**
- 무엇을 **외부로 빼고(오프로딩)**
- 무엇을 **필요할 때만 다시 가져오며(리트리벌)**
- 무엇을 **격리하고(아이솔레이션)**
- 무엇을 **캐시로 비용/지연을 줄이는지(캐싱)**

를 체계적으로 설계/운영하는 문제로 확장됩니다. (LangChain PDF: “Context grows w/ agents”, “Typical task… 50 tool calls”, “hundreds of turns”)

---

## 2) 왜 지금 “Context Engineering”인가?

### 2.1 에이전트에서 컨텍스트는 ‘성장’한다

LangChain PDF는 에이전트가 등장하면서 컨텍스트가 본질적으로 커진다고 강조합니다.

- 한 작업이 **많은 도구 호출(tool calls)**을 필요로 하고(예: Manus의 “around 50 tool calls”),
- 운영 환경에서는 **수백 턴**의 대화/관찰이 누적됩니다.

### 2.2 컨텍스트가 커질수록 성능이 떨어질 수 있다 (“context rot”)

LangChain PDF는 `context-rot` 자료를 인용하며 **컨텍스트 길이가 늘수록 성능이 하락할 수 있음**을 지적합니다.  
즉, “더 많이 넣으면 더 똑똑해진다”는 직관이 깨집니다.

### 2.3 컨텍스트 실패 모드(실패 유형)들이 반복 관측된다

LangChain PDF는 컨텍스트가 커질 때의 대표적인 실패를 4가지로 제시합니다.

- **Context Poisoning**: 잘못된 정보가 섞여 이후 의사결정/행동을 오염
- **Context Distraction**: 중요한 목표보다 반복/쉬운 패턴에 끌림(장기 컨텍스트에서 새 계획보다 반복 행동 선호 등)
- **Context Confusion**: 도구가 많고 비슷할수록 잘못된 도구 호출/비존재 도구 호출 등 혼란 증가
- **Context Clash**: 연속된 관찰/도구 결과가 서로 모순될 때 성능 하락

이 실패 모드들은 “프롬프트를 더 잘 쓰자”로는 해결이 어렵고, **컨텍스트의 구조·유지·정리·검증 전략**이 필요합니다.

### 2.4 Manus 관점: “모델을 건드리기보다 컨텍스트를 설계하라”

Manus PDF는 Context Engineering을 “앱과 모델 사이의 가장 실용적인 경계”라고 강조합니다.

- “Context Engineering is the clearest and most practical boundary between application and model.” (Manus PDF)

이 관점에서 Manus는 제품 개발 과정에서 흔히 빠지는 2가지 함정을 지적합니다.

- **The First Trap**: “차라리 우리 모델을 따로 학습(파인튜닝)하지?”라는 유혹  
  → 현실적으로 모델 반복 속도가 제품 반복 속도를 제한할 수 있고(PMF 이전에는 특히), 이 선택이 제품 개발을 늦출 수 있음
- **The Second Trap**: “액션 스페이스+리워드+롤아웃(RL)로 최적화하자”  
  → 하지만 이후 MCP 같은 확장 요구가 들어오면 다시 설계를 뒤엎게 될 수 있으니, 기반 모델 회사가 이미 잘하는 영역을 불필요하게 재구축하지 말 것

요약하면, “모델을 바꾸는 일”을 서두르기 전에 **컨텍스트를 어떻게 구성/축소/격리/리트리벌할지**를 먼저 해결하라는 메시지입니다.

---

## 3) Context Engineering의 5가지 핵심 레버(지렛대)

LangChain PDF(및 영상 전개)는 공통적으로 다음 5가지를 핵심 테마로 제시합니다.

### 3.1 Offload (컨텍스트 오프로딩)

**핵심 아이디어**: 모든 정보를 메시지 히스토리에 남길 필요가 없습니다. 토큰이 큰 결과는 **파일시스템/외부 저장소로 보내고**, 요약/참조(포인터)만 남깁니다.

- LangChain PDF: “Use file system for notes / todo.md / tok-heavy context / long-term memories”
- 영상: “you don't need all context to live in this messages history… offload it… it can be retrieved later”

**실무 패턴**
- 대용량 tool output은 파일로 저장하고, 대화에는 `저장 경로 + 요약 + 재로드 방법`만 남기기
- “작업 브리프/계획(todo)”를 파일로 유지해 긴 대화에서도 방향성을 잃지 않기

### 3.2 Reduce (컨텍스트 축소: Prune/Compaction/Summarization)

**핵심 아이디어**: 컨텍스트가 커질수록 성능/비용이 악화될 수 있으므로, “넣는 기술”뿐 아니라 “빼는 기술”이 필요합니다.

LangChain PDF는 “Summarize / prune message history”, “Summarize / prune tool call outputs”를 언급하면서도, 정보 손실 위험을 경고합니다.

Manus PDF는 **Compaction vs Summarization**을 별도 토픽으로 둡니다.

- **Compaction(압축/정리)**: 불필요한 원문을 제거하거나, 구조화된 형태로 재배치/정돈해 “같은 의미를 더 적은 토큰으로” 담기
- **Summarization(요약)**: 모델이 자연어 요약을 생성해 토큰을 줄이되, 정보 손실 가능

**실무 패턴**
- “도구 결과 원문”은 저장(Offload)하고 대화에는 “관찰(Observations) 요약”만 유지
- 일정 기준(예: 컨텍스트 사용량 임계치, 턴 수, 주기)마다 요약/정리 실행
- 요약은 “결정/근거/미해결 질문/다음 행동” 같은 스키마로 구조화(손실 최소화)

### 3.3 Retrieve (필요할 때만 가져오기)

**핵심 아이디어**: 오프로딩/축소로 비운 자리를 “아무거나로 채우지 말고”, **현재 스텝에 필요한 것만** 검색·리트리벌로 가져옵니다.

LangChain PDF는 “Mix of retrieval methods + re-ranking”, “Systems to assemble retrievals into prompts”, “Retrieve relevant tools based upon tool descriptions” 등 “검색 결과를 프롬프트에 조립하는 시스템”을 강조합니다.

**실무 패턴**
- 파일/노트/로그에서 grep/glob/키워드 기반 검색(결정적이고 디버그 가능)
- 리트리벌 결과는 “왜 가져왔는지(근거)”와 함께 삽입해 모델이 사용 이유를 이해하도록 구성
- 도구 설명도 리트리벌 대상: “필요한 도구만 로딩”하여 혼란(Context Confusion) 완화

### 3.4 Isolate (컨텍스트 격리)

**핵심 아이디어**: 한 컨텍스트에 모든 역할을 때려 넣으면 오염/충돌이 늘어납니다. 작업을 쪼개 **서브 에이전트(서브 컨텍스트)**로 격리합니다.

LangChain PDF는 multi-agent로 컨텍스트를 분리하되, 의사결정 충돌 위험을 경고합니다(“Multi-agents make conflicting decisions…”).  
Manus PDF는 “언어/동시성(concurrency)에서 배운 지혜”를 빌려오며, Go 블로그의 문구를 인용합니다.

- “Do not communicate by sharing memory; instead, share memory by communicating.” (Manus PDF)

즉, **공유 메모리(거대한 공용 컨텍스트)**로 동기화하려 하지 말고,
역할별 컨텍스트를 분리한 뒤 **명시적 메시지/산출물(요약, 브리프, 결과물)**로만 조율하라는 뜻입니다.

### 3.5 Cache (반복 컨텍스트 캐싱)

**핵심 아이디어**: 에이전트는 시스템 프롬프트/도구 설명/정책 같은 “상대적으로 불변(stable)”한 토큰이 반복됩니다. 이를 캐싱하면 비용과 지연을 크게 줄일 수 있습니다.

LangChain PDF:
- “Cache agent instructions, tool descriptions to prefix”
- “Add mutable context / recent observations to suffix”
- (특정 프로바이더 기준) “Cached input tokens … 10x cheaper!” 같은 비용 레버를 언급

**실무 패턴**
- 프롬프트를 **prefix(불변: 지침/도구/정책)** + **suffix(가변: 최근 관찰/상태)**로 분리해 캐시 효율 극대화
- 캐시 안정성을 해치지 않도록: 자주 변하는 내용을 system/prefix에 섞지 않기

---

## 4) “도구”도 컨텍스트를 더럽힌다: Tool Offloading과 계층적 액션 스페이스

Manus PDF는 오프로딩을 “메모리(파일)로만” 생각하면 부족하다고 말합니다.

- “tools themselves also clutter context”
- “Too many tools → confusion, invalid calls”

그래서 **도구 자체도 오프로딩/계층화**해야 한다고 제안합니다(“Offload tools through a hierarchical action space”).

Manus PDF가 제시하는 3단계 추상화(계층):

1. **Function Calling**  
   - 장점: 스키마 안전, 표준적  
   - 단점: 변경 시 캐시가 자주 깨짐, 도구가 많아지면 혼란 증가(Context Confusion)
2. **Sandbox Utilities (CLI/셸 유틸리티)**  
   - 모델 컨텍스트를 바꾸지 않고도 기능을 확장 가능  
   - 큰 출력은 파일로 저장시키기 쉬움(오프로딩과 결합)
3. **Packages & APIs (스크립트로 사전 승인된 API 호출)**  
   - 데이터가 크거나 연쇄 호출이 필요한 작업에 유리  
   - 목표: “Keep model context clean, use memory for reasoning only.”

요지는 “모든 것을 함수 목록으로 나열”하는 대신, **상위 레벨의 안정적인 인터페이스**를 제공해 컨텍스트를 작게 유지하고, 필요 시 하위 레벨로 내려가게 만드는 설계입니다.

---

## 5) 운영 관점 체크리스트 (실무 적용용)

아래는 위 소스들의 공통 테마를 “운영 가능한 체크리스트”로 정리한 것입니다.

1. **컨텍스트 예산(Budget) 정의**: 모델 윈도우/비용/지연을 고려해 “언제 줄일지” 임계치를 정한다.
2. **오프로딩 기본값**: 큰 tool output은 원문을 남기지 말고 파일/스토리지로 보내며, 포인터를 남긴다.
3. **축소 전략 계층화**: (가능하면) Compaction/Prune → Summarization 순으로 비용·손실을 제어한다.
4. **리트리벌 조립(Assembly)**: 검색은 끝이 아니라 시작이다. “무엇을, 왜, 어떤 순서로” 넣는 조립 로직이 필요하다.
5. **격리 기준 수립**: “오염될 수 있는 작업(탐색/긴 출력/다단계 분석)”을 서브 에이전트로 분리한다.
6. **캐시 친화 프롬프트**: prefix(불변) / suffix(가변)로 분리한다.
7. **실패 모드 방어**: Poisoning/Distraction/Confusion/Clash를 로그로 관측하고, 완화책(정리·검증·도구 로딩 제한·모순 검사)을 붙인다.
8. **과잉 엔지니어링 경계**: Manus PDF의 경고처럼 “더 많은 컨텍스트 ≠ 더 많은 지능”. 최대 성과는 종종 “추가”가 아니라 “제거”에서 나온다.

---

## 6) `Context_Engineering_Research.ipynb` 평가: 이 문서를 ‘잘 표현’하고 있는가?

### 결론(요약)

- **5대 전략(Offload/Reduce/Retrieve/Isolate/Cache)을 설명하는 데는 충분히 좋습니다.**
- 최근 업데이트로 **“왜 문제인지”, 실패 모드, Tool Offloading** 개요가 추가되어 설명력은 좋아졌지만, 이를 뒷받침하는 **재현 실험/운영 규칙**은 여전히 보완이 필요합니다.

### 잘 표현하는 부분

- **5가지 핵심 전략을 명시적으로 구조화**해 설명합니다(표 + 섹션 구성).
- Offloading/Reduction/Retrieval/Isolation/Caching을 각각:
  - “원리(트리거/효과)”
  - “구현(DeepAgents 미들웨어/설정)”
  - “간단한 시뮬레이션 실험”
  로 연결해, “개념 → 구현” 흐름이 명확합니다.

### 부족한 부분(이 문서 대비 갭)

- **실패 모드 4종(Poisoning/Distraction/Confusion/Clash)**을 “설명”하긴 하지만, 노트북의 **실험/설계에 직접 반영**(재현→완화 비교)되어 있지 않습니다.
- Manus PDF의 중요한 관점인 **“도구도 컨텍스트를 더럽힌다”**(Tool Offloading)도 개요는 추가되었지만, **도구 과다/유사 도구로 인한 혼란**을 줄이는 실험이 없습니다.
- Manus PDF의 메시지인 **“과잉 컨텍스트/과잉 설계 경계(‘Removing, not adding’)”**가 체크리스트/의사결정 가이드로 정리되어 있지 않습니다.

### “Context Engineering 설명”을 완성도 높게 만들기 위한 보완 제안(노트북 기준)

노트북을 “5가지 전략 데모”에서 “Context Engineering 설명서”로 끌어올리려면, 아래 4가지만 추가해도 체감이 큽니다.

1. (완료) **서론 강화(문제 정의)**: 컨텍스트 성장, context rot, 실패 모드 4종을 첫 섹션에서 명시.
2. (추가됨 - 시뮬레이션) **실패 모드별 미니 실험**: Confusion(도구 과다/유사 도구)·Clash(모순 tool output)·Distraction(장기 로그에서 반복행동)·Poisoning(검증되지 않은 사실) 재현/완화 시뮬레이션 추가.
3. (완료) **Tool Offloading/계층 설계 섹션**: “도구를 리트리벌로 로딩/제한”하거나 “상위 래퍼 도구로 단순화”하는 패턴 소개.
4. **‘삭제’ 중심의 운영 규칙**: 언제 넣고/빼고/격리할지 규칙(임계치, 주기, 스키마)과 로그 지표(토큰/비용/실패율) 추가.
