"""Middleware for providing subagents to an agent via a `task` tool."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, NotRequired, TypedDict, cast

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.types import Command


class SubAgent(TypedDict):
    """에이전트에 대한 사양(Specification)입니다.

    사용자 정의 에이전트를 지정할 때, `SubAgentMiddleware`의 `default_middleware`가
    먼저 적용되고, 그 다음에 이 사양에 지정된 `middleware`가 적용됩니다.
    기본값을 제외하고 사용자 정의 미들웨어만 사용하려면, `SubAgentMiddleware`에
    `default_middleware=[]`를 전달하십시오.
    """

    name: str
    """에이전트의 이름."""

    description: str
    """에이전트의 설명."""

    system_prompt: str
    """에이전트에 사용할 시스템 프롬프트."""

    tools: Sequence[BaseTool | Callable | dict[str, Any]]
    """에이전트에 사용할 도구들."""

    model: NotRequired[str | BaseChatModel]
    """에이전트의 모델. 기본값은 `default_model`입니다."""

    middleware: NotRequired[list[AgentMiddleware]]
    """`default_middleware` 뒤에 추가할 추가 미들웨어."""

    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]
    """에이전트에 사용할 도구 설정."""


class CompiledSubAgent(TypedDict):
    """미리 컴파일된 에이전트 사양."""

    name: str
    """에이전트의 이름."""

    description: str
    """에이전트의 설명."""

    runnable: Runnable
    """에이전트에 사용할 Runnable."""


DEFAULT_SUBAGENT_PROMPT = "사용자가 요청하는 목표를 완료하기 위해, 당신은 여러 표준 도구에 접근할 수 있습니다."

# State keys that are excluded when passing state to subagents and when returning
# updates from subagents.
# When returning updates:
# 1. The messages key is handled explicitly to ensure only the final message is included
# 2. The todos and structured_response keys are excluded as they do not have a defined reducer
#    and no clear meaning for returning them from a subagent to the main agent.
_EXCLUDED_STATE_KEYS = {"messages", "todos", "structured_response"}

TASK_TOOL_DESCRIPTION = """격리된 컨텍스트 창(isolated context windows)을 가진 복잡하고 다단계적인 독립 작업을 처리하기 위해 일회성(ephemeral) 서브 에이전트를 실행합니다.

사용 가능한 에이전트 유형과 그들이 접근할 수 있는 도구:
{available_agents}

Task 도구를 사용할 때는 subagent_type 매개변수를 지정하여 사용할 에이전트 유형을 선택해야 합니다.

## 사용 참고 사항:
1. 성능을 극대화하기 위해 가능한 경우 여러 에이전트를 동시에(concurrently) 실행하십시오. 이를 위해 다중 도구 사용(multiple tool uses)이 포함된 단일 메시지를 사용하십시오.
2. 에이전트가 완료되면 단일 메시지를 반환합니다. 에이전트가 반환한 결과는 사용자에게 보이지 않습니다. 사용자에게 결과를 보여주려면 결과에 대한 간결한 요약이 담긴 텍스트 메시지를 사용자에게 보내야 합니다.
3. 각 에이전트 호출은 상태비저장(stateless)입니다. 서브 에이전트에게 추가 메시지를 보낼 수 없으며, 서브 에이전트도 최종 보고서 이외에는 당신과 통신할 수 없습니다. 따라서 프롬프트에는 에이전트가 자율적으로 수행해야 할 작업에 대한 매우 자세한 설명이 포함되어야 하며, 에이전트가 최종적이고 유일한 메시지로 어떤 정보를 반환해야 하는지 정확히 지정해야 합니다.
4. 에이전트의 출력은 일반적으로 신뢰할 수 있어야 합니다.
5. 에이전트는 사용자의 의도를 알지 못하므로 콘텐츠 생성, 분석 수행, 또는 단순 연구(검색, 파일 읽기, 웹 가져오기 등) 중 무엇을 수행해야 하는지 명확하게 알려주십시오.
6. 에이전트 설명에 선제적으로(proactively) 사용해야 한다고 언급되어 있다면, 사용자가 먼저 요청하지 않아도 최선을 다해 사용해 보십시오. 판단력을 발휘하십시오.
7. 범용(general-purpose) 에이전트만 제공되는 경우 모든 작업에 해당 에이전트를 사용해야 합니다. 메인 에이전트와 동일한 모든 기능을 갖추고 있으므로, 컨텍스트와 토큰 사용을 격리하고 특정하고 복잡한 작업을 완료하는 데 매우 적합합니다.

### 범용 에이전트 사용 예시:

<example_agent_descriptions>
"general-purpose": use this agent for general purpose tasks, it has access to all tools as the main agent.
</example_agent_descriptions>

<example>
User: "I want to conduct research on the accomplishments of Lebron James, Michael Jordan, and Kobe Bryant, and then compare them."
Assistant: *Uses the task tool in parallel to conduct isolated research on each of the three players*
Assistant: *Synthesizes the results of the three isolated research tasks and responds to the User*
<commentary>
연구는 그 자체로 복잡하고 다단계적인 작업입니다.
각 개별 선수의 연구는 다른 선수의 연구에 의존하지 않습니다.
어시스턴트는 task 도구를 사용하여 복잡한 목표를 세 가지 독립적인 작업으로 나눕니다.
각 연구 작업은 한 선수에 대한 컨텍스트와 토큰만 신경 쓰면 되며, 도구 결과로 각 선수에 대한 종합된 정보를 반환합니다.
이는 각 연구 작업이 각 선수를 깊이 있게 연구하는 데 토큰과 컨텍스트를 사용할 수 있음을 의미하며, 최종 결과는 종합된 정보이므로 선수들을 서로 비교할 때 장기적으로 토큰을 절약할 수 있습니다.
</commentary>
</example>

<example>
User: "Analyze a single large code repository for security vulnerabilities and generate a report."
Assistant: *Launches a single `task` subagent for the repository analysis*
Assistant: *Receives report and integrates results into final summary*
<commentary>
서브 에이전트는 단 하나라도 크고 컨텍스트가 많은 작업을 격리하는 데 사용됩니다. 이는 메인 스레드가 세부 사항으로 과부하되는 것을 방지합니다.
사용자가 후속 질문을 하면 분석 및 도구 호출의 전체 기록 대신 참조할 간결한 보고서가 있으므로 시간과 비용을 절약할 수 있습니다.
</commentary>
</example>

<example>
User: "Schedule two meetings for me and prepare agendas for each."
Assistant: *Calls the task tool in parallel to launch two `task` subagents (one per meeting) to prepare agendas*
Assistant: *Returns final schedules and agendas*
<commentary>
작업은 개별적으로는 간단하지만, 서브 에이전트는 의제 준비를 격리하는 데 도움이 됩니다.
각 서브 에이전트는 한 회의의 의제만 신경 쓰면 됩니다.
</commentary>
</example>

<example>
User: "I want to order a pizza from Dominos, order a burger from McDonald's, and order a salad from Subway."
Assistant: *Calls tools directly in parallel to order a pizza from Dominos, a burger from McDonald's, and a salad from Subway*
<commentary>
목표가 매우 간단하고 명확하며 몇 가지 사소한 도구 호출만 필요하므로 어시스턴트는 task 도구를 사용하지 않았습니다.
작업을 직접 완료하고 `task` 도구를 사용하지 않는 것이 더 좋습니다.
</commentary>
</example>

### Example usage with custom agents:

<example_agent_descriptions>
"content-reviewer": use this agent after you are done creating significant content or documents
"greeting-responder": use this agent when to respond to user greetings with a friendly joke
"research-analyst": use this agent to conduct thorough research on complex topics
</example_agent_description>

<example>
user: "Please write a function that checks if a number is prime"
assistant: Sure let me write a function that checks if a number is prime
assistant: First let me use the Write tool to write a function that checks if a number is prime
assistant: I'm going to use the Write tool to write the following code:
<code>
function isPrime(n) {{
  if (n <= 1) return false
  for (let i = 2; i * i <= n; i++) {{
    if (n % i === 0) return false
  }}
  return true
}}
</code>
<commentary>
상당한 콘텐츠가 생성되었고 작업이 완료되었으므로, 이제 content-reviewer 에이전트를 사용하여 작업을 검토합니다.
</commentary>
assistant: Now let me use the content-reviewer agent to review the code
assistant: Uses the Task tool to launch with the content-reviewer agent
</example>

<example>
user: "Can you help me research the environmental impact of different renewable energy sources and create a comprehensive report?"
<commentary>
이것은 철저한 분석을 수행하기 위해 research-analyst 에이전트를 사용하는 것이 도움이 되는 복잡한 연구 작업입니다.
</commentary>
assistant: I'll help you research the environmental impact of renewable energy sources. Let me use the research-analyst agent to conduct comprehensive research on this topic.
assistant: Uses the Task tool to launch with the research-analyst agent, providing detailed instructions about what research to conduct and what format the report should take
</example>

<example>
user: "Hello"
<commentary>
사용자가 인사를 하고 있으므로, greeting-responder 에이전트를 사용하여 친절한 농담으로 응답하십시오.
</commentary>
assistant: "I'm going to use the Task tool to launch with the greeting-responder agent"
</example>"""  # noqa: E501

TASK_SYSTEM_PROMPT = """## `task` (서브 에이전트 스포너(spawner))

당신은 격리된 작업을 처리하는 일회성 서브 에이전트를 실행하기 위한 `task` 도구에 접근할 수 있습니다. 이 에이전트들은 일회적(ephemeral)입니다 — 작업 기간 동안에만 존재하며 단일 결과를 반환합니다.

task 도구를 사용해야 하는 경우:
- 작업이 복잡하고 다단계적이며 완전히 격리하여 위임할 수 있는 경우
- 작업이 다른 작업과 독립적이며 병렬로 실행할 수 있는 경우
- 작업에 집중적인 추론이나 많은 토큰/컨텍스트 사용이 필요하여 오케스트레이터 스레드를 부풀릴(bloat) 수 있는 경우
- 샌드박싱이 신뢰성을 향상시키는 경우 (예: 코드 실행, 구조화된 검색, 데이터 포맷팅)
- 서브 에이전트의 중간 단계가 아니라 출력에만 관심이 있는 경우 (예: 많은 연구를 수행한 후 종합된 보고서를 반환하거나, 간결하고 관련성 있는 답변을 얻기 위해 일련의 계산 또는 조회를 수행하는 경우)

서브 에이전트 생명주기:
1. **생성(Spawn)** → 명확한 역할, 지침 및 예상 출력 제공
2. **실행(Run)** → 서브 에이전트가 자율적으로 작업 완료
3. **반환(Return)** → 서브 에이전트가 단일 구조화된 결과를 제공
4. **조정(Reconcile)** → 결과를 메인 스레드에 통합하거나 합성

task 도구를 사용하지 말아야 하는 경우:
- 서브 에이전트가 완료된 후 중간 추론이나 단계를 확인해야 하는 경우 (task 도구는 이를 숨깁니다)
- 작업이 사소한 경우 (몇 번의 도구 호출 또는 간단한 조회)
- 위임이 토큰 사용량, 복잡성 또는 컨텍스트 전환을 줄이지 않는 경우
- 분할이 이점 없이 지연 시간만 추가하는 경우

## 기억해야 할 중요한 Task 도구 사용 참고 사항
- 가능하면 수행하는 작업을 병렬화하십시오. 이는 도구 호출(tool_calls)과 작업(tasks) 모두에 해당합니다. 완료해야 할 독립적인 단계가 있을 때마다 - 도구 호출을 하거나 작업을 병렬로 시작(kick off)하여 더 빠르게 완료하십시오. 이는 사용자에게 매우 중요한 시간을 절약해 줍니다.
- 다중 파트 목표 내에서 독립적인 작업을 격리(silo)하려면 `task` 도구를 사용하는 것을 기억하십시오.
- 여러 단계가 걸리고 에이전트가 완료해야 하는 다른 작업과 독립적인 복잡한 작업이 있을 때마다 `task` 도구를 사용해야 합니다. 이 에이전트들은 매우 유능하고 효율적입니다."""  # noqa: E501


DEFAULT_GENERAL_PURPOSE_DESCRIPTION = "복잡한 질문 연구, 파일 및 콘텐츠 검색, 다중 단계 작업 실행을 위한 범용 에이전트입니다. 키워드나 파일을 검색할 때 처음 몇 번의 시도로 올바른 일치 항목을 찾을 수 있을지 확신이 서지 않는다면, 이 에이전트를 사용하여 검색을 수행하십시오. 이 에이전트는 메인 에이전트와 동일한 모든 도구에 접근할 수 있습니다."  # noqa: E501


def _get_subagents(
    *,
    default_model: str | BaseChatModel,
    default_tools: Sequence[BaseTool | Callable | dict[str, Any]],
    default_middleware: list[AgentMiddleware] | None,
    default_interrupt_on: dict[str, bool | InterruptOnConfig] | None,
    subagents: list[SubAgent | CompiledSubAgent],
    general_purpose_agent: bool,
) -> tuple[dict[str, Any], list[str]]:
    """사양(specifications)에서 서브 에이전트 인스턴스를 생성합니다.

    Args:
        default_model: 지정하지 않은 서브 에이전트를 위한 기본 모델.
        default_tools: 지정하지 않은 서브 에이전트를 위한 기본 도구.
        default_middleware: 모든 서브 에이전트에 적용할 미들웨어. `None`인 경우 기본 미들웨어가 적용되지 않습니다.
        default_interrupt_on: 기본 범용 서브 에이전트에 사용할 도구 설정입니다.
            이는 자체 도구 설정을 지정하지 않은 서브 에이전트에 대한 폴백(fallback)이기도 합니다.
        subagents: 에이전트 사양 또는 미리 컴파일된 에이전트 목록.
        general_purpose_agent: 범용 서브 에이전트 포함 여부.

    Returns:
        (agent_dict, description_list) 튜플. agent_dict는 에이전트 이름을 runnable 인스턴스에 매핑하고,
        description_list는 포맷된 설명을 포함합니다.
    """
    # Use empty list if None (no default middleware)
    default_subagent_middleware = default_middleware or []

    agents: dict[str, Any] = {}
    subagent_descriptions = []

    # Create general-purpose agent if enabled
    if general_purpose_agent:
        general_purpose_middleware = [*default_subagent_middleware]
        if default_interrupt_on:
            general_purpose_middleware.append(HumanInTheLoopMiddleware(interrupt_on=default_interrupt_on))
        general_purpose_subagent = create_agent(
            default_model,
            system_prompt=DEFAULT_SUBAGENT_PROMPT,
            tools=default_tools,
            middleware=general_purpose_middleware,
        )
        agents["general-purpose"] = general_purpose_subagent
        subagent_descriptions.append(f"- general-purpose: {DEFAULT_GENERAL_PURPOSE_DESCRIPTION}")

    # Process custom subagents
    for agent_ in subagents:
        subagent_descriptions.append(f"- {agent_['name']}: {agent_['description']}")
        if "runnable" in agent_:
            custom_agent = cast("CompiledSubAgent", agent_)
            agents[custom_agent["name"]] = custom_agent["runnable"]
            continue
        _tools = agent_.get("tools", list(default_tools))

        subagent_model = agent_.get("model", default_model)

        _middleware = (
            [*default_subagent_middleware, *agent_["middleware"]]
            if "middleware" in agent_
            else [*default_subagent_middleware]
        )

        interrupt_on = agent_.get("interrupt_on", default_interrupt_on)
        if interrupt_on:
            _middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

        agents[agent_["name"]] = create_agent(
            subagent_model,
            system_prompt=agent_["system_prompt"],
            tools=_tools,
            middleware=_middleware,
        )
    return agents, subagent_descriptions


def _create_task_tool(
    *,
    default_model: str | BaseChatModel,
    default_tools: Sequence[BaseTool | Callable | dict[str, Any]],
    default_middleware: list[AgentMiddleware] | None,
    default_interrupt_on: dict[str, bool | InterruptOnConfig] | None,
    subagents: list[SubAgent | CompiledSubAgent],
    general_purpose_agent: bool,
    task_description: str | None = None,
) -> BaseTool:
    """서브 에이전트를 호출하기 위한 task 도구를 생성합니다.

    Args:
        default_model: 서브 에이전트용 기본 모델.
        default_tools: 서브 에이전트용 기본 도구.
        default_middleware: 모든 서브 에이전트에 적용할 미들웨어.
        default_interrupt_on: 기본 범용 서브 에이전트에 사용할 도구 설정입니다.
            이는 자체 도구 설정을 지정하지 않은 서브 에이전트에 대한 폴백이기도 합니다.
        subagents: 서브 에이전트 사양 목록.
        general_purpose_agent: 범용 에이전트 포함 여부.
        task_description: task 도구에 대한 사용자 정의 설명. `None`인 경우
            기본 템플릿을 사용합니다. `{available_agents}` 플레이스홀더를 지원합니다.

    Returns:
        유형별로 서브 에이전트를 호출할 수 있는 StructuredTool.
    """
    subagent_graphs, subagent_descriptions = _get_subagents(
        default_model=default_model,
        default_tools=default_tools,
        default_middleware=default_middleware,
        default_interrupt_on=default_interrupt_on,
        subagents=subagents,
        general_purpose_agent=general_purpose_agent,
    )
    subagent_description_str = "\n".join(subagent_descriptions)

    def _return_command_with_state_update(result: dict, tool_call_id: str) -> Command:
        state_update = {k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS}
        # Strip trailing whitespace to prevent API errors with Anthropic
        message_text = result["messages"][-1].text.rstrip() if result["messages"][-1].text else ""
        return Command(
            update={
                **state_update,
                "messages": [ToolMessage(message_text, tool_call_id=tool_call_id)],
            }
        )

    def _validate_and_prepare_state(
        subagent_type: str, description: str, runtime: ToolRuntime
    ) -> tuple[Runnable, dict]:
        """Prepare state for invocation."""
        subagent = subagent_graphs[subagent_type]
        # Create a new state dict to avoid mutating the original
        subagent_state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}
        subagent_state["messages"] = [HumanMessage(content=description)]
        return subagent, subagent_state

    # Use custom description if provided, otherwise use default template
    if task_description is None:
        task_description = TASK_TOOL_DESCRIPTION.format(available_agents=subagent_description_str)
    elif "{available_agents}" in task_description:
        # If custom description has placeholder, format with agent descriptions
        task_description = task_description.format(available_agents=subagent_description_str)

    def task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"{subagent_type} 서브 에이전트는 존재하지 않으므로 호출할 수 없습니다. 허용된 유형은 다음과 같습니다: {allowed_types}"
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
        result = subagent.invoke(subagent_state, runtime.config)
        if not runtime.tool_call_id:
            value_error_msg = "서브 에이전트 호출에는 도구 호출 ID가 필요합니다"
            raise ValueError(value_error_msg)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    async def atask(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"{subagent_type} 서브 에이전트는 존재하지 않으므로 호출할 수 없습니다. 허용된 유형은 다음과 같습니다: {allowed_types}"
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
        result = await subagent.ainvoke(subagent_state, runtime.config)
        if not runtime.tool_call_id:
            value_error_msg = "서브 에이전트 호출에는 도구 호출 ID가 필요합니다"
            raise ValueError(value_error_msg)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    return StructuredTool.from_function(
        name="task",
        func=task,
        coroutine=atask,
        description=task_description,
    )


class SubAgentMiddleware(AgentMiddleware):
    """`task` 도구를 통해 에이전트에게 서브 에이전트를 제공하기 위한 미들웨어.

    이 미들웨어는 서브 에이전트를 호출하는 데 사용할 수 있는 `task` 도구를 에이전트에 추가합니다.
    서브 에이전트는 여러 단계가 필요한 복잡한 작업이나 해결하기 위해 많은 컨텍스트가 필요한 작업을 처리하는 데 유용합니다.

    서브 에이전트의 주된 이점은 다중 단계 작업을 처리한 다음,
    깨끗하고 간결한 응답을 메인 에이전트에게 반환할 수 있다는 것입니다.

    서브 에이전트는 좁은 도구 집합과 집중이 필요한 다양한 전문 분야에도 적합합니다.

    이 미들웨어에는 격리된 컨텍스트에서 메인 에이전트와 동일한 작업을 처리하는 데 사용할 수 있는
    기본 범용 서브 에이전트가 함께 제공됩니다.

    Args:
        default_model: 서브 에이전트에 사용할 모델.
            LanguageModelLike 또는 init_chat_model을 위한 dict일 수 있습니다.
        default_tools: 기본 범용 서브 에이전트에 사용할 도구.
        default_middleware: 모든 서브 에이전트에 적용할 기본 미들웨어. `None`(기본값)인 경우
            기본 미들웨어가 적용되지 않습니다. 사용자 정의 미들웨어를 지정하려면 목록을 전달하십시오.
        default_interrupt_on: 기본 범용 서브 에이전트에 사용할 도구 설정입니다.
            이는 자체 도구 설정을 지정하지 않은 서브 에이전트에 대한 폴백이기도 합니다.
        subagents: 에이전트에 제공할 추가 서브 에이전트 목록.
        system_prompt: 전체 시스템 프롬프트 재정의. 제공된 경우 에이전트의
            시스템 프롬프트를 완전히 대체합니다.
        general_purpose_agent: 범용 에이전트 포함 여부. 기본값은 `True`입니다.
        task_description: task 도구에 대한 사용자 정의 설명. `None`인 경우
            기본 설명 템플릿을 사용합니다.

    Example:
        ```python
        from langchain.agents.middleware.subagents import SubAgentMiddleware
        from langchain.agents import create_agent

        # Basic usage with defaults (no default middleware)
        agent = create_agent(
            "openai:gpt-4o",
            middleware=[
                SubAgentMiddleware(
                    default_model="openai:gpt-4o",
                    subagents=[],
                )
            ],
        )

        # Add custom middleware to subagents
        agent = create_agent(
            "openai:gpt-4o",
            middleware=[
                SubAgentMiddleware(
                    default_model="openai:gpt-4o",
                    default_middleware=[TodoListMiddleware()],
                    subagents=[],
                )
            ],
        )
        ```
    """

    def __init__(
        self,
        *,
        default_model: str | BaseChatModel,
        default_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
        default_middleware: list[AgentMiddleware] | None = None,
        default_interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
        subagents: list[SubAgent | CompiledSubAgent] | None = None,
        system_prompt: str | None = TASK_SYSTEM_PROMPT,
        general_purpose_agent: bool = True,
        task_description: str | None = None,
    ) -> None:
        """SubAgentMiddleware를 초기화합니다."""
        super().__init__()
        self.system_prompt = system_prompt
        task_tool = _create_task_tool(
            default_model=default_model,
            default_tools=default_tools or [],
            default_middleware=default_middleware,
            default_interrupt_on=default_interrupt_on,
            subagents=subagents or [],
            general_purpose_agent=general_purpose_agent,
            task_description=task_description,
        )
        self.tools = [task_tool]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """시스템 프롬프트를 업데이트하여 서브 에이전트 사용 지침을 포함합니다."""
        if self.system_prompt is not None:
            system_prompt = (
                request.system_prompt + "\n\n" + self.system_prompt if request.system_prompt else self.system_prompt
            )
            return handler(request.override(system_prompt=system_prompt))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) 시스템 프롬프트를 업데이트하여 서브 에이전트 사용 지침을 포함합니다."""
        if self.system_prompt is not None:
            system_prompt = (
                request.system_prompt + "\n\n" + self.system_prompt if request.system_prompt else self.system_prompt
            )
            return await handler(request.override(system_prompt=system_prompt))
        return await handler(request)
