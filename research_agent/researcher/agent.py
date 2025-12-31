"""자율적 연구 에이전트 팩토리.

이 모듈은 자체 계획, 반성, 컨텍스트 관리 기능을 갖춘
독립적인 연구 DeepAgent를 생성합니다.
"""

from datetime import datetime

from deepagents import create_deep_agent
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from research_agent.researcher.prompts import AUTONOMOUS_RESEARCHER_INSTRUCTIONS
from research_agent.tools import tavily_search, think_tool


def create_researcher_agent(
    model: str | BaseChatModel | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
) -> CompiledStateGraph:
    """Create an autonomous researcher DeepAgent.

    This agent has its own:
    - Planning loop (write_todos via TodoListMiddleware)
    - Research loop (tavily_search + think_tool)
    - Context management (SummarizationMiddleware)
    - File access (FilesystemMiddleware) for intermediate results

    Essentially a "research SubGraph" that operates autonomously.

    Args:
        model: LLM to use. Defaults to gpt-4.1 with temperature=0.
        backend: Backend for file operations. If provided,
                 researcher can save intermediate results to filesystem.

    Returns:
        CompiledStateGraph: A fully autonomous research agent that can be
        used standalone or as a CompiledSubAgent in an orchestrator.

    Example:
        # Standalone usage
        researcher = create_researcher_agent()
        result = researcher.invoke({
            "messages": [HumanMessage("Research quantum computing trends")]
        })

        # As SubAgent in orchestrator
        subagent = get_researcher_subagent()
        orchestrator = create_deep_agent(subagents=[subagent, ...])
    """
    if model is None:
        model = ChatOpenAI(model="gpt-4.1", temperature=0.0)

    # Format prompt with current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    formatted_prompt = AUTONOMOUS_RESEARCHER_INSTRUCTIONS.format(date=current_date)

    return create_deep_agent(
        model=model,
        tools=[tavily_search, think_tool],
        system_prompt=formatted_prompt,
        backend=backend,
    )


def get_researcher_subagent(
    model: str | BaseChatModel | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
) -> dict:
    """Get researcher as a CompiledSubAgent for use in orchestrator.

    This function creates an autonomous researcher agent and wraps it
    in the CompiledSubAgent format expected by SubAgentMiddleware.

    Args:
        model: LLM to use. Defaults to gpt-4.1.
        backend: Backend for file operations.

    Returns:
        dict: CompiledSubAgent with keys:
            - name: "researcher"
            - description: Used by orchestrator to decide when to delegate
            - runnable: The autonomous researcher agent

    Example:
        from research_agent.researcher import get_researcher_subagent

        researcher = get_researcher_subagent(model=model, backend=backend)

        agent = create_deep_agent(
            model=model,
            subagents=[researcher, explorer, synthesizer],
            ...
        )
    """
    researcher = create_researcher_agent(model=model, backend=backend)

    return {
        "name": "researcher",
        "description": (
            "Autonomous deep research agent with self-planning and "
            "'breadth-first, depth-second' methodology. Use for comprehensive "
            "topic research requiring multiple search iterations and synthesis. "
            "The agent plans its own research phases, reflects after each search, "
            "and synthesizes findings into structured output. "
            "Best for: complex topics, multi-faceted questions, trend analysis."
        ),
        "runnable": researcher,
    }
