"""SubAgents module for research_agent.

This module implements a Claude Code-inspired SubAgent system with:
- Multiple specialized agent types (researcher, explorer, synthesizer)
- SubAgent registry for dynamic agent management
- Type-based routing via the task tool

Architecture:
    Main Orchestrator Agent
        ├── researcher SubAgent (deep web research)
        ├── explorer SubAgent (fast codebase exploration)
        └── synthesizer SubAgent (research synthesis)

Usage:
    from research_agent.subagents import get_all_subagents

    agent = create_deep_agent(
        model=model,
        subagents=get_all_subagents(),
        ...
    )
"""

from research_agent.subagents.definitions import (
    EXPLORER_AGENT,
    RESEARCHER_AGENT,
    SYNTHESIZER_AGENT,
    get_all_subagents,
)
from research_agent.subagents.registry import SubAgentRegistry

__all__ = [
    "SubAgentRegistry",
    "RESEARCHER_AGENT",
    "EXPLORER_AGENT",
    "SYNTHESIZER_AGENT",
    "get_all_subagents",
]
