"""Autonomous researcher subagent module.

This module provides a self-planning, self-reflecting research agent
that follows a "breadth-first, depth-second" methodology.

Usage:
    from research_agent.researcher import get_researcher_subagent

    researcher = get_researcher_subagent(model=model, backend=backend)
    # Returns CompiledSubAgent for use in create_deep_agent(subagents=[...])
"""

from research_agent.researcher.agent import (
    create_researcher_agent,
    get_researcher_subagent,
)
from research_agent.researcher.prompts import AUTONOMOUS_RESEARCHER_INSTRUCTIONS

__all__ = [
    "create_researcher_agent",
    "get_researcher_subagent",
    "AUTONOMOUS_RESEARCHER_INSTRUCTIONS",
]
