"""SubAgent definitions for research_agent.

This module defines specialized SubAgent specifications following the
Claude Code subagent_type pattern. Each SubAgent has:
- Unique name (used as subagent_type)
- Clear description for delegation decisions
- Specialized system prompt
- Curated tool set
- Optional model override

SubAgent Types:
    researcher: Deep web research with reflection
    explorer: Fast read-only codebase/document exploration
    synthesizer: Research synthesis and report generation
"""

from datetime import datetime

from research_agent.prompts import (
    EXPLORER_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SYNTHESIZER_INSTRUCTIONS,
)

# Current date for dynamic prompts
_current_date = datetime.now().strftime("%Y-%m-%d")


# =============================================================================
# EXPLORER SubAgent
# =============================================================================

EXPLORER_AGENT = {
    "name": "explorer",
    "description": "Fast read-only exploration of codebases and documents. Use for finding files, searching patterns, and quick information retrieval. Cannot modify files.",
    "system_prompt": EXPLORER_INSTRUCTIONS,
    "tools": [],  # Will be populated with read-only tools at runtime
    "capabilities": ["explore", "search", "read"],
}


# =============================================================================
# RESEARCHER SubAgent
# =============================================================================

RESEARCHER_AGENT = {
    "name": "researcher",
    "description": "Deep web research with reflection. Use for comprehensive topic research, gathering sources, and in-depth analysis. Includes tavily_search and think_tool.",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=_current_date),
    "tools": [],  # Will be populated with tavily_search, think_tool at runtime
    "capabilities": ["research", "web", "analysis"],
}


# =============================================================================
# SYNTHESIZER SubAgent
# =============================================================================

SYNTHESIZER_AGENT = {
    "name": "synthesizer",
    "description": "Synthesize multiple research findings into coherent reports. Use for combining sub-agent results, creating summaries, and writing final reports.",
    "system_prompt": SYNTHESIZER_INSTRUCTIONS,
    "tools": [],  # Will be populated with read_file, write_file, think_tool at runtime
    "capabilities": ["synthesize", "write", "analysis"],
}


# =============================================================================
# Utility Functions
# =============================================================================


def get_all_subagents() -> list[dict]:
    """Get all SubAgent definitions as a list.

    Returns:
        List of SubAgent specification dictionaries.

    Note:
        Tools are empty and should be populated at agent creation time
        based on available tools in the runtime.
    """
    return [
        RESEARCHER_AGENT,
        EXPLORER_AGENT,
        SYNTHESIZER_AGENT,
    ]


def get_subagent_by_name(name: str) -> dict | None:
    """Get a specific SubAgent definition by name.

    Args:
        name: SubAgent name (e.g., "researcher", "explorer", "synthesizer")

    Returns:
        SubAgent specification dict if found, None otherwise.
    """
    agents = {
        "researcher": RESEARCHER_AGENT,
        "explorer": EXPLORER_AGENT,
        "synthesizer": SYNTHESIZER_AGENT,
    }
    return agents.get(name)


def get_subagent_descriptions() -> str:
    """Get formatted descriptions of all SubAgents.

    Returns:
        Formatted string listing all SubAgents and their descriptions.
    """
    descriptions = []
    for agent in get_all_subagents():
        descriptions.append(f"- **{agent['name']}**: {agent['description']}")
    return "\n".join(descriptions)
