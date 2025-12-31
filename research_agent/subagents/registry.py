"""SubAgent Registry for managing specialized agent types.

This module provides a registry pattern inspired by Claude Code's subagent_type system,
allowing dynamic registration and lookup of SubAgent specifications.

The registry supports:
- Type-based agent lookup (by name)
- Capability-based filtering (by tags)
- Runtime agent discovery

Example:
    registry = SubAgentRegistry()
    registry.register(RESEARCHER_AGENT)
    registry.register(EXPLORER_AGENT)

    # Get specific agent
    researcher = registry.get("researcher")

    # Get all agents with "research" capability
    research_agents = registry.get_by_capability("research")
"""

from typing import Any, TypedDict

from typing_extensions import NotRequired


class SubAgentSpec(TypedDict):
    """Specification for a SubAgent.

    This follows the DeepAgents SubAgent TypedDict pattern with additional
    fields for capability-based routing.
    """

    name: str
    """Unique identifier for the SubAgent (used as subagent_type)."""

    description: str
    """Description shown to main agent for delegation decisions."""

    system_prompt: str
    """System prompt defining SubAgent behavior."""

    tools: list[Any]
    """Tools available to this SubAgent."""

    model: NotRequired[str]
    """Optional model override (defaults to parent's model)."""

    capabilities: NotRequired[list[str]]
    """Capability tags for filtering (e.g., ['research', 'web'])."""


class SubAgentRegistry:
    """Registry for managing SubAgent specifications.

    This class provides Claude Code-style SubAgent management with:
    - Registration and deregistration of agents
    - Name-based lookup (subagent_type matching)
    - Capability-based filtering

    Example:
        registry = SubAgentRegistry()

        # Register agents
        registry.register({
            "name": "researcher",
            "description": "Deep web research",
            "system_prompt": "...",
            "tools": [...],
            "capabilities": ["research", "web"],
        })

        # Lookup by name
        agent = registry.get("researcher")

        # Filter by capability
        web_agents = registry.get_by_capability("web")
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._agents: dict[str, SubAgentSpec] = {}

    def register(self, agent_spec: SubAgentSpec) -> None:
        """Register a SubAgent specification.

        Args:
            agent_spec: SubAgent specification dictionary.

        Raises:
            ValueError: If agent with same name already registered.
        """
        name = agent_spec["name"]
        if name in self._agents:
            msg = f"SubAgent '{name}' is already registered"
            raise ValueError(msg)
        self._agents[name] = agent_spec

    def unregister(self, name: str) -> None:
        """Remove a SubAgent from the registry.

        Args:
            name: Name of the SubAgent to remove.

        Raises:
            KeyError: If agent not found.
        """
        if name not in self._agents:
            msg = f"SubAgent '{name}' not found in registry"
            raise KeyError(msg)
        del self._agents[name]

    def get(self, name: str) -> SubAgentSpec | None:
        """Get a SubAgent specification by name.

        Args:
            name: SubAgent name (subagent_type).

        Returns:
            SubAgent specification if found, None otherwise.
        """
        return self._agents.get(name)

    def list_all(self) -> list[SubAgentSpec]:
        """List all registered SubAgent specifications.

        Returns:
            List of all SubAgent specs.
        """
        return list(self._agents.values())

    def list_names(self) -> list[str]:
        """List all registered SubAgent names.

        Returns:
            List of SubAgent names.
        """
        return list(self._agents.keys())

    def get_by_capability(self, capability: str) -> list[SubAgentSpec]:
        """Get SubAgents that have a specific capability.

        Args:
            capability: Capability tag to filter by.

        Returns:
            List of SubAgents with the specified capability.
        """
        return [
            agent
            for agent in self._agents.values()
            if capability in agent.get("capabilities", [])
        ]

    def get_descriptions(self) -> dict[str, str]:
        """Get a mapping of agent names to descriptions.

        Useful for displaying available agents to the main orchestrator.

        Returns:
            Dictionary mapping agent names to their descriptions.
        """
        return {name: agent["description"] for name, agent in self._agents.items()}

    def __contains__(self, name: str) -> bool:
        """Check if a SubAgent is registered.

        Args:
            name: SubAgent name to check.

        Returns:
            True if agent is registered.
        """
        return name in self._agents

    def __len__(self) -> int:
        """Get number of registered SubAgents."""
        return len(self._agents)
