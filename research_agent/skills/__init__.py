"""Skills module for research_agent.

This module implements the Agent Skills pattern with progressive disclosure:
1. At session start, parse YAML frontmatter from SKILL.md files
2. Inject skill metadata (name + description) into system prompt
3. Agent reads full SKILL.md content when the skill is relevant

Public API:
- SkillsMiddleware: Middleware to integrate skills into agent execution
- list_skills: Load skill metadata from directories
- SkillMetadata: TypedDict for skill metadata structure
"""

from research_agent.skills.load import SkillMetadata, list_skills
from research_agent.skills.middleware import SkillsMiddleware

__all__ = [
    "SkillsMiddleware",
    "list_skills",
    "SkillMetadata",
]
