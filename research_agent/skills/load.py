"""Skill loader for parsing and loading agent skills from SKILL.md files.

This module implements the Anthropic Agent Skills pattern via YAML frontmatter parsing.
Each skill is a directory containing a SKILL.md file with:
- YAML frontmatter (name, description required)
- Markdown instructions for the agent
- Optional supporting files (scripts, configs, etc.)

Example SKILL.md structure:
```markdown
---
name: web-research
description: A structured approach to conducting thorough web research
---

# Web Research Skill

## When to Use
- When the user requests topic research
...
```

Adapted from deepagents-cli for use in research_agent project.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import NotRequired, TypedDict

import yaml

logger = logging.getLogger(__name__)

# Maximum file size for SKILL.md files (10MB) - DoS protection
MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024

# Agent Skills specification constraints (https://agentskills.io/specification)
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024


class SkillMetadata(TypedDict):
    """Skill metadata following Agent Skills specification."""

    name: str
    """Skill name (max 64 chars, lowercase alphanumeric and hyphens)."""

    description: str
    """Description of what the skill does (max 1024 chars)."""

    path: str
    """Path to the SKILL.md file."""

    source: str
    """Source of the skill ('user' or 'project')."""

    # Optional fields per Agent Skills specification
    license: NotRequired[str | None]
    """License name or reference to bundled license file."""

    compatibility: NotRequired[str | None]
    """Environment requirements (max 500 chars)."""

    metadata: NotRequired[dict[str, str] | None]
    """Arbitrary key-value mapping for additional metadata."""

    allowed_tools: NotRequired[str | None]
    """Space-separated list of pre-approved tools."""


def _is_safe_path(path: Path, base_dir: Path) -> bool:
    """Check if path is safely contained within base_dir.

    Prevents directory traversal attacks via symlinks or path manipulation.
    Resolves both paths to canonical form (following symlinks) and verifies
    the target path is within the base directory.

    Args:
        path: Path to validate
        base_dir: Base directory that should contain the path

    Returns:
        True if path is safely within base_dir, False otherwise
    """
    try:
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()
        resolved_path.relative_to(resolved_base)
        return True
    except ValueError:
        # Path is not a subdirectory of base_dir
        return False
    except (OSError, RuntimeError):
        # Error resolving path (e.g., circular symlinks)
        return False


def _validate_skill_name(name: str, directory_name: str) -> tuple[bool, str]:
    """Validate skill name per Agent Skills specification.

    Requirements:
    - Max 64 characters
    - Lowercase alphanumeric and hyphens only (a-z, 0-9, -)
    - Cannot start or end with hyphen
    - No consecutive hyphens
    - Must match parent directory name

    Args:
        name: Skill name from YAML frontmatter
        directory_name: Parent directory name

    Returns:
        (is_valid, error_message) tuple. Error message is empty if valid.
    """
    if not name:
        return False, "Name is required"
    if len(name) > MAX_SKILL_NAME_LENGTH:
        return False, "Name exceeds 64 characters"
    # Pattern: lowercase alphanumeric, single hyphens between segments
    if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", name):
        return False, "Name must use only lowercase alphanumeric and single hyphens"
    if name != directory_name:
        return False, f"Name '{name}' must match directory name '{directory_name}'"
    return True, ""


def _parse_skill_metadata(skill_md_path: Path, source: str) -> SkillMetadata | None:
    """Parse YAML frontmatter from SKILL.md file per Agent Skills specification.

    Args:
        skill_md_path: Path to SKILL.md file
        source: Skill source ('user' or 'project')

    Returns:
        SkillMetadata with all fields, or None if parsing fails
    """
    try:
        # Security: Check file size to prevent DoS
        file_size = skill_md_path.stat().st_size
        if file_size > MAX_SKILL_FILE_SIZE:
            logger.warning(
                "Skipping %s: file too large (%d bytes)", skill_md_path, file_size
            )
            return None

        content = skill_md_path.read_text(encoding="utf-8")

        # Match YAML frontmatter between --- delimiters
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if not match:
            logger.warning(
                "Skipping %s: no valid YAML frontmatter found", skill_md_path
            )
            return None

        frontmatter_str = match.group(1)

        # Parse YAML with safe_load for proper nested structure support
        try:
            frontmatter_data = yaml.safe_load(frontmatter_str)
        except yaml.YAMLError as e:
            logger.warning("Invalid YAML in %s: %s", skill_md_path, e)
            return None

        if not isinstance(frontmatter_data, dict):
            logger.warning("Skipping %s: frontmatter is not a mapping", skill_md_path)
            return None

        # Validate required fields
        name = frontmatter_data.get("name")
        description = frontmatter_data.get("description")

        if not name or not description:
            logger.warning(
                "Skipping %s: missing required 'name' or 'description'", skill_md_path
            )
            return None

        # Validate name format per spec (warn but load for backward compatibility)
        directory_name = skill_md_path.parent.name
        is_valid, error = _validate_skill_name(str(name), directory_name)
        if not is_valid:
            logger.warning(
                "Skill '%s' in %s does not follow Agent Skills spec: %s. "
                "Consider renaming for spec compliance.",
                name,
                skill_md_path,
                error,
            )

        # Validate description length (spec: max 1024 chars)
        description_str = str(description)
        if len(description_str) > MAX_SKILL_DESCRIPTION_LENGTH:
            logger.warning(
                "Description in %s exceeds %d chars, truncating",
                skill_md_path,
                MAX_SKILL_DESCRIPTION_LENGTH,
            )
            description_str = description_str[:MAX_SKILL_DESCRIPTION_LENGTH]

        return SkillMetadata(
            name=str(name),
            description=description_str,
            path=str(skill_md_path),
            source=source,
            license=frontmatter_data.get("license"),
            compatibility=frontmatter_data.get("compatibility"),
            metadata=frontmatter_data.get("metadata"),
            allowed_tools=frontmatter_data.get("allowed-tools"),
        )

    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Error reading %s: %s", skill_md_path, e)
        return None


def _list_skills_from_dir(skills_dir: Path, source: str) -> list[SkillMetadata]:
    """List all skills from a single skills directory (internal helper).

    Scans the skills directory for subdirectories containing SKILL.md files,
    parses YAML frontmatter, and returns skill metadata.

    Skills organization:
    skills/
    ├── skill-name/
    │   ├── SKILL.md        # Required: instructions with YAML frontmatter
    │   ├── script.py       # Optional: supporting files
    │   └── config.json     # Optional: supporting files

    Args:
        skills_dir: Path to skills directory
        source: Skill source ('user' or 'project')

    Returns:
        List of skill metadata dictionaries with name, description, path, and source
    """
    skills_dir = skills_dir.expanduser()
    if not skills_dir.exists():
        return []

    # Resolve base directory for security checks
    try:
        resolved_base = skills_dir.resolve()
    except (OSError, RuntimeError):
        return []

    skills: list[SkillMetadata] = []

    # Iterate over subdirectories
    for skill_dir in skills_dir.iterdir():
        # Security: catch symlinks pointing outside skills directory
        if not _is_safe_path(skill_dir, resolved_base):
            continue

        if not skill_dir.is_dir():
            continue

        # Look for SKILL.md file
        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            continue

        # Security: validate SKILL.md path before reading
        if not _is_safe_path(skill_md_path, resolved_base):
            continue

        # Parse metadata
        metadata = _parse_skill_metadata(skill_md_path, source=source)
        if metadata:
            skills.append(metadata)

    return skills


def list_skills(
    *,
    user_skills_dir: Path | None = None,
    project_skills_dir: Path | None = None,
) -> list[SkillMetadata]:
    """List skills from user and/or project directories.

    When both directories are provided, project skills with the same name
    as user skills will override the user skills.

    Args:
        user_skills_dir: Path to user-level skills directory
        project_skills_dir: Path to project-level skills directory

    Returns:
        Merged list of skill metadata from both sources, with project skills
        taking precedence over user skills when names conflict
    """
    all_skills: dict[str, SkillMetadata] = {}

    # Load user skills first (baseline)
    if user_skills_dir:
        user_skills = _list_skills_from_dir(user_skills_dir, source="user")
        for skill in user_skills:
            all_skills[skill["name"]] = skill

    # Load project skills second (override/extend)
    if project_skills_dir:
        project_skills = _list_skills_from_dir(project_skills_dir, source="project")
        for skill in project_skills:
            # Project skills override user skills with same name
            all_skills[skill["name"]] = skill

    return list(all_skills.values())
