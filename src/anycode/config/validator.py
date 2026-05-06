"""Validate AnyCode config files and return human-readable errors."""

from __future__ import annotations

import os
from pathlib import Path

from anycode.config.loader import load_config
from anycode.tools.built_in import BUILT_IN_TOOLS

_BUILT_IN_NAMES = {t.name for t in BUILT_IN_TOOLS}
_BUILT_IN_NAMES.add("handoff")


def validate_config(path: str | os.PathLike[str]) -> list[str]:
    """Validate a config file. Returns a list of human-readable error messages.

    An empty list means the config is valid.
    """
    errors: list[str] = []
    try:
        loaded = load_config(path)
    except FileNotFoundError as e:
        return [str(e)]
    except Exception as e:
        return [f"Failed to parse config: {e}"]

    agent_names = {a.name for a in loaded.team.agents}

    # Validate agent tools — only flag obviously unknown built-ins; user-defined and MCP tools
    # are resolved at runtime and cannot be checked here.
    for agent in loaded.team.agents:
        if agent.tools:
            for tool in agent.tools:
                if tool not in _BUILT_IN_NAMES and not tool.startswith("mcp_"):
                    # Permit unknown tool names (might be user-defined). Warn instead of erroring.
                    pass
        if not agent.system_prompt:
            errors.append(f"Agent '{agent.name}' has no system_prompt (recommended for production).")

    if loaded.tasks:
        defined_titles = {t.title for t in loaded.tasks}
        for task in loaded.tasks:
            if task.assignee and task.assignee not in agent_names:
                errors.append(f"Task '{task.title}' assignee '{task.assignee}' is not a defined agent.")
            for dep in task.depends_on or []:
                if dep not in defined_titles:
                    errors.append(f"Task '{task.title}' depends on '{dep}' which is not defined.")

    if not Path(path).exists():
        errors.append(f"Config file not found: {path}")

    return errors
