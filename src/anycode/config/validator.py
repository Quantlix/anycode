"""Validate AnyCode config files and return human-readable errors."""

from __future__ import annotations

import os
from pathlib import Path

from anycode.config.loader import load_config
from anycode.security.redaction import safe_exception_message


def validate_config(path: str | os.PathLike[str]) -> list[str]:
    """Validate a config file. Returns a list of human-readable error messages.

    An empty list means the config is valid.
    """
    errors: list[str] = []
    try:
        loaded = load_config(path)
    except FileNotFoundError as e:
        return [safe_exception_message(e)]
    except Exception as e:
        return [f"Failed to parse config: {safe_exception_message(e)}"]

    agent_names = {a.name for a in loaded.team.agents}

    # Tool names referenced by agents are resolved at runtime (built-ins, MCP tools, and
    # user-defined tools registered via `ToolRegistry`). We cannot reliably check unknown
    # names here, so tool validation is intentionally skipped.
    #
    # Missing `system_prompt` is a recommendation (not a hard error) and is therefore not
    # reported by this validator. Use `anycode inspect team <path>` to surface advisories.

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
