"""Central enforcement helpers for tool security policy."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from anycode.types import ToolSecurityPolicy, ToolUseContext

_SHELL_CONTROL_TOKENS = frozenset({"&", "&&", "|", "||", ";", "<", ">", ">>", "<<"})


class ToolSecurityError(PermissionError):
    """Raised when a tool operation violates its configured security policy."""


def check_tool_access(tool_name: str, policy: ToolSecurityPolicy | None) -> None:
    """Reject a tool that is denied or absent from a non-empty allowlist."""
    if policy is None:
        return
    if tool_name in policy.denied_tools:
        raise ToolSecurityError(f'Tool "{tool_name}" is denied by the security policy.')
    if policy.allowed_tools and tool_name not in policy.allowed_tools:
        raise ToolSecurityError(f'Tool "{tool_name}" is not present in the security policy allowlist.')


def resolve_tool_path(path: str | None, context: ToolUseContext) -> Path:
    """Resolve a tool path and enforce configured workspace containment."""
    policy = context.security_policy
    if path is None:
        raw = Path(context.cwd or (policy.workspace_root if policy else None) or os.getcwd())
    else:
        raw = Path(path)

    if policy is None:
        return raw

    base = Path(context.cwd or policy.workspace_root or os.getcwd())
    candidate = raw if raw.is_absolute() else base / raw
    resolved = candidate.resolve(strict=False)

    configured_roots = tuple(root for root in (policy.workspace_root, *policy.allowed_path_roots) if root)
    if not configured_roots:
        return resolved

    roots = tuple(Path(root).resolve(strict=False) for root in configured_roots)
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise ToolSecurityError(f'Path "{path or resolved}" is outside the allowed workspace roots.')
    return resolved


def validate_shell_command(command: str, context: ToolUseContext) -> None:
    """Enforce shell disablement and a conservative executable allowlist."""
    policy = context.security_policy
    if policy is None:
        return
    if not policy.allow_shell:
        raise ToolSecurityError("Shell execution is disabled by the security policy.")
    if not policy.allowed_shell_commands:
        return

    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError as error:
        raise ToolSecurityError(f"Shell command could not be parsed safely: {error}") from error

    if not tokens:
        raise ToolSecurityError("Shell command is empty.")
    if any(token in _SHELL_CONTROL_TOKENS for token in tokens) or "`" in command or "$(" in command:
        raise ToolSecurityError("Shell control operators are not allowed with a command allowlist.")

    executable = Path(tokens[0]).name.casefold()
    allowed = {Path(item).name.casefold() for item in policy.allowed_shell_commands}
    if executable not in allowed:
        raise ToolSecurityError(f'Shell executable "{tokens[0]}" is not present in the security policy allowlist.')


def build_subprocess_environment(context: ToolUseContext) -> dict[str, str] | None:
    """Return a filtered child environment, or ``None`` to inherit the parent."""
    policy = context.security_policy
    if policy is None or policy.inherit_environment:
        return None
    return {name: os.environ[name] for name in policy.allowed_environment_variables if name in os.environ}
