"""MCP server configuration validation."""

from __future__ import annotations

from anycode.constants import MCP_TRANSPORT_SSE, MCP_TRANSPORT_STDIO, MCP_TRANSPORT_STREAMABLE_HTTP
from anycode.types import MCPServerConfig


def validate_server_config(config: MCPServerConfig) -> list[str]:
    """Validate an MCP server configuration, returning a list of error messages (empty = valid)."""
    errors: list[str] = []

    if not config.name:
        errors.append("MCP server config requires a non-empty 'name'.")

    if config.transport == MCP_TRANSPORT_STDIO:
        if not config.command:
            errors.append(f"MCP server '{config.name}': stdio transport requires a 'command'.")
    elif config.transport in (MCP_TRANSPORT_SSE, MCP_TRANSPORT_STREAMABLE_HTTP):
        if not config.url:
            errors.append(f"MCP server '{config.name}': {config.transport} transport requires a 'url'.")
    else:
        errors.append(f"MCP server '{config.name}': unknown transport '{config.transport}'.")

    if config.timeout <= 0:
        errors.append(f"MCP server '{config.name}': timeout must be positive.")

    return errors
