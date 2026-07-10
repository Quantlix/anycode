"""MCP server configuration validation."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from anycode.constants import MCP_TRANSPORT_SSE, MCP_TRANSPORT_STDIO, MCP_TRANSPORT_STREAMABLE_HTTP
from anycode.types import MCPServerConfig


def validate_server_config(config: MCPServerConfig) -> list[str]:
    """Validate an MCP server configuration, returning a list of error messages (empty = valid)."""
    errors: list[str] = []

    if not config.name:
        errors.append("MCP server config requires a non-empty 'name'.")

    if config.transport == MCP_TRANSPORT_STDIO:
        if not config.trust.allow_stdio:
            errors.append(f"MCP server '{config.name}': stdio transport is disabled by the trust policy.")
        if not config.command:
            errors.append(f"MCP server '{config.name}': stdio transport requires a 'command'.")
    elif config.transport in (MCP_TRANSPORT_SSE, MCP_TRANSPORT_STREAMABLE_HTTP):
        if not config.url:
            errors.append(f"MCP server '{config.name}': {config.transport} transport requires a 'url'.")
        else:
            errors.extend(_validate_http_target(config))
    else:
        errors.append(f"MCP server '{config.name}': unknown transport '{config.transport}'.")

    if config.timeout <= 0:
        errors.append(f"MCP server '{config.name}': timeout must be positive.")

    return errors


def _validate_http_target(config: MCPServerConfig) -> list[str]:
    assert config.url is not None
    parsed = urlparse(config.url)
    host = (parsed.hostname or "").casefold()
    errors: list[str] = []

    if parsed.scheme not in ("http", "https") or not host:
        return [f"MCP server '{config.name}': URL must be an absolute HTTP(S) URL."]
    if parsed.username is not None or parsed.password is not None:
        errors.append(f"MCP server '{config.name}': credentials must not be embedded in the URL.")

    allowed_hosts = {item.casefold() for item in config.trust.allowed_hosts}
    explicitly_allowed = host in allowed_hosts
    if allowed_hosts and not explicitly_allowed:
        errors.append(f"MCP server '{config.name}': host '{host}' is not present in the trust policy allowlist.")

    is_loopback = host == "localhost"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        is_loopback = address.is_loopback
        if (address.is_private or address.is_link_local) and not is_loopback and not config.trust.allow_private_networks and not explicitly_allowed:
            errors.append(f"MCP server '{config.name}': private-network target '{host}' is not allowed by the trust policy.")

    if parsed.scheme != "https" and not is_loopback and not config.trust.allow_insecure_http:
        errors.append(f"MCP server '{config.name}': remote HTTP transport requires HTTPS.")
    return errors
