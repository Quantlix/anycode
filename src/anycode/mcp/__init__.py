"""Model Context Protocol client and tool bridge.

Exports resolve lazily so the optional `mcp` SDK is imported only when used."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anycode._lazy import build_export_map, lazy_getattr

_EXPORTS = build_export_map(
    {
        "anycode.mcp.bridge": (
            "discover_and_register",
            "mcp_tool_to_definition",
            "schema_to_pydantic_model",
        ),
        "anycode.mcp.client": ("MCPClient",),
        "anycode.mcp.config": ("validate_server_config",),
    },
)


def __getattr__(name: str) -> Any:
    return lazy_getattr(__name__, name, _EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from anycode.mcp.bridge import discover_and_register, mcp_tool_to_definition, schema_to_pydantic_model
    from anycode.mcp.client import MCPClient
    from anycode.mcp.config import validate_server_config

__all__ = [
    "MCPClient",
    "discover_and_register",
    "mcp_tool_to_definition",
    "schema_to_pydantic_model",
    "validate_server_config",
]
