"""MCP (Model Context Protocol) integration for AnyCode."""

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
