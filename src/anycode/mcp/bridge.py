"""MCP bridge — converts MCP tools to AnyCode ToolDefinitions."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, create_model

from anycode.constants import MCP_TOOL_PREFIX
from anycode.mcp.client import MCPClient
from anycode.tools.registry import ToolRegistry
from anycode.types import MCPToolInfo, ToolDefinition, ToolResult, ToolUseContext

logger = logging.getLogger(__name__)

# JSON Schema type → Python type mapping
_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
}


def _json_type_to_python(prop_schema: dict[str, Any]) -> type:
    """Convert a JSON Schema property definition to a Python type."""
    json_type = prop_schema.get("type", "string")

    if json_type == "array":
        return list
    elif json_type == "object":
        return dict
    elif isinstance(json_type, list):
        # Union type like ["string", "null"]
        non_null = [t for t in json_type if t != "null"]
        if non_null:
            return _JSON_TYPE_MAP.get(non_null[0], str)
        return str

    return _JSON_TYPE_MAP.get(json_type, str)


def schema_to_pydantic_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Dynamically create a Pydantic model from a JSON Schema definition.

    Handles required/optional fields, basic types, arrays, and nested objects.
    """
    fields: dict[str, Any] = {}
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    for prop_name, prop_schema in properties.items():
        python_type = _json_type_to_python(prop_schema)
        is_required = prop_name in required_fields

        if is_required:
            fields[prop_name] = (python_type, ...)
        else:
            fields[prop_name] = (python_type | None, None)

    if not fields:
        fields["_placeholder"] = (str | None, None)

    return create_model(name, **fields)


def _build_tool_name(server_name: str, tool_name: str) -> str:
    """Build a prefixed tool name to avoid collisions: mcp_{server}_{tool}."""
    safe_server = server_name.replace("-", "_").replace(".", "_")
    safe_tool = tool_name.replace("-", "_").replace(".", "_")
    return f"{MCP_TOOL_PREFIX}_{safe_server}_{safe_tool}"


def mcp_tool_to_definition(
    server_name: str,
    tool_info: MCPToolInfo,
    client: MCPClient,
) -> ToolDefinition:
    """Convert an MCP tool into an AnyCode ToolDefinition.

    The resulting tool, when executed, calls the MCP server via the client.
    """
    prefixed_name = _build_tool_name(server_name, tool_info.name)
    input_model = schema_to_pydantic_model(
        f"MCP_{server_name}_{tool_info.name}_Input",
        tool_info.input_schema,
    )

    original_tool_name = tool_info.name
    bound_client = client

    async def _execute(validated_input: BaseModel, context: ToolUseContext) -> ToolResult:
        if not bound_client.is_connected:
            return ToolResult(
                data=f'MCP server "{server_name}" is disconnected — tool "{original_tool_name}" is unavailable.',
                is_error=True,
            )

        arguments = validated_input.model_dump(exclude_none=True)
        # Remove placeholder field if present
        arguments.pop("_placeholder", None)

        result = await bound_client.call_tool(original_tool_name, arguments)
        return ToolResult(
            data=result.get("content", ""),
            is_error=result.get("is_error", False),
        )

    return ToolDefinition(
        name=prefixed_name,
        description=f"[MCP:{server_name}] {tool_info.description}",
        input_model=input_model,
        execute=_execute,
    )


async def discover_and_register(
    client: MCPClient,
    server_name: str,
    registry: ToolRegistry,
) -> list[str]:
    """Discover tools from an MCP server and register them in the ToolRegistry.

    Returns the list of registered tool names.
    """
    tools = await client.discover_tools()
    registered: list[str] = []

    for tool_info in tools:
        try:
            definition = mcp_tool_to_definition(server_name, tool_info, client)
            registry.register(definition)
            registered.append(definition.name)
            logger.info("Registered MCP tool: %s", definition.name)
        except Exception as e:
            logger.warning("Failed to register MCP tool '%s' from server '%s': %s", tool_info.name, server_name, e)

    return registered
