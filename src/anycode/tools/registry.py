"""Tool declaration, registration, and schema conversion."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from anycode.types import LLMToolDef, ToolDefinition, ToolResult


def define_tool(
    *,
    name: str,
    description: str,
    input_model: type[BaseModel],
    execute: Callable[..., Awaitable[ToolResult]],
) -> ToolDefinition:
    """Create a typed tool definition."""
    return ToolDefinition(name=name, description=description, input_model=input_model, execute=execute)


class ToolRegistry:
    """Named store that converts entries to JSON Schema for LLM providers."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f'ToolRegistry: "{tool.name}" is already registered. Choose a unique name or remove the existing entry first.')
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def has(self, name: str) -> bool:
        return name in self._tools

    def deregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def register_from_mcp(self, tools: list[ToolDefinition]) -> None:
        """Batch-register MCP tools, skipping duplicates."""
        for tool in tools:
            if tool.name not in self._tools:
                self._tools[tool.name] = tool

    def deregister_prefix(self, prefix: str) -> int:
        """Remove all tools whose name starts with *prefix*. Returns count removed."""
        to_remove = [name for name in self._tools if name.startswith(prefix)]
        for name in to_remove:
            del self._tools[name]
        return len(to_remove)

    def to_tool_defs(self) -> list[LLMToolDef]:
        return [
            LLMToolDef(
                name=tool.name,
                description=tool.description,
                input_schema=_model_to_json_schema(tool.input_model),
            )
            for tool in self._tools.values()
        ]

    def to_llm_tools(self) -> list[dict[str, Any]]:
        """Anthropic-specific format with input_schema."""
        result = []
        for tool in self._tools.values():
            schema = _model_to_json_schema(tool.input_model)
            result.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        **({"required": schema["required"]} if "required" in schema else {}),
                    },
                }
            )
        return result


def _model_to_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to JSON Schema suitable for LLM tool definitions."""
    schema = model.model_json_schema()
    # Remove pydantic metadata fields not needed by LLMs
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema
