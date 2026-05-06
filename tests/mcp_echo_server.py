"""In-process MCP Echo Server for testing.

Provides deterministic tools, resources, and prompts for exercising
the full MCP lifecycle without relying on external services.

Run standalone:  uv run python tests/mcp_echo_server.py
"""

from __future__ import annotations

import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    Resource,
    TextContent,
    Tool,
)
from pydantic import AnyUrl

ECHO_SERVER_NAME = "echo-test"

server = Server(ECHO_SERVER_NAME)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_TOOLS: list[Tool] = [
    Tool(
        name="echo",
        description="Echoes back the provided message.",
        inputSchema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    ),
    Tool(
        name="add",
        description="Returns the sum of two numbers.",
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    ),
    Tool(
        name="complex_input",
        description="Accepts nested objects (tags array + metadata dict).",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object"},
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="slow_task",
        description="Waits for `seconds` before responding (for timeout tests).",
        inputSchema={
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
            "required": ["seconds"],
        },
    ),
    Tool(
        name="error_tool",
        description="Always returns an error result.",
        inputSchema={
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    ),
    Tool(
        name="multi_content",
        description="Returns multiple TextContent items.",
        inputSchema={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        },
    ),
]


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return _TOOLS


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    args = arguments or {}

    if name == "echo":
        return [TextContent(type="text", text=args.get("message", ""))]

    if name == "add":
        result = float(args.get("a", 0)) + float(args.get("b", 0))
        return [TextContent(type="text", text=str(result))]

    if name == "complex_input":
        return [TextContent(type="text", text=json.dumps(args, sort_keys=True))]

    if name == "slow_task":
        seconds = float(args.get("seconds", 1))
        await asyncio.sleep(seconds)
        return [TextContent(type="text", text=f"completed after {seconds}s")]

    if name == "error_tool":
        raise ValueError(f"Intentional error: {args.get('reason', 'no reason')}")

    if name == "multi_content":
        count = int(args.get("count", 2))
        return [TextContent(type="text", text=f"part-{i}") for i in range(count)]

    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

_RESOURCES: list[Resource] = [
    Resource(uri=AnyUrl("test://docs/readme"), name="readme", description="Project readme content", mimeType="text/plain"),
    Resource(uri=AnyUrl("test://data/config.json"), name="config", description="Config file", mimeType="application/json"),
]

_RESOURCE_DATA: dict[str, str] = {
    "test://docs/readme": "# Echo Server\nA test MCP server for integration testing.",
    "test://data/config.json": json.dumps({"version": "1.0", "debug": True}),
}


@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    return _RESOURCES


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    key = str(uri)
    if key in _RESOURCE_DATA:
        return _RESOURCE_DATA[key]
    raise ValueError(f"Resource not found: {uri}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PROMPTS: list[Prompt] = [
    Prompt(
        name="greeting",
        description="Generates a greeting message.",
        arguments=[
            PromptArgument(name="name", description="Name to greet", required=True),
        ],
    ),
    Prompt(
        name="summarize",
        description="Summarize a topic.",
        arguments=[
            PromptArgument(name="topic", description="Topic to summarize", required=True),
            PromptArgument(name="style", description="Summary style", required=False),
        ],
    ),
]


@server.list_prompts()
async def handle_list_prompts() -> list[Prompt]:
    return _PROMPTS


@server.get_prompt()
async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
    args = arguments or {}

    if name == "greeting":
        return GetPromptResult(
            description="A greeting message",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=f"Hello, {args.get('name', 'World')}!"),
                )
            ],
        )

    if name == "summarize":
        style = args.get("style", "concise")
        return GetPromptResult(
            description="A summary request",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=f"Summarize '{args.get('topic', '')}' in a {style} style."),
                )
            ],
        )

    raise ValueError(f"Unknown prompt: {name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        init_options = server.create_initialization_options()
        await server.run(read_stream, write_stream, init_options)


if __name__ == "__main__":
    asyncio.run(main())
