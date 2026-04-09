# Demo 10 — MCP Tool Integration
# Execute: uv run python examples/10_mcp_tools.py
#
# Demonstrates:
#   1. Configuring MCP server connections (stdio and HTTP)
#   2. Dynamic Pydantic model generation from JSON Schema
#   3. Tool name prefixing to avoid collisions
#   4. MCPClient connection lifecycle and tool discovery
#   5. Converting MCP tools to AnyCode ToolDefinitions
#   6. Local echo server for guaranteed end-to-end testing
#
# Sections A-C are pure utilities (no server needed).
# Section D attempts a real MCP server connection (requires npx + an MCP server).
# Section E connects to the local echo server (always works).

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from anycode.mcp.bridge import _build_tool_name, mcp_tool_to_definition, schema_to_pydantic_model
from anycode.mcp.client import MCPClient
from anycode.mcp.config import validate_server_config
from anycode.tools.registry import ToolRegistry
from anycode.types import MCPServerConfig, MCPToolInfo

load_dotenv()

SEPARATOR = "-" * 60


async def main() -> None:
    print("=== MCP Integration Demo ===\n")

    # --- Section A: Configuration validation ---
    print(f"{SEPARATOR}")
    print("Section A: Configuration Validation")

    github_token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    valid_config = MCPServerConfig(
        name="github",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": github_token},
    )
    errors = validate_server_config(valid_config)
    print(f"  stdio config: {'VALID' if not errors else errors}")

    sse_config = MCPServerConfig(
        name="my-api",
        transport="sse",
        url="http://localhost:3000/sse",
    )
    errors = validate_server_config(sse_config)
    print(f"  SSE config:   {'VALID' if not errors else errors}")

    bad_config = MCPServerConfig(name="bad", transport="stdio")
    errors = validate_server_config(bad_config)
    print(f"  bad config:   {errors}")

    # --- Section B: Schema-to-Pydantic conversion ---
    print(f"\n{SEPARATOR}")
    print("Section B: Dynamic Pydantic Model Generation")

    schema = {
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
            "include_forks": {"type": "boolean"},
        },
        "required": ["query"],
    }
    model = schema_to_pydantic_model("SearchInput", schema)
    instance = model(query="anycode", max_results=10)
    print(f"  model:    {model.__name__}")
    print(f"  instance: query={instance.query}, max_results={instance.max_results}")

    # --- Section C: Tool name building and manual registration ---
    print(f"\n{SEPARATOR}")
    print("Section C: Tool Name Prefixing & Manual Registration")

    name = _build_tool_name("github", "search-repos")
    print(f"  github + search-repos = {name}")

    tool_infos = [
        MCPToolInfo(
            server="github",
            name="search_repos",
            description="Search GitHub repositories",
            input_schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        ),
        MCPToolInfo(
            server="github",
            name="get_issue",
            description="Get a GitHub issue by number",
            input_schema={
                "properties": {"repo": {"type": "string"}, "number": {"type": "integer"}},
                "required": ["repo", "number"],
            },
        ),
    ]

    print(f"  tool_infos: {len(tool_infos)} MCP tool definitions ready")
    for info in tool_infos:
        prefixed = _build_tool_name("github", info.name)
        print(f"    {info.name} -> {prefixed}")

    # --- Section D: Real MCP server connection ---
    print(f"\n{SEPARATOR}")
    print("Section D: MCPClient Connection Lifecycle")

    client = MCPClient(valid_config)
    print(f"  client created for server: {client.name}")
    print(f"  connected: {client.is_connected}")

    try:
        await client.connect()
        print(f"  connected: {client.is_connected}")

        await client.discover_tools()
        print(f"  discovered {len(client.discovered_tools)} tools:")
        for tool in client.discovered_tools[:5]:
            print(f"    - {tool.name}: {tool.description[:60]}...")

        # Register discovered tools
        registry = ToolRegistry()
        for tool in client.discovered_tools:
            defn = mcp_tool_to_definition("github", tool, client)
            registry.register(defn)

        print(f"\n  registered {len(registry.list())} tools in ToolRegistry")

        # Call a real tool if we have any
        if client.discovered_tools:
            result = await client.call_tool("list_commits", {"owner": "Quantlix", "repo": "anycode", "per_page": 2})
            print(f"\n  list_commits(Quantlix/anycode, per_page=2):")
            for line in result["content"].split("\n")[:5]:
                print(f"    {line[:100]}")


        # Cleanup
        registry.deregister_prefix("mcp_github")
        print(f"  after deregister_prefix('mcp_github'): {len(registry.list())} tools")

        await client.disconnect()
        print(f"  disconnected: {not client.is_connected}")
    except Exception as e:
        print(f"  connection failed (expected if npx/server not installed): {type(e).__name__}: {e}")
        print("  tip: install the MCP server with: npx -y @modelcontextprotocol/server-github")
        try:
            await client.disconnect()
        except Exception:
            pass

    # --- Section E: Local echo server (guaranteed to work) ---
    print(f"\n{SEPARATOR}")
    print("Section E: Local Echo Server (end-to-end)")

    echo_script = str(Path(__file__).parent.parent / "tests" / "mcp_echo_server.py")
    echo_config = MCPServerConfig(
        name="echo-test",
        transport="stdio",
        command=sys.executable,
        args=[echo_script],
        timeout=15,
    )

    async with MCPClient(echo_config) as echo_client:
        tools = await echo_client.discover_tools()
        print(f"  discovered {len(tools)} tools from echo server:")
        for tool in tools:
            print(f"    - {tool.name}: {tool.description}")

        # Register into ToolRegistry
        echo_registry = ToolRegistry()
        for tool in tools:
            defn = mcp_tool_to_definition("echo-test", tool, echo_client)
            echo_registry.register(defn)
        print(f"\n  registered {len(echo_registry.list())} tools")

        # Call echo tool
        result = await echo_client.call_tool("echo", {"message": "Hello from AnyCode!"})
        print(f"\n  echo('Hello from AnyCode!') = {result['content']}")

        # Call add tool
        result = await echo_client.call_tool("add", {"a": 42, "b": 58})
        print(f"  add(42, 58) = {result['content']}")

        # Call complex_input tool
        result = await echo_client.call_tool(
            "complex_input",
            {
                "name": "demo",
                "tags": ["mcp", "anycode"],
                "metadata": {"version": "1.0"},
            },
        )
        print(f"  complex_input(...) = {result['content']}")

        # Call error tool (demonstrates error handling)
        result = await echo_client.call_tool("error_tool", {"reason": "demo error"})
        print(f"  error_tool: is_error={result['is_error']}, content={result['content'][:60]}")

        echo_registry.deregister_prefix("mcp_echo")
        print(f"\n  after deregister: {len(echo_registry.list())} tools")

    print("  echo server disconnected")

    print(f"\n{SEPARATOR}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
