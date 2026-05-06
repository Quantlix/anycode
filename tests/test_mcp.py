"""Tests for MCP integration (config, client, bridge)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from anycode.mcp.bridge import _build_tool_name, discover_and_register, mcp_tool_to_definition, schema_to_pydantic_model
from anycode.mcp.client import MCPClient
from anycode.mcp.config import validate_server_config
from anycode.tools.registry import ToolRegistry, define_tool
from anycode.types import MCPServerConfig, MCPToolInfo, ToolResult

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestValidateServerConfig:
    def test_valid_stdio_config(self) -> None:
        cfg = MCPServerConfig(name="test", transport="stdio", command="node", args=["server.js"])
        assert validate_server_config(cfg) == []

    def test_valid_sse_config(self) -> None:
        cfg = MCPServerConfig(name="test", transport="sse", url="http://localhost:3000/sse")
        assert validate_server_config(cfg) == []

    def test_valid_streamable_http_config(self) -> None:
        cfg = MCPServerConfig(name="test", transport="streamable-http", url="http://localhost:3000/mcp")
        assert validate_server_config(cfg) == []

    def test_stdio_missing_command(self) -> None:
        cfg = MCPServerConfig(name="test", transport="stdio")
        errors = validate_server_config(cfg)
        assert any("command" in e for e in errors)

    def test_sse_missing_url(self) -> None:
        cfg = MCPServerConfig(name="test", transport="sse")
        errors = validate_server_config(cfg)
        assert any("url" in e for e in errors)

    def test_empty_name(self) -> None:
        cfg = MCPServerConfig(name="", transport="stdio", command="node")
        errors = validate_server_config(cfg)
        assert any("non-empty" in e for e in errors)

    def test_negative_timeout(self) -> None:
        cfg = MCPServerConfig(name="test", transport="stdio", command="node", timeout=-1)
        errors = validate_server_config(cfg)
        assert any("timeout" in e for e in errors)


# ---------------------------------------------------------------------------
# Bridge — schema_to_pydantic_model
# ---------------------------------------------------------------------------


class TestSchemaToPydanticModel:
    def test_simple_string_props(self) -> None:
        schema = {"properties": {"name": {"type": "string"}}, "required": ["name"]}
        model = schema_to_pydantic_model("TestModel", schema)
        assert issubclass(model, BaseModel)
        instance = model(name="hello")
        assert instance.name == "hello"

    def test_optional_field(self) -> None:
        schema = {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name"]}
        model = schema_to_pydantic_model("OptModel", schema)
        instance = model(name="hello")
        assert instance.age is None

    def test_number_and_boolean(self) -> None:
        schema = {
            "properties": {"score": {"type": "number"}, "active": {"type": "boolean"}},
            "required": ["score", "active"],
        }
        model = schema_to_pydantic_model("NumBool", schema)
        instance = model(score=3.14, active=True)
        assert instance.score == 3.14
        assert instance.active is True

    def test_array_and_object(self) -> None:
        schema = {
            "properties": {"tags": {"type": "array"}, "meta": {"type": "object"}},
            "required": ["tags", "meta"],
        }
        model = schema_to_pydantic_model("ArrayObj", schema)
        instance = model(tags=["a", "b"], meta={"k": "v"})
        assert len(instance.tags) == 2

    def test_empty_schema_gets_placeholder(self) -> None:
        schema = {"properties": {}}
        model = schema_to_pydantic_model("Empty", schema)
        instance = model()
        assert hasattr(instance, "_placeholder")

    def test_union_type_picks_non_null(self) -> None:
        schema = {"properties": {"val": {"type": ["string", "null"]}}, "required": ["val"]}
        model = schema_to_pydantic_model("UnionModel", schema)
        instance = model(val="hello")
        assert instance.val == "hello"


# ---------------------------------------------------------------------------
# Bridge — tool name building
# ---------------------------------------------------------------------------


class TestBuildToolName:
    def test_basic_name(self) -> None:
        assert _build_tool_name("github", "search_repos") == "mcp_github_search_repos"

    def test_name_with_dashes(self) -> None:
        assert _build_tool_name("my-server", "get-data") == "mcp_my_server_get_data"

    def test_name_with_dots(self) -> None:
        assert _build_tool_name("my.server", "get.data") == "mcp_my_server_get_data"


# ---------------------------------------------------------------------------
# Bridge — mcp_tool_to_definition
# ---------------------------------------------------------------------------


class TestMCPToolToDefinition:
    def test_creates_tool_definition(self) -> None:
        tool_info = MCPToolInfo(
            server="github",
            name="search",
            description="Search repos",
            input_schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
        client = MagicMock()
        client.is_connected = True

        defn = mcp_tool_to_definition("github", tool_info, client)
        assert defn.name == "mcp_github_search"
        assert "MCP:github" in defn.description
        assert defn.input_model is not None

    async def test_execute_calls_client(self) -> None:
        tool_info = MCPToolInfo(
            server="github",
            name="search",
            description="Search repos",
            input_schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
        client = MagicMock()
        client.is_connected = True
        client.call_tool = AsyncMock(return_value={"content": "found 5 repos", "is_error": False})

        defn = mcp_tool_to_definition("github", tool_info, client)
        validated = defn.input_model(query="test")
        ctx = MagicMock()

        result = await defn.execute(validated, ctx)
        assert isinstance(result, ToolResult)
        assert result.data == "found 5 repos"
        assert result.is_error is False
        client.call_tool.assert_called_once_with("search", {"query": "test"})

    async def test_execute_when_disconnected(self) -> None:
        tool_info = MCPToolInfo(
            server="github",
            name="search",
            description="Search repos",
            input_schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
        client = MagicMock()
        client.is_connected = False

        defn = mcp_tool_to_definition("github", tool_info, client)
        validated = defn.input_model(query="test")
        ctx = MagicMock()

        result = await defn.execute(validated, ctx)
        assert result.is_error is True
        assert "disconnected" in result.data.lower()


# ---------------------------------------------------------------------------
# Bridge — discover_and_register
# ---------------------------------------------------------------------------


class TestDiscoverAndRegister:
    async def test_registers_discovered_tools(self) -> None:
        mock_client = MagicMock()
        mock_client.is_connected = True
        mock_client.discover_tools = AsyncMock(
            return_value=[
                MCPToolInfo(server="test", name="tool_a", description="A", input_schema={"properties": {"x": {"type": "string"}}}),
                MCPToolInfo(server="test", name="tool_b", description="B", input_schema={"properties": {"y": {"type": "integer"}}}),
            ]
        )
        mock_client.call_tool = AsyncMock(return_value={"content": "ok", "is_error": False})

        registry = ToolRegistry()
        names = await discover_and_register(mock_client, "test", registry)

        assert len(names) == 2
        assert "mcp_test_tool_a" in names
        assert "mcp_test_tool_b" in names
        assert registry.has("mcp_test_tool_a")
        assert registry.has("mcp_test_tool_b")


# ---------------------------------------------------------------------------
# Registry — deregister_prefix
# ---------------------------------------------------------------------------


class TestDeregisterPrefix:
    def test_removes_matching_tools(self) -> None:
        registry = ToolRegistry()

        async def noop(v: BaseModel, ctx: MagicMock) -> ToolResult:
            return ToolResult(data="ok")

        for name in ["mcp_github_search", "mcp_github_list", "local_tool"]:

            class DummyInput(BaseModel):
                pass

            t = define_tool(name=name, description="test", input_model=DummyInput, execute=noop)
            registry.register(t)

        registry.deregister_prefix("mcp_github")
        assert not registry.has("mcp_github_search")
        assert not registry.has("mcp_github_list")
        assert registry.has("local_tool")


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------


class TestMCPClient:
    def test_init_validates_config(self) -> None:

        with pytest.raises(ValueError, match="Invalid MCP server config"):
            MCPClient(MCPServerConfig(name="", transport="stdio", command="node"))

    def test_initial_state(self) -> None:

        client = MCPClient(MCPServerConfig(name="test", transport="stdio", command="node"))
        assert client.name == "test"
        assert client.is_connected is False
        assert client.discovered_tools == []

    async def test_discover_tools_when_not_connected(self) -> None:

        client = MCPClient(MCPServerConfig(name="test", transport="stdio", command="node"))
        with pytest.raises(RuntimeError, match="not connected"):
            await client.discover_tools()

    async def test_call_tool_when_not_connected(self) -> None:

        client = MCPClient(MCPServerConfig(name="test", transport="stdio", command="node"))
        with pytest.raises(RuntimeError, match="not connected"):
            await client.call_tool("test_tool", {"key": "value"})

    async def test_disconnect_idempotent(self) -> None:

        client = MCPClient(MCPServerConfig(name="test", transport="stdio", command="node"))
        await client.disconnect()
        assert client.is_connected is False
