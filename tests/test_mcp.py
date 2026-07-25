"""Tests for MCP integration (config, client, bridge)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from anycode import AgentConfig, AnyCode
from anycode.mcp import bridge as mcp_bridge_module
from anycode.mcp import client as mcp_client_module
from anycode.mcp.bridge import _build_tool_name, discover_and_register, mcp_tool_to_definition, schema_to_pydantic_model
from anycode.mcp.client import MCPClient
from anycode.mcp.config import validate_server_config
from anycode.tools.registry import ToolRegistry, define_tool
from anycode.types import MCPServerConfig, MCPToolInfo, MCPTrustPolicy, TeamConfig, ToolResult

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

    def test_remote_plaintext_http_is_rejected(self) -> None:
        cfg = MCPServerConfig(name="remote", transport="streamable-http", url="http://example.com/mcp")
        assert any("requires HTTPS" in error for error in validate_server_config(cfg))

    def test_http_host_allowlist_is_enforced(self) -> None:
        cfg = MCPServerConfig(
            name="remote",
            transport="streamable-http",
            url="https://example.com/mcp",
            trust=MCPTrustPolicy(allowed_hosts=("trusted.example",)),
        )
        assert any("allowlist" in error for error in validate_server_config(cfg))

    def test_private_network_literal_is_rejected(self) -> None:
        cfg = MCPServerConfig(name="remote", transport="streamable-http", url="https://10.0.0.8/mcp")
        assert any("private-network" in error for error in validate_server_config(cfg))

    def test_stdio_can_be_disabled(self) -> None:
        cfg = MCPServerConfig(
            name="local",
            transport="stdio",
            command="node",
            trust=MCPTrustPolicy(allow_stdio=False),
        )
        assert any("stdio transport is disabled" in error for error in validate_server_config(cfg))


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


class TestEngineMCPVisibility:
    def test_mcp_tools_require_explicit_agent_server_scope(self) -> None:
        engine = AnyCode()
        engine._mcp_clients = {"alpha": object(), "alpha_admin": object()}

        class DummyInput(BaseModel):
            pass

        async def noop(v: BaseModel, ctx: MagicMock) -> ToolResult:
            return ToolResult(data="ok")

        for name in ("mcp_alpha_read", "mcp_alpha_admin_write"):
            engine._mcp_tool_registry.register(define_tool(name=name, description="test", input_model=DummyInput, execute=noop))
        engine._mcp_tools_by_server = {
            "alpha": {"mcp_alpha_read"},
            "alpha_admin": {"mcp_alpha_admin_write"},
        }

        unscoped = engine.build_agent(AgentConfig(name="unscoped", model="fake", provider="openai"))
        scoped = engine.build_agent(AgentConfig(name="scoped", model="fake", provider="openai", tools=[], mcp_servers=["alpha"]))

        assert "mcp_alpha_read" not in unscoped.get_tools()
        assert "mcp_alpha_admin_write" not in unscoped.get_tools()
        assert "mcp_alpha_read" in scoped.get_tools()
        assert "mcp_alpha_admin_write" not in scoped.get_tools()
        assert scoped.config.tools == ["mcp_alpha_read"]

    async def test_connection_updates_agents_built_from_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.connect = AsyncMock()
        client.disconnect = AsyncMock()

        class DummyInput(BaseModel):
            pass

        async def noop(_value: BaseModel, _context: MagicMock) -> ToolResult:
            return ToolResult(data="ok")

        tool = define_tool(name="mcp_alpha_read", description="test", input_model=DummyInput, execute=noop)

        async def register_tool(_client: object, _server_name: str, registry: ToolRegistry) -> list[str]:
            registry.register(tool)
            return [tool.name]

        monkeypatch.setattr(mcp_client_module, "MCPClient", MagicMock(return_value=client))
        monkeypatch.setattr(mcp_bridge_module, "discover_and_register", register_tool)

        config = AgentConfig(name="configured", model="fake", provider="openai", tools=[], mcp_servers=["alpha"])
        engine = AnyCode(
            {
                "mcp_servers": [
                    MCPServerConfig(name="alpha", transport="stdio", command="node"),
                ]
            }
        )
        engine.create_team("configured-team", TeamConfig(name="configured-team", agents=[config]))
        agent = engine._pool.get("configured")
        assert agent is not None
        assert "mcp_alpha_read" not in agent.get_tools()

        await engine.connect_mcp_servers()

        assert "mcp_alpha_read" in agent.get_tools()
        assert agent.config.tools == ["mcp_alpha_read"]

        await engine.disconnect_mcp_servers()

        assert "mcp_alpha_read" not in agent.get_tools()
        assert agent.config.tools == []

    async def test_failed_discovery_disconnects_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.connect = AsyncMock()
        client.disconnect = AsyncMock()

        monkeypatch.setattr(mcp_client_module, "MCPClient", MagicMock(return_value=client))
        monkeypatch.setattr(
            mcp_bridge_module,
            "discover_and_register",
            AsyncMock(side_effect=RuntimeError("discovery failed")),
        )

        engine = AnyCode(
            {
                "mcp_servers": [
                    MCPServerConfig(name="alpha", transport="stdio", command="node"),
                ]
            }
        )
        await engine.connect_mcp_servers()

        client.disconnect.assert_awaited_once()
        assert engine._mcp_clients == {}
        assert engine._mcp_tools_by_server == {}

    async def test_failed_agent_attachment_rolls_back_and_can_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.connect = AsyncMock()
        client.disconnect = AsyncMock()

        class DummyInput(BaseModel):
            pass

        async def noop(_value: BaseModel, _context: MagicMock) -> ToolResult:
            return ToolResult(data="ok")

        tool = define_tool(name="mcp_alpha_read", description="test", input_model=DummyInput, execute=noop)

        async def register_tool(_client: object, _server_name: str, registry: ToolRegistry) -> list[str]:
            registry.register(tool)
            return [tool.name]

        attach = MagicMock(side_effect=[RuntimeError("attachment failed"), None])
        monkeypatch.setattr(mcp_client_module, "MCPClient", MagicMock(return_value=client))
        monkeypatch.setattr(mcp_bridge_module, "discover_and_register", register_tool)

        engine = AnyCode(
            {
                "mcp_servers": [
                    MCPServerConfig(name="alpha", transport="stdio", command="node"),
                ]
            }
        )
        monkeypatch.setattr(engine, "_attach_mcp_tools_to_agents", attach)

        await engine.connect_mcp_servers()

        assert engine._mcp_clients == {}
        assert engine._mcp_tools_by_server == {}
        assert not engine._mcp_tool_registry.has(tool.name)
        client.disconnect.assert_awaited_once()

        await engine.connect_mcp_servers()

        assert engine._mcp_clients == {"alpha": client}
        assert engine._mcp_tools_by_server == {"alpha": {tool.name}}
        assert engine._mcp_tool_registry.has(tool.name)
        assert attach.call_count == 2

    async def test_cancelled_connection_preserves_cancellation_when_cleanup_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.connect = AsyncMock()
        client.disconnect = AsyncMock(side_effect=RuntimeError("cleanup failed"))
        monkeypatch.setattr(mcp_client_module, "MCPClient", MagicMock(return_value=client))
        monkeypatch.setattr(
            mcp_bridge_module,
            "discover_and_register",
            AsyncMock(side_effect=asyncio.CancelledError),
        )

        engine = AnyCode(
            {
                "mcp_servers": [
                    MCPServerConfig(name="alpha", transport="stdio", command="node"),
                ]
            }
        )

        with pytest.raises(asyncio.CancelledError):
            await engine.connect_mcp_servers()

        client.disconnect.assert_awaited_once()
        assert engine._mcp_clients == {}
        assert engine._mcp_tools_by_server == {}

    async def test_disconnect_clears_state_when_agent_detachment_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.disconnect = AsyncMock()
        engine = AnyCode()
        engine._mcp_clients = {"alpha": client}
        engine._mcp_tools_by_server = {"alpha": {"mcp_alpha_read"}}
        monkeypatch.setattr(
            engine,
            "_detach_mcp_tools_from_agents",
            MagicMock(side_effect=RuntimeError("detachment failed")),
        )

        await engine.disconnect_mcp_servers()

        client.disconnect.assert_awaited_once()
        assert engine._mcp_clients == {}
        assert engine._mcp_tools_by_server == {}
        assert engine._mcp_tool_registry.list() == []


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

    async def test_server_read_only_hint_does_not_bypass_side_effect_controls(self) -> None:
        client = MCPClient(MCPServerConfig(name="test", transport="stdio", command="node"))
        client._session = MagicMock()
        client._session.list_tools = AsyncMock(
            return_value=SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="claimed-read-only",
                        description="advisory metadata",
                        inputSchema={},
                        annotations=SimpleNamespace(readOnlyHint=True),
                    )
                ]
            )
        )

        tools = await client.discover_tools()

        assert tools[0].side_effecting is True

    async def test_call_tool_when_not_connected(self) -> None:

        client = MCPClient(MCPServerConfig(name="test", transport="stdio", command="node"))
        with pytest.raises(RuntimeError, match="not connected"):
            await client.call_tool("test_tool", {"key": "value"})

    async def test_disconnect_idempotent(self) -> None:

        client = MCPClient(MCPServerConfig(name="test", transport="stdio", command="node"))
        await client.disconnect()
        assert client.is_connected is False

    async def test_failed_initialization_closes_session_and_transport(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session_exit = AsyncMock()
        transport_exit = AsyncMock()

        class FakeTransport:
            async def __aenter__(self) -> tuple[object, object]:
                return object(), object()

            async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
                await transport_exit(exc_type, exc_value, traceback)

        class FakeSession:
            def __init__(self, read_stream: object, write_stream: object) -> None:
                pass

            async def __aenter__(self) -> FakeSession:
                return self

            async def initialize(self) -> None:
                raise RuntimeError("initialization failed")

            async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
                await session_exit(exc_type, exc_value, traceback)

        monkeypatch.setattr(mcp_client_module, "ClientSession", FakeSession)
        monkeypatch.setattr(mcp_client_module, "StdioServerParameters", MagicMock())
        monkeypatch.setattr(mcp_client_module, "stdio_client", lambda _params: FakeTransport())

        client = MCPClient(MCPServerConfig(name="test", transport="stdio", command="node"))
        with pytest.raises(RuntimeError, match="initialization failed"):
            await client.connect()

        session_exit.assert_awaited_once()
        transport_exit.assert_awaited_once()
        assert client.is_connected is False
