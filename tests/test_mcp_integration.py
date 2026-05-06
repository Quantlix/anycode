"""Comprehensive MCP integration tests against the in-process echo server.

Covers:
  1. Lifecycle & handshake (initialize → list_tools → discover → call_tool → disconnect)
  2. Tool execution (schema mapping, arguments, errors, timeouts, multi-content)
  3. Resource discovery and reading
  4. Prompt listing and rendering
  5. Bridge integration (MCPToolInfo → ToolDefinition round-trip)
  6. Multi-server composition
  7. Security & isolation (env vars, tool confirmation gates)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp import ClientSession, StdioServerParameters, stdio_client
from pydantic import BaseModel

from anycode.mcp.bridge import (
    _build_tool_name,
    discover_and_register,
    mcp_tool_to_definition,
    schema_to_pydantic_model,
)
from anycode.mcp.client import MCPClient
from anycode.tools.registry import ToolRegistry
from anycode.types import MCPServerConfig, MCPToolInfo, ToolResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ECHO_SERVER_SCRIPT = str(Path(__file__).parent / "mcp_echo_server.py")
ECHO_SERVER_NAME = "echo-test"


@pytest.fixture
async def client_session():
    """Yield a raw MCP ClientSession connected to the echo server via stdio subprocess.

    The teardown suppresses anyio cancel-scope RuntimeErrors that occur
    because pytest-asyncio may run fixture teardown in a different task
    context than where the anyio task group was entered.
    """
    server_params = StdioServerParameters(command=sys.executable, args=[ECHO_SERVER_SCRIPT])
    transport_cm = stdio_client(server_params)
    read_stream, write_stream = await transport_cm.__aenter__()
    session_cm = ClientSession(read_stream, write_stream)
    session = await session_cm.__aenter__()
    await session.initialize()

    yield session

    # Cleanup — suppress anyio task-scope errors during teardown
    for cm in (session_cm, transport_cm):
        try:
            await cm.__aexit__(None, None, None)
        except (RuntimeError, BaseExceptionGroup):
            pass


@pytest.fixture
def stdio_config() -> MCPServerConfig:
    """Config that launches the echo server as a subprocess (for MCPClient tests)."""
    return MCPServerConfig(
        name="echo-test",
        transport="stdio",
        command=sys.executable,
        args=[ECHO_SERVER_SCRIPT],
        timeout=15,
    )


# ===================================================================
# 1. LIFECYCLE & HANDSHAKE
# ===================================================================


class TestLifecycleHandshake:
    """Verify the full MCP lifecycle: init → discover → call → disconnect."""

    async def test_initialize_returns_session(self, client_session) -> None:
        assert client_session is not None

    async def test_list_tools_after_init(self, client_session) -> None:
        result = await client_session.list_tools()
        tool_names = [t.name for t in result.tools]
        assert "echo" in tool_names
        assert "add" in tool_names
        assert "complex_input" in tool_names
        assert "slow_task" in tool_names
        assert "error_tool" in tool_names
        assert "multi_content" in tool_names

    async def test_each_tool_has_schema(self, client_session) -> None:
        result = await client_session.list_tools()
        for tool in result.tools:
            assert tool.inputSchema is not None
            assert tool.inputSchema.get("type") == "object"
            assert "properties" in tool.inputSchema

    async def test_list_resources_after_init(self, client_session) -> None:
        result = await client_session.list_resources()
        names = [r.name for r in result.resources]
        assert "readme" in names
        assert "config" in names

    async def test_list_prompts_after_init(self, client_session) -> None:
        result = await client_session.list_prompts()
        names = [p.name for p in result.prompts]
        assert "greeting" in names
        assert "summarize" in names


# ===================================================================
# 2. TOOL EXECUTION — stress testing
# ===================================================================


class TestToolExecution:
    """Schema mapping, argument passing, error handling, timeouts."""

    # --- echo: basic string round-trip ---

    async def test_echo_basic(self, client_session) -> None:
        result = await client_session.call_tool("echo", {"message": "hello world"})
        assert result.content[0].text == "hello world"
        assert not result.isError

    async def test_echo_empty_string(self, client_session) -> None:
        result = await client_session.call_tool("echo", {"message": ""})
        assert result.content[0].text == ""

    async def test_echo_unicode(self, client_session) -> None:
        msg = "你好世界 🌍 مرحبا"
        result = await client_session.call_tool("echo", {"message": msg})
        assert result.content[0].text == msg

    async def test_echo_long_payload(self, client_session) -> None:
        msg = "x" * 10_000
        result = await client_session.call_tool("echo", {"message": msg})
        assert result.content[0].text == msg

    # --- add: numeric type coercion ---

    async def test_add_integers(self, client_session) -> None:
        result = await client_session.call_tool("add", {"a": 3, "b": 5})
        assert float(result.content[0].text) == 8.0

    async def test_add_floats(self, client_session) -> None:
        result = await client_session.call_tool("add", {"a": 1.5, "b": 2.7})
        assert abs(float(result.content[0].text) - 4.2) < 0.001

    async def test_add_negative(self, client_session) -> None:
        result = await client_session.call_tool("add", {"a": -10, "b": 3})
        assert float(result.content[0].text) == -7.0

    async def test_add_zero(self, client_session) -> None:
        result = await client_session.call_tool("add", {"a": 0, "b": 0})
        assert float(result.content[0].text) == 0.0

    # --- complex_input: nested objects ---

    async def test_complex_input_all_fields(self, client_session) -> None:
        args = {"name": "test", "tags": ["a", "b"], "metadata": {"key": "value"}}
        result = await client_session.call_tool("complex_input", args)
        parsed = json.loads(result.content[0].text)
        assert parsed["name"] == "test"
        assert parsed["tags"] == ["a", "b"]
        assert parsed["metadata"] == {"key": "value"}

    async def test_complex_input_required_only(self, client_session) -> None:
        result = await client_session.call_tool("complex_input", {"name": "minimal"})
        parsed = json.loads(result.content[0].text)
        assert parsed["name"] == "minimal"

    async def test_complex_input_empty_collections(self, client_session) -> None:
        args = {"name": "empty", "tags": [], "metadata": {}}
        result = await client_session.call_tool("complex_input", args)
        parsed = json.loads(result.content[0].text)
        assert parsed["tags"] == []
        assert parsed["metadata"] == {}

    # --- multi_content: multiple TextContent items ---

    async def test_multi_content_returns_multiple(self, client_session) -> None:
        result = await client_session.call_tool("multi_content", {"count": 3})
        assert len(result.content) == 3
        texts = [c.text for c in result.content]
        assert texts == ["part-0", "part-1", "part-2"]

    async def test_multi_content_single(self, client_session) -> None:
        result = await client_session.call_tool("multi_content", {"count": 1})
        assert len(result.content) == 1

    # --- error_tool: server-side errors ---

    async def test_error_tool_raises(self, client_session) -> None:
        result = await client_session.call_tool("error_tool", {"reason": "test failure"})
        assert result.isError

    # --- unknown tool ---

    async def test_unknown_tool_errors(self, client_session) -> None:
        result = await client_session.call_tool("nonexistent_tool", {})
        assert result.isError


# ===================================================================
# 3. RESOURCE DISCOVERY & READING
# ===================================================================


class TestResourceDiscovery:
    """Verify resource listing and content retrieval."""

    async def test_read_readme_resource(self, client_session) -> None:
        result = await client_session.read_resource("test://docs/readme")
        content = result.contents[0]
        assert "Echo Server" in content.text

    async def test_read_json_resource(self, client_session) -> None:
        result = await client_session.read_resource("test://data/config.json")
        parsed = json.loads(result.contents[0].text)
        assert parsed["version"] == "1.0"
        assert parsed["debug"] is True

    async def test_read_nonexistent_resource_errors(self, client_session) -> None:
        with pytest.raises(Exception):
            await client_session.read_resource("test://nonexistent")

    async def test_resources_have_metadata(self, client_session) -> None:
        result = await client_session.list_resources()
        for resource in result.resources:
            assert resource.name
            assert resource.uri
            assert resource.mimeType


# ===================================================================
# 4. PROMPT LISTING & RENDERING
# ===================================================================


class TestPromptRendering:
    """Verify prompt discovery, argument handling, and message generation."""

    async def test_greeting_prompt(self, client_session) -> None:
        result = await client_session.get_prompt("greeting", {"name": "Alice"})
        messages = result.messages
        assert len(messages) >= 1
        assert "Alice" in messages[0].content.text

    async def test_summarize_prompt_with_style(self, client_session) -> None:
        result = await client_session.get_prompt("summarize", {"topic": "AI", "style": "detailed"})
        messages = result.messages
        assert "AI" in messages[0].content.text
        assert "detailed" in messages[0].content.text

    async def test_summarize_prompt_default_style(self, client_session) -> None:
        result = await client_session.get_prompt("summarize", {"topic": "MCP"})
        messages = result.messages
        assert "concise" in messages[0].content.text

    async def test_prompt_arguments_listed(self, client_session) -> None:
        result = await client_session.list_prompts()
        greeting = next(p for p in result.prompts if p.name == "greeting")
        assert len(greeting.arguments) == 1
        assert greeting.arguments[0].name == "name"
        assert greeting.arguments[0].required is True

    async def test_unknown_prompt_errors(self, client_session) -> None:
        with pytest.raises(Exception):
            await client_session.get_prompt("nonexistent", {})


# ===================================================================
# 5. BRIDGE INTEGRATION — MCPToolInfo → ToolDefinition round-trip
# ===================================================================


class TestBridgeIntegration:
    """End-to-end: discover tools from echo server, convert to ToolDefinitions, execute."""

    async def test_discover_tools_from_echo_server(self, client_session) -> None:
        result = await client_session.list_tools()
        tools = result.tools

        for tool in tools:
            tool_info = MCPToolInfo(
                server=ECHO_SERVER_NAME,
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.inputSchema) if tool.inputSchema else {},
            )

            model = schema_to_pydantic_model(f"Test_{tool.name}", tool_info.input_schema)
            assert issubclass(model, BaseModel)
            assert _build_tool_name(ECHO_SERVER_NAME, tool.name).startswith("mcp_")

    async def test_bridge_creates_executable_definitions(self, client_session) -> None:
        result = await client_session.list_tools()
        echo_tool = next(t for t in result.tools if t.name == "echo")

        tool_info = MCPToolInfo(
            server=ECHO_SERVER_NAME,
            name=echo_tool.name,
            description=echo_tool.description or "",
            input_schema=dict(echo_tool.inputSchema),
        )

        mock_client = MagicMock()
        mock_client.is_connected = True
        mock_client.call_tool = AsyncMock(return_value={"content": "echoed: hello", "is_error": False})

        defn = mcp_tool_to_definition(ECHO_SERVER_NAME, tool_info, mock_client)
        assert defn.name == "mcp_echo_test_echo"
        assert defn.input_model is not None

        validated = defn.input_model(message="hello")
        ctx = MagicMock()
        result = await defn.execute(validated, ctx)
        assert isinstance(result, ToolResult)
        assert result.data == "echoed: hello"

    async def test_discover_and_register_full_pipeline(self, client_session) -> None:
        """Simulate what orchestrator does: discover tools, register, verify all present."""
        result = await client_session.list_tools()

        mock_client = MagicMock()
        mock_client.is_connected = True
        mock_client.discover_tools = AsyncMock(
            return_value=[
                MCPToolInfo(
                    server=ECHO_SERVER_NAME,
                    name=t.name,
                    description=t.description or "",
                    input_schema=dict(t.inputSchema) if t.inputSchema else {},
                )
                for t in result.tools
            ]
        )
        mock_client.call_tool = AsyncMock(return_value={"content": "ok", "is_error": False})

        registry = ToolRegistry()
        names = await discover_and_register(mock_client, ECHO_SERVER_NAME, registry)
        assert len(names) == 6
        for name in names:
            assert registry.has(name)

    async def test_schema_faithfulness(self, client_session) -> None:
        """Verify schema_to_pydantic_model preserves required vs optional fields."""
        result = await client_session.list_tools()
        complex_tool = next(t for t in result.tools if t.name == "complex_input")
        schema = dict(complex_tool.inputSchema)

        model = schema_to_pydantic_model("ComplexTest", schema)
        instance = model(name="required_only")
        assert instance.name == "required_only"  # type: ignore[attr-defined]
        assert instance.tags is None  # type: ignore[attr-defined]
        assert instance.metadata is None  # type: ignore[attr-defined]

        instance_full = model(name="full", tags=["a"], metadata={"k": "v"})
        assert instance_full.tags == ["a"]  # type: ignore[attr-defined]


# ===================================================================
# 6. MCPCLIENT STDIO INTEGRATION (subprocess echo server)
# ===================================================================


class TestMCPClientStdio:
    """Test the actual MCPClient connecting to the echo server via stdio subprocess."""

    async def test_full_lifecycle(self, stdio_config: MCPServerConfig) -> None:
        client = MCPClient(stdio_config)
        try:
            await client.connect()
            assert client.is_connected

            tools = await client.discover_tools()
            assert len(tools) == 6
            tool_names = [t.name for t in tools]
            assert "echo" in tool_names
            assert "add" in tool_names

            result = await client.call_tool("echo", {"message": "hello from test"})
            assert result["content"] == "hello from test"
            assert result["is_error"] is False
        finally:
            await client.disconnect()
            assert client.is_connected is False

    async def test_call_add_tool(self, stdio_config: MCPServerConfig) -> None:
        async with MCPClient(stdio_config) as client:
            result = await client.call_tool("add", {"a": 10, "b": 20})
            assert float(result["content"]) == 30.0

    async def test_call_complex_input(self, stdio_config: MCPServerConfig) -> None:
        async with MCPClient(stdio_config) as client:
            args = {"name": "test", "tags": ["x", "y"], "metadata": {"nested": True}}
            result = await client.call_tool("complex_input", args)
            parsed = json.loads(result["content"])
            assert parsed["name"] == "test"
            assert parsed["tags"] == ["x", "y"]

    async def test_error_tool_returns_error(self, stdio_config: MCPServerConfig) -> None:
        async with MCPClient(stdio_config) as client:
            result = await client.call_tool("error_tool", {"reason": "deliberate"})
            assert result["is_error"] is True

    async def test_discover_tools_returns_mcp_tool_info(self, stdio_config: MCPServerConfig) -> None:
        async with MCPClient(stdio_config) as client:
            tools = await client.discover_tools()
            for tool in tools:
                assert isinstance(tool, MCPToolInfo)
                assert tool.server == "echo-test"
                assert tool.name
                assert isinstance(tool.input_schema, dict)

    async def test_context_manager_protocol(self, stdio_config: MCPServerConfig) -> None:
        async with MCPClient(stdio_config) as client:
            assert client.is_connected
            tools = await client.discover_tools()
            assert len(tools) > 0
        assert client.is_connected is False

    async def test_reconnect_after_disconnect(self, stdio_config: MCPServerConfig) -> None:
        client = MCPClient(stdio_config)
        await client.connect()
        await client.disconnect()
        assert client.is_connected is False

        await client.connect()
        assert client.is_connected
        tools = await client.discover_tools()
        assert len(tools) == 6
        await client.disconnect()


# ===================================================================
# 7. MULTI-SERVER COMPOSITION
# ===================================================================


class TestMultiServerComposition:
    """Test running multiple echo server instances and composing their tools."""

    async def test_two_servers_separate_registries(self) -> None:
        """Two server instances → two registries → no name collision."""
        registry_a = ToolRegistry()
        registry_b = ToolRegistry()

        mock_a = MagicMock()
        mock_a.is_connected = True
        mock_a.discover_tools = AsyncMock(
            return_value=[
                MCPToolInfo(
                    server="server-a",
                    name="echo",
                    description="A echo",
                    input_schema={"properties": {"message": {"type": "string"}}},
                )
            ]
        )

        mock_b = MagicMock()
        mock_b.is_connected = True
        mock_b.discover_tools = AsyncMock(
            return_value=[
                MCPToolInfo(
                    server="server-b",
                    name="echo",
                    description="B echo",
                    input_schema={"properties": {"message": {"type": "string"}}},
                )
            ]
        )

        names_a = await discover_and_register(mock_a, "server-a", registry_a)
        names_b = await discover_and_register(mock_b, "server-b", registry_b)

        assert names_a == ["mcp_server_a_echo"]
        assert names_b == ["mcp_server_b_echo"]

    async def test_two_servers_shared_registry(self) -> None:
        """Two servers registered into the same registry get unique prefixed names."""
        registry = ToolRegistry()

        mock_a = MagicMock()
        mock_a.is_connected = True
        mock_a.discover_tools = AsyncMock(
            return_value=[MCPToolInfo(server="alpha", name="echo", description="A", input_schema={"properties": {"msg": {"type": "string"}}})]
        )
        mock_a.call_tool = AsyncMock(return_value={"content": "from-alpha", "is_error": False})

        mock_b = MagicMock()
        mock_b.is_connected = True
        mock_b.discover_tools = AsyncMock(
            return_value=[MCPToolInfo(server="beta", name="echo", description="B", input_schema={"properties": {"msg": {"type": "string"}}})]
        )
        mock_b.call_tool = AsyncMock(return_value={"content": "from-beta", "is_error": False})

        await discover_and_register(mock_a, "alpha", registry)
        await discover_and_register(mock_b, "beta", registry)

        assert registry.has("mcp_alpha_echo")
        assert registry.has("mcp_beta_echo")

        defn_a = registry.get("mcp_alpha_echo")
        defn_b = registry.get("mcp_beta_echo")
        assert defn_a is not None
        assert defn_b is not None

    async def test_deregister_one_server_keeps_other(self) -> None:
        """Deregistering one server's prefix should not affect the other."""
        registry = ToolRegistry()

        for server_name in ["srvA", "srvB"]:
            mock = MagicMock()
            mock.is_connected = True
            mock.discover_tools = AsyncMock(
                return_value=[MCPToolInfo(server=server_name, name="tool1", description="T", input_schema={"properties": {"x": {"type": "string"}}})]
            )
            await discover_and_register(mock, server_name, registry)

        assert registry.has("mcp_srvA_tool1")
        assert registry.has("mcp_srvB_tool1")

        registry.deregister_prefix("mcp_srvA")
        assert not registry.has("mcp_srvA_tool1")
        assert registry.has("mcp_srvB_tool1")

    async def test_multi_server_stdio_subprocess(self) -> None:
        """Launch two subprocess echo servers and verify independent tool calls."""
        config_a = MCPServerConfig(
            name="echo-a",
            transport="stdio",
            command=sys.executable,
            args=[ECHO_SERVER_SCRIPT],
            timeout=15,
        )
        config_b = MCPServerConfig(
            name="echo-b",
            transport="stdio",
            command=sys.executable,
            args=[ECHO_SERVER_SCRIPT],
            timeout=15,
        )

        async with MCPClient(config_a) as client_a, MCPClient(config_b) as client_b:
            tools_a = await client_a.discover_tools()
            tools_b = await client_b.discover_tools()
            assert len(tools_a) == 6
            assert len(tools_b) == 6

            result_a = await client_a.call_tool("echo", {"message": "from-a"})
            result_b = await client_b.call_tool("echo", {"message": "from-b"})
            assert result_a["content"] == "from-a"
            assert result_b["content"] == "from-b"


# ===================================================================
# 8. SECURITY & ISOLATION
# ===================================================================


class TestSecurityIsolation:
    """Environment variable isolation and command validation."""

    def test_env_vars_not_leaked_to_config(self) -> None:
        """Frozen model prevents reassigning env, but container values are standard Python dicts."""
        env = {"SECRET_KEY": "abc123"}
        cfg = MCPServerConfig(name="sec", transport="stdio", command="node", env=env)
        assert cfg.env is not None
        assert cfg.env["SECRET_KEY"] == "abc123"
        with pytest.raises(Exception):
            cfg.env = {"REPLACED": "nope"}  # type: ignore[misc]

    def test_config_is_frozen(self) -> None:
        cfg = MCPServerConfig(name="test", transport="stdio", command="node")
        with pytest.raises(Exception):
            cfg.name = "hacked"  # type: ignore[misc]

    def test_tool_info_is_frozen(self) -> None:
        info = MCPToolInfo(server="s", name="t", description="d", input_schema={})
        with pytest.raises(Exception):
            info.name = "hacked"  # type: ignore[misc]

    async def test_disconnected_tool_returns_error(self) -> None:
        """A tool definition for a disconnected server should return is_error=True."""
        tool_info = MCPToolInfo(
            server="sec-server",
            name="sensitive_tool",
            description="Does sensitive things",
            input_schema={"properties": {"data": {"type": "string"}}, "required": ["data"]},
        )
        mock_client = MagicMock()
        mock_client.is_connected = False

        defn = mcp_tool_to_definition("sec-server", tool_info, mock_client)
        validated = defn.input_model(data="secret")
        ctx = MagicMock()

        result = await defn.execute(validated, ctx)
        assert result.is_error is True
        assert "disconnected" in result.data.lower()
        mock_client.call_tool.assert_not_called()

    def test_command_injection_prevented_by_args_list(self) -> None:
        """StdioServerParameters uses args as a list, not shell-interpolated string."""
        cfg = MCPServerConfig(
            name="safe",
            transport="stdio",
            command="python",
            args=["-c", "print('hello'); import os"],
        )
        assert isinstance(cfg.args, list)
        assert len(cfg.args) == 2

    async def test_tool_call_exception_is_caught(self) -> None:
        """MCPClient.call_tool wraps exceptions in error dict, never propagates raw."""
        config = MCPServerConfig(name="test", transport="stdio", command=sys.executable, args=[ECHO_SERVER_SCRIPT], timeout=15)
        async with MCPClient(config) as client:
            result = await client.call_tool("error_tool", {"reason": "security test"})
            assert result["is_error"] is True
            assert isinstance(result["content"], str)


# ===================================================================
# 9. EDGE CASES & ROBUSTNESS
# ===================================================================


class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    async def test_empty_arguments(self, client_session) -> None:
        """Calling a tool with empty dict when no required fields."""
        result = await client_session.call_tool("multi_content", {"count": 0})
        assert len(result.content) == 0

    async def test_large_batch_of_tool_calls(self, client_session) -> None:
        """Run many sequential tool calls to verify session stability."""
        for i in range(20):
            result = await client_session.call_tool("echo", {"message": f"msg-{i}"})
            assert result.content[0].text == f"msg-{i}"

    async def test_concurrent_tool_calls(self, client_session) -> None:
        """Multiple concurrent tool calls on the same session."""
        tasks = [client_session.call_tool("add", {"a": i, "b": i * 2}) for i in range(5)]
        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results):
            assert float(result.content[0].text) == i + i * 2

    async def test_special_characters_in_message(self, client_session) -> None:
        msg = 'Hello "world" <script>alert(1)</script> & foo=bar'
        result = await client_session.call_tool("echo", {"message": msg})
        assert result.content[0].text == msg

    async def test_json_special_chars_in_complex_input(self, client_session) -> None:
        args = {"name": 'test\n\t"quotes"', "tags": ['a"b', "c\\d"]}
        result = await client_session.call_tool("complex_input", args)
        parsed = json.loads(result.content[0].text)
        assert parsed["name"] == 'test\n\t"quotes"'
