"""MCP client — manages connection lifecycle to a single MCP server."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from anycode.constants import MCP_TRANSPORT_SSE, MCP_TRANSPORT_STDIO, MCP_TRANSPORT_STREAMABLE_HTTP
from anycode.mcp.config import validate_server_config
from anycode.types import MCPServerConfig, MCPToolInfo

try:
    from mcp import ClientSession, StdioServerParameters, stdio_client

    try:
        from mcp import sse_client  # type: ignore[attr-defined]
    except ImportError:
        sse_client: Any = None
except ImportError:
    ClientSession: Any = None
    StdioServerParameters: Any = None
    sse_client: Any = None
    stdio_client: Any = None

logger = logging.getLogger(__name__)


class MCPClient:
    """Manages connection, tool discovery, and tool execution for a single MCP server."""

    def __init__(self, config: MCPServerConfig) -> None:
        errors = validate_server_config(config)
        if errors:
            raise ValueError(f"Invalid MCP server config: {'; '.join(errors)}")

        self._config = config
        self._session: Any = None
        self._transport_cm: Any = None
        self._session_cm: Any = None
        self._read_stream: Any = None
        self._write_stream: Any = None
        self._process: asyncio.subprocess.Process | None = None
        self._connected = False
        self._tools: list[MCPToolInfo] = []

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def discovered_tools(self) -> list[MCPToolInfo]:
        return list(self._tools)

    async def connect(self) -> None:
        """Connect to the MCP server and initialize the session."""
        if ClientSession is None:
            raise ImportError('mcp package is required for MCP integration. Install it with: pip install "anycode-py[mcp]"')

        if self._config.transport == MCP_TRANSPORT_STDIO:
            await self._connect_stdio(ClientSession)
        elif self._config.transport in (MCP_TRANSPORT_SSE, MCP_TRANSPORT_STREAMABLE_HTTP):
            await self._connect_http(ClientSession)
        else:
            raise ValueError(f"Unsupported MCP transport: {self._config.transport}")

        self._connected = True
        logger.info("MCP client connected to server '%s'", self._config.name)

    async def _connect_stdio(self, session_cls: type) -> None:
        """Connect via stdio transport (subprocess)."""
        env = dict(self._config.env) if self._config.env else None
        server_params = StdioServerParameters(
            command=self._config.command,
            args=list(self._config.args) if self._config.args else [],
            env=env,
        )

        transport_cm = stdio_client(server_params)
        read_stream, write_stream = await asyncio.wait_for(
            transport_cm.__aenter__(),
            timeout=self._config.timeout,
        )
        self._transport_cm = transport_cm
        self._read_stream = read_stream
        self._write_stream = write_stream

        session_cm = session_cls(read_stream, write_stream)
        await asyncio.wait_for(session_cm.__aenter__(), timeout=self._config.timeout)
        await asyncio.wait_for(session_cm.initialize(), timeout=self._config.timeout)
        self._session = session_cm
        self._session_cm = session_cm

    async def _connect_http(self, session_cls: type) -> None:
        """Connect via SSE or streamable-http transport."""
        if not self._config.url:
            raise ValueError(f"MCP server '{self._config.name}': HTTP transport requires a 'url'.")
        if sse_client is None:
            raise ImportError(
                f"MCP server '{self._config.name}': SSE/HTTP transport requires the 'mcp' package with SSE support. "
                'Install it with: pip install "anycode-py[mcp]"'
            )

        transport_cm = sse_client(self._config.url)
        read_stream, write_stream = await asyncio.wait_for(
            transport_cm.__aenter__(),
            timeout=self._config.timeout,
        )
        self._transport_cm = transport_cm
        self._read_stream = read_stream
        self._write_stream = write_stream

        session_cm = session_cls(read_stream, write_stream)
        await asyncio.wait_for(session_cm.__aenter__(), timeout=self._config.timeout)
        await asyncio.wait_for(session_cm.initialize(), timeout=self._config.timeout)
        self._session = session_cm
        self._session_cm = session_cm

    async def discover_tools(self) -> list[MCPToolInfo]:
        """Discover available tools from the MCP server."""
        if not self._session:
            raise RuntimeError(f"MCP client '{self._config.name}' is not connected.")

        result = await asyncio.wait_for(
            self._session.list_tools(),
            timeout=self._config.timeout,
        )

        self._tools = [
            MCPToolInfo(
                server=self._config.name,
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.inputSchema) if tool.inputSchema else {},
            )
            for tool in result.tools
        ]

        logger.info("Discovered %d tools from MCP server '%s'", len(self._tools), self._config.name)
        return list(self._tools)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool on the MCP server and return the result."""
        if not self._session:
            raise RuntimeError(f"MCP client '{self._config.name}' is not connected.")

        try:
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments),
                timeout=self._config.timeout,
            )

            content_parts: list[str] = []
            is_error = False

            for item in result.content:
                if hasattr(item, "text"):
                    content_parts.append(item.text)
                elif hasattr(item, "data"):
                    content_parts.append(str(item.data))

            if hasattr(result, "isError") and result.isError:
                is_error = True

            return {
                "content": "\n".join(content_parts) if content_parts else "",
                "is_error": is_error,
            }

        except Exception as e:
            logger.error("MCP tool call '%s' on server '%s' failed: %s", tool_name, self._config.name, e)
            return {"content": str(e), "is_error": True}

    async def disconnect(self) -> None:
        """Disconnect from the MCP server and clean up resources."""
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None
            self._session_cm = None

        if self._transport_cm:
            try:
                await self._transport_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._transport_cm = None

        self._read_stream = None
        self._write_stream = None
        self._connected = False
        self._tools = []
        logger.info("MCP client disconnected from server '%s'", self._config.name)

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()
