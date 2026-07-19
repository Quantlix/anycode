---
title: "Connect MCP Servers to AnyCode Agents (stdio and HTTP with Auth)"
description: "Connect AnyCode to MCP servers over stdio, streamable HTTP, or SSE with environment-based bearer auth, per-agent tool scope, and safe lifecycle cleanup."
keywords: AnyCode MCP, Model Context Protocol, MCPServerConfig, streamable-http, MCP bearer token, auth_token_env, MCP stdio, MCP tools agents
---

# Connect MCP Servers

AnyCode agents can call tools from any Model Context Protocol (MCP) server. Declare servers with `MCPServerConfig`, pass them to the engine via `mcp_servers`, and discovered tools run through the same validation and execution path as built-in tools. Three transports are supported: `stdio` (subprocess), `streamable-http` (preferred for remote servers), and legacy `sse`.

HTTP transports require the MCP extra:

```bash
pip install "anycode-py[mcp]"
```

## Configure a server

`MCPServerConfig` fields:

| Field | Type | Default | Applies to |
| --- | --- | --- | --- |
| `name` | `str` | required | all — used to prefix tool names |
| `transport` | `"stdio" \| "sse" \| "streamable-http"` | required | all |
| `command` / `args` | `str` / `list[str]` | `None` | `stdio` |
| `env` | `dict[str, str]` | `None` | `stdio` |
| `url` | `str` | `None` | `streamable-http`, `sse` |
| `headers` | `dict[str, str]` | `None` | HTTP — static headers on every request |
| `auth_token_env` | `str` | `None` | HTTP — env var **name** resolved to `Authorization: Bearer <token>` |
| `timeout` | `float` | `30.0` | all — applied to connect, initialize, discovery, and each tool call |
| `trust` | `MCPTrustPolicy` | secure transport defaults | all — stdio permission, HTTPS, private-network, and host controls |

=== "stdio"

    ```python
    from anycode import MCPServerConfig

    config = MCPServerConfig(
        name="files",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/data"],
    )
    ```

=== "streamable-http with auth"

    ```python
    from anycode import MCPServerConfig

    config = MCPServerConfig(
        name="secured-api",
        transport="streamable-http",
        url="https://mcp.example.com/mcp",
        headers={"X-Client": "anycode-demo"},
        auth_token_env="MCP_API_TOKEN",  # env var NAME, not the secret itself
    )
    ```

!!! tip "Secrets stay out of your config"
    `auth_token_env` stores the *name* of an environment variable. The token value is read from `os.environ` at connect time and sent as `Authorization: Bearer <token>` — it is never stored on the config object, and never appears in prompts or logs. Static `headers` are merged with the resolved auth header. If the configured variable is missing or empty, connection fails before opening the transport rather than falling back to anonymous access.

Configs are validated on construction: `stdio` requires `command`, the HTTP transports require `url`, `timeout` must be positive, and an invalid config raises `ValueError`.

## Restrict transport trust

Remote MCP uses HTTPS by default. Plaintext HTTP is accepted only for loopback development endpoints unless `allow_insecure_http=True` is explicit. Private IP literals are rejected unless allowed, and `allowed_hosts` can reduce remote access to a fixed set.

```python title="trusted_mcp.py"
from anycode import MCPServerConfig, MCPTrustPolicy

config = MCPServerConfig(
    name="internal-search",
    transport="streamable-http",
    url="https://mcp.example.com/mcp",
    auth_token_env="MCP_API_TOKEN",
    trust=MCPTrustPolicy(
        allow_stdio=False,
        allowed_hosts=("mcp.example.com",),
    ),
)
```

For stdio servers, set `allow_stdio=False` in environments where spawning a local subprocess is outside the deployment's trust model. Host allowlisting also reduces server-side request forgery exposure, but DNS and network egress controls still belong at the container or network layer.

## Register with the engine

Pass servers on the engine config; agents opt in by server name:

```python title="engine_with_mcp.py"
from anycode import AnyCode, AgentConfig, MCPServerConfig

engine = AnyCode(config={
    "mcp_servers": [
        MCPServerConfig(name="files", transport="stdio", command="npx",
                        args=["-y", "@modelcontextprotocol/server-filesystem", "/data"]),
    ],
})

agent = AgentConfig(
    name="reader",
    provider="anthropic",
    model="claude-haiku-4-5",
    mcp_servers=["files"],   # only this server's tools are visible to this agent
)
```

Discovered tools are registered under the name `mcp_<server>_<tool>` (dashes and dots in the server name become underscores). An agent's `mcp_servers` list filters which servers' tools it can see; omit it to expose none.

## Use the client directly

For scripting or debugging, `MCPClient` is an async context manager:

```python title="mcp_client.py"
from anycode import MCPClient

async with MCPClient(config) as client:
    tools = await client.discover_tools()      # list[MCPToolInfo]
    result = await client.call_tool("search", {"query": "agents"})
```

Behavior worth knowing:

- Every step — transport connect, session init, `discover_tools`, `call_tool` — is wrapped in `asyncio.wait_for(..., timeout=config.timeout)`.
- Engine connection is best-effort per server: one failed server is logged and cleaned up without blocking other servers. Agents that requested the failed server receive none of its tools, so production deployments must monitor MCP connection errors.
- `call_tool` does not raise on tool failure; it returns `{"content": "<error>", "is_error": True}` so the calling agent sees a normal error result.
- There is **no auto-reconnect**. A dropped connection surfaces as errors until you reconnect; `disconnect()` is best-effort and resets state.

See [`examples/33_mcp_http_auth.py`](https://github.com/Quantlix/anycode/blob/main/examples/33_mcp_http_auth.py) for a runnable demo that validates a config, resolves auth headers (masked when printed), and connects live when `MCP_HTTP_URL` is set.

## The complete, runnable program

The snippets above are fragments. Here is one file that exercises both transports: it spawns a local `stdio` filesystem server and lists its tools, then configures a `streamable-http` server with a bearer token, resolves the auth header (masked so nothing leaks), and connects live when `MCP_HTTP_URL` is set. Every network step degrades gracefully, so the program runs and prints useful output even with no server available.

```python title="mcp_demo.py"
import asyncio
import os

from dotenv import load_dotenv

from anycode import MCPClient, MCPServerConfig
from anycode.mcp.client import resolve_auth_headers

load_dotenv()


def mask_auth(headers: dict[str, str] | None) -> dict[str, str]:
    """Redact bearer values so a printout never leaks a token."""
    if not headers:
        return {}
    return {
        key: ("Bearer ****" if key.lower() == "authorization" else value)
        for key, value in headers.items()
    }


async def demo_stdio() -> None:
    """Spawn a local stdio MCP server (filesystem) and list its tools."""
    config = MCPServerConfig(
        name="files",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."],
        timeout=20,
    )
    print(f"[stdio] server '{config.name}' via {config.command}")
    try:
        async with MCPClient(config) as client:
            tools = await client.discover_tools()
            print(f"  discovered {len(tools)} tools:")
            for tool in tools[:5]:
                print(f"    - {tool.name}: {tool.description[:60]}")
    except Exception as exc:  # noqa: BLE001 - stays runnable without Node/npx
        print(f"  connection failed: {type(exc).__name__}: {exc}")
        print("  tip: install Node.js so npx can fetch the filesystem server")


async def demo_http() -> None:
    """Configure a streamable-http server with a bearer token and connect when set."""
    config = MCPServerConfig(
        name="secured-api",
        transport="streamable-http",
        url=os.environ.get("MCP_HTTP_URL", "https://mcp.example.com/mcp"),
        headers={"X-Client": "anycode-demo"},
        auth_token_env="MCP_API_TOKEN",  # env var NAME, never the secret itself
    )
    print(f"[http] server '{config.name}' -> {config.url}")

    # Resolve the auth header with a throwaway token so the demo never prints a real one.
    # (resolve_auth_headers raises if auth_token_env is set but the variable is empty.)
    had_token = "MCP_API_TOKEN" in os.environ
    if not had_token:
        os.environ["MCP_API_TOKEN"] = "demo-token-value"
    try:
        print(f"  resolved headers: {mask_auth(resolve_auth_headers(config))}")
    finally:
        if not had_token:
            del os.environ["MCP_API_TOKEN"]

    if not os.environ.get("MCP_HTTP_URL"):
        print("  skipped live connect: set MCP_HTTP_URL (and MCP_API_TOKEN) to try a real server")
        return
    try:
        async with MCPClient(config) as client:
            tools = await client.discover_tools()
            print(f"  connected — discovered {len(tools)} tools")
    except Exception as exc:  # noqa: BLE001 - stays runnable without a live server
        print(f"  connection failed: {type(exc).__name__}: {exc}")
        print('  tip: HTTP transports need the extra: pip install "anycode-py[mcp]"')


async def main() -> None:
    await demo_stdio()
    print()
    await demo_http()


if __name__ == "__main__":
    asyncio.run(main())
```

Run it from the project root:

```bash
uv run python mcp_demo.py
```

!!! tip "Tested copy"
    See [`examples/10_mcp_tools.py`](https://github.com/Quantlix/anycode/blob/main/examples/10_mcp_tools.py) for the `stdio` path (config validation, tool discovery, and a guaranteed local echo server) and [`examples/33_mcp_http_auth.py`](https://github.com/Quantlix/anycode/blob/main/examples/33_mcp_http_auth.py) for the authenticated `streamable-http` path.

## See also

- [Work with tools](tools.md) — the shared tool execution path
- [Configuration reference](../reference/configuration.md) — `MCPServerConfig` in context
