# Demo 33 — MCP over Authenticated HTTP
# Execute: uv run python examples/33_mcp_http_auth.py
#
# Demonstrates the phase-11 MCP HTTP-auth surface:
#   1. MCPServerConfig for the streamable-http transport (the modern successor
#      to SSE) with static headers plus an env-sourced Bearer token
#   2. resolve_auth_headers(): secrets are read from the environment at call
#      time and merged into request headers, never stored on the config or
#      surfaced into model context / logs
#   3. Config validation for HTTP transports
#
# Sections A-C need no server. Section D attempts a live connection only when
# MCP_HTTP_URL is set, and degrades gracefully otherwise.

import asyncio
import os

from dotenv import load_dotenv

from anycode.mcp.client import MCPClient, resolve_auth_headers
from anycode.mcp.config import validate_server_config
from anycode.types import MCPServerConfig

load_dotenv()

SEPARATOR = "-" * 60


def _mask(headers: dict[str, str] | None) -> dict[str, str]:
    """Redact bearer values so a demo print never leaks a token."""
    if not headers:
        return {}
    out = {}
    for key, value in headers.items():
        out[key] = "Bearer ****" if key.lower() == "authorization" else value
    return out


async def main() -> None:
    print("=== MCP HTTP Auth Demo ===\n")

    # --- Section A: streamable-http config with auth ---
    print(SEPARATOR)
    print("Section A: streamable-http config\n")

    config = MCPServerConfig(
        name="secured-api",
        transport="streamable-http",
        url="https://mcp.example.com/mcp",
        headers={"X-Client": "anycode-demo"},
        auth_token_env="MCP_API_TOKEN",  # value read from env at call time
    )
    print(f"  transport:      {config.transport}")
    print(f"  url:            {config.url}")
    print(f"  static headers: {config.headers}")
    print(f"  auth_token_env: {config.auth_token_env!r}  (stores the var NAME, never the value)")

    errors = validate_server_config(config)
    print(f"  validation: {'VALID' if not errors else errors}")

    # --- Section B: header resolution with the token present ---
    print(f"\n{SEPARATOR}")
    print("Section B: resolve_auth_headers (token set)\n")

    os.environ["MCP_API_TOKEN"] = "s3cret-demo-value"
    try:
        resolved = resolve_auth_headers(config)
        print(f"  resolved headers: {_mask(resolved)}")
        print(f"  Authorization injected: {'Authorization' in (resolved or {})}")
        print(f"  static header preserved: {resolved.get('X-Client') if resolved else None}")
    finally:
        del os.environ["MCP_API_TOKEN"]

    # --- Section C: no token in the environment ---
    print(f"\n{SEPARATOR}")
    print("Section C: resolve_auth_headers (token absent)\n")

    # Resolution is fail-closed: a config that declares auth_token_env refuses to build
    # headers when that variable is missing, rather than connecting unauthenticated.
    try:
        resolve_auth_headers(config)
        print("  unexpected: resolution succeeded without a token")
    except ValueError as error:
        print(f"  refused, as intended: {error}")

    header_only = MCPServerConfig(name="plain", transport="sse", url="https://mcp.example.com/sse")
    print(f"  config with no headers/token resolves to: {resolve_auth_headers(header_only)}")

    # --- Section D: live connection (optional) ---
    print(f"\n{SEPARATOR}")
    print("Section D: live streamable-http connection\n")

    live_url = os.environ.get("MCP_HTTP_URL")
    if not live_url:
        print("  skipped: set MCP_HTTP_URL (and MCP_API_TOKEN if the server needs it)")
        print(f"\n{SEPARATOR}\nDone.")
        return

    live_config = MCPServerConfig(
        name="live-http",
        transport="streamable-http",
        url=live_url,
        auth_token_env="MCP_API_TOKEN",
        timeout=15,
    )
    try:
        async with MCPClient(live_config) as client:
            tools = await client.discover_tools()
            print(f"  connected to {live_url}")
            print(f"  discovered {len(tools)} tools:")
            for tool in tools[:5]:
                print(f"    - {tool.name}: {tool.description[:60]}")
    except Exception as e:  # noqa: BLE001 - example stays runnable without a live server
        print(f"  connection failed: {type(e).__name__}: {e}")
        print('  tip: streamable-http needs a recent mcp package: pip install "anycode-py[mcp]"')

    print(f"\n{SEPARATOR}\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
