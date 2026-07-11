"""MCP HTTP transport auth resolution."""

from __future__ import annotations

import os

import pytest

from anycode.mcp.client import resolve_auth_headers
from anycode.types import MCPServerConfig


def test_resolve_auth_headers_merges_static_and_token(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("MCP_TEST_TOKEN", "secret-value")
    config = MCPServerConfig(
        name="remote",
        transport="streamable-http",
        url="https://example.com/mcp",
        headers={"X-Custom": "v1"},
        auth_token_env="MCP_TEST_TOKEN",
    )
    headers = resolve_auth_headers(config)
    assert headers is not None
    assert headers["X-Custom"] == "v1"
    assert headers["Authorization"] == "Bearer secret-value"


def test_resolve_auth_headers_none_when_unset() -> None:
    config = MCPServerConfig(name="remote", transport="streamable-http", url="https://example.com/mcp")
    assert resolve_auth_headers(config) is None


def test_resolve_auth_headers_rejects_missing_env() -> None:
    os.environ.pop("MCP_MISSING_TOKEN", None)
    config = MCPServerConfig(
        name="remote",
        transport="streamable-http",
        url="https://example.com/mcp",
        auth_token_env="MCP_MISSING_TOKEN",
    )
    with pytest.raises(ValueError, match="MCP_MISSING_TOKEN.*not set or is empty"):
        resolve_auth_headers(config)
