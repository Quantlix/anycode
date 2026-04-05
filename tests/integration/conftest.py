"""Integration-test fixtures — requires Docker services running."""

from __future__ import annotations

import socket

import pytest


def _is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _skip_unless(service: str, host: str, port: int) -> None:
    if not _is_port_open(host, port):
        pytest.skip(f"{service} not available at {host}:{port}")


@pytest.fixture(autouse=True)
def _require_integration(request: pytest.FixtureRequest) -> None:
    """Auto-applied: integration marker is required for this directory."""


@pytest.fixture
def require_redis() -> None:
    _skip_unless("Redis", "localhost", 6380)


@pytest.fixture
def require_postgres() -> None:
    _skip_unless("PostgreSQL", "localhost", 5433)


@pytest.fixture
def require_chromadb() -> None:
    _skip_unless("ChromaDB", "localhost", 8100)
