"""Shared fixtures, markers, and service availability guards for anycode tests."""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env.test for test environment variables
_env_test = Path(__file__).resolve().parent.parent / ".env.test"
if _env_test.exists():
    load_dotenv(_env_test, override=False)

REDIS_URL = os.environ.get("ANYCODE_TEST_REDIS_URL", "redis://localhost:6380/0")
POSTGRES_URL = os.environ.get("ANYCODE_TEST_POSTGRES_URL", "postgresql://anycode_test:testpass@localhost:5433/anycode_test")
CHROMADB_URL = os.environ.get("ANYCODE_TEST_CHROMADB_URL", "http://localhost:8100")


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture
def redis_available() -> bool:
    return _is_port_open("localhost", 6380)


@pytest.fixture
def postgres_available() -> bool:
    return _is_port_open("localhost", 5433)


@pytest.fixture
def chromadb_available() -> bool:
    return _is_port_open("localhost", 8100)


@pytest.fixture
def redis_url() -> str:
    return REDIS_URL


@pytest.fixture
def postgres_url() -> str:
    return POSTGRES_URL


@pytest.fixture
def chromadb_url() -> str:
    return CHROMADB_URL
