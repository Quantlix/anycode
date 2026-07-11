"""Tests for CI compatibility and optional-dependency coverage."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).parent.parent
_CI_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"
_PACKAGE_WORKFLOW = _ROOT / ".github" / "workflows" / "package-validation.yml"
_COMPOSE_FILE = _ROOT / "docker-compose.yml"


def _load_ci_workflow() -> dict[str, Any]:
    return yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))


def _load_project() -> dict[str, Any]:
    return tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_ci_covers_supported_python_and_operating_systems() -> None:
    matrix = _load_ci_workflow()["jobs"]["tests"]["strategy"]["matrix"]
    supported_python = {
        classifier.rsplit(" :: ", 1)[-1]
        for classifier in _load_project()["classifiers"]
        if classifier.startswith("Programming Language :: Python :: 3.")
    }

    assert set(matrix["python-version"]) == supported_python == {"3.12", "3.13"}
    assert set(matrix["os"]) == {"ubuntu-latest", "windows-latest", "macos-latest"}


def test_ci_covers_every_declared_optional_extra() -> None:
    declared = set(_load_project()["optional-dependencies"])
    entries = _load_ci_workflow()["jobs"]["optional-extras"]["strategy"]["matrix"]["include"]
    covered = {entry["extra"] for entry in entries}

    assert covered == declared | {"core"}
    assert len(entries) == len(covered)
    assert next(entry for entry in entries if entry["extra"] == "core")["sync-args"] == ""
    assert all(entry["sync-args"] == f"--extra {entry['extra']}" for entry in entries if entry["extra"] != "core")
    assert all(entry["imports"] for entry in entries)


def test_ci_runs_all_service_backed_integration_tests() -> None:
    integration = _load_ci_workflow()["jobs"]["integration"]
    compose = yaml.safe_load(_COMPOSE_FILE.read_text(encoding="utf-8"))
    commands = "\n".join(step.get("run", "") for step in integration["steps"])
    chromadb_options = integration["services"]["chromadb"]["options"]

    assert set(integration["services"]) == {"redis", "chromadb"}
    assert all(service["image"] == compose["services"][name]["image"] for name, service in integration["services"].items())
    assert "curl" not in chromadb_options
    assert "uv sync --locked --group dev" in commands
    assert "/api/v2/heartbeat" in commands
    assert "python -m pytest tests/integration -m integration" in commands


def test_compose_chromadb_healthcheck_uses_available_tools() -> None:
    compose = yaml.safe_load(_COMPOSE_FILE.read_text(encoding="utf-8"))
    command = " ".join(compose["services"]["chromadb"]["healthcheck"]["test"])

    assert "curl" not in command
    assert "/api/v2/heartbeat" in command


def test_package_validation_owns_built_distribution_checks() -> None:
    workflow = yaml.safe_load(_PACKAGE_WORKFLOW.read_text(encoding="utf-8"))
    validate = workflow["jobs"]["validate"]
    commands = "\n".join(step.get("run", "") for step in validate["steps"])

    assert set(validate["strategy"]["matrix"]["python-version"]) == {"3.12", "3.13"}
    assert validate["timeout-minutes"] == 30
    assert "uv sync --locked --group dev" in commands
    assert commands.count("python -m pip install") == 3
    assert "[cli]" in commands
    assert "from anycode import AnyCode" in commands
    assert "pytest" not in commands
