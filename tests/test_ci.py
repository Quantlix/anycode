"""Tests for CI compatibility and optional-dependency coverage."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import yaml
from mkdocs.config import load_config

_ROOT = Path(__file__).parent.parent
_CI_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"
_DOCS_WORKFLOW = _ROOT / ".github" / "workflows" / "docs.yml"
_PACKAGE_WORKFLOW = _ROOT / ".github" / "workflows" / "package-validation.yml"
_PUBLISH_WORKFLOWS = (
    _ROOT / ".github" / "workflows" / "publish-testpypi.yml",
    _ROOT / ".github" / "workflows" / "publish-pypi.yml",
)
_COMPOSE_FILE = _ROOT / "docker-compose.yml"
_MKDOCS_CONFIG = _ROOT / "mkdocs.yml"
_DOCS_OVERRIDE = _ROOT / "overrides" / "main.html"


def _load_ci_workflow() -> dict[str, Any]:
    return yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))


def _load_workflow(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _job_commands(job: dict[str, Any]) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


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
    workflow = _load_workflow(_PACKAGE_WORKFLOW)
    validate = workflow["jobs"]["validate"]
    commands = _job_commands(validate)

    assert set(validate["strategy"]["matrix"]["python-version"]) == {"3.12", "3.13"}
    assert validate["timeout-minutes"] == 30
    assert "uv sync --locked --group dev" in commands
    assert commands.count("python -m pip install") == 3
    assert "[cli]" in commands
    assert "from anycode import AnyCode" in commands
    assert "pytest" not in commands


def test_documentation_deploys_only_from_release_tag_pushes() -> None:
    jobs = _load_workflow(_DOCS_WORKFLOW)["jobs"]
    validate_commands = _job_commands(jobs["validate"])
    deploy_commands = _job_commands(jobs["deploy"])

    assert "workflow_dispatch" in jobs["validate"]["if"]
    assert "refs/heads/main" in jobs["validate"]["if"]
    assert "github.event_name == 'push'" in jobs["deploy"]["if"]
    assert "refs/tags/v" in jobs["deploy"]["if"]
    assert "uv sync --locked --group dev" in validate_commands
    assert "python -m mkdocs build --strict" in validate_commands
    assert "scripts/check_docs.py" in validate_commands
    assert "check_versions.py --tag" in deploy_commands
    assert "steps.ver.outputs.prerelease" in deploy_commands
    assert "--update-aliases" in deploy_commands
    assert "mike set-default --push latest" in deploy_commands


def test_documentation_version_switcher_and_social_metadata_have_single_owners() -> None:
    config = load_config(config_file=str(_MKDOCS_CONFIG))
    version = config["extra"]["version"]
    override = _DOCS_OVERRIDE.read_text(encoding="utf-8")

    assert version == {"provider": "mike", "default": "latest", "alias": True}
    assert '<meta property="og:' not in override
    assert '<meta name="twitter:' not in override


def test_publish_workflows_run_the_release_gate_with_trusted_publishing() -> None:
    for path in _PUBLISH_WORKFLOWS:
        workflow = _load_workflow(path)
        build = workflow["jobs"]["build"]
        publish = workflow["jobs"]["publish"]
        commands = _job_commands(build)

        assert build["permissions"] == {"contents": "read"}
        assert "uv sync --locked --group dev" in commands
        assert "python -m ruff check ." in commands
        assert "python -m ruff format --check src/" in commands
        assert "python -m pyright" in commands
        assert "python -m pytest" in commands
        assert "python -m mkdocs build --strict" in commands
        assert "scripts/check_docs.py" in commands
        assert "python -m build --no-isolation" in commands
        assert "python -m twine check --strict" in commands
        assert publish["permissions"] == {"id-token": "write"}
        assert publish["environment"]["name"] in {"testpypi", "pypi"}
        assert "skip-existing" not in str(publish)
