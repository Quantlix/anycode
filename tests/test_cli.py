"""Tests for the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

typer = pytest.importorskip("typer")
from typer.testing import CliRunner  # noqa: E402

from anycode.cli.main import app  # noqa: E402

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "anycode" in result.stdout.lower()


def test_inspect_tools() -> None:
    result = runner.invoke(app, ["inspect", "tools"])
    assert result.exit_code == 0
    assert "Built-in Tools" in result.stdout


def test_inspect_providers() -> None:
    result = runner.invoke(app, ["inspect", "providers"])
    assert result.exit_code == 0
    assert "anthropic" in result.stdout


def test_init_creates_project(tmp_path: Path) -> None:
    project = tmp_path / "myproj"
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 0, result.stdout
    assert (project / "team.yaml").exists()
    assert (project / "main.py").exists()
    assert (project / ".env.example").exists()
    assert (project / ".gitignore").exists()


def test_inspect_team_validates(tmp_path: Path) -> None:
    project = tmp_path / "p"
    runner.invoke(app, ["init", str(project)])
    result = runner.invoke(app, ["inspect", "team", str(project / "team.yaml")])
    assert result.exit_code == 0
    assert "backend-crew" in result.stdout


def test_inspect_config_only_validates(tmp_path: Path) -> None:
    project = tmp_path / "p"
    runner.invoke(app, ["init", str(project)])
    result = runner.invoke(app, ["inspect", "config", str(project / "team.yaml")])
    assert result.exit_code == 0
    assert "OK" in result.stdout


def test_init_refuses_existing_nonempty(tmp_path: Path) -> None:
    project = tmp_path / "p"
    project.mkdir()
    (project / "x.txt").write_text("hello")
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code != 0
