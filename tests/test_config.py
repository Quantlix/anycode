"""Tests for YAML/TOML config loader (Phase 4.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from anycode.config.loader import load_config
from anycode.config.validator import validate_config

_YAML_BASIC = """\
name: demo-team
agents:
  - name: alice
    model: claude-haiku-4-5
    provider: anthropic
    system_prompt: Be helpful.
    tools: []
tasks:
  - title: Plan
    description: Plan the work.
    assignee: alice
"""

_YAML_ENV = """\
name: env-team
agents:
  - name: bot
    model: ${ANYCODE_TEST_MODEL}
    provider: anthropic
    system_prompt: hi
"""

_YAML_BAD = """\
name: bad
agents:
  - name: alice
    model: m
    provider: anthropic
    system_prompt: hi
tasks:
  - title: t1
    description: d
    assignee: nobody
  - title: t2
    description: d
    assignee: alice
    depends_on: [missing-task]
"""

_TOML_BASIC = """\
name = "toml-team"

[[agents]]
name = "alice"
model = "claude-haiku-4-5"
provider = "anthropic"
system_prompt = "Be helpful."
tools = []
"""


def test_load_yaml(tmp_path: Path) -> None:
    p = tmp_path / "team.yaml"
    p.write_text(_YAML_BASIC, encoding="utf-8")
    loaded = load_config(p)
    assert loaded.team.name == "demo-team"
    assert loaded.team.agents[0].name == "alice"
    assert loaded.tasks and loaded.tasks[0].title == "Plan"


def test_load_toml(tmp_path: Path) -> None:
    p = tmp_path / "team.toml"
    p.write_text(_TOML_BASIC, encoding="utf-8")
    loaded = load_config(p)
    assert loaded.team.name == "toml-team"
    assert loaded.team.agents[0].name == "alice"


def test_env_substitution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANYCODE_TEST_MODEL", "gpt-4o-mini")
    p = tmp_path / "team.yaml"
    p.write_text(_YAML_ENV, encoding="utf-8")
    loaded = load_config(p)
    assert loaded.team.agents[0].model == "gpt-4o-mini"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_validate_catches_unknown_assignee_and_dep(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(_YAML_BAD, encoding="utf-8")
    issues = validate_config(p)
    assert any("nobody" in i for i in issues)
    assert any("missing-task" in i for i in issues)


def test_unsupported_extension(tmp_path: Path) -> None:
    p = tmp_path / "team.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_to_orchestrator_config(tmp_path: Path) -> None:
    p = tmp_path / "team.yaml"
    p.write_text(_YAML_BASIC, encoding="utf-8")
    loaded = load_config(p)
    config = loaded.to_orchestrator_config()
    assert config is not None
