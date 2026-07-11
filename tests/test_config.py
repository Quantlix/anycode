"""Tests for YAML/TOML config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from anycode.config.loader import UnknownConfigFieldError, UnsupportedConfigVersionError, load_config
from anycode.config.validator import validate_config
from anycode.types import OrchestratorConfig

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

_YAML_PROVIDER_CAPACITY = """\
name: capacity-team
provider_resilience:
  max_concurrency: 3
  requests_per_minute: 120
  capacity_scope: shared-openai
  capacity_wait_timeout_seconds: 10
agents:
  - name: default-capacity
    model: gpt-4o-mini
    provider: openai
  - name: override-capacity
    model: gpt-4o-mini
    provider: openai
    provider_resilience:
      max_concurrency: 1
      capacity_scope: isolated-openai
"""

_YAML_TOOL_IDEMPOTENCY = """\
name: durable-tools
tool_idempotency:
  backend: sqlite
  path: .state/tool-claims.db
  redact_sensitive_data: false
agents:
  - name: operator
    model: gpt-4o-mini
    provider: openai
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


def test_unversioned_config_loads_as_v1(tmp_path: Path) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text(_YAML_BASIC, encoding="utf-8")

    assert load_config(path).format_version == 1


def test_future_config_version_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "future.yaml"
    path.write_text(f"format_version: 2\n{_YAML_BASIC}", encoding="utf-8")

    with pytest.raises(UnsupportedConfigVersionError, match="config format version 2"):
        load_config(path)


@pytest.mark.parametrize(
    "payload, field",
    [
        (f"future_root: true\n{_YAML_BASIC}", "future_root"),
        (_YAML_BASIC.replace("    tools: []", "    tools: []\n    future_agent: true"), "future_agent"),
        (_YAML_BASIC.replace("    assignee: alice", "    assignee: alice\n    future_task: true"), "future_task"),
        (f"context_engineering:\n  future_context: true\n{_YAML_BASIC}", "future_context"),
        (f"context_engineering:\n  window:\n    future_window: true\n{_YAML_BASIC}", "future_window"),
        (
            f"context_engineering:\n  sections:\n    tool_results:\n      future_section: true\n{_YAML_BASIC}",
            "future_section",
        ),
        (
            f"routing:\n  rules:\n    - condition: complex\n      target_model: gpt-4o-mini\n      future_rule: true\n{_YAML_BASIC}",
            "future_rule",
        ),
        (
            _YAML_PROVIDER_CAPACITY.replace(
                "  max_concurrency: 3",
                "  max_concurrency: 3\n  retry:\n    future_retry_option: true",
            ),
            "future_retry_option",
        ),
    ],
)
def test_unknown_config_fields_fail_closed(tmp_path: Path, payload: str, field: str) -> None:
    path = tmp_path / "unknown.yaml"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(UnknownConfigFieldError, match=field):
        load_config(path)


def test_programmatic_config_extra_behavior_is_unchanged() -> None:
    config = OrchestratorConfig.model_validate({"future_only": True})

    assert "future_only" not in config.model_dump()


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


def test_loads_global_and_per_agent_provider_capacity(tmp_path: Path) -> None:
    path = tmp_path / "capacity.yaml"
    path.write_text(_YAML_PROVIDER_CAPACITY, encoding="utf-8")

    loaded = load_config(path)
    orchestrator = loaded.to_orchestrator_config()

    assert orchestrator.provider_resilience is not None
    assert orchestrator.provider_resilience.max_concurrency == 3
    assert orchestrator.provider_resilience.requests_per_minute == 120
    assert loaded.team.agents[0].provider_resilience is None
    assert loaded.team.agents[1].provider_resilience is not None
    assert loaded.team.agents[1].provider_resilience.max_concurrency == 1


def test_loads_tool_idempotency_store_config(tmp_path: Path) -> None:
    path = tmp_path / "idempotency.yaml"
    path.write_text(_YAML_TOOL_IDEMPOTENCY, encoding="utf-8")

    loaded = load_config(path)
    config = loaded.to_orchestrator_config()

    assert config.tool_idempotency.backend == "sqlite"
    assert config.tool_idempotency.path == ".state/tool-claims.db"
    assert config.tool_idempotency.redact_sensitive_data is False
