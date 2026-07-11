"""Tests for the harness component registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from anycode.harness import (
    HarnessRegistry,
    build_default_registry,
    build_manifest,
    diff_manifests,
    load_manifest,
    save_manifest,
)
from anycode.harness.component import (
    REDACTED_CHECKSUM_MARKER,
    compute_checksum,
    make_component,
    redact_for_checksum,
)
from anycode.tools.built_in import register_built_in_tools
from anycode.tools.registry import ToolRegistry
from anycode.types import (
    AgentConfig,
    ContextPolicy,
    HarnessComponent,
    OrchestratorConfig,
    RoutingConfig,
    RoutingRule,
    TeamConfig,
    VerificationSensorConfig,
)

# -- checksum / redaction --


def test_checksum_is_deterministic_for_same_payload() -> None:
    payload = {"a": 1, "b": [1, 2, 3], "c": {"d": "e"}}
    assert compute_checksum(payload) == compute_checksum(payload)


def test_checksum_changes_when_payload_changes() -> None:
    payload_a = {"x": 1}
    payload_b = {"x": 2}
    assert compute_checksum(payload_a) != compute_checksum(payload_b)


def test_redact_for_checksum_masks_sensitive_keys() -> None:
    payload = {
        "model": "claude-sonnet-4-5",
        "api_key": "sk-secret-value",
        "nested": {"PASSWORD": "abc", "ok": "hi"},
    }
    redacted = redact_for_checksum(payload)
    assert redacted["api_key"] == REDACTED_CHECKSUM_MARKER
    assert redacted["nested"]["PASSWORD"] == REDACTED_CHECKSUM_MARKER
    assert redacted["nested"]["ok"] == "hi"


def test_checksum_redacts_secrets_before_hashing() -> None:
    a = {"api_key": "sk-1"}
    b = {"api_key": "sk-2"}
    assert compute_checksum(a) == compute_checksum(b)


# -- make_component --


def test_make_component_defaults_core_components_to_non_editable() -> None:
    component = make_component(
        id="tool:read",
        kind="tool",
        source="file_read",
        owner="core",
        description="read",
        payload={"x": 1},
    )
    assert component.editable is False
    assert component.owner == "core"


def test_make_component_config_components_default_editable() -> None:
    component = make_component(
        id="prompt:agent:foo",
        kind="prompt",
        source="config",
        owner="config",
        description="prompt",
        payload={"text": "hello"},
    )
    assert component.editable is True


def test_make_component_override_editable() -> None:
    component = make_component(
        id="prompt:locked",
        kind="prompt",
        source="config",
        owner="config",
        description="locked",
        editable=False,
        payload={},
    )
    assert component.editable is False


# -- registry behaviour --


def test_registry_register_and_get() -> None:
    registry = HarnessRegistry()
    comp = make_component(
        id="prompt:a",
        kind="prompt",
        source="x",
        owner="config",
        description="d",
        payload={"a": 1},
    )
    registry.register(comp)
    assert registry.get("prompt:a") is comp
    assert "prompt:a" in registry
    assert len(registry) == 1


def test_registry_rejects_duplicate_ids() -> None:
    registry = HarnessRegistry()
    comp = make_component(
        id="prompt:dup",
        kind="prompt",
        source="x",
        owner="config",
        description="d",
        payload={},
    )
    registry.register(comp)
    with pytest.raises(ValueError):
        registry.register(comp)


def test_registry_filters_by_kind_and_editability() -> None:
    registry = HarnessRegistry()
    registry.register(make_component(id="t:a", kind="tool", source="s", owner="core", description="", payload={}))
    registry.register(make_component(id="p:a", kind="prompt", source="s", owner="config", description="", payload={"text": "hi"}))
    tools = registry.list(kind="tool")
    prompts = registry.list(kind="prompt")
    editable = registry.list(editable=True)
    assert {c.id for c in tools} == {"t:a"}
    assert {c.id for c in prompts} == {"p:a"}
    assert {c.id for c in editable} == {"p:a"}


# -- build_default_registry --


def _team_with_agent() -> TeamConfig:
    return TeamConfig(
        name="demo",
        agents=[
            AgentConfig(
                name="alice",
                model="claude-sonnet-4-5",
                provider="anthropic",
                system_prompt="You are helpful.",
                tools=["read"],
                context_policy=ContextPolicy(enabled=True, mode="manual"),
                verification=(VerificationSensorConfig(name="ruff", kind="computational"),),
            )
        ],
    )


def test_build_default_registry_from_team_and_orchestrator() -> None:
    team = _team_with_agent()
    orchestrator = OrchestratorConfig(
        routing=RoutingConfig(
            enabled=True,
            default_model="claude-sonnet-4-5",
            rules=[RoutingRule(condition="complexity == expert", target_model="claude-opus-4-1")],
        ),
        verification=(VerificationSensorConfig(name="pytest", kind="computational"),),
    )
    tool_registry = ToolRegistry()
    register_built_in_tools(tool_registry)
    registry = build_default_registry(
        team=team,
        orchestrator=orchestrator,
        tools=tool_registry,
    )
    component_ids = {c.id for c in registry.list()}
    assert "prompt:agent:alice" in component_ids
    assert "provider:agent:alice" in component_ids
    assert "context_policy:agent:alice" in component_ids
    assert "verification:agent:alice:ruff" in component_ids
    assert "verification:global:pytest" in component_ids
    assert "routing_policy:global" in component_ids
    assert any(cid.startswith("tool:") for cid in component_ids)


def test_build_default_registry_skips_duplicate_agents() -> None:
    team = _team_with_agent()
    registry = build_default_registry(team=team, agents=list(team.agents))
    # alice should appear once
    matching = [c for c in registry.list() if c.id == "provider:agent:alice"]
    assert len(matching) == 1


# -- manifest --


def test_build_manifest_is_deterministic(tmp_path: Path) -> None:
    registry = build_default_registry(team=_team_with_agent())
    a = build_manifest(registry)
    b = build_manifest(registry)
    assert a.checksum == b.checksum
    assert tuple(c.id for c in a.components) == tuple(c.id for c in b.components)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    registry = build_default_registry(team=_team_with_agent())
    snapshot = build_manifest(registry, notes="test")
    target = save_manifest(snapshot, tmp_path / "manifest.json")
    restored = load_manifest(target)
    assert restored.checksum == snapshot.checksum
    assert restored.notes == "test"
    assert tuple(c.id for c in restored.components) == tuple(c.id for c in snapshot.components)


def test_manifest_persistence_redacts_notes(tmp_path: Path) -> None:
    snapshot = build_manifest(build_default_registry(team=_team_with_agent()), notes="api_key=plain-value")

    target = save_manifest(snapshot, tmp_path / "manifest.json")

    assert "plain-value" not in target.read_text(encoding="utf-8")
    assert load_manifest(target).notes == "<redacted-secret>"


def test_diff_manifests_detects_drift() -> None:
    team_a = _team_with_agent()
    team_b = TeamConfig(
        name="demo",
        agents=[team_a.agents[0].model_copy(update={"system_prompt": "Different prompt."})],
    )
    snap_a = build_manifest(build_default_registry(team=team_a))
    snap_b = build_manifest(build_default_registry(team=team_b))
    diff = diff_manifests(snap_a, snap_b)
    assert diff["drift_detected"] is True
    changed_ids = {entry["id"] for entry in diff["changed"]}
    assert "prompt:agent:alice" in changed_ids


def test_diff_manifests_no_drift_for_identical_state() -> None:
    team = _team_with_agent()
    snap_a = build_manifest(build_default_registry(team=team))
    snap_b = build_manifest(build_default_registry(team=team))
    diff = diff_manifests(snap_a, snap_b)
    assert diff["drift_detected"] is False
    assert diff["changed"] == []


def test_manifest_includes_checksum_for_evaluation_reports() -> None:
    registry = build_default_registry(team=_team_with_agent())
    snapshot = build_manifest(registry)
    assert isinstance(snapshot.checksum, str) and len(snapshot.checksum) == 64
    # Components are immutable
    assert isinstance(snapshot.components[0], HarnessComponent)
    with pytest.raises(Exception):
        snapshot.components[0].id = "x"  # type: ignore[misc]
