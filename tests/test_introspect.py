"""Tests for the machine-readable API description and the `anycode api` command."""

from __future__ import annotations

import inspect
import json

import pytest

import anycode
from anycode import CORE_SURFACE, ApiEntry, ApiMap, describe
from anycode.introspect import render_entry, render_text, to_json

typer_testing = pytest.importorskip("typer.testing")

from anycode.cli.main import app  # noqa: E402

runner = typer_testing.CliRunner()

# Output budgets, in characters. The core surface is what an agent should read first;
# the full dump is a reference, not something to paste into a prompt.
MAX_CORE_TEXT_CHARS = 8_000
MAX_CORE_JSON_CHARS = 12_000


def test_describe_covers_the_whole_public_surface() -> None:
    api = describe()
    assert isinstance(api, ApiMap)
    assert set(api.names()) == set(anycode.__all__)


def test_every_entry_has_a_module_and_a_signature() -> None:
    for entry in describe().entries:
        assert entry.module.startswith("anycode"), entry.name
        assert entry.signature, entry.name


def test_describe_one_symbol_matches_the_real_signature() -> None:
    entry = describe("Agent")
    assert isinstance(entry, ApiEntry)
    assert entry.kind == "class"
    assert entry.module == "anycode.core.agent"
    assert entry.signature == f"Agent{inspect.signature(anycode.Agent)}"
    assert entry.summary.startswith("High-level agent")


def test_describe_rejects_a_private_symbol() -> None:
    with pytest.raises(AttributeError, match="not part of the AnyCode public API"):
        describe("_lazy")


def test_core_surface_is_a_subset_of_the_public_api() -> None:
    assert set(CORE_SURFACE) <= set(anycode.__all__)
    assert describe(core=True).names() == CORE_SURFACE


def test_kind_filter_narrows_the_map() -> None:
    functions = describe(kind="function")
    assert functions.entries
    assert {entry.kind for entry in functions.entries} == {"function"}


def test_models_report_their_fields_not_pydantic_boilerplate() -> None:
    entry = describe("ToolResult")
    assert entry.kind == "model"
    assert entry.signature == "ToolResult(data: str, is_error: bool | None, retry_safe: bool | None)"
    assert "Usage Documentation" not in entry.summary


def test_constants_show_their_value_rather_than_their_type_docstring() -> None:
    entry = describe("START")
    assert entry.signature == "START: str = '__start__'"
    assert entry.summary == ""


def test_render_text_groups_by_module() -> None:
    rendered = render_text(describe(core=True))
    assert "anycode.core.agent" in rendered
    assert "Agent(" in rendered


def test_render_text_can_omit_signatures() -> None:
    rendered = render_text(describe(core=True), show_signature=False)
    assert "Agent  —" in rendered
    assert "Agent(config" not in rendered


def test_render_entry_includes_kind_and_module() -> None:
    rendered = render_entry(describe("Crew"))
    assert "kind:   class" in rendered
    assert "module: anycode.crew" in rendered


def test_to_json_is_stable_and_ordered() -> None:
    first = json.dumps(to_json(describe(core=True)))
    second = json.dumps(to_json(describe(core=True)))
    assert first == second
    payload = json.loads(first)
    assert payload["count"] == len(CORE_SURFACE)
    assert list(payload["symbols"][0]) == ["name", "kind", "module", "signature", "summary"]


def test_compact_json_drops_signatures() -> None:
    payload = to_json(describe(core=True), compact=True)
    assert "signature" not in payload["symbols"][0]
    assert len(json.dumps(payload)) < len(json.dumps(to_json(describe(core=True))))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_api_core_stays_within_its_budget() -> None:
    result = runner.invoke(app, ["api", "--core"])
    assert result.exit_code == 0
    assert len(result.output) < MAX_CORE_TEXT_CHARS
    for name in CORE_SURFACE:
        assert name in result.output


def test_api_core_json_stays_within_its_budget() -> None:
    result = runner.invoke(app, ["api", "--core", "--json"])
    assert result.exit_code == 0
    assert len(result.output) < MAX_CORE_JSON_CHARS
    payload = json.loads(result.output)
    assert payload["count"] == len(CORE_SURFACE)


def test_api_full_json_parses() -> None:
    result = runner.invoke(app, ["api", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["count"] == len(anycode.__all__)


def test_api_single_symbol() -> None:
    result = runner.invoke(app, ["api", "Workflow"])
    assert result.exit_code == 0
    assert "module: anycode.workflow" in result.output


def test_api_unknown_symbol_exits_nonzero() -> None:
    result = runner.invoke(app, ["api", "Nope"])
    assert result.exit_code == 1
    assert "not part of the AnyCode public API" in result.output


def test_api_rejects_an_unknown_kind() -> None:
    result = runner.invoke(app, ["api", "--kind", "widget"])
    assert result.exit_code == 1
    assert "Unknown kind" in result.output
