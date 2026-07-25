"""Tests for the state-graph workflow runtime."""

from __future__ import annotations

import threading
from typing import Annotated

import pytest
from pydantic import BaseModel, ConfigDict

from anycode import END, START, Agent, Command, Workflow, WorkflowError
from anycode.providers.fake import FakeAdapter, FakeResponse
from anycode.workflow import add, keep_first, keep_last, merge
from anycode.workflow.state import apply_patch, coerce_state, collect_reducers, merge_concurrent_patches


class Draft(BaseModel):
    model_config = ConfigDict(frozen=True)

    topic: str = ""
    draft: str = ""
    critique: str = ""
    rounds: int = 0
    notes: Annotated[list[str], add] = []


# ---------------------------------------------------------------------------
# Linear execution
# ---------------------------------------------------------------------------


async def test_linear_graph_merges_patches_in_order() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("first", lambda state: {"draft": "d1"})
    workflow.add_node("second", lambda state: {"critique": f"about {state.draft}"})
    workflow.add_edge(START, "first")
    workflow.add_edge("first", "second")
    workflow.add_edge("second", END)

    result = await workflow.compile().run(Draft(topic="x"))
    assert result.success
    assert result.state.draft == "d1"
    assert result.state.critique == "about d1"
    assert result.path == ("first", "second")
    assert result.steps == 2


async def test_node_without_outgoing_edge_terminates() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("only", lambda state: {"draft": "done"})
    workflow.set_entry("only")

    result = await workflow.compile().run()
    assert result.success
    assert result.state.draft == "done"


async def test_untyped_workflow_uses_dict_state() -> None:
    workflow = Workflow()
    workflow.add_node("bump", lambda state: {"count": state.get("count", 0) + 1})
    workflow.set_entry("bump")

    result = await workflow.compile().run({"count": 4})
    assert result.state == {"count": 5}


async def test_returning_a_state_instance_is_treated_as_a_patch() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("replace", lambda state: Draft(topic="t", draft="whole"))
    workflow.set_entry("replace")

    result = await workflow.compile().run()
    assert result.state.draft == "whole"
    assert result.state.topic == "t"


async def test_returning_none_leaves_state_untouched() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("noop", lambda state: None)
    workflow.set_entry("noop")

    result = await workflow.compile().run(Draft(draft="kept"))
    assert result.state.draft == "kept"


# ---------------------------------------------------------------------------
# Conditional routing and loops
# ---------------------------------------------------------------------------


def _review_loop(limit: int = 3) -> Workflow:
    workflow = Workflow(Draft)
    workflow.add_node("write", lambda state: {"draft": f"draft-{state.rounds + 1}", "rounds": state.rounds + 1})
    workflow.add_node("review", lambda state: {"critique": "APPROVED" if state.rounds >= limit else "more work"})
    workflow.add_edge(START, "write")
    workflow.add_edge("write", "review")
    workflow.add_conditional_edge("review", lambda state: END if "APPROVED" in state.critique else "write")
    return workflow


async def test_conditional_edge_loops_until_approved() -> None:
    result = await _review_loop(limit=3).compile().run(Draft(topic="t"))
    assert result.success
    assert result.state.rounds == 3
    assert result.path.count("write") == 3


async def test_conditional_edge_with_path_map() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("check", lambda state: {"critique": "ok"})
    workflow.add_node("fix", lambda state: {"draft": "fixed"})
    workflow.add_edge(START, "check")
    workflow.add_conditional_edge("check", lambda state: "pass" if state.critique == "ok" else "fail", {"pass": END, "fail": "fix"})
    workflow.add_edge("fix", END)

    result = await workflow.compile().run()
    assert result.path == ("check",)


async def test_router_returning_an_unmapped_key_is_rejected() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("check", lambda state: {})
    workflow.add_node("fix", lambda state: {})
    workflow.add_edge(START, "check")
    workflow.add_conditional_edge("check", lambda state: "surprise", {"pass": END, "fail": "fix"})
    workflow.add_edge("fix", END)

    with pytest.raises(WorkflowError, match="not in its path map"):
        await workflow.compile().run()


async def test_max_steps_stops_a_runaway_loop() -> None:
    result = await _review_loop(limit=99).compile().run(Draft(topic="t"), max_steps=6)
    assert not result.success
    assert result.stop_reason is not None
    assert result.stop_reason.code == "max_steps"
    assert result.stop_reason.recoverable
    assert result.steps == 6


async def test_max_steps_must_be_positive() -> None:
    workflow = _review_loop()
    with pytest.raises(WorkflowError, match="at least 1"):
        await workflow.compile().run(Draft(), max_steps=0)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


async def test_command_overrides_the_static_edge() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("start", lambda state: Command(goto="skipped_to", update={"draft": "via command"}), goto=["skipped_to"])
    workflow.add_node("normal", lambda state: {"critique": "should not run"})
    workflow.add_node("skipped_to", lambda state: {"critique": "arrived"})
    workflow.add_edge(START, "start")
    workflow.add_edge("start", "normal")
    workflow.add_edge("normal", END)
    workflow.add_edge("skipped_to", END)

    result = await workflow.compile().run()
    assert result.path == ("start", "skipped_to")
    assert result.state.draft == "via command"
    assert result.state.critique == "arrived"


async def test_command_can_terminate() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("start", lambda state: Command(goto=END, update={"draft": "early exit"}))
    workflow.add_node("never", lambda state: {"critique": "unreachable in practice"})
    workflow.add_edge(START, "start")
    workflow.add_edge("start", "never")
    workflow.add_edge("never", END)

    result = await workflow.compile().run()
    assert result.path == ("start",)
    assert result.state.draft == "early exit"


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------


def test_built_in_reducers() -> None:
    assert add([1], [2]) == [1, 2]
    assert add([1], 2) == [1, 2]
    assert add("a", "b") == "ab"
    assert add(1, 2) == 3
    assert add(None, [1]) == [1]
    assert merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert keep_first("held", "new") == "held"
    assert keep_first("", "new") == "new"
    assert keep_last("old", "new") == "new"


def test_add_reducer_rejects_incompatible_types() -> None:
    with pytest.raises(WorkflowError, match="cannot combine"):
        add(True, 1)


def test_merge_reducer_rejects_non_mappings() -> None:
    with pytest.raises(WorkflowError, match="needs two mappings"):
        merge(1, 2)


def test_reducers_are_collected_from_annotations() -> None:
    assert collect_reducers(Draft) == {"notes": add}


async def test_annotated_field_accumulates_across_steps() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("one", lambda state: {"notes": ["a"]})
    workflow.add_node("two", lambda state: {"notes": ["b"]})
    workflow.add_edge(START, "one")
    workflow.add_edge("one", "two")
    workflow.add_edge("two", END)

    result = await workflow.compile().run()
    assert result.state.notes == ["a", "b"]


def test_apply_patch_replaces_unreduced_fields() -> None:
    state = apply_patch(Draft(draft="old"), {"draft": "new"}, schema=Draft, reducers={})
    assert state.draft == "new"


# ---------------------------------------------------------------------------
# Patch validation
# ---------------------------------------------------------------------------


async def test_unknown_patch_field_names_the_valid_set() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("bad", lambda state: {"nope": 1})
    workflow.set_entry("bad")

    result = await workflow.compile().run()
    assert not result.success
    assert result.error is not None
    assert "unknown state field(s): nope" in result.error


async def test_non_mapping_return_is_rejected() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("bad", lambda state: 42)
    workflow.set_entry("bad")

    result = await workflow.compile().run()
    assert not result.success
    assert result.error is not None
    assert "must return a dict of changed fields" in result.error


def test_coerce_state_accepts_mappings_and_instances() -> None:
    assert coerce_state(Draft, {"draft": "d"}).draft == "d"
    assert coerce_state(Draft, None).draft == ""
    assert coerce_state(None, None) == {}
    with pytest.raises(WorkflowError, match="Expected Draft"):
        coerce_state(Draft, 3)
    with pytest.raises(WorkflowError, match="needs a mapping"):
        coerce_state(None, 3)


# ---------------------------------------------------------------------------
# Compile-time validation
# ---------------------------------------------------------------------------


def test_compile_reports_a_missing_entry_point() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("a", lambda state: {})
    with pytest.raises(WorkflowError, match="no entry point"):
        workflow.compile()


def test_compile_reports_an_empty_graph() -> None:
    with pytest.raises(WorkflowError, match="no nodes are registered"):
        Workflow(Draft).compile()


def test_compile_reports_an_edge_to_an_unregistered_node() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("a", lambda state: {})
    workflow.add_edge(START, "a")
    workflow.add_edge("a", "ghost")
    with pytest.raises(WorkflowError, match='"a" -> "ghost" points at an unregistered node'):
        workflow.compile()


def test_compile_reports_an_unreachable_node() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("a", lambda state: {})
    workflow.add_node("orphan", lambda state: {})
    workflow.add_edge(START, "a")
    workflow.add_edge("a", END)
    workflow.add_edge("orphan", END)
    with pytest.raises(WorkflowError, match='"orphan" is unreachable'):
        workflow.compile()


def test_compile_reports_a_graph_that_cannot_terminate() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("a", lambda state: {})
    workflow.add_node("b", lambda state: {})
    workflow.add_edge(START, "a")
    workflow.add_edge("a", "b")
    workflow.add_edge("b", "a")
    with pytest.raises(WorkflowError, match="can never reach END"):
        workflow.compile()


def test_compile_reports_a_bad_path_map_target() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("a", lambda state: {})
    workflow.add_edge(START, "a")
    workflow.add_conditional_edge("a", lambda state: "x", {"x": "ghost"})
    with pytest.raises(WorkflowError, match='maps "x" to unregistered node "ghost"'):
        workflow.compile()


def test_compile_aggregates_every_problem() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("a", lambda state: {})
    workflow.add_node("orphan", lambda state: {})
    workflow.add_edge(START, "a")
    workflow.add_edge("a", "ghost")
    with pytest.raises(WorkflowError) as error:
        workflow.compile()
    assert str(error.value).count("- ") >= 2


def test_duplicate_node_names_are_rejected() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("a", lambda state: {})
    with pytest.raises(WorkflowError, match="already registered"):
        workflow.add_node("a", lambda state: {})


def test_reserved_node_names_are_rejected() -> None:
    with pytest.raises(WorkflowError, match="reserved"):
        Workflow(Draft).add_node(END, lambda state: {})


def test_a_node_cannot_have_both_edge_kinds() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("a", lambda state: {})
    workflow.add_edge("a", END)
    with pytest.raises(WorkflowError, match="already has a static edge"):
        workflow.add_conditional_edge("a", lambda state: END)


def test_two_entry_points_are_rejected() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("a", lambda state: {})
    workflow.add_node("b", lambda state: {})
    workflow.set_entry("a")
    with pytest.raises(WorkflowError, match="exactly one entry point"):
        workflow.set_entry("b")


def test_state_schema_must_be_a_model() -> None:
    with pytest.raises(WorkflowError, match="Pydantic model class"):
        Workflow(dict)  # type: ignore[arg-type]


def test_node_target_must_be_runnable() -> None:
    with pytest.raises(WorkflowError, match="Pass a function"):
        Workflow(Draft).add_node("a", 3)


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


async def test_fan_out_runs_targets_concurrently_and_merges() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("split", lambda state: {})
    workflow.add_node("left", lambda state: {"notes": ["left"]})
    workflow.add_node("right", lambda state: {"notes": ["right"]})
    workflow.add_edge(START, "split")
    workflow.add_edge("split", "left")
    workflow.add_edge("split", "right")
    workflow.add_edge("left", END)
    workflow.add_edge("right", END)

    result = await workflow.compile().run()
    assert result.success
    assert sorted(result.state.notes) == ["left", "right"]


async def test_conflicting_concurrent_writes_are_rejected() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("split", lambda state: {})
    workflow.add_node("left", lambda state: {"draft": "L"})
    workflow.add_node("right", lambda state: {"draft": "R"})
    workflow.add_edge(START, "split")
    workflow.add_edge("split", "left")
    workflow.add_edge("split", "right")
    workflow.add_edge("left", END)
    workflow.add_edge("right", END)

    with pytest.raises(WorkflowError, match="both wrote"):
        await workflow.compile().run()


def test_merge_concurrent_patches_uses_reducers() -> None:
    combined = merge_concurrent_patches([("a", {"notes": ["x"]}), ("b", {"notes": ["y"]})], {"notes": add})
    assert combined == {"notes": ["x", "y"]}


# ---------------------------------------------------------------------------
# Node kinds
# ---------------------------------------------------------------------------


async def test_sync_node_runs_off_the_event_loop() -> None:
    seen: list[int] = []

    def blocking(state: Draft) -> dict[str, str]:
        seen.append(threading.get_ident())
        return {"draft": "sync"}

    workflow = Workflow(Draft)
    workflow.add_node("blocking", blocking)
    workflow.set_entry("blocking")

    await workflow.compile().run()
    assert seen and seen[0] != threading.get_ident()


async def test_agent_node_reads_and_writes_state(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _create(*_args: object, **_kwargs: object) -> FakeAdapter:
        return FakeAdapter(responses=[FakeResponse(text="agent said so")])

    monkeypatch.setattr("anycode.core.agent.create_adapter", _create)
    agent = Agent(name="writer", model="m", provider="openai", tools=[])

    class AgentState(BaseModel):
        model_config = ConfigDict(frozen=True)

        question: str = ""
        answer: str = ""

    workflow = Workflow(AgentState)
    workflow.add_node("ask", agent, input_key="question", output_key="answer")
    workflow.set_entry("ask")

    result = await workflow.compile().run(AgentState(question="why?"))
    assert result.state.answer == "agent said so"


async def test_agent_node_reports_a_missing_input_field(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _create(*_args: object, **_kwargs: object) -> FakeAdapter:
        return FakeAdapter(responses=[FakeResponse(text="unused")])

    monkeypatch.setattr("anycode.core.agent.create_adapter", _create)
    agent = Agent(name="writer", model="m", provider="openai", tools=[])

    workflow = Workflow(Draft)
    workflow.add_node("ask", agent, input_key="topic", output_key="draft")
    workflow.set_entry("ask")

    result = await workflow.compile().run(Draft())
    assert not result.success
    assert result.error is not None
    assert 'reads state field "topic"' in result.error


async def test_sub_workflow_node_merges_its_final_state() -> None:
    inner = Workflow(Draft, name="inner")
    inner.add_node("inner_step", lambda state: {"critique": "from inner"})
    inner.set_entry("inner_step")

    outer = Workflow(Draft, name="outer")
    outer.add_node("prepare", lambda state: {"draft": "outer draft"})
    outer.add_node("delegate", inner.compile())
    outer.add_edge(START, "prepare")
    outer.add_edge("prepare", "delegate")
    outer.add_edge("delegate", END)

    result = await outer.compile().run()
    assert result.state.draft == "outer draft"
    assert result.state.critique == "from inner"


# ---------------------------------------------------------------------------
# Decorator, streaming, sync, and rendering
# ---------------------------------------------------------------------------


async def test_node_decorator_registers_by_function_name() -> None:
    workflow = Workflow(Draft)

    @workflow.node
    def written(state: Draft) -> dict[str, str]:
        return {"draft": "decorated"}

    workflow.set_entry("written")
    result = await workflow.compile().run()
    assert result.state.draft == "decorated"
    assert written(Draft()) == {"draft": "decorated"}


async def test_node_decorator_accepts_an_explicit_name() -> None:
    workflow = Workflow(Draft)

    @workflow.node(name="custom")
    def anything(state: Draft) -> dict[str, str]:
        return {"draft": "named"}

    workflow.set_entry("custom")
    assert (await workflow.compile().run()).state.draft == "named"


async def test_stream_emits_the_expected_event_sequence() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("one", lambda state: {"draft": "d"})
    workflow.add_node("two", lambda state: {"critique": "c"})
    workflow.add_edge(START, "one")
    workflow.add_edge("one", "two")
    workflow.add_edge("two", END)

    events = [event async for event in workflow.compile().stream()]
    assert [event.type for event in events] == [
        "node_start",
        "node_end",
        "route",
        "node_start",
        "node_end",
        "route",
        "done",
    ]
    assert events[-1].result is not None
    assert events[-1].result.success


def test_run_sync_executes_the_graph() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("only", lambda state: {"draft": "blocking"})
    workflow.set_entry("only")

    assert workflow.compile().run_sync().state.draft == "blocking"


def test_to_mermaid_contains_every_node_and_edge() -> None:
    compiled = _review_loop().compile()
    diagram = compiled.to_mermaid()
    assert "flowchart TD" in diagram
    assert "write --> review" in diagram
    assert "__start__ --> write" in diagram
    assert "review -.->|?| write" in diagram
    assert "review -.->|?| __end__" in diagram


def test_to_dict_describes_the_graph() -> None:
    described = _review_loop().compile().to_dict()
    assert described["entry"] == "write"
    assert described["state_schema"] == "Draft"
    assert {node["name"] for node in described["nodes"]} == {"write", "review"}
    assert described["conditional_edges"][0]["dynamic"] is True


def test_compiled_workflow_reports_its_shape() -> None:
    compiled = _review_loop().compile()
    assert compiled.nodes == ("write", "review")
    assert compiled.entry == "write"
    assert repr(compiled) == "CompiledWorkflow(name='workflow', nodes=2, entry='write')"


def test_editing_the_builder_does_not_change_a_compiled_graph() -> None:
    workflow = Workflow(Draft)
    workflow.add_node("a", lambda state: {})
    workflow.set_entry("a")
    compiled = workflow.compile()
    workflow.add_node("b", lambda state: {})
    assert compiled.nodes == ("a",)
