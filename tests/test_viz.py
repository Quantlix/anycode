"""Tests for DAG and timeline visualization."""

from __future__ import annotations

import json

import pytest

from anycode import AgentRunResult, TaskQueue, TeamRunResult, TokenUsage, render_dag, render_timeline
from anycode.tasks.task import create_task


@pytest.fixture
def queue() -> TaskQueue:
    q = TaskQueue()
    a = create_task(title="Plan", description="d", assignee="planner")
    q.add(a)
    q.update(a.id, status="completed")
    b = create_task(title="Build", description="d", assignee="builder", depends_on=[a.id])
    q.add(b)
    q.update(b.id, status="in_progress")
    c = create_task(title="Test", description="d", assignee="qa", depends_on=[b.id])
    q.add(c)
    return q


def test_render_dag_mermaid_includes_classdef(queue: TaskQueue) -> None:
    out = render_dag(queue, format="mermaid")
    assert out.startswith("graph TD")
    tasks = queue.list()
    assert all(f"t_{t.id.replace('-', '_')}" in out for t in tasks)
    assert "-->" in out
    assert "classDef completed" in out
    assert "classDef in_progress" in out


def test_render_dag_dot(queue: TaskQueue) -> None:
    out = render_dag(queue, format="dot")
    assert out.startswith("digraph tasks")
    assert "->" in out


def test_render_dag_json_structure(queue: TaskQueue) -> None:
    data = json.loads(render_dag(queue, format="json"))
    titles = {n["title"] for n in data["nodes"]}
    assert titles == {"Plan", "Build", "Test"}
    assert len(data["edges"]) == 2


def test_render_dag_ascii_has_titles(queue: TaskQueue) -> None:
    out = render_dag(queue, format="ascii")
    assert "Plan" in out and "Build" in out and "Test" in out


def test_render_dag_invalid_format(queue: TaskQueue) -> None:
    with pytest.raises(ValueError):
        render_dag(queue, format="xml")  # type: ignore[arg-type]


def test_render_timeline_basic() -> None:
    result = TeamRunResult(
        success=True,
        agent_results={
            "alice": AgentRunResult(
                success=True, output="ok", messages=[], token_usage=TokenUsage(input_tokens=100, output_tokens=50), tool_calls=[]
            ),
            "bob": AgentRunResult(success=True, output="ok", messages=[], token_usage=TokenUsage(input_tokens=200, output_tokens=100), tool_calls=[]),
        },
        total_token_usage=TokenUsage(input_tokens=300, output_tokens=150),
    )
    out = render_timeline(result, width=20)
    assert "alice" in out and "bob" in out
