"""Render task DAGs in Mermaid, DOT, JSON, and ASCII formats."""

from __future__ import annotations

import json
from typing import Literal

from anycode.tasks.queue import TaskQueue
from anycode.types import Task

DagFormat = Literal["mermaid", "dot", "json", "ascii"]

_STATUS_COLORS = {
    "completed": "#22c55e",
    "failed": "#ef4444",
    "blocked": "#94a3b8",
    "pending": "#f59e0b",
    "in_progress": "#3b82f6",
}


def render_dag(queue: TaskQueue, *, format: DagFormat = "mermaid", show_status: bool = True) -> str:
    """Render the task DAG. Supported formats: mermaid, dot, json, ascii."""
    tasks = queue.list()
    if format == "mermaid":
        return _to_mermaid(tasks, show_status=show_status)
    if format == "dot":
        return _to_dot(tasks, show_status=show_status)
    if format == "json":
        return _to_json(tasks)
    if format == "ascii":
        return _to_ascii(tasks)
    raise ValueError(f"Unsupported DAG format: {format!r}")


def _node_id(task: Task) -> str:
    return f"t_{task.id.replace('-', '_')}"


def _escape_mermaid_label(label: str) -> str:
    return label.replace('"', '\\"').replace("\n", " ")


def _to_mermaid(tasks: list[Task], *, show_status: bool) -> str:
    lines: list[str] = ["graph TD"]
    if not tasks:
        lines.append('    empty["<empty DAG>"]')
        return "\n".join(lines)

    used_classes: set[str] = set()
    for t in tasks:
        node_id = _node_id(t)
        label = _escape_mermaid_label(t.title)
        if t.assignee:
            label = f"{label}<br/><small>{_escape_mermaid_label(t.assignee)}</small>"
        suffix = f":::{t.status}" if show_status else ""
        lines.append(f'    {node_id}["{label}"]{suffix}')
        if show_status:
            used_classes.add(t.status)

    for t in tasks:
        if t.depends_on:
            for dep_id in t.depends_on:
                dep = next((x for x in tasks if x.id == dep_id), None)
                if dep:
                    lines.append(f"    {_node_id(dep)} --> {_node_id(t)}")

    if show_status:
        for cls in sorted(used_classes):
            color = _STATUS_COLORS.get(cls, "#6b7280")
            lines.append(f"    classDef {cls} fill:{color},color:#fff")

    return "\n".join(lines)


def _to_dot(tasks: list[Task], *, show_status: bool) -> str:
    lines: list[str] = [
        "digraph tasks {",
        "    rankdir=LR;",
        "    node [shape=box, style=rounded];",
    ]
    for t in tasks:
        attrs: list[str] = [f'label="{t.title}"']
        if show_status:
            color = _STATUS_COLORS.get(t.status, "#6b7280")
            attrs.append(f'fillcolor="{color}"')
            attrs.append('style="filled,rounded"')
            attrs.append('fontcolor="white"')
        lines.append(f"    {_node_id(t)} [{', '.join(attrs)}];")
    for t in tasks:
        for dep_id in t.depends_on or []:
            dep = next((x for x in tasks if x.id == dep_id), None)
            if dep:
                lines.append(f"    {_node_id(dep)} -> {_node_id(t)};")
    lines.append("}")
    return "\n".join(lines)


def _to_json(tasks: list[Task]) -> str:
    nodes = [
        {
            "id": t.id,
            "title": t.title,
            "status": t.status,
            "assignee": t.assignee,
        }
        for t in tasks
    ]
    edges: list[dict[str, str]] = []
    for t in tasks:
        for dep_id in t.depends_on or []:
            edges.append({"from": dep_id, "to": t.id})
    return json.dumps({"nodes": nodes, "edges": edges}, indent=2)


def _to_ascii(tasks: list[Task]) -> str:
    if not tasks:
        return "<empty DAG>"
    by_id = {t.id: t for t in tasks}
    children: dict[str, list[str]] = {t.id: [] for t in tasks}
    for t in tasks:
        for dep in t.depends_on or []:
            if dep in children:
                children[dep].append(t.id)

    roots = [t for t in tasks if not (t.depends_on or [])]
    seen: set[str] = set()
    lines: list[str] = []

    def _walk(node_id: str, prefix: str, is_last: bool) -> None:
        if node_id in seen:
            return
        seen.add(node_id)
        node = by_id[node_id]
        connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
        marker = {"completed": "\u2713", "failed": "\u2717", "blocked": "\u2298", "in_progress": "\u25b6"}.get(node.status, "\u25cb")
        lines.append(f"{prefix}{connector}{marker} {node.title}" + (f" [{node.assignee}]" if node.assignee else ""))
        kids = children.get(node_id, [])
        new_prefix = prefix + ("    " if is_last else "\u2502   ")
        for idx, child in enumerate(kids):
            _walk(child, new_prefix, idx == len(kids) - 1)

    for idx, root in enumerate(roots):
        _walk(root.id, "", idx == len(roots) - 1)

    # Append any tasks not reached (cyclic / orphan)
    for t in tasks:
        if t.id not in seen:
            lines.append(f"\u25cb {t.title} (orphan)")

    return "\n".join(lines)
