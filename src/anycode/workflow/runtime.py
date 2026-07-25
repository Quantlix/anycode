"""Execution of a compiled workflow graph: stepping, streaming, and result assembly."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterator, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from anycode.helpers.sync_runner import iterate_async_blocking, run_coroutine_blocking
from anycode.helpers.usage_tracker import EMPTY_USAGE, merge_usage
from anycode.security.redaction import safe_exception_message
from anycode.types import StopReason, TokenUsage
from anycode.workflow.graph import END, START, ConditionalEdge, Node
from anycode.workflow.state import (
    Reducer,
    StateSchema,
    WorkflowError,
    apply_patch,
    coerce_state,
    merge_concurrent_patches,
)

DEFAULT_MAX_STEPS = 25

EventType = Literal["node_start", "node_end", "route", "done", "error"]


class WorkflowResult(BaseModel):
    """Outcome of one workflow run."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    success: bool
    state: Any
    steps: int = 0
    path: tuple[str, ...] = ()
    usage: TokenUsage = EMPTY_USAGE
    stop_reason: StopReason | None = None
    error: str | None = None


class WorkflowEvent(BaseModel):
    """One observable moment during execution."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    type: EventType
    node: str | None = None
    step: int = 0
    patch: dict[str, Any] | None = None
    targets: tuple[str, ...] = ()
    state: Any = None
    result: WorkflowResult | None = None
    error: str | None = None


class CompiledWorkflow:
    """An immutable, runnable graph. Build one with :meth:`~anycode.workflow.graph.Workflow.compile`."""

    def __init__(
        self,
        *,
        name: str,
        state_schema: StateSchema,
        nodes: dict[str, Node],
        edges: dict[str, list[str]],
        conditional: dict[str, ConditionalEdge],
        dynamic: dict[str, tuple[str, ...]],
        entry: str,
        reducers: Mapping[str, Reducer],
    ) -> None:
        self.name = name
        self.state_schema = state_schema
        self._nodes = nodes
        self._edges = edges
        self._conditional = conditional
        self._dynamic = dynamic
        self._entry = entry
        self._reducers = dict(reducers)

    # -- execution --

    async def run(self, state: Any = None, *, max_steps: int = DEFAULT_MAX_STEPS) -> WorkflowResult:
        """Run to completion and return the final result."""
        outcome: WorkflowResult | None = None
        async for event in self.stream(state, max_steps=max_steps):
            if event.result is not None:
                outcome = event.result
        assert outcome is not None
        return outcome

    def run_sync(self, state: Any = None, *, max_steps: int = DEFAULT_MAX_STEPS) -> WorkflowResult:
        """Blocking form of :meth:`run`."""
        return run_coroutine_blocking(
            self.run(state, max_steps=max_steps),
            sync_call="CompiledWorkflow.run_sync()",
            async_call="await workflow.run(...)",
        )

    def stream_sync(self, state: Any = None, *, max_steps: int = DEFAULT_MAX_STEPS) -> Iterator[WorkflowEvent]:
        """Blocking form of :meth:`stream`."""
        return iterate_async_blocking(
            lambda: self.stream(state, max_steps=max_steps),
            sync_call="CompiledWorkflow.stream_sync()",
            async_call="async for event in workflow.stream(...)",
        )

    async def stream(self, state: Any = None, *, max_steps: int = DEFAULT_MAX_STEPS) -> AsyncGenerator[WorkflowEvent, None]:
        """Run the graph, yielding an event per node start, node end, and routing decision."""
        if max_steps < 1:
            raise WorkflowError("max_steps must be at least 1.")

        current = coerce_state(self.state_schema, state)
        frontier: tuple[str, ...] = (self._entry,)
        path: list[str] = []
        usage = EMPTY_USAGE
        steps = 0

        while frontier:
            if steps + len(frontier) > max_steps:
                yield self._final_event(
                    WorkflowResult(
                        success=False,
                        state=current,
                        steps=steps,
                        path=tuple(path),
                        usage=usage,
                        stop_reason=StopReason(
                            code="max_steps",
                            message=f"Workflow '{self.name}' hit its {max_steps}-step limit without reaching END.",
                            recoverable=True,
                        ),
                    )
                )
                return

            for node_name in frontier:
                yield WorkflowEvent(type="node_start", node=node_name, step=steps, state=current)

            try:
                outcomes = await asyncio.gather(*(self._nodes[node_name].run(current) for node_name in frontier))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                message = safe_exception_message(error)
                yield WorkflowEvent(type="error", node=frontier[0] if len(frontier) == 1 else None, step=steps, error=message)
                yield self._final_event(
                    WorkflowResult(
                        success=False,
                        state=current,
                        steps=steps,
                        path=tuple(path),
                        usage=usage,
                        error=message,
                        stop_reason=StopReason(code="unknown", message=message, recoverable=False),
                    )
                )
                return

            patches: list[tuple[str, dict[str, Any]]] = []
            overrides: dict[str, tuple[str, ...]] = {}
            for node_name, outcome in zip(frontier, outcomes, strict=True):
                steps += 1
                path.append(node_name)
                usage = merge_usage(usage, outcome.usage)
                patches.append((node_name, outcome.patch))
                if outcome.goto is not None:
                    overrides[node_name] = outcome.goto
                yield WorkflowEvent(type="node_end", node=node_name, step=steps, patch=outcome.patch)

            combined = merge_concurrent_patches(patches, self._reducers)
            current = apply_patch(current, combined, schema=self.state_schema, reducers=self._reducers)

            next_nodes: list[str] = []
            for node_name in frontier:
                targets = overrides.get(node_name) or self._route(node_name, current)
                yield WorkflowEvent(type="route", node=node_name, step=steps, targets=targets, state=current)
                for target in targets:
                    if target == END:
                        continue
                    if target not in self._nodes:
                        raise WorkflowError(f'Node "{node_name}" routed to "{target}", which is not a registered node.')
                    if target not in next_nodes:
                        next_nodes.append(target)

            frontier = tuple(next_nodes)

        yield self._final_event(
            WorkflowResult(success=True, state=current, steps=steps, path=tuple(path), usage=usage),
        )

    def _route(self, node_name: str, state: Any) -> tuple[str, ...]:
        edge = self._conditional.get(node_name)
        if edge is not None:
            return edge.resolve(node_name, state)
        return tuple(self._edges.get(node_name, (END,)))

    @staticmethod
    def _final_event(result: WorkflowResult) -> WorkflowEvent:
        return WorkflowEvent(type="done", step=result.steps, state=result.state, result=result)

    # -- introspection --

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(self._nodes)

    @property
    def entry(self) -> str:
        return self._entry

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly description of the graph."""
        return {
            "name": self.name,
            "entry": self._entry,
            "state_schema": self.state_schema.__name__ if self.state_schema else "dict",
            "nodes": [{"name": name, "kind": node.kind} for name, node in self._nodes.items()],
            "edges": [{"from": source, "to": target} for source, targets in self._edges.items() for target in targets],
            "conditional_edges": [
                {"from": source, "path_map": edge.path_map or {}, "dynamic": edge.path_map is None} for source, edge in self._conditional.items()
            ],
            "declared_goto": [{"from": source, "to": target} for source, targets in self._dynamic.items() for target in targets],
        }

    def to_mermaid(self) -> str:
        """Render the graph as a Mermaid flowchart."""
        lines = ["flowchart TD", f'    {START}(["START"])', f'    {END}(["END"])']
        for name, node in self._nodes.items():
            lines.append(f'    {name}["{name}<br/><i>{node.kind}</i>"]')
        lines.append(f"    {START} --> {self._entry}")
        for source, targets in self._edges.items():
            lines.extend(f"    {source} --> {target}" for target in targets)
        for source, edge in self._conditional.items():
            if edge.path_map:
                lines.extend(f"    {source} -.->|{key}| {target}" for key, target in edge.path_map.items())
            else:
                # Without a path map the targets are only known at runtime, so every
                # node stays a candidate. Supply path_map= for a precise diagram.
                candidates = [name for name in self._nodes if name != source]
                lines.extend(f"    {source} -.->|?| {target}" for target in (*candidates, END))
        for source, targets in self._dynamic.items():
            lines.extend(f"    {source} -.->|goto| {target}" for target in targets)
        for name in self._nodes:
            if name not in self._edges and name not in self._conditional:
                lines.append(f"    {name} --> {END}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CompiledWorkflow(name={self.name!r}, nodes={len(self._nodes)}, entry={self._entry!r})"
