"""Workflow graph construction: nodes, edges, conditional edges, and compile-time validation."""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from anycode.helpers.usage_tracker import EMPTY_USAGE
from anycode.types import TokenUsage
from anycode.workflow.state import (
    Reducer,
    StateSchema,
    WorkflowError,
    collect_reducers,
    normalize_patch,
    state_get,
)

if TYPE_CHECKING:
    from anycode.workflow.runtime import CompiledWorkflow

START = "__start__"
"""Virtual node every graph enters from."""

END = "__end__"
"""Virtual node that terminates a branch."""

DEFAULT_INPUT_KEY = "prompt"
DEFAULT_OUTPUT_KEY = "output"

Router = Callable[[Any], str | Sequence[str]]
NodeResult = Mapping[str, Any] | BaseModel | None


class Command(BaseModel):
    """Returned by a node to set the next hop dynamically, optionally with a state patch."""

    model_config = ConfigDict(frozen=True)

    goto: str | None = None
    update: dict[str, Any] | None = None


@dataclass(frozen=True)
class NodeOutcome:
    patch: dict[str, Any]
    usage: TokenUsage
    goto: tuple[str, ...] | None


@dataclass(frozen=True)
class Node:
    name: str
    run: Callable[[Any], Awaitable[NodeOutcome]]
    kind: str


@dataclass(frozen=True)
class ConditionalEdge:
    router: Router
    path_map: dict[str, str] | None

    def resolve(self, source: str, state: Any) -> tuple[str, ...]:
        decision = self.router(state)
        keys = [decision] if isinstance(decision, str) else list(decision)
        if not keys:
            raise WorkflowError(f'The router on "{source}" returned no target. Return a node name, a list of names, or END.')
        if self.path_map is None:
            return tuple(keys)
        resolved: list[str] = []
        for key in keys:
            if key not in self.path_map:
                raise WorkflowError(f'The router on "{source}" returned "{key}", which is not in its path map: {sorted(self.path_map)}.')
            resolved.append(self.path_map[key])
        return tuple(resolved)


class Workflow:
    """Builder for a state graph: register nodes, connect them, then :meth:`compile`."""

    def __init__(self, state_schema: StateSchema = None, *, name: str = "workflow") -> None:
        if state_schema is not None and not (isinstance(state_schema, type) and issubclass(state_schema, BaseModel)):
            raise WorkflowError("Workflow(state_schema=...) takes a Pydantic model class, or nothing for dict state.")
        self.name = name
        self.state_schema = state_schema
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, list[str]] = {}
        self._conditional: dict[str, ConditionalEdge] = {}
        self._dynamic: dict[str, tuple[str, ...]] = {}
        self._entry: str | None = None

    # -- registration --

    def add_node(
        self,
        name: str,
        target: object = None,
        *,
        input_key: str = DEFAULT_INPUT_KEY,
        output_key: str = DEFAULT_OUTPUT_KEY,
        goto: Sequence[str] | None = None,
    ) -> Workflow:
        """Register *target* under *name*. Accepts a function, an Agent, a Crew, or a compiled workflow.

        ``goto`` declares the nodes this one may jump to by returning a
        :class:`Command`. Declaring them keeps reachability analysis honest — a node
        reached only by ``Command`` is otherwise indistinguishable from an orphan.
        """
        if name in (START, END):
            raise WorkflowError(f'"{name}" is reserved. Choose another node name.')
        if name in self._nodes:
            raise WorkflowError(f'A node named "{name}" is already registered.')
        if target is None:
            raise WorkflowError(f'Node "{name}" needs something to run. Pass a function, an Agent, a Crew, or a compiled workflow.')
        self._nodes[name] = _build_node(name, target, self.state_schema, input_key=input_key, output_key=output_key)
        if goto:
            self._dynamic[name] = tuple(goto)
        return self

    def node(self, target: object = None, /, *, name: str | None = None, **options: Any) -> Any:
        """Decorator form of :meth:`add_node`. Usable bare (``@wf.node``) or configured."""
        if callable(target) and name is None and not options:
            self.add_node(getattr(target, "__name__", "node"), target)
            return target

        def _decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.add_node(name or getattr(fn, "__name__", "node"), fn, **options)
            return fn

        if target is not None:
            return _decorate(target)  # type: ignore[arg-type]
        return _decorate

    def add_edge(self, source: str, target: str) -> Workflow:
        """Connect *source* to *target*. Repeating a source fans out to every target."""
        if source == START:
            self.set_entry(target)
            return self
        if source == END:
            raise WorkflowError("END has no outgoing edges.")
        if source in self._conditional:
            raise WorkflowError(f'"{source}" already has a conditional edge. A node routes one way or the other, not both.')
        self._edges.setdefault(source, []).append(target)
        return self

    def add_conditional_edge(self, source: str, router: Router, path_map: dict[str, str] | None = None) -> Workflow:
        """Route out of *source* by calling *router* with the current state."""
        if source in (START, END):
            raise WorkflowError(f'"{source}" cannot have a conditional edge.')
        if source in self._edges:
            raise WorkflowError(f'"{source}" already has a static edge. A node routes one way or the other, not both.')
        if source in self._conditional:
            raise WorkflowError(f'"{source}" already has a conditional edge.')
        if not callable(router):
            raise WorkflowError(f'The router for "{source}" must be callable and take the state.')
        self._conditional[source] = ConditionalEdge(router=router, path_map=dict(path_map) if path_map else None)
        return self

    def set_entry(self, name: str) -> Workflow:
        """Choose the node the graph starts at."""
        if self._entry is not None and self._entry != name:
            raise WorkflowError(f'This workflow already starts at "{self._entry}". A graph has exactly one entry point.')
        self._entry = name
        return self

    # -- compilation --

    def compile(self) -> CompiledWorkflow:
        """Validate the graph and return an immutable, runnable copy."""
        problems = self._validate()
        if problems:
            raise WorkflowError(f"Workflow '{self.name}' cannot be compiled:\n" + "\n".join(f"  - {problem}" for problem in problems))

        assert self._entry is not None
        from anycode.workflow.runtime import CompiledWorkflow

        return CompiledWorkflow(
            name=self.name,
            state_schema=self.state_schema,
            nodes=dict(self._nodes),
            edges={source: list(targets) for source, targets in self._edges.items()},
            conditional=dict(self._conditional),
            dynamic=dict(self._dynamic),
            entry=self._entry,
            reducers=collect_reducers(self.state_schema),
        )

    def _validate(self) -> list[str]:
        problems: list[str] = []
        if not self._nodes:
            problems.append("no nodes are registered.")
        if self._entry is None:
            problems.append('no entry point. Call add_edge(START, "first") or set_entry("first").')
        elif self._entry not in self._nodes:
            problems.append(f'the entry point "{self._entry}" is not a registered node.')

        for source, targets in self._edges.items():
            if source not in self._nodes:
                problems.append(f'an edge starts at "{source}", which is not a registered node.')
            for target in targets:
                if target != END and target not in self._nodes:
                    problems.append(f'the edge "{source}" -> "{target}" points at an unregistered node.')

        for source, edge in self._conditional.items():
            if source not in self._nodes:
                problems.append(f'a conditional edge starts at "{source}", which is not a registered node.')
            for key, target in (edge.path_map or {}).items():
                if target != END and target not in self._nodes:
                    problems.append(f'the path map on "{source}" maps "{key}" to unregistered node "{target}".')

        for source, targets in self._dynamic.items():
            for target in targets:
                if target != END and target not in self._nodes:
                    problems.append(f'"{source}" declares goto="{target}", which is not a registered node.')

        if self._entry in self._nodes:
            reachable = self._reachable_from_entry()
            for name in self._nodes:
                if name not in reachable:
                    problems.append(f'"{name}" is unreachable from the entry point.')
            for name in sorted(self._terminating_nodes(reachable)):
                problems.append(f'"{name}" can never reach END, so the workflow would loop forever.')

        return problems

    def _static_targets(self, name: str) -> tuple[str, ...] | None:
        """Targets known without running the graph, or ``None`` when only a router decides."""
        declared = self._dynamic.get(name, ())
        if name in self._edges:
            return (*self._edges[name], *declared)
        edge = self._conditional.get(name)
        if edge is None:
            return (END, *declared)
        if edge.path_map is None:
            return None
        return (*edge.path_map.values(), *declared)

    def _reachable_from_entry(self) -> set[str]:
        assert self._entry is not None
        seen: set[str] = set()
        frontier = [self._entry]
        while frontier:
            current = frontier.pop()
            if current in seen or current == END:
                continue
            seen.add(current)
            targets = self._static_targets(current)
            if targets is None:
                # A router with no path map may reach any node; treat all as reachable.
                return set(self._nodes)
            frontier.extend(targets)
        return seen

    def _terminating_nodes(self, reachable: set[str]) -> set[str]:
        """Reachable nodes that provably cannot reach END."""
        can_end: set[str] = set()
        changed = True
        while changed:
            changed = False
            for name in reachable:
                if name in can_end:
                    continue
                targets = self._static_targets(name)
                if targets is None or any(target == END or target in can_end for target in targets):
                    can_end.add(name)
                    changed = True
        return reachable - can_end

    def __repr__(self) -> str:
        return f"Workflow(name={self.name!r}, nodes={len(self._nodes)}, entry={self._entry!r})"


# ---------------------------------------------------------------------------
# Node adapters
# ---------------------------------------------------------------------------


def _build_node(
    name: str,
    target: object,
    schema: StateSchema,
    *,
    input_key: str,
    output_key: str,
) -> Node:
    from anycode.core.agent import Agent
    from anycode.crew import Crew
    from anycode.workflow.runtime import CompiledWorkflow

    if isinstance(target, Agent):
        return Node(name=name, run=_agent_runner(name, target, input_key, output_key, schema), kind="agent")
    if isinstance(target, Crew):
        return Node(name=name, run=_crew_runner(name, target, input_key, output_key, schema), kind="crew")
    if isinstance(target, CompiledWorkflow):
        return Node(name=name, run=_subgraph_runner(target), kind="workflow")
    if callable(target):
        return Node(name=name, run=_callable_runner(name, target, schema), kind="function")
    raise WorkflowError(f'Node "{name}" got {type(target).__name__}. Pass a function, an Agent, a Crew, or a compiled workflow.')


def _outcome(name: str, schema: StateSchema, result: object, usage: TokenUsage = EMPTY_USAGE) -> NodeOutcome:
    if isinstance(result, Command):
        patch = normalize_patch(name, schema, result.update or {})
        goto = (result.goto,) if result.goto else None
        return NodeOutcome(patch=patch, usage=usage, goto=goto)
    return NodeOutcome(patch=normalize_patch(name, schema, result), usage=usage, goto=None)


def _callable_runner(name: str, fn: Callable[..., Any], schema: StateSchema) -> Callable[[Any], Awaitable[NodeOutcome]]:
    is_coroutine = inspect.iscoroutinefunction(fn)

    async def _run(state: Any) -> NodeOutcome:
        result = await fn(state) if is_coroutine else await asyncio.to_thread(functools.partial(fn, state))
        return _outcome(name, schema, result)

    return _run


def _require_input(name: str, state: Any, input_key: str) -> str:
    value = state_get(state, input_key)
    if value is None or value == "":
        raise WorkflowError(f'Node "{name}" reads state field "{input_key}", which is empty. Set it upstream or pass input_key= to add_node.')
    return str(value)


def _agent_runner(name: str, agent: Any, input_key: str, output_key: str, schema: StateSchema) -> Callable[[Any], Awaitable[NodeOutcome]]:
    async def _run(state: Any) -> NodeOutcome:
        result = await agent.run(_require_input(name, state, input_key))
        return _outcome(name, schema, {output_key: result.output}, result.token_usage)

    return _run


def _crew_runner(name: str, crew: Any, input_key: str, output_key: str, schema: StateSchema) -> Callable[[Any], Awaitable[NodeOutcome]]:
    async def _run(state: Any) -> NodeOutcome:
        goal = None if crew.tasks else _require_input(name, state, input_key)
        result = await crew.run(goal)
        return _outcome(name, schema, {output_key: result.output}, result.usage)

    return _run


def _subgraph_runner(subgraph: Any) -> Callable[[Any], Awaitable[NodeOutcome]]:
    async def _run(state: Any) -> NodeOutcome:
        result = await subgraph.run(state)
        from anycode.workflow.state import state_to_dict

        return NodeOutcome(patch=state_to_dict(result.state), usage=result.usage, goto=None)

    return _run


__all__ = [
    "END",
    "START",
    "Command",
    "ConditionalEdge",
    "Node",
    "NodeOutcome",
    "Reducer",
    "Router",
    "Workflow",
]
