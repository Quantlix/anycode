"""State containers, patches, and reducers for workflow graphs.

A node returns a *patch* — a mapping of the fields it changed. Patches merge into the
running state field by field; a field annotated with a reducer accumulates instead of
being replaced.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel

Patch = Mapping[str, Any]
Reducer = Callable[[Any, Any], Any]
StateSchema = type[BaseModel] | None


class WorkflowError(RuntimeError):
    """Raised when a workflow graph is invalid or a node produces an unusable result."""


# ---------------------------------------------------------------------------
# Built-in reducers
# ---------------------------------------------------------------------------


def add(current: Any, incoming: Any) -> Any:
    """Accumulate: concatenate sequences and strings, union sets, sum numbers."""
    if current is None:
        return incoming
    if incoming is None:
        return current
    if isinstance(current, list):
        return [*current, *incoming] if isinstance(incoming, list) else [*current, incoming]
    if isinstance(current, tuple):
        return (*current, *incoming) if isinstance(incoming, tuple) else (*current, incoming)
    if isinstance(current, set):
        return current | (incoming if isinstance(incoming, set) else {incoming})
    if isinstance(current, str) and isinstance(incoming, str):
        return current + incoming
    if isinstance(current, bool) or isinstance(incoming, bool):
        raise WorkflowError(f"The add reducer cannot combine {type(current).__name__} with {type(incoming).__name__}.")
    if isinstance(current, int | float) and isinstance(incoming, int | float):
        return current + incoming
    raise WorkflowError(f"The add reducer cannot combine {type(current).__name__} with {type(incoming).__name__}.")


def merge(current: Any, incoming: Any) -> Any:
    """Shallow-merge two mappings; the incoming keys win."""
    if current is None:
        return incoming
    if incoming is None:
        return current
    if isinstance(current, Mapping) and isinstance(incoming, Mapping):
        return {**current, **incoming}
    raise WorkflowError(f"The merge reducer needs two mappings, got {type(current).__name__} and {type(incoming).__name__}.")


def keep_first(current: Any, incoming: Any) -> Any:
    """Keep the value already in state once it is set."""
    return incoming if current in (None, "", [], {}, ()) else current


def keep_last(_current: Any, incoming: Any) -> Any:
    """Replace with the incoming value. This is the default behavior."""
    return incoming


BUILT_IN_REDUCERS: tuple[Reducer, ...] = (add, merge, keep_first, keep_last)


# ---------------------------------------------------------------------------
# State inspection and patching
# ---------------------------------------------------------------------------


def collect_reducers(schema: StateSchema) -> dict[str, Reducer]:
    """Read per-field reducers declared as ``Annotated[T, reducer]`` on the state model."""
    if schema is None:
        return {}
    reducers: dict[str, Reducer] = {}
    for name, field in schema.model_fields.items():
        for metadata in field.metadata:
            if callable(metadata) and not isinstance(metadata, type):
                reducers[name] = metadata
                break
    return reducers


def state_fields(schema: StateSchema) -> tuple[str, ...]:
    return tuple(schema.model_fields) if schema is not None else ()


def coerce_state(schema: StateSchema, value: Any) -> Any:
    """Normalize a caller-supplied initial state into the graph's state type."""
    if schema is None:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, BaseModel):
            return value.model_dump()
        raise WorkflowError(f"An untyped workflow needs a mapping as its initial state, got {type(value).__name__}.")
    if value is None:
        return schema()
    if isinstance(value, schema):
        return value
    if isinstance(value, Mapping):
        return schema.model_validate(dict(value))
    raise WorkflowError(f"Expected {schema.__name__} or a mapping as the initial state, got {type(value).__name__}.")


def state_get(state: Any, key: str) -> Any:
    if isinstance(state, Mapping):
        return state.get(key)
    return getattr(state, key, None)


def state_to_dict(state: Any) -> dict[str, Any]:
    if isinstance(state, Mapping):
        return dict(state)
    if isinstance(state, BaseModel):
        return state.model_dump()
    return {}


def normalize_patch(node: str, schema: StateSchema, result: Any) -> dict[str, Any]:
    """Turn whatever a node returned into a plain patch mapping."""
    if result is None:
        return {}
    if isinstance(result, Mapping):
        patch = dict(result)
    elif isinstance(result, BaseModel):
        patch = result.model_dump()
    else:
        raise WorkflowError(
            f'Node "{node}" returned {type(result).__name__}. A node must return a dict of changed fields, a state instance, a Command, or None.'
        )

    if schema is not None:
        known = set(schema.model_fields)
        unknown = sorted(set(patch) - known)
        if unknown:
            raise WorkflowError(
                f'Node "{node}" returned unknown state field(s): {", ".join(unknown)}. {schema.__name__} declares: {", ".join(sorted(known))}.'
            )
    return patch


def apply_patch(state: Any, patch: Patch, *, schema: StateSchema, reducers: Mapping[str, Reducer]) -> Any:
    """Merge *patch* into *state*, honoring per-field reducers."""
    if not patch:
        return state

    updates: dict[str, Any] = {}
    for key, incoming in patch.items():
        reducer = reducers.get(key)
        updates[key] = reducer(state_get(state, key), incoming) if reducer else incoming

    if schema is None:
        return {**state_to_dict(state), **updates}
    return state.model_copy(update=updates)


def merge_concurrent_patches(patches: list[tuple[str, dict[str, Any]]], reducers: Mapping[str, Reducer]) -> dict[str, Any]:
    """Combine patches produced in the same step, rejecting conflicting plain writes."""
    combined: dict[str, Any] = {}
    owner: dict[str, str] = {}
    for node, patch in patches:
        for key, value in patch.items():
            if key not in combined:
                combined[key] = value
                owner[key] = node
                continue
            reducer = reducers.get(key)
            if reducer is None:
                raise WorkflowError(
                    f'Nodes "{owner[key]}" and "{node}" both wrote "{key}" in the same step. '
                    f"Annotate that field with a reducer, e.g. Annotated[list[str], add], or have only one node write it."
                )
            combined[key] = reducer(combined[key], value)
    return combined
