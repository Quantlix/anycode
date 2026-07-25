"""Machine-readable description of the public API.

An AI coding agent should not have to read 27k lines of source to learn what AnyCode
offers. ``describe()`` renders the whole public surface as one line per symbol, and the
``anycode api`` command prints it as a table or as JSON.
"""

from __future__ import annotations

import inspect
from typing import Any, Literal, Protocol, get_args, get_origin, overload

from pydantic import BaseModel, ConfigDict

import anycode

# The symbols that cover the overwhelming majority of real use. Everything else stays
# public and documented; this is the front door, not a fence.
CORE_SURFACE: tuple[str, ...] = (
    "Agent",
    "tool",
    "Crew",
    "TaskSpec",
    "Workflow",
    "START",
    "END",
    "AnyCode",
    "AgentConfig",
    "TeamConfig",
    "ToolResult",
    "ToolRegistry",
    "ToolExecutor",
    "SubAgentSpec",
    "create_adapter",
)

EntryKind = Literal["class", "model", "protocol", "function", "type", "constant", "module"]

SIGNATURE_UNAVAILABLE = "(…)"


class ApiEntry(BaseModel):
    """One public symbol."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: EntryKind
    module: str
    signature: str
    summary: str


class ApiMap(BaseModel):
    """The public surface as data."""

    model_config = ConfigDict(frozen=True)

    version: str
    entries: tuple[ApiEntry, ...]

    def get(self, name: str) -> ApiEntry | None:
        return next((entry for entry in self.entries if entry.name == name), None)

    def names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries)

    def by_module(self) -> dict[str, tuple[ApiEntry, ...]]:
        grouped: dict[str, list[ApiEntry]] = {}
        for entry in self.entries:
            grouped.setdefault(entry.module, []).append(entry)
        return {module: tuple(items) for module, items in sorted(grouped.items())}


def package_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("anycode-py")
    except PackageNotFoundError:  # pragma: no cover - source checkout without an install
        return "unknown"


def _kind_of(value: object) -> EntryKind:
    if inspect.ismodule(value):
        return "module"
    if inspect.isclass(value):
        if issubclass(value, BaseModel):
            return "model"
        if Protocol in getattr(value, "__mro__", ()) or getattr(value, "_is_protocol", False):
            return "protocol"
        return "class"
    if inspect.isroutine(value):
        return "function"
    if get_origin(value) is not None or get_args(value):
        return "type"
    return "constant"


def _model_signature(name: str, model: type[BaseModel]) -> str:
    fields = ", ".join(f"{field}: {_annotation_name(info.annotation)}" for field, info in model.model_fields.items())
    return f"{name}({fields})"


def _annotation_name(annotation: object) -> str:
    if annotation is None:
        return "None"
    rendered = getattr(annotation, "__name__", None) or str(annotation)
    return rendered.replace("anycode.types.", "").replace("typing.", "")


def _signature_of(name: str, value: object, kind: EntryKind) -> str:
    if kind == "model" and isinstance(value, type) and issubclass(value, BaseModel):
        return _model_signature(name, value)
    if kind in ("class", "protocol", "function"):
        try:
            return f"{name}{inspect.signature(value)}"  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return f"{name}{SIGNATURE_UNAVAILABLE}"
    if kind == "module":
        return name
    return f"{name}: {type(value).__name__} = {_short_repr(value)}"


def _short_repr(value: object, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _summary_of(value: object, kind: EntryKind) -> str:
    # A constant inherits its type's docstring, which says nothing about the constant.
    if kind in ("constant", "type"):
        return ""
    doc = inspect.getdoc(value)
    if not doc or doc == inspect.getdoc(BaseModel):
        return ""
    first = doc.strip().split("\n\n", 1)[0]
    return " ".join(first.split())


def _module_of(name: str) -> str:
    mapped = anycode._EXPORTS.get(name)
    return mapped[0] if mapped else "anycode"


def _entry_for(name: str) -> ApiEntry:
    value = getattr(anycode, name)
    kind = _kind_of(value)
    return ApiEntry(
        name=name,
        kind=kind,
        module=_module_of(name),
        signature=_signature_of(name, value, kind),
        summary=_summary_of(value, kind),
    )


@overload
def describe(name: str, /) -> ApiEntry: ...


@overload
def describe(name: None = ..., /, *, kind: EntryKind | None = ..., core: bool = ...) -> ApiMap: ...


def describe(
    name: str | None = None,
    /,
    *,
    kind: EntryKind | None = None,
    core: bool = False,
) -> ApiMap | ApiEntry:
    """Describe the public API, or one symbol of it.

    ``describe()`` returns every public symbol. ``describe("Agent")`` returns one entry.
    ``core=True`` narrows the map to the handful of symbols in :data:`CORE_SURFACE`.
    """
    if name is not None:
        if name not in anycode.__all__:
            raise AttributeError(f'"{name}" is not part of the AnyCode public API. Run `anycode api` to list it.')
        return _entry_for(name)

    selected = CORE_SURFACE if core else tuple(sorted(anycode.__all__))
    entries = tuple(_entry_for(symbol) for symbol in selected)
    if kind is not None:
        entries = tuple(entry for entry in entries if entry.kind == kind)
    return ApiMap(version=package_version(), entries=entries)


def render_text(api: ApiMap, *, show_module: bool = True, show_signature: bool = True) -> str:
    """Render an :class:`ApiMap` as compact, readable lines."""
    lines: list[str] = []
    for module, entries in api.by_module().items():
        if show_module:
            lines.append(f"\n{module}")
        for entry in entries:
            summary = f"  — {entry.summary}" if entry.summary else ""
            head = entry.signature if show_signature else entry.name
            lines.append(f"  {head}{summary}")
    return "\n".join(lines).strip()


def render_entry(entry: ApiEntry) -> str:
    """Render one symbol in full."""
    parts = [entry.signature, f"kind:   {entry.kind}", f"module: {entry.module}"]
    if entry.summary:
        parts.append(f"\n{entry.summary}")
    return "\n".join(parts)


def to_json(api: ApiMap, *, compact: bool = False) -> dict[str, Any]:
    """A stable, key-ordered dictionary suitable for machine consumption.

    ``compact`` drops signatures, which is roughly a quarter of the tokens when an agent
    only needs to know what exists and where it lives.
    """
    return {
        "version": api.version,
        "count": len(api.entries),
        "symbols": [
            {
                "name": entry.name,
                "kind": entry.kind,
                "module": entry.module,
                **({} if compact else {"signature": entry.signature}),
                "summary": entry.summary,
            }
            for entry in api.entries
        ],
    }
