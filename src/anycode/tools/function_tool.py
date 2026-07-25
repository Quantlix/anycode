"""Turn plain Python functions into tool definitions.

A function's signature becomes the input schema, its docstring becomes the description,
and its return value is coerced into a :class:`~anycode.types.ToolResult`.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import re
from collections.abc import Callable, Iterable
from typing import Any, overload

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from anycode.types import ToolDefinition, ToolResult, ToolUseContext

# ---------------------------------------------------------------------------
# Naming and docstring grammar
# ---------------------------------------------------------------------------

TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]{0,63}$")
CONTEXT_PARAMETER_NAMES = frozenset({"ctx", "context", "tool_context"})
UNANNOTATED_PARAMETER_TYPE = str

_SECTION_HEADER = re.compile(
    r"^(Args|Arguments|Parameters|Params|Returns?|Raises|Yields?|Examples?|Notes?|Attributes|Warnings?)\s*:\s*$",
    re.IGNORECASE,
)
_ARGUMENT_SECTIONS = frozenset({"args", "arguments", "parameters", "params"})
_ARGUMENT_LINE = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*(?:\((?P<type>[^)]*)\))?\s*:\s*(?P<description>.*)$")


ToolSpec = ToolDefinition | Callable[..., Any] | str
"""Every form accepted wherever AnyCode takes a tool."""


class ToolDefinitionError(ValueError):
    """Raised when a function cannot be turned into a usable tool."""


# ---------------------------------------------------------------------------
# Docstring parsing
# ---------------------------------------------------------------------------


def parse_docstring(docstring: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into ``(summary, {parameter: description})``.

    Never raises: an unparseable docstring degrades to a summary with no parameter
    descriptions.
    """
    if not docstring or not docstring.strip():
        return "", {}

    lines = inspect.cleandoc(docstring).splitlines()
    summary_lines: list[str] = []
    parameter_docs: dict[str, str] = {}
    current_section = "summary"
    current_parameter: str | None = None

    for line in lines:
        header = _SECTION_HEADER.match(line.strip())
        if header:
            current_section = header.group(1).lower()
            current_parameter = None
            continue

        if current_section == "summary":
            summary_lines.append(line)
            continue

        if current_section not in _ARGUMENT_SECTIONS:
            continue

        stripped = line.strip()
        if not stripped:
            current_parameter = None
            continue

        argument = _ARGUMENT_LINE.match(stripped)
        if argument:
            parameter_name = argument.group("name")
            current_parameter = parameter_name
            parameter_docs[parameter_name] = argument.group("description").strip()
        elif current_parameter is not None:
            parameter_docs[current_parameter] = f"{parameter_docs[current_parameter]} {stripped}".strip()

    return " ".join(" ".join(summary_lines).split()), parameter_docs


# ---------------------------------------------------------------------------
# Signature to Pydantic model
# ---------------------------------------------------------------------------


def _resolve_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    try:
        from typing import get_type_hints

        return get_type_hints(fn, include_extras=True)
    except Exception:
        return dict(getattr(fn, "__annotations__", {}))


def _is_context_annotation(annotation: Any) -> bool:
    return annotation is ToolUseContext or (isinstance(annotation, type) and issubclass(annotation, ToolUseContext))


def _build_input_model(
    fn: Callable[..., Any],
    tool_name: str,
    parameter_docs: dict[str, str],
) -> tuple[type[BaseModel], tuple[str, ...], str | None]:
    """Return ``(input_model, field_names, context_parameter_name)`` for *fn*."""
    signature = inspect.signature(fn)
    hints = _resolve_hints(fn)

    fields: dict[str, Any] = {}
    field_names: list[str] = []
    context_parameter: str | None = None

    for parameter_name, parameter in signature.parameters.items():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ToolDefinitionError(
                f'Tool "{tool_name}" cannot use *{parameter_name} — variadic parameters have no JSON Schema equivalent. '
                "Declare each parameter explicitly."
            )

        annotation = hints.get(parameter_name, parameter.annotation)
        unannotated = annotation is inspect.Parameter.empty

        if (not unannotated and _is_context_annotation(annotation)) or (unannotated and parameter_name in CONTEXT_PARAMETER_NAMES):
            if context_parameter is not None:
                raise ToolDefinitionError(
                    f'Tool "{tool_name}" declares two context parameters ("{context_parameter}" and "{parameter_name}"). Keep only one.'
                )
            context_parameter = parameter_name
            continue

        if unannotated:
            annotation = UNANNOTATED_PARAMETER_TYPE

        default = PydanticUndefined if parameter.default is inspect.Parameter.empty else parameter.default
        field_info = FieldInfo.from_annotated_attribute(annotation, default)
        if not field_info.description:
            documented = parameter_docs.get(parameter_name)
            if documented:
                field_info.description = documented

        fields[parameter_name] = (field_info.annotation, field_info)
        field_names.append(parameter_name)

    model_name = f"{_pascal_case(tool_name)}Input"
    model = create_model(model_name, __module__=getattr(fn, "__module__", __name__), **fields)
    return model, tuple(field_names), context_parameter


def _pascal_case(value: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[^A-Za-z0-9]+", value) if part) or "Tool"


# ---------------------------------------------------------------------------
# Return-value coercion
# ---------------------------------------------------------------------------


def coerce_tool_result(value: object) -> ToolResult:
    """Normalize any function return value into a :class:`ToolResult`."""
    if isinstance(value, ToolResult):
        return value
    if value is None:
        return ToolResult(data="")
    if isinstance(value, str):
        return ToolResult(data=value)
    if isinstance(value, BaseModel):
        return ToolResult(data=value.model_dump_json())
    if isinstance(value, dict | list | tuple | int | float | bool):
        return ToolResult(data=json.dumps(value, default=str))
    return ToolResult(data=str(value))


# ---------------------------------------------------------------------------
# Definition builder
# ---------------------------------------------------------------------------


def build_tool_definition(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    side_effecting: bool = False,
    idempotency_key_field: str | None = "idempotency_key",
) -> ToolDefinition:
    """Build a :class:`ToolDefinition` from *fn* without attaching anything to it."""
    if not callable(fn):
        raise ToolDefinitionError(f"Expected a callable to build a tool from, got {type(fn).__name__}.")

    resolved_name = name or getattr(fn, "__name__", "")
    if not TOOL_NAME_PATTERN.match(resolved_name):
        raise ToolDefinitionError(
            f'Invalid tool name "{resolved_name}". Names must start with a letter or underscore, '
            "use only letters, digits, hyphens, or underscores, and be at most 64 characters. "
            'Pass @tool(name="...") to override.'
        )

    summary, parameter_docs = parse_docstring(inspect.getdoc(fn))
    resolved_description = (description or summary).strip()
    if not resolved_description:
        raise ToolDefinitionError(
            f'Tool "{resolved_name}" has no description. Add a docstring to {resolved_name}() or pass @tool(description="...").'
        )

    input_model, field_names, context_parameter = _build_input_model(fn, resolved_name, parameter_docs)
    is_coroutine = inspect.iscoroutinefunction(fn)

    async def execute(params: BaseModel, context: ToolUseContext) -> ToolResult:
        kwargs: dict[str, Any] = {field: getattr(params, field) for field in field_names}
        if context_parameter is not None:
            kwargs[context_parameter] = context
        if is_coroutine:
            return coerce_tool_result(await fn(**kwargs))
        return coerce_tool_result(await asyncio.to_thread(functools.partial(fn, **kwargs)))

    return ToolDefinition(
        name=resolved_name,
        description=resolved_description,
        input_model=input_model,
        execute=execute,
        side_effecting=side_effecting,
        idempotency_key_field=idempotency_key_field,
    )


def function_tool[F: Callable[..., Any]](
    fn: F,
    *,
    name: str | None = None,
    description: str | None = None,
    side_effecting: bool = False,
    idempotency_key_field: str | None = "idempotency_key",
) -> F:
    """Attach a generated :class:`ToolDefinition` to *fn* and return *fn* unchanged."""
    definition = build_tool_definition(
        fn,
        name=name,
        description=description,
        side_effecting=side_effecting,
        idempotency_key_field=idempotency_key_field,
    )
    setattr(fn, "tool_definition", definition)
    setattr(fn, "__anycode_tool__", definition)
    return fn


@overload
def tool[F: Callable[..., Any]](fn: F, /) -> F: ...


@overload
def tool[F: Callable[..., Any]](
    *,
    name: str | None = ...,
    description: str | None = ...,
    side_effecting: bool = ...,
    idempotency_key_field: str | None = ...,
) -> Callable[[F], F]: ...


def tool[F: Callable[..., Any]](
    fn: F | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    side_effecting: bool = False,
    idempotency_key_field: str | None = "idempotency_key",
) -> F | Callable[[F], F]:
    """Expose a Python function as an agent tool.

    Usable bare (``@tool``) or configured (``@tool(name="search")``). The decorated
    function is returned unchanged and stays directly callable; its generated
    :class:`ToolDefinition` is available as ``fn.tool_definition``.
    """
    if fn is not None:
        return function_tool(fn)

    def _decorate(inner: F) -> F:
        return function_tool(
            inner,
            name=name,
            description=description,
            side_effecting=side_effecting,
            idempotency_key_field=idempotency_key_field,
        )

    return _decorate


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _named_tools() -> dict[str, ToolDefinition]:
    from anycode.handoff.tool import HANDOFF_TOOL_DEF
    from anycode.tools.built_in import BUILT_IN_TOOLS

    named = {definition.name: definition for definition in BUILT_IN_TOOLS}
    named[HANDOFF_TOOL_DEF.name] = HANDOFF_TOOL_DEF
    return named


def builtin_tool_names() -> tuple[str, ...]:
    """Names accepted by :func:`as_tool_definition` when given a string."""
    return tuple(sorted(_named_tools()))


def resolve_tool_specs(specs: Iterable[object]) -> list[ToolDefinition]:
    """Normalize a sequence of tool specifications, rejecting duplicate names."""
    resolved: list[ToolDefinition] = []
    seen: set[str] = set()
    for spec in specs:
        definition = as_tool_definition(spec)
        if definition.name in seen:
            raise ToolDefinitionError(f'Duplicate tool name "{definition.name}" in the tool list. Every tool needs a unique name.')
        seen.add(definition.name)
        resolved.append(definition)
    return resolved


def as_tool_definition(spec: object) -> ToolDefinition:
    """Normalize a tool specification into a :class:`ToolDefinition`.

    Accepts a ``ToolDefinition``, a ``@tool``-decorated function, any plain callable,
    or the name of a bundled tool.
    """
    if isinstance(spec, ToolDefinition):
        return spec

    if isinstance(spec, str):
        named = _named_tools()
        found = named.get(spec)
        if found is None:
            raise ToolDefinitionError(f'Unknown built-in tool "{spec}". Available: {", ".join(sorted(named))}.')
        return found

    if callable(spec):
        attached = getattr(spec, "tool_definition", None)
        if isinstance(attached, ToolDefinition):
            return attached
        return build_tool_definition(spec)

    raise ToolDefinitionError(
        f"Cannot use {type(spec).__name__} as a tool. Pass a ToolDefinition, a function "
        "(optionally decorated with @tool), or the name of a built-in tool."
    )
