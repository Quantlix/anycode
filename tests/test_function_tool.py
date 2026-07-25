"""Tests for the @tool decorator and tool-specification normalization."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from anycode import (
    AgentInfo,
    ToolDefinition,
    ToolDefinitionError,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    ToolUseContext,
    as_tool_definition,
    tool,
)
from anycode.tools.function_tool import build_tool_definition, builtin_tool_names, coerce_tool_result, parse_docstring


def _context(name: str = "tester") -> ToolUseContext:
    return ToolUseContext(agent=AgentInfo(name=name, role="test", model="test-model"))


# ---------------------------------------------------------------------------
# Bare decorator
# ---------------------------------------------------------------------------


@tool
def greet(name: str, excited: bool = False) -> str:
    """Greet a person by name.

    Args:
        name: Who to greet.
        excited: Whether to add an exclamation mark.
    """
    return f"Hello {name}{'!' if excited else '.'}"


def test_bare_decorator_builds_definition() -> None:
    definition = as_tool_definition(greet)
    assert isinstance(definition, ToolDefinition)
    assert definition.name == "greet"
    assert definition.description == "Greet a person by name."
    assert definition.side_effecting is False


def test_schema_marks_required_and_optional_fields() -> None:
    schema = as_tool_definition(greet).input_model.model_json_schema()
    assert schema["required"] == ["name"]
    assert schema["properties"]["name"]["description"] == "Who to greet."
    assert schema["properties"]["excited"]["description"] == "Whether to add an exclamation mark."
    assert schema["properties"]["excited"]["default"] is False


def test_decorated_function_stays_directly_callable() -> None:
    assert greet("Ada", excited=True) == "Hello Ada!"


# ---------------------------------------------------------------------------
# Parametrized decorator
# ---------------------------------------------------------------------------


@tool(name="publish-note", description="Publish a note somewhere durable.", side_effecting=True)
async def publish(body: str, idempotency_key: str = "") -> str:
    return f"published:{body}:{idempotency_key}"


def test_parametrized_decorator_overrides_metadata() -> None:
    definition = as_tool_definition(publish)
    assert definition.name == "publish-note"
    assert definition.description == "Publish a note somewhere durable."
    assert definition.side_effecting is True


# ---------------------------------------------------------------------------
# Docstring parsing
# ---------------------------------------------------------------------------


def test_parse_docstring_handles_sections_and_continuations() -> None:
    summary, args = parse_docstring(
        """Do a thing.

        Extra summary line.

        Args:
            first: The first argument
                which wraps onto a second line.
            second (int): The second argument.

        Returns:
            Something irrelevant.
        """
    )
    assert summary == "Do a thing. Extra summary line."
    assert args["first"] == "The first argument which wraps onto a second line."
    assert args["second"] == "The second argument."
    assert "Returns" not in args


def test_parse_docstring_degrades_on_unstructured_text() -> None:
    summary, args = parse_docstring("Just a sentence with a colon: no sections here.")
    assert summary == "Just a sentence with a colon: no sections here."
    assert args == {}


def test_parse_docstring_handles_empty_input() -> None:
    assert parse_docstring(None) == ("", {})
    assert parse_docstring("   ") == ("", {})


# ---------------------------------------------------------------------------
# Annotated constraints
# ---------------------------------------------------------------------------


@tool
def scaled(value: Annotated[int, Field(ge=0, le=10, description="Clamped value.")]) -> int:
    """Double a bounded value."""
    return value * 2


def test_annotated_constraints_survive_into_schema() -> None:
    schema = as_tool_definition(scaled).input_model.model_json_schema()
    assert schema["properties"]["value"]["minimum"] == 0
    assert schema["properties"]["value"]["maximum"] == 10
    assert schema["properties"]["value"]["description"] == "Clamped value."


def test_annotated_constraints_reject_bad_input() -> None:
    model = as_tool_definition(scaled).input_model
    with pytest.raises(ValueError):
        model.model_validate({"value": 99})


# ---------------------------------------------------------------------------
# Async and sync execution
# ---------------------------------------------------------------------------


async def test_async_function_is_awaited() -> None:
    @tool
    async def slow_echo(text: str) -> str:
        """Echo text."""
        await asyncio.sleep(0)
        return text.upper()

    definition = as_tool_definition(slow_echo)
    result = await definition.execute(definition.input_model.model_validate({"text": "hi"}), _context())
    assert result.data == "HI"


async def test_sync_function_runs_off_the_event_loop() -> None:
    seen: list[int] = []

    @tool
    def blocking(marker: int) -> int:
        """Record the calling thread."""
        seen.append(threading.get_ident())
        return marker

    definition = as_tool_definition(blocking)
    result = await definition.execute(definition.input_model.model_validate({"marker": 7}), _context())
    assert result.data == "7"
    assert seen and seen[0] != threading.get_ident()


# ---------------------------------------------------------------------------
# Context injection
# ---------------------------------------------------------------------------


async def test_context_injected_by_annotation_and_excluded_from_schema() -> None:
    @tool
    def whoami(prefix: str, ctx_param: ToolUseContext) -> str:
        """Report the calling agent."""
        return f"{prefix}{ctx_param.agent.name}"

    definition = as_tool_definition(whoami)
    assert set(definition.input_model.model_fields) == {"prefix"}
    result = await definition.execute(definition.input_model.model_validate({"prefix": "agent="}), _context("scout"))
    assert result.data == "agent=scout"


async def test_context_injected_by_conventional_name() -> None:
    @tool
    def named_context(value: str, ctx) -> str:  # noqa: ANN001 - deliberately unannotated
        """Use the conventional context parameter name."""
        return f"{value}:{ctx.agent.name}"

    definition = as_tool_definition(named_context)
    assert set(definition.input_model.model_fields) == {"value"}
    result = await definition.execute(definition.input_model.model_validate({"value": "v"}), _context("runner"))
    assert result.data == "v:runner"


def test_two_context_parameters_are_rejected() -> None:
    def doubled(ctx: ToolUseContext, context: ToolUseContext) -> str:
        """Two contexts."""
        return "x"

    with pytest.raises(ToolDefinitionError, match="two context parameters"):
        build_tool_definition(doubled)


# ---------------------------------------------------------------------------
# Return coercion
# ---------------------------------------------------------------------------


class _Payload(BaseModel):
    value: int


@pytest.mark.parametrize(
    ("returned", "expected"),
    [
        (ToolResult(data="raw", is_error=True), "raw"),
        ("plain", "plain"),
        (None, ""),
        (_Payload(value=3), '{"value":3}'),
        ({"a": 1}, '{"a": 1}'),
        ([1, 2], "[1, 2]"),
        (True, "true"),
        (4.5, "4.5"),
    ],
)
def test_return_value_coercion(returned: object, expected: str) -> None:
    assert coerce_tool_result(returned).data == expected


def test_tool_result_passthrough_preserves_flags() -> None:
    original = ToolResult(data="boom", is_error=True, retry_safe=False)
    assert coerce_tool_result(original) is original


def test_unknown_type_falls_back_to_str() -> None:
    class Opaque:
        def __str__(self) -> str:
            return "opaque"

    assert coerce_tool_result(Opaque()).data == "opaque"


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_missing_description_is_rejected() -> None:
    def undocumented(value: str) -> str:
        return value

    with pytest.raises(ToolDefinitionError, match="has no description"):
        build_tool_definition(undocumented)


def test_variadic_parameters_are_rejected() -> None:
    def variadic(*args: str) -> str:
        """Takes anything."""
        return "x"

    with pytest.raises(ToolDefinitionError, match="variadic parameters"):
        build_tool_definition(variadic)


def test_invalid_name_is_rejected() -> None:
    def fine(value: str) -> str:
        """Fine."""
        return value

    with pytest.raises(ToolDefinitionError, match="Invalid tool name"):
        build_tool_definition(fine, name="not a valid name!")


def test_unannotated_parameter_defaults_to_string() -> None:
    @tool
    def loose(value) -> str:  # noqa: ANN001 - deliberately unannotated
        """Accept anything as text."""
        return str(value)

    schema = as_tool_definition(loose).input_model.model_json_schema()
    assert schema["properties"]["value"]["type"] == "string"


def test_zero_parameter_tool_is_valid() -> None:
    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    definition = as_tool_definition(ping)
    assert definition.input_model.model_fields == {}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_as_tool_definition_accepts_every_supported_form() -> None:
    definition = as_tool_definition(greet)
    assert as_tool_definition(definition) is definition
    assert as_tool_definition("file_read").name == "file_read"

    def undecorated(value: str) -> str:
        """Undecorated but documented."""
        return value

    assert as_tool_definition(undecorated).name == "undecorated"


def test_as_tool_definition_rejects_other_values() -> None:
    with pytest.raises(ToolDefinitionError, match="Cannot use int as a tool"):
        as_tool_definition(3)


def test_unknown_builtin_name_lists_alternatives() -> None:
    with pytest.raises(ToolDefinitionError, match="Available: bash"):
        as_tool_definition("nope")


def test_builtin_tool_names_includes_bundled_tools() -> None:
    names = builtin_tool_names()
    assert {"bash", "file_read", "file_write", "file_edit", "grep", "list_files", "handoff"} <= set(names)


# ---------------------------------------------------------------------------
# End-to-end through the executor
# ---------------------------------------------------------------------------


async def test_round_trip_through_registry_and_executor() -> None:
    @tool
    def add(left: int, right: int = 1) -> dict:
        """Add two integers."""
        return {"sum": left + right}

    registry = ToolRegistry()
    registry.register(as_tool_definition(add))
    executor = ToolExecutor(registry)

    result = await executor.execute("add", {"left": 4, "right": 5}, _context())
    assert not result.is_error
    assert json.loads(result.data) == {"sum": 9}


async def test_executor_reports_validation_errors() -> None:
    registry = ToolRegistry()
    registry.register(as_tool_definition(greet))
    executor = ToolExecutor(registry)

    result = await executor.execute("greet", {"excited": True}, _context())
    assert result.is_error
    assert "Invalid input" in result.data
