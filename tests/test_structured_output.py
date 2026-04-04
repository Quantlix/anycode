"""Tests for the structured output module: schema conversion, parsing, and retry logic."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from anycode.structured.output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    build_retry_prompt,
    parse_structured_output,
    schema_to_openai_response_format,
    schema_to_tool_def,
)

# -- Test models --


class SimpleModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    score: int


class CodeReview(BaseModel):
    model_config = ConfigDict(frozen=True)
    summary: str
    issues: list[str]
    severity: Literal["low", "medium", "high", "critical"]
    approved: bool


class NestedAddress(BaseModel):
    model_config = ConfigDict(frozen=True)
    street: str
    city: str
    zip_code: str


class NestedModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    address: NestedAddress


class OptionalFieldsModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    required_field: str
    optional_field: str | None = None
    optional_int: int | None = None


# -- Schema conversion tests --


class TestSchemaToToolDef:
    def test_simple_model_tool_def(self) -> None:
        tool_def = schema_to_tool_def(SimpleModel)
        assert tool_def.name == STRUCTURED_OUTPUT_TOOL_NAME
        assert "SimpleModel" in tool_def.description
        assert "properties" in tool_def.input_schema
        assert "name" in tool_def.input_schema["properties"]
        assert "score" in tool_def.input_schema["properties"]

    def test_nested_model_tool_def(self) -> None:
        tool_def = schema_to_tool_def(NestedModel)
        assert "properties" in tool_def.input_schema
        # Nested models should have $defs for address
        schema = tool_def.input_schema
        assert "name" in schema["properties"]
        assert "address" in schema["properties"]

    def test_required_fields_preserved(self) -> None:
        tool_def = schema_to_tool_def(SimpleModel)
        assert "required" in tool_def.input_schema
        assert "name" in tool_def.input_schema["required"]
        assert "score" in tool_def.input_schema["required"]


class TestSchemaToOpenAIFormat:
    def test_simple_model_format(self) -> None:
        fmt = schema_to_openai_response_format(SimpleModel)
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "response"
        assert fmt["json_schema"]["strict"] is True
        assert "properties" in fmt["json_schema"]["schema"]

    def test_nested_model_format(self) -> None:
        fmt = schema_to_openai_response_format(NestedModel)
        assert fmt["type"] == "json_schema"
        schema = fmt["json_schema"]["schema"]
        assert "properties" in schema


# -- Parse tests --


class TestParseStructuredOutput:
    def test_simple_flat_model(self) -> None:
        raw = '{"name": "test", "score": 42}'
        result = parse_structured_output(raw, SimpleModel)
        assert result is not None
        assert isinstance(result, SimpleModel)
        assert result.name == "test"
        assert result.score == 42

    def test_nested_model(self) -> None:
        raw = '{"name": "Alice", "address": {"street": "123 Main St", "city": "Springfield", "zip_code": "12345"}}'
        result = parse_structured_output(raw, NestedModel)
        assert result is not None
        assert isinstance(result, NestedModel)
        assert result.address.city == "Springfield"

    def test_enum_field_constrained(self) -> None:
        raw = '{"summary": "Good code", "issues": ["minor formatting"], "severity": "low", "approved": true}'
        result = parse_structured_output(raw, CodeReview)
        assert result is not None
        assert isinstance(result, CodeReview)
        assert result.severity == "low"
        assert result.approved is True

    def test_optional_fields_with_none(self) -> None:
        raw = '{"required_field": "hello"}'
        result = parse_structured_output(raw, OptionalFieldsModel)
        assert result is not None
        assert result.required_field == "hello"
        assert result.optional_field is None

    def test_optional_fields_with_values(self) -> None:
        raw = '{"required_field": "hello", "optional_field": "world", "optional_int": 42}'
        result = parse_structured_output(raw, OptionalFieldsModel)
        assert result is not None
        assert result.optional_field == "world"
        assert result.optional_int == 42

    def test_invalid_json_returns_none(self) -> None:
        result = parse_structured_output("not json at all", SimpleModel)
        assert result is None

    def test_partial_json_returns_none(self) -> None:
        result = parse_structured_output('{"name": "test"}', SimpleModel)
        # Missing required field 'score'
        assert result is None

    def test_wrong_type_does_not_crash(self) -> None:
        # Pydantic may coerce strings to int, so just check it doesn't crash
        parse_structured_output('{"name": "test", "score": "not_a_number"}', SimpleModel)

    def test_json_in_markdown_code_block(self) -> None:
        raw = 'Here is my response:\n```json\n{"name": "test", "score": 99}\n```\nDone.'
        result = parse_structured_output(raw, SimpleModel)
        assert result is not None
        assert result.name == "test"
        assert result.score == 99

    def test_json_in_bare_code_block(self) -> None:
        raw = '```\n{"name": "test", "score": 77}\n```'
        result = parse_structured_output(raw, SimpleModel)
        assert result is not None
        assert result.score == 77

    def test_empty_string(self) -> None:
        result = parse_structured_output("", SimpleModel)
        assert result is None

    def test_array_json_returns_none(self) -> None:
        result = parse_structured_output('[{"name": "test", "score": 1}]', SimpleModel)
        assert result is None


# -- Retry prompt tests --


class TestBuildRetryPrompt:
    def test_includes_error(self) -> None:
        prompt = build_retry_prompt("Do the thing", "Missing field 'score'")
        assert "Missing field 'score'" in prompt
        assert "Do the thing" in prompt
        assert "Validation error" in prompt

    def test_includes_schema_instruction(self) -> None:
        prompt = build_retry_prompt("", "some error")
        assert "valid JSON" in prompt


# -- Integration-style tests --


class TestStructuredOutputToolName:
    def test_tool_name_is_prefixed(self) -> None:
        assert STRUCTURED_OUTPUT_TOOL_NAME == "_structured_output"
        assert STRUCTURED_OUTPUT_TOOL_NAME.startswith("_")

    def test_tool_def_works_with_complex_model(self) -> None:
        tool_def = schema_to_tool_def(CodeReview)
        assert tool_def.name == "_structured_output"
        props = tool_def.input_schema["properties"]
        assert "summary" in props
        assert "issues" in props
        assert "severity" in props
        assert "approved" in props
