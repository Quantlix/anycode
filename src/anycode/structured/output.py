"""Schema conversion, response parsing, and retry-on-validation-failure for structured output."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

from anycode.types import LLMToolDef

STRUCTURED_OUTPUT_TOOL_NAME = "_structured_output"


def schema_to_tool_def(schema_class: type[BaseModel]) -> LLMToolDef:
    """Convert a Pydantic model class to an LLMToolDef for the Anthropic tool-use trick."""
    raw_schema = schema_class.model_json_schema()
    input_schema: dict[str, Any] = {"type": "object"}
    if "properties" in raw_schema:
        input_schema["properties"] = raw_schema["properties"]
    if "required" in raw_schema:
        input_schema["required"] = raw_schema["required"]
    if "$defs" in raw_schema:
        input_schema["$defs"] = raw_schema["$defs"]

    return LLMToolDef(
        name=STRUCTURED_OUTPUT_TOOL_NAME,
        description=f"Return your response in this exact JSON format matching the {schema_class.__name__} schema.",
        input_schema=input_schema,
    )


def schema_to_openai_response_format(schema_class: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model class to OpenAI's response_format with json_schema."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": schema_class.model_json_schema(),
            "strict": True,
        },
    }


def parse_structured_output(raw: str, schema_class: type[T]) -> T | None:
    """Attempt to parse raw LLM output as a validated Pydantic model instance.

    Tries direct JSON parse first, then extracts JSON from markdown code blocks.
    Returns None if parsing or validation fails.
    """
    # Try direct parse
    parsed = _try_parse_json(raw)
    if parsed is not None:
        return _try_validate(parsed, schema_class)

    # Try extracting from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if json_match:
        parsed = _try_parse_json(json_match.group(1).strip())
        if parsed is not None:
            return _try_validate(parsed, schema_class)

    return None


def build_retry_prompt(original_prompt: str, error: str) -> str:
    """Build a re-prompt message including the validation error."""
    return (
        f"{original_prompt}\n\n"
        "Your previous response could not be validated. Please try again.\n"
        f"Validation error: {error}\n"
        "Respond with valid JSON matching the required schema exactly."
    )


def _try_parse_json(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _try_validate(data: dict[str, Any], schema_class: type[T]) -> T | None:
    try:
        return schema_class.model_validate(data)
    except Exception:
        return None
