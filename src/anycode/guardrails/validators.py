"""Input/output content validators with Protocol-based extensibility."""

from __future__ import annotations

from anycode.constants import MAX_VALIDATION_RETRIES  # noqa: F401
from anycode.types import AgentInfo, OutputValidator, ValidationResult


async def run_validators(
    output: str,
    validators: list[OutputValidator],
    context: AgentInfo,
) -> ValidationResult:
    """Run all validators sequentially. Return first failure, or valid result."""
    for validator in validators:
        result = await validator.validate(output, context)
        if not result.valid:
            return result
    return ValidationResult(valid=True)


class MaxLengthValidator:
    """Rejects output exceeding a character limit."""

    def __init__(self, max_length: int) -> None:
        self._max_length = max_length

    async def validate(self, output: str, context: AgentInfo) -> ValidationResult:
        if len(output) > self._max_length:
            return ValidationResult(
                valid=False,
                reason=f"Output exceeds maximum length of {self._max_length} characters (got {len(output)}).",
                retry=True,
            )
        return ValidationResult(valid=True)


class ContainsValidator:
    """Rejects output that does not contain a required substring."""

    def __init__(self, required: str) -> None:
        self._required = required

    async def validate(self, output: str, context: AgentInfo) -> ValidationResult:
        if self._required not in output:
            return ValidationResult(
                valid=False,
                reason=f"Output must contain '{self._required}'.",
                retry=True,
            )
        return ValidationResult(valid=True)


class BlocklistValidator:
    """Rejects output containing any blocked term."""

    def __init__(self, blocked_terms: list[str]) -> None:
        self._blocked = [t.lower() for t in blocked_terms]

    async def validate(self, output: str, context: AgentInfo) -> ValidationResult:
        lower = output.lower()
        for term in self._blocked:
            if term in lower:
                return ValidationResult(
                    valid=False,
                    reason=f"Output contains blocked term: '{term}'.",
                    retry=True,
                )
        return ValidationResult(valid=True)
