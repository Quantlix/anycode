"""Shared helpers for provider-prefixed sandbox secret references."""

from __future__ import annotations

from anycode.contracts.models import ContractError
from anycode.sandbox.models import SandboxSpec


def strip_secret_prefix(provider: str, reference: str) -> str:
    """Return the secret name behind ``<provider>:``, rejecting foreign prefixes."""
    prefix = f"{provider}:"
    if not reference.startswith(prefix):
        raise ValueError(f"Secret reference {reference!r} is not scoped to provider {provider!r}; expected the {prefix!r} prefix")
    return reference.removeprefix(prefix)


def validate_secret_references(provider: str, spec: SandboxSpec) -> ContractError | None:
    """Reject references scoped to another provider before any sandbox is created."""
    for name, reference in spec.secret_references.items():
        if not reference.startswith(f"{provider}:"):
            return ContractError(
                code="sandbox_secret_reference_invalid",
                message=f"Secret reference for {name!r} must use the '{provider}:' prefix",
            )
    return None
