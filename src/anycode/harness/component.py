"""Helpers for building :class:`HarnessComponent` records.

The registry treats every harness component as an immutable artifact with a stable
identity, an owner, an editability flag, and a deterministic checksum. Sensitive
fields (API keys, free-form environment values) are redacted before being hashed so
that drift detection never leaks secrets through manifest comparisons.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from anycode.types import (
    HarnessComponent,
    HarnessComponentKind,
    HarnessComponentOwner,
)

REDACTED_CHECKSUM_MARKER = "<redacted>"

_SENSITIVE_HINTS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "credential",
    "authorization",
    "auth_token",
    "private_key",
    "client_secret",
)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(hint in lowered for hint in _SENSITIVE_HINTS)


def redact_for_checksum(payload: Any) -> Any:
    """Return a copy of *payload* with secret-looking values replaced.

    This is used to derive checksums and to surface a stable representation of a
    component without leaking values that should never appear in evaluation reports.
    """

    if isinstance(payload, Mapping):
        return {key: REDACTED_CHECKSUM_MARKER if _is_sensitive_key(str(key)) else redact_for_checksum(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [redact_for_checksum(item) for item in payload]
    return payload


def compute_checksum(payload: Any) -> str:
    """Deterministic SHA-256 hex digest of any JSON-serialisable payload."""

    cleaned = redact_for_checksum(payload)
    encoded = json.dumps(cleaned, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_component(
    *,
    id: str,
    kind: HarnessComponentKind,
    source: str,
    owner: HarnessComponentOwner,
    description: str,
    editable: bool | None = None,
    payload: Any = None,
    metadata: Mapping[str, str] | None = None,
) -> HarnessComponent:
    """Build a :class:`HarnessComponent` with a checksum derived from *payload*.

    ``editable`` defaults to ``True`` for config/user/plugin components and to
    ``False`` for core components — core code is visible but not safely rewritable
    by the evolution loop. Callers may always override the default explicitly.
    """

    if editable is None:
        editable = owner != "core"
    checksum = compute_checksum(payload if payload is not None else {"id": id, "kind": kind})
    return HarnessComponent(
        id=id,
        kind=kind,
        source=source,
        editable=editable,
        owner=owner,
        checksum=checksum,
        description=description,
        metadata=dict(metadata or {}),
    )
