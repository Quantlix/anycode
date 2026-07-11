"""Secret redaction for telemetry, persistence, reports, and audit records."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

REDACTED_SECRET = "<redacted-secret>"
DEFAULT_MAX_EXCEPTION_CHARS = 2_000

_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "private_key",
        "secret",
        "token",
    }
)
_SENSITIVE_KEY_SUFFIXES = (
    "_api_key",
    "_credential",
    "_credentials",
    "_password",
    "_private_key",
    "_secret",
    "_token",
)

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"(?i)\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)(?<![\w-])(?:[\w-]*api[_-]?key|token|[\w-]*(?:access|auth|refresh|id|session)[_-]?token|"
        r"[\w-]*(?:client[_-]?secret|secret[_-]?access[_-]?key|password|passwd|credential(?:s)?))"
        r"\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
    ),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
)


def redact_text(text: str) -> str:
    """Replace recognizable credentials in free-form text."""
    cleaned = text
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub(REDACTED_SECRET, cleaned)
    return cleaned


def redact_sensitive(value: Any, *, key: str | None = None) -> Any:
    """Recursively redact secret-looking keys and values while preserving shape."""
    if key is not None and _is_sensitive_key(key):
        return REDACTED_SECRET
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, (bytes, bytearray)):
        return redact_text(bytes(value).decode("utf-8", errors="replace"))
    if isinstance(value, Mapping):
        return {item_key: redact_sensitive(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item) for item in value)
    return value


def safe_exception_message(error: BaseException, *, max_chars: int = DEFAULT_MAX_EXCEPTION_CHARS) -> str:
    """Render an exception without leaking recognizable credentials or raising."""
    try:
        message = str(error)
    except Exception:  # noqa: BLE001 - exception rendering must never mask the original failure
        message = type(error).__name__
    cleaned = redact_text(message) or type(error).__name__
    return cleaned[:max_chars]


def _is_sensitive_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_KEY_SUFFIXES)
