"""Secret redaction at persistence and telemetry boundaries."""

from __future__ import annotations

from anycode.security.redaction import REDACTED_SECRET, redact_sensitive, redact_text, safe_exception_message


def test_redact_text_covers_common_credential_formats() -> None:
    text = "openai=sk-1234567890abcdef1234567890 aws=AKIAABCDEFGHIJKLMNOP github=ghp_abcdefghijklmnopqrst Authorization: Bearer abcdefghijklmnop"

    cleaned = redact_text(text)

    assert cleaned.count(REDACTED_SECRET) == 4
    assert "sk-" not in cleaned
    assert "AKIA" not in cleaned
    assert "ghp_" not in cleaned
    assert "Bearer" not in cleaned


def test_redact_sensitive_preserves_shape_and_scrubs_nested_keys() -> None:
    payload = {
        "headers": {"Authorization": "Bearer secret-value"},
        "config": {"api_key": "plain-value", "region": "us-east-1", "input_tokens": 42},
        "items": ["safe", "token=abcdefghijklmnop"],
    }

    cleaned = redact_sensitive(payload)

    assert cleaned == {
        "headers": {"Authorization": REDACTED_SECRET},
        "config": {"api_key": REDACTED_SECRET, "region": "us-east-1", "input_tokens": 42},
        "items": ["safe", REDACTED_SECRET],
    }


def test_redact_text_scrubs_environment_assignments_and_private_keys() -> None:
    text = (
        "OPENAI_API_KEY=plain-value AWS_SECRET_ACCESS_KEY='another-value'\n-----BEGIN PRIVATE KEY-----\nprivate-material\n-----END PRIVATE KEY-----"
    )

    cleaned = redact_text(text)

    assert "plain-value" not in cleaned
    assert "another-value" not in cleaned
    assert "private-material" not in cleaned


def test_safe_exception_message_redacts_bounds_and_never_raises() -> None:
    error = RuntimeError("provider rejected Bearer abcdefghijklmnop " + "x" * 50)

    assert safe_exception_message(error, max_chars=30) == "provider rejected <redacted-se"

    class BrokenError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("cannot render")

    assert safe_exception_message(BrokenError()) == "BrokenError"
