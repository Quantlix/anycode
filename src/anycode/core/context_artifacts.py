"""On-disk artifact offload for large tool outputs.

Large tool outputs are written to a deterministic location with a content
digest, leaving an inline placeholder summary for the model and a recovery
hint that the agent can use to fetch the full payload later.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Final

from anycode.security.redaction import redact_text
from anycode.types import ContextArtifact

_DEFAULT_HEAD_CHARS: Final[int] = 400
_DEFAULT_TAIL_CHARS: Final[int] = 400
_FILENAME_DIGEST_LENGTH: Final[int] = 16


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def offload_text(
    text: str,
    artifact_dir: str | os.PathLike[str],
    *,
    label: str = "tool_output",
    source_event_id: str | None = None,
    head_chars: int = _DEFAULT_HEAD_CHARS,
    tail_chars: int = _DEFAULT_TAIL_CHARS,
    redact_sensitive_data: bool = True,
) -> ContextArtifact:
    """Write `text` to `artifact_dir` and return a recoverable artifact handle."""
    if redact_sensitive_data:
        text = redact_text(text)
    target_dir = Path(artifact_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = _hash(text)
    artifact_id = digest[:_FILENAME_DIGEST_LENGTH]
    target = target_dir / f"{label}-{artifact_id}.txt"
    if not target.exists():
        target.write_text(text, encoding="utf-8")
    encoded_size = len(text.encode("utf-8"))
    head = text[:head_chars]
    tail = text[-tail_chars:] if len(text) > tail_chars else ""
    recovery_hint = f"Full output offloaded to '{target}'. Use the file_read tool with this path to retrieve the complete content."
    return ContextArtifact(
        artifact_id=artifact_id,
        path=str(target),
        bytes=encoded_size,
        digest=digest,
        head_excerpt=head,
        tail_excerpt=tail,
        recovery_hint=recovery_hint,
        source_event_id=source_event_id,
    )


def restore_text(artifact: ContextArtifact) -> str:
    """Read an offloaded artifact back into memory and verify its digest."""
    path = Path(artifact.path)
    payload = path.read_text(encoding="utf-8")
    if _hash(payload) != artifact.digest:
        raise ValueError(f"Digest mismatch when restoring artifact {artifact.artifact_id}")
    return payload


def render_placeholder(artifact: ContextArtifact) -> str:
    """Inline placeholder text the agent sees in place of an offloaded payload."""
    return (
        f"[OFFLOADED ARTIFACT id={artifact.artifact_id} bytes={artifact.bytes}]\n"
        f"head:\n{artifact.head_excerpt}\n"
        f"tail:\n{artifact.tail_excerpt}\n"
        f"recovery: {artifact.recovery_hint}"
    )
