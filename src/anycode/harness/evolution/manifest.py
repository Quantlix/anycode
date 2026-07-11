"""Change manifest construction and persistence."""

from __future__ import annotations

import difflib
import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anycode.harness.component import compute_checksum
from anycode.helpers.uuid7 import uuid7
from anycode.security.redaction import redact_sensitive
from anycode.types import (
    EvidencePacket,
    HarnessChangeEdit,
    HarnessChangeManifest,
    HarnessChangePrediction,
)


def diff_payloads(
    *,
    component_id: str,
    before: Any,
    after: Any,
    note: str = "",
) -> HarnessChangeEdit:
    """Build a :class:`HarnessChangeEdit` from two arbitrary payloads.

    The textual diff uses pretty-printed JSON for stability across runs.
    """

    before_json = json.dumps(before, indent=2, sort_keys=True, default=str).splitlines(keepends=True)
    after_json = json.dumps(after, indent=2, sort_keys=True, default=str).splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(
            before_json,
            after_json,
            fromfile=f"{component_id}@before",
            tofile=f"{component_id}@after",
            n=3,
        )
    )
    return HarnessChangeEdit(
        component_id=component_id,
        before_checksum=compute_checksum(before),
        after_checksum=compute_checksum(after),
        diff="".join(diff_lines),
        note=note,
    )


def build_change_manifest(
    *,
    component_ids: Sequence[str],
    summary: str,
    predictions: Sequence[HarnessChangePrediction],
    rollback_plan: str,
    edits: Iterable[HarnessChangeEdit] = (),
    evidence_packets: Iterable[EvidencePacket] = (),
    safety_review_required: bool = True,
    manifest_id: str | None = None,
    created_at: datetime | None = None,
) -> HarnessChangeManifest:
    """Validated constructor for :class:`HarnessChangeManifest`.

    Raises ``ValueError`` if the manifest would be ambiguous — for example, no
    predictions or component ids — so that downstream acceptance logic can rely
    on a populated structure.
    """

    if not component_ids:
        raise ValueError("change manifest must reference at least one component_id")
    if not predictions:
        raise ValueError("change manifest must include at least one falsifiable prediction")
    if not summary.strip():
        raise ValueError("change manifest must have a non-empty summary")
    if not rollback_plan.strip():
        raise ValueError("change manifest must have an explicit rollback plan")

    return HarnessChangeManifest(
        id=manifest_id or f"change-{uuid7()}",
        component_ids=tuple(component_ids),
        evidence_packet_ids=tuple(packet.id for packet in evidence_packets),
        summary=summary.strip(),
        predictions=tuple(predictions),
        rollback_plan=rollback_plan.strip(),
        safety_review_required=safety_review_required,
        edits=tuple(edits),
        created_at=created_at or datetime.now(UTC),
    )


def save_change_manifest(manifest: HarnessChangeManifest, path: str | Path, *, redact_sensitive_data: bool = True) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    if redact_sensitive_data:
        payload = redact_sensitive(payload)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return target


def load_change_manifest(path: str | Path) -> HarnessChangeManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return HarnessChangeManifest.model_validate(payload)


__all__ = [
    "build_change_manifest",
    "diff_payloads",
    "load_change_manifest",
    "save_change_manifest",
]
