"""Persistence and drift detection for :class:`HarnessManifest`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anycode.types import HarnessManifest


def save_manifest(manifest: HarnessManifest, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump()
    target.write_text(json.dumps(payload, indent=2, default=str, sort_keys=True), encoding="utf-8")
    return target


def load_manifest(path: str | Path) -> HarnessManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return HarnessManifest.model_validate(payload)


def diff_manifests(baseline: HarnessManifest, candidate: HarnessManifest) -> dict[str, Any]:
    """Return a structured diff of two manifests.

    The result has four buckets:

    - ``added`` — component ids present only in *candidate*.
    - ``removed`` — component ids present only in *baseline*.
    - ``changed`` — component ids in both manifests whose checksum differs.
    - ``unchanged`` — component ids in both manifests with matching checksums.
    """

    base_by_id = {c.id: c for c in baseline.components}
    cand_by_id = {c.id: c for c in candidate.components}

    added = sorted(cid for cid in cand_by_id if cid not in base_by_id)
    removed = sorted(cid for cid in base_by_id if cid not in cand_by_id)
    changed: list[dict[str, str]] = []
    unchanged: list[str] = []
    for cid in sorted(cand_by_id.keys() & base_by_id.keys()):
        base_c = base_by_id[cid]
        cand_c = cand_by_id[cid]
        if base_c.checksum != cand_c.checksum:
            changed.append({"id": cid, "before": base_c.checksum, "after": cand_c.checksum})
        else:
            unchanged.append(cid)

    return {
        "baseline_checksum": baseline.checksum,
        "candidate_checksum": candidate.checksum,
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged": unchanged,
        "drift_detected": bool(added or removed or changed),
    }


__all__ = ["diff_manifests", "load_manifest", "save_manifest"]
