"""Filesystem persistence for trajectory evidence bundles.

Layout for one run::

    <root>/<run_id>/
      summary.json         # RunSummary
      packets.json         # EvidencePacket[]
      failure_map.json     # FailureMapEntry[]
      timeline.json        # TrajectoryEvent[]
      raw_trace.jsonl      # raw normalized events (one per line)

Every file is deterministic JSON so that bundles diff cleanly between runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anycode.harness.distill import distill_evidence
from anycode.security.redaction import redact_sensitive
from anycode.types import AgentRunResult, TrajectoryEvent, TrajectoryEvidence


class EvidenceStore:
    """Append-only on-disk store for trajectory evidence bundles."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def bundle_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def write_bundle(
        self,
        evidence: TrajectoryEvidence,
        *,
        raw_events: list[dict[str, Any]] | None = None,
    ) -> Path:
        target = self.bundle_dir(evidence.run_summary.run_id)
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "summary.json", evidence.run_summary.model_dump())
        _write_json(
            target / "failure_map.json",
            [entry.model_dump() for entry in evidence.failure_map],
        )
        _write_json(
            target / "timeline.json",
            [event.model_dump() for event in evidence.decision_timeline],
        )
        _write_json(
            target / "packets.json",
            [packet.model_dump() for packet in evidence.evidence_packets],
        )
        raw_path = target / "raw_trace.jsonl"
        if raw_events is not None:
            with raw_path.open("w", encoding="utf-8") as fh:
                for record in raw_events:
                    fh.write(json.dumps(redact_sensitive(record), default=str, sort_keys=True) + "\n")
        return target

    def list_runs(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(child.name for child in self.root.iterdir() if child.is_dir())

    def load_bundle(self, run_id: str) -> TrajectoryEvidence:
        target = self.bundle_dir(run_id)
        if not target.exists():
            raise FileNotFoundError(f"evidence bundle '{run_id}' not found under {self.root}")
        summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
        failure_map = json.loads((target / "failure_map.json").read_text(encoding="utf-8"))
        timeline = json.loads((target / "timeline.json").read_text(encoding="utf-8"))
        packets = json.loads((target / "packets.json").read_text(encoding="utf-8"))
        raw_path = target / "raw_trace.jsonl"
        return TrajectoryEvidence.model_validate(
            {
                "run_summary": summary,
                "failure_map": failure_map,
                "decision_timeline": timeline,
                "evidence_packets": packets,
                "raw_trace_path": str(raw_path) if raw_path.exists() else None,
            }
        )


class EvidenceCollector:
    """Incrementally collect raw events for a run, then build a bundle on close."""

    def __init__(
        self,
        *,
        run_id: str,
        task: str,
        store: EvidenceStore | None = None,
        manifest_checksum: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.task = task
        self.store = store
        self.manifest_checksum = manifest_checksum
        self._raw_events: list[dict[str, Any]] = []

    def record(self, event: dict[str, Any]) -> None:
        self._raw_events.append(dict(event))

    @property
    def raw_events(self) -> list[dict[str, Any]]:
        return list(self._raw_events)

    def finalize(self, result: AgentRunResult) -> TrajectoryEvidence:
        evidence = distill_evidence(
            result,
            run_id=self.run_id,
            task=self.task,
            manifest_checksum=self.manifest_checksum,
        )
        if self.store is not None:
            target = self.store.write_bundle(evidence, raw_events=self._raw_events)
            evidence = evidence.model_copy(update={"raw_trace_path": str(target / "raw_trace.jsonl")})
        return evidence


def write_evidence_bundle(
    evidence: TrajectoryEvidence,
    root: str | Path,
    *,
    raw_events: list[dict[str, Any]] | None = None,
) -> Path:
    """One-shot helper: write *evidence* under ``<root>/<run_id>/``."""

    store = EvidenceStore(root)
    return store.write_bundle(evidence, raw_events=raw_events)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(redact_sensitive(payload), indent=2, sort_keys=True, default=str, ensure_ascii=False),
        encoding="utf-8",
    )


def serialize_timeline(events: list[TrajectoryEvent]) -> list[dict[str, Any]]:
    return [event.model_dump() for event in events]


__all__ = [
    "EvidenceCollector",
    "EvidenceStore",
    "serialize_timeline",
    "write_evidence_bundle",
]
