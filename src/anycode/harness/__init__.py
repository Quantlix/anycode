"""Adaptive harness evolution surface.

The :mod:`anycode.harness` package exposes:

- A read-only registry that describes every editable harness component.
- A trajectory evidence pipeline that distills runs into actionable artifacts.
- A governed evolution loop that proposes, isolates, evaluates, and accepts changes.
- An experimental meta-optimization layer that evaluates evolution blueprints.

The runtime never imports :mod:`anycode.harness.evolution` or
:mod:`anycode.harness.meta` by default — production runs only need the registry
and evidence modules. Evolution and meta-evolution are opt-in entry points for
evaluation tooling.
"""

from anycode.harness.component import (
    REDACTED_CHECKSUM_MARKER,
    redact_for_checksum,
)
from anycode.harness.distill import distill_evidence, replay_raw_trace
from anycode.harness.evidence import (
    EvidenceCollector,
    EvidenceStore,
    write_evidence_bundle,
)
from anycode.harness.failure_taxonomy import (
    DEFAULT_TAXONOMY_VERSION,
    categorize_event,
    categorize_run,
)
from anycode.harness.manifest import (
    diff_manifests,
    load_manifest,
    save_manifest,
)
from anycode.harness.registry import (
    HarnessRegistry,
    build_default_registry,
    build_manifest,
    register_component,
)

__all__ = [
    "REDACTED_CHECKSUM_MARKER",
    "EvidenceCollector",
    "EvidenceStore",
    "HarnessRegistry",
    "DEFAULT_TAXONOMY_VERSION",
    "build_default_registry",
    "build_manifest",
    "categorize_event",
    "categorize_run",
    "diff_manifests",
    "distill_evidence",
    "load_manifest",
    "redact_for_checksum",
    "register_component",
    "replay_raw_trace",
    "save_manifest",
    "write_evidence_bundle",
]
