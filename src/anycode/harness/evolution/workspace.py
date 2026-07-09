"""Isolated workspace for evaluating candidate harness changes.

The workspace is an on-disk copy of the editable harness state (today: a tree
of JSON snapshots representing component payloads, since AnyCode keeps most of
its harness in code or in-memory ``OrchestratorConfig`` objects). Candidate
edits are applied to the workspace and either accepted (returned as a patch) or
rolled back atomically.

The workspace never modifies the source repository directly. The evolution
loop emits patch files that humans review and apply.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anycode.harness.component import compute_checksum


@dataclass
class HarnessWorkspace:
    """An isolated working copy of harness component payloads."""

    root: Path
    components: dict[str, dict[str, Any]]

    def read(self, component_id: str) -> dict[str, Any]:
        if component_id not in self.components:
            raise KeyError(f"component '{component_id}' is not materialised in workspace")
        return json.loads(json.dumps(self.components[component_id], sort_keys=True, default=str))

    def write(self, component_id: str, payload: dict[str, Any]) -> str:
        if component_id not in self.components:
            raise KeyError(f"component '{component_id}' is not materialised in workspace")
        self.components[component_id] = json.loads(json.dumps(payload, sort_keys=True, default=str))
        path = self.root / f"{_safe_filename(component_id)}.json"
        path.write_text(
            json.dumps(self.components[component_id], indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return compute_checksum(self.components[component_id])

    def rollback(self, component_id: str, payload: dict[str, Any]) -> None:
        self.write(component_id, payload)

    def list(self) -> list[str]:
        return sorted(self.components.keys())

    def checksum(self, component_id: str) -> str:
        return compute_checksum(self.read(component_id))


def _safe_filename(component_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in component_id)
    return safe or "component"


def materialize_workspace(
    payloads: Mapping[str, dict[str, Any]],
    *,
    root: str | Path | None = None,
) -> HarnessWorkspace:
    """Build a :class:`HarnessWorkspace` from a mapping of component payloads.

    If *root* is not provided, a temporary directory is created. Callers may
    rely on ``isolated_workspace`` (a context manager below) for automatic
    cleanup.
    """

    if root is None:
        root = Path(tempfile.mkdtemp(prefix="anycode-evolve-"))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    materialised: dict[str, dict[str, Any]] = {}
    for component_id, payload in payloads.items():
        normalized = json.loads(json.dumps(payload, sort_keys=True, default=str))
        materialised[component_id] = normalized
        (root / f"{_safe_filename(component_id)}.json").write_text(
            json.dumps(normalized, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return HarnessWorkspace(root=root, components=materialised)


@contextmanager
def isolated_workspace(
    payloads: dict[str, dict[str, Any]],
    *,
    cleanup: bool = True,
):
    """Context-managed temporary workspace that is removed on exit when ``cleanup`` is set."""

    ws = materialize_workspace(payloads)
    try:
        yield ws
    finally:
        if cleanup and ws.root.exists():
            shutil.rmtree(ws.root, ignore_errors=True)


__all__ = [
    "HarnessWorkspace",
    "isolated_workspace",
    "materialize_workspace",
]
