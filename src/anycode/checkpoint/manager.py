"""Checkpoint lifecycle manager — automatic save, load, prune, and spec-change detection."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from anycode.checkpoint.store import FilesystemCheckpointStore
from anycode.constants import CHECKPOINT_FORMAT_VERSION
from anycode.helpers.uuid7 import uuid7
from anycode.types import AgentRunResult, CheckpointConfig, CheckpointData, CheckpointStore, Task, TokenUsage


class CheckpointManager:
    """Orchestrates checkpoint save/load/prune with spec-change detection."""

    def __init__(self, config: CheckpointConfig, store: CheckpointStore | None = None) -> None:
        self._config = config
        self._store: CheckpointStore = store or FilesystemCheckpointStore(config.path)

    @property
    def store(self) -> CheckpointStore:
        return self._store

    async def auto_save(
        self,
        workflow_id: str,
        tasks: list[Task],
        agent_results: dict[str, AgentRunResult],
        wave_index: int,
        total_usage: TokenUsage,
        metadata: dict[str, Any] | None = None,
    ) -> CheckpointData:
        checkpoint = CheckpointData(
            id=str(uuid7()),
            workflow_id=workflow_id,
            version=CHECKPOINT_FORMAT_VERSION,
            tasks=tasks,
            agent_results=agent_results,
            wave_index=wave_index,
            total_token_usage=total_usage,
            created_at=datetime.now(UTC),
            metadata=metadata,
        )
        await self._store.save(checkpoint)
        await self._store.prune(workflow_id, self._config.keep_last)
        return checkpoint

    async def load_latest(self, workflow_id: str) -> CheckpointData | None:
        return await self._store.latest(workflow_id)

    @staticmethod
    def compute_spec_hash(tasks: list[Task]) -> str:
        content = "|".join(f"{t.title}:{t.description}" for t in sorted(tasks, key=lambda t: t.title))
        return hashlib.sha256(content.encode()).hexdigest()

    def detect_spec_change(self, current_tasks: list[Task], checkpoint: CheckpointData) -> bool:
        current_hash = self.compute_spec_hash(current_tasks)
        checkpoint_hash = self.compute_spec_hash(checkpoint.tasks)
        return current_hash != checkpoint_hash
