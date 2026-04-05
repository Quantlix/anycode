# 2.2 Workflow Checkpointing & Resume

> **Priority:** High
> **Complexity:** High
> **Status:** Complete
> **Phase:** 2 — Persistence & Resilience

## Problem

A 50-task multi-agent pipeline that crashes at task 35 must restart from task 1. This wastes tokens, time, and money. Long-running workflows (codebase migration, test generation, documentation) are risky without crash recovery.

No major competitor has checkpoint/resume for DAG-based agent workflows. This is a killer USP for enterprise adoption.

## Solution

After each execution wave, serialize the complete workflow state (task statuses, agent results, conversation context) to a persistent store. On restart, load the checkpoint and resume from the next incomplete wave.

## Files to Create

```
src/anycode/checkpoint/
├── __init__.py
├── store.py          # CheckpointStore Protocol + filesystem/SQLite implementations
├── manager.py        # Automatic checkpoint creation and lifecycle
└── serializer.py     # Pydantic model (de)serialization including LLMMessage content
```

## Files to Modify

| File | Change |
|------|--------|
| `src/anycode/types.py` | Add `CheckpointData`, `CheckpointConfig` models |
| `src/anycode/core/orchestrator.py` | Save checkpoint after each wave; accept `resume_from` parameter |
| `src/anycode/tasks/queue.py` | Add `serialize()` / `deserialize()` methods |

## New Types

```python
class CheckpointConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    backend: Literal["filesystem", "sqlite"] = "filesystem"
    path: str = ".anycode/checkpoints"
    keep_last: int = 5  # Retain last N checkpoints


class CheckpointData(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    workflow_id: str
    version: int = 1
    tasks: list[Task]
    agent_results: dict[str, AgentRunResult]
    wave_index: int
    total_token_usage: TokenUsage
    created_at: datetime
    metadata: dict[str, Any] | None = None


@runtime_checkable
class CheckpointStore(Protocol):
    async def save(self, checkpoint: CheckpointData) -> None: ...
    async def load(self, checkpoint_id: str) -> CheckpointData | None: ...
    async def latest(self, workflow_id: str) -> CheckpointData | None: ...
    async def list(self, workflow_id: str) -> list[str]: ...
    async def delete(self, checkpoint_id: str) -> None: ...
    async def prune(self, workflow_id: str, keep_last: int) -> None: ...
```

## API Design

```python
from anycode import AnyCode, CheckpointConfig

# Enable checkpointing
engine = AnyCode(config={
    "checkpoint": CheckpointConfig(
        enabled=True,
        path=".anycode/checkpoints",
    ),
})

# First run — checkpoints saved automatically
result = await engine.run_tasks(team, specs)

# If it crashes mid-way, resume:
result = await engine.run_tasks(
    team, specs,
    resume_from="latest",  # or a specific checkpoint ID
)
```

## Checkpoint Flow

```
run_tasks():
  1. If resume_from is set:
     a. Load checkpoint
     b. Restore task queue state (skip completed tasks)
     c. Restore agent results collected so far
     d. Set wave index to checkpoint.wave_index + 1
  2. Execute waves:
     for wave in remaining_waves:
       a. Execute all tasks in wave (parallel)
       b. Save checkpoint:
          - All task states
          - All agent results so far
          - Current wave index
          - Cumulative token usage
       c. Prune old checkpoints (keep last N)
  3. Return final result
```

## Serialization Strategy

The main challenge is serializing `LLMMessage` content blocks which contain:
- `TextBlock` — straightforward text
- `ToolUseBlock` — contains `dict[str, Any]` input
- `ToolResultBlock` — contains string content
- `ImageBlock` — contains base64 data (potentially large)

Strategy:
- Use Pydantic's `model_dump(mode="json")` for all models
- Store as JSON files (filesystem) or JSONB (SQLite)
- Image blocks: store base64 inline for small images; reference external files for large ones
- Version the checkpoint format to support future migrations

## Filesystem Layout

```
.anycode/checkpoints/
├── workflow-abc123/
│   ├── checkpoint-001.json
│   ├── checkpoint-002.json
│   └── checkpoint-003.json
└── workflow-def456/
    └── checkpoint-001.json
```

## Implementation Notes

- Checkpoints are saved after each wave, not after each task (wave is the atomic unit)
- `resume_from="latest"` loads the most recent checkpoint for the given task specs
- Workflow ID is derived from a hash of the task spec titles + descriptions
- Checkpoint files are human-readable JSON for debugging
- SQLite backend stores checkpoints as JSONB for efficient querying
- Old checkpoints are pruned automatically based on `keep_last`
- Checkpoint version field enables future format migrations

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Serialization of arbitrary `Any` types in tool inputs | High | JSON-serialize with fallback to `str()` for non-serializable |
| Checkpoint file corruption | Medium | Write to temp file, then atomic rename |
| Checkpoint format changes between versions | Medium | Version field + migration functions |
| Large image data in checkpoints | Medium | Externalize >1MB images to separate files |
| Task spec changes between runs invalidate checkpoint | High | Detect spec changes via hash; warn and refuse to resume |

## Tests

**File:** `tests/test_checkpoint.py`

- [ ] Checkpoint saves all task states correctly
- [ ] Checkpoint restores and skips completed tasks
- [ ] Resume continues from correct wave index
- [ ] Filesystem store creates valid JSON files
- [ ] SQLite store persists and retrieves checkpoints
- [ ] Checkpoint pruning keeps only last N
- [ ] `resume_from="latest"` finds most recent checkpoint
- [ ] Modified task specs prevent resume (hash mismatch)
- [ ] Checkpoint includes all agent results
- [ ] Token usage is cumulative across checkpoint + resumed execution
- [ ] Empty workflow produces no checkpoint
- [ ] Concurrent writes don't corrupt checkpoint
- [ ] Round-trip serialization preserves all LLMMessage content types

## Estimated Effort

High — 5-7 working sessions
