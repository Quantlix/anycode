# Demo 07 — Workflow Checkpointing & Resume
# Execute: uv run python examples/07_checkpointing.py
#
# Demonstrates:
#   1. Filesystem checkpoint store — save, load, list, prune
#   2. SQLite checkpoint store — full lifecycle
#   3. Checkpoint serialization — round-trip with all content block types
#   4. CheckpointManager — auto-save, load latest, spec-change detection
#   5. Simulated crash + resume workflow
#
# No external services required (uses local filesystem and in-memory SQLite).

import asyncio
import json
import sys
import tempfile
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from anycode.checkpoint.manager import CheckpointManager
from anycode.checkpoint.serializer import deserialize_checkpoint, serialize_checkpoint
from anycode.checkpoint.store import FilesystemCheckpointStore, SQLiteCheckpointStore
from anycode.tasks.task import create_task
from anycode.types import (
    AgentRunResult,
    CheckpointConfig,
    CheckpointData,
    ImageBlock,
    ImageSource,
    LLMMessage,
    TextBlock,
    TokenUsage,
    ToolCallRecord,
    ToolResultBlock,
    ToolUseBlock,
)

SEPARATOR = "=" * 60
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def _make_checkpoint(
    checkpoint_id: str = "cp-001",
    workflow_id: str = "wf-demo",
    wave_index: int = 0,
    task_title: str = "Demo task",
    task_status: str = "completed",
) -> CheckpointData:
    task = create_task(title=task_title, description=f"Task for wave {wave_index}")
    task = task.model_copy(update={"status": task_status, "result": f"output-{wave_index}"})
    messages = [
        LLMMessage(role="user", content=[TextBlock(text="Process this task")]),
        LLMMessage(
            role="assistant",
            content=[
                TextBlock(text="I'll use a tool"),
                ToolUseBlock(id="tu-1", name="file_read", input={"path": "/src/main.py"}),
            ],
        ),
        LLMMessage(role="user", content=[ToolResultBlock(tool_use_id="tu-1", content="def main(): pass")]),
        LLMMessage(role="assistant", content=[TextBlock(text="Task completed successfully")]),
    ]
    return CheckpointData(
        id=checkpoint_id,
        workflow_id=workflow_id,
        version=1,
        tasks=[task],
        agent_results={
            "worker-1": AgentRunResult(
                success=True,
                output=f"Wave {wave_index} done",
                messages=messages,
                token_usage=TokenUsage(input_tokens=200 * (wave_index + 1), output_tokens=100 * (wave_index + 1)),
                tool_calls=[ToolCallRecord(tool_name="file_read", input={"path": "/src/main.py"}, output="def main(): pass", duration=0.3)],
            )
        },
        wave_index=wave_index,
        total_token_usage=TokenUsage(input_tokens=200 * (wave_index + 1), output_tokens=100 * (wave_index + 1)),
        created_at=datetime.now(UTC),
        metadata={"wave": wave_index, "demo": True},
    )


# --- Section 1: Filesystem Checkpoint Store ---


async def demo_filesystem_store() -> None:
    print(f"\n{SEPARATOR}")
    print("  1. Filesystem Checkpoint Store")
    print(SEPARATOR)

    with tempfile.TemporaryDirectory() as tmp:
        store = FilesystemCheckpointStore(tmp)

        # Save multiple checkpoints
        print("\n--- Save Checkpoints ---")
        for i in range(5):
            cp = _make_checkpoint(f"cp-{i:03d}", wave_index=i)
            await store.save(cp)
            print(f"Saved checkpoint {cp.id} (wave={cp.wave_index})")

        # List
        print("\n--- List Checkpoints ---")
        ids = await store.list_checkpoints("wf-demo")
        print(f"Checkpoints: {ids}")

        # Load latest
        print("\n--- Load Latest ---")
        latest = await store.latest("wf-demo")
        assert latest is not None
        print(f"Latest: id={latest.id}, wave={latest.wave_index}, tokens={latest.total_token_usage}")

        # Load specific
        print("\n--- Load Specific ---")
        cp2 = await store.load("cp-002")
        assert cp2 is not None
        print(f"Loaded cp-002: wave={cp2.wave_index}, tasks={len(cp2.tasks)}, results={list(cp2.agent_results.keys())}")

        # Prune
        print("\n--- Prune (keep_last=2) ---")
        await store.prune("wf-demo", keep_last=2)
        remaining = await store.list_checkpoints("wf-demo")
        print(f"After prune: {remaining}")

        # Delete
        print("\n--- Delete ---")
        await store.delete(remaining[0])
        final = await store.list_checkpoints("wf-demo")
        print(f"After delete: {final}")

        # Non-existent
        missing = await store.load("non-existent")
        print(f"\nLoad non-existent: {missing}  ✓")

        # Verify JSON files are human-readable
        print("\n--- JSON File Inspection ---")
        wf_dir = Path(tmp) / "wf-demo"
        if wf_dir.exists():
            for f in sorted(wf_dir.glob("*.json")):
                data = json.loads(f.read_text())
                print(f"  {f.name}: id={data['id']}, wave={data['wave_index']}, keys={list(data.keys())}")


# --- Section 2: SQLite Checkpoint Store ---


async def demo_sqlite_store() -> None:
    print(f"\n{SEPARATOR}")
    print("  2. SQLite Checkpoint Store")
    print(SEPARATOR)

    store = SQLiteCheckpointStore(":memory:")
    await store.setup()

    try:
        # Save
        print("\n--- Save Checkpoints ---")
        for i in range(4):
            cp = _make_checkpoint(f"sql-cp-{i:03d}", workflow_id="wf-sql", wave_index=i)
            await store.save(cp)
            print(f"Saved {cp.id} (wave={cp.wave_index})")

        # List
        ids = await store.list_checkpoints("wf-sql")
        print(f"\nList: {ids}")

        # Latest
        latest = await store.latest("wf-sql")
        assert latest is not None
        print(f"Latest: {latest.id}, wave={latest.wave_index}")

        # Load
        loaded = await store.load("sql-cp-001")
        assert loaded is not None
        print(f"Loaded sql-cp-001: wave={loaded.wave_index}")

        # Prune
        await store.prune("wf-sql", keep_last=2)
        after_prune = await store.list_checkpoints("wf-sql")
        print(f"After prune(keep_last=2): {after_prune}")

        # Delete
        await store.delete(after_prune[0])
        final = await store.list_checkpoints("wf-sql")
        print(f"After delete: {final}")

    finally:
        await store.teardown()


# --- Section 3: Serialization Round-Trip ---


async def demo_serialization() -> None:
    print(f"\n{SEPARATOR}")
    print("  3. Checkpoint Serialization Round-Trip")
    print(SEPARATOR)

    # TextBlock + ToolUseBlock + ToolResultBlock
    print("\n--- Text + Tool Blocks ---")
    cp = _make_checkpoint()
    raw = serialize_checkpoint(cp)
    restored = deserialize_checkpoint(raw)
    print(f"Original id={cp.id}, Restored id={restored.id}, Match={cp.id == restored.id}")
    print(f"Tasks preserved: {len(restored.tasks)} (title={restored.tasks[0].title!r})")
    print(f"Messages preserved: {len(restored.agent_results['worker-1'].messages)}")

    msg2 = restored.agent_results["worker-1"].messages[1]
    print(f"Message[1] blocks: {[type(b).__name__ for b in msg2.content]}")
    assert isinstance(msg2.content[0], TextBlock)
    assert isinstance(msg2.content[1], ToolUseBlock)
    print(f"ToolUseBlock.name={msg2.content[1].name}, input={msg2.content[1].input}")

    msg3 = restored.agent_results["worker-1"].messages[2]
    assert isinstance(msg3.content[0], ToolResultBlock)
    print(f"ToolResultBlock.content={msg3.content[0].content!r}")

    # ImageBlock
    print("\n--- ImageBlock Round-Trip ---")
    img_messages = [LLMMessage(role="user", content=[ImageBlock(source=ImageSource(media_type="image/png", data="aGVsbG8="))])]
    img_cp = CheckpointData(
        id="cp-img",
        workflow_id="wf-img",
        version=1,
        tasks=[],
        agent_results={"agent-1": AgentRunResult(success=True, output="", messages=img_messages, token_usage=TokenUsage(), tool_calls=[])},
        wave_index=0,
        total_token_usage=TokenUsage(),
        created_at=datetime.now(UTC),
    )
    img_raw = serialize_checkpoint(img_cp)
    img_restored = deserialize_checkpoint(img_raw)
    img_block = img_restored.agent_results["agent-1"].messages[0].content[0]
    assert isinstance(img_block, ImageBlock)
    print(f"ImageBlock preserved: media_type={img_block.source.media_type}, data={img_block.source.data!r}")

    # Token usage
    print("\n--- Token Usage Round-Trip ---")
    print(f"Original: input={cp.total_token_usage.input_tokens}, output={cp.total_token_usage.output_tokens}")
    print(f"Restored: input={restored.total_token_usage.input_tokens}, output={restored.total_token_usage.output_tokens}")
    assert cp.total_token_usage.input_tokens == restored.total_token_usage.input_tokens

    # Tool call records
    tc = restored.agent_results["worker-1"].tool_calls
    print(f"\nTool calls: {len(tc)} (name={tc[0].tool_name}, duration={tc[0].duration}s)")

    # JSON structure inspection
    print("\n--- JSON Structure ---")
    parsed = json.loads(raw)
    print(f"Top-level keys: {list(parsed.keys())}")
    print(f"Version: {parsed['version']}")
    print(f"JSON size: {len(raw)} bytes")


# --- Section 4: CheckpointManager ---


async def demo_checkpoint_manager() -> None:
    print(f"\n{SEPARATOR}")
    print("  4. CheckpointManager")
    print(SEPARATOR)

    with tempfile.TemporaryDirectory() as tmp:
        config = CheckpointConfig(enabled=True, path=tmp, keep_last=3)
        mgr = CheckpointManager(config)

        # Auto-save
        print("\n--- Auto-Save ---")
        task1 = create_task(title="Analyze code", description="Static analysis")
        task2 = create_task(title="Generate tests", description="Unit test generation")

        cp1 = await mgr.auto_save(
            "wf-mgr",
            [task1, task2],
            {"worker": AgentRunResult(success=True, output="analysis done", messages=[], token_usage=TokenUsage(input_tokens=500, output_tokens=200), tool_calls=[])},
            wave_index=0,
            total_usage=TokenUsage(input_tokens=500, output_tokens=200),
        )
        print(f"Wave 0 saved: {cp1.id}")

        cp2 = await mgr.auto_save(
            "wf-mgr",
            [task1.model_copy(update={"status": "completed"}), task2],
            {
                "worker": AgentRunResult(success=True, output="analysis done", messages=[], token_usage=TokenUsage(input_tokens=500, output_tokens=200), tool_calls=[]),
                "tester": AgentRunResult(success=True, output="tests generated", messages=[], token_usage=TokenUsage(input_tokens=800, output_tokens=400), tool_calls=[]),
            },
            wave_index=1,
            total_usage=TokenUsage(input_tokens=1300, output_tokens=600),
        )
        print(f"Wave 1 saved: {cp2.id}")

        # Load latest
        print("\n--- Load Latest ---")
        latest = await mgr.load_latest("wf-mgr")
        assert latest is not None
        print(f"Latest: {latest.id}, wave={latest.wave_index}, agents={list(latest.agent_results.keys())}")
        print(f"Total tokens: input={latest.total_token_usage.input_tokens}, output={latest.total_token_usage.output_tokens}")

        # Spec-change detection
        print("\n--- Spec-Change Detection ---")
        same_tasks = [create_task(title="Analyze code", description="Static analysis"), create_task(title="Generate tests", description="Unit test generation")]
        changed_tasks = [create_task(title="Analyze code", description="Static analysis"), create_task(title="Deploy app", description="Production deployment")]

        changed_same = mgr.detect_spec_change(same_tasks, latest)
        changed_diff = mgr.detect_spec_change(changed_tasks, latest)
        print(f"Same tasks → spec changed: {changed_same}")
        print(f"Different tasks → spec changed: {changed_diff}")

        # Hash determinism
        print("\n--- Hash Determinism ---")
        hash1 = CheckpointManager.compute_spec_hash(same_tasks)
        hash2 = CheckpointManager.compute_spec_hash(same_tasks)
        print(f"Hash 1: {hash1[:16]}...")
        print(f"Hash 2: {hash2[:16]}...")
        print(f"Deterministic: {hash1 == hash2}")


# --- Section 5: Simulated Crash & Resume ---


async def demo_crash_resume() -> None:
    print(f"\n{SEPARATOR}")
    print("  5. Simulated Crash & Resume Workflow")
    print(SEPARATOR)

    with tempfile.TemporaryDirectory() as tmp:
        config = CheckpointConfig(enabled=True, path=tmp, keep_last=5)

        # --- First run: process waves 0-2, "crash" at wave 3 ---
        print("\n--- First Run (waves 0-2, crash at wave 3) ---")
        mgr = CheckpointManager(config)
        workflow_id = "wf-pipeline"
        tasks = [create_task(title=f"Task-{i}", description=f"Process step {i}") for i in range(5)]

        completed_results: dict[str, AgentRunResult] = {}
        total_in, total_out = 0, 0

        for wave in range(3):  # Process waves 0, 1, 2
            task_name = f"Task-{wave}"
            result = AgentRunResult(
                success=True,
                output=f"Wave {wave} completed",
                messages=[LLMMessage(role="assistant", content=[TextBlock(text=f"Done with wave {wave}")])],
                token_usage=TokenUsage(input_tokens=100, output_tokens=50),
                tool_calls=[],
            )
            completed_results[f"agent-{wave}"] = result
            total_in += 100
            total_out += 50

            updated_tasks = []
            for t in tasks:
                if t.title == task_name:
                    updated_tasks.append(t.model_copy(update={"status": "completed", "result": f"wave-{wave}"}))
                else:
                    updated_tasks.append(t)
            tasks = updated_tasks

            cp = await mgr.auto_save(
                workflow_id, tasks, dict(completed_results), wave_index=wave, total_usage=TokenUsage(input_tokens=total_in, output_tokens=total_out)
            )
            print(f"  Wave {wave} completed, checkpoint saved: {cp.id}")

        print("  💥 Crash at wave 3!")

        # --- Resume ---
        print("\n--- Resume From Latest Checkpoint ---")
        mgr2 = CheckpointManager(config)
        restored = await mgr2.load_latest(workflow_id)
        assert restored is not None
        print(f"Restored checkpoint: {restored.id}")
        print(f"Resume from wave: {restored.wave_index + 1}")
        print(f"Completed agents: {list(restored.agent_results.keys())}")
        print(f"Token usage so far: input={restored.total_token_usage.input_tokens}, output={restored.total_token_usage.output_tokens}")

        # Check task statuses
        print("\n--- Task Statuses at Resume ---")
        for t in restored.tasks:
            print(f"  {t.title}: status={t.status}, result={t.result}")

        # Continue from wave 3
        print("\n--- Continue Execution (waves 3-4) ---")
        tasks = list(restored.tasks)  # Restore tasks
        completed_results = dict(restored.agent_results)
        total_in = restored.total_token_usage.input_tokens
        total_out = restored.total_token_usage.output_tokens

        for wave in range(restored.wave_index + 1, 5):
            task_name = f"Task-{wave}"
            result = AgentRunResult(
                success=True,
                output=f"Wave {wave} completed (resumed)",
                messages=[LLMMessage(role="assistant", content=[TextBlock(text=f"Resumed wave {wave}")])],
                token_usage=TokenUsage(input_tokens=100, output_tokens=50),
                tool_calls=[],
            )
            completed_results[f"agent-{wave}"] = result
            total_in += 100
            total_out += 50

            updated_tasks = []
            for t in tasks:
                if t.title == task_name:
                    updated_tasks.append(t.model_copy(update={"status": "completed", "result": f"wave-{wave}-resumed"}))
                else:
                    updated_tasks.append(t)
            tasks = updated_tasks

            cp = await mgr2.auto_save(
                workflow_id, tasks, dict(completed_results), wave_index=wave, total_usage=TokenUsage(input_tokens=total_in, output_tokens=total_out)
            )
            print(f"  Wave {wave} completed (resumed), checkpoint: {cp.id}")

        # Final state
        print("\n--- Final State ---")
        final = await mgr2.load_latest(workflow_id)
        assert final is not None
        print(f"All waves complete. Total agents: {len(final.agent_results)}")
        print(f"Total tokens: input={final.total_token_usage.input_tokens}, output={final.total_token_usage.output_tokens}")
        for t in final.tasks:
            print(f"  {t.title}: status={t.status}")

        # Verify pruning
        all_ids = await mgr2.store.list_checkpoints(workflow_id)
        print(f"\nCheckpoints on disk: {len(all_ids)} (keep_last={config.keep_last})")


# --- Main ---


class _OutputCapture:
    def __init__(self) -> None:
        self._buffer = StringIO()
        self._stdout = sys.stdout

    def write(self, text: str) -> None:
        self._stdout.write(text)
        self._buffer.write(text)

    def flush(self) -> None:
        self._stdout.flush()

    def get_output(self) -> str:
        return self._buffer.getvalue()


async def main() -> None:
    capture = _OutputCapture()
    sys.stdout = capture  # type: ignore[assignment]

    print("AnyCode — Checkpointing & Resume Demo (Phase 2.2)")

    await demo_filesystem_store()
    await demo_sqlite_store()
    await demo_serialization()
    await demo_checkpoint_manager()
    await demo_crash_resume()

    print(f"\n{SEPARATOR}")
    print("  All checkpointing demos completed successfully!")
    print(SEPARATOR)

    sys.stdout = capture._stdout
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "07-checkpointing-output.txt"
    output_file.write_text(capture.get_output(), encoding="utf-8")
    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
