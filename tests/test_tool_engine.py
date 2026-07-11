"""Tool execution engine tests: async file IO, bash hardening, discovery, and parallel dispatch."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel

from anycode.tools.bash import BashInput, bash_tool
from anycode.tools.executor import ToolExecutor
from anycode.tools.file_edit import FileEditInput, file_edit_tool
from anycode.tools.file_read import FileReadInput, file_read_tool
from anycode.tools.file_write import FileWriteInput, file_write_tool
from anycode.tools.grep import GrepInput, grep_tool
from anycode.tools.idempotency import IdempotencyClaim, SQLiteToolIdempotencyStore
from anycode.tools.list_files import ListFilesInput, list_files_tool
from anycode.tools.registry import ToolRegistry, define_tool
from anycode.types import AgentInfo, ToolResult, ToolSecurityPolicy, ToolUseContext

CTX = ToolUseContext(agent=AgentInfo(name="t", role="assistant", model="m"))
PY = "python"


class _SideEffectInput(BaseModel):
    value: str
    idempotency_key: str | None = None


async def test_executor_enforces_security_policy_for_custom_tools() -> None:
    registry = ToolRegistry()

    class _Input(BaseModel):
        pass

    async def _execute(_input: object, _context: ToolUseContext) -> ToolResult:
        return ToolResult(data="should not run")

    registry.register(define_tool(name="custom", description="custom", input_model=_Input, execute=_execute))
    executor = ToolExecutor(registry)
    context = CTX.model_copy(update={"security_policy": ToolSecurityPolicy(allowed_tools=("other",))})

    result = await executor.execute("custom", {}, context)

    assert result.is_error is True
    assert "allowlist" in result.data


async def test_executor_redacts_secrets_from_tool_exceptions() -> None:
    registry = ToolRegistry()

    class _Input(BaseModel):
        pass

    async def _execute(_input: object, _context: ToolUseContext) -> ToolResult:
        raise RuntimeError("request failed with Bearer abcdefghijklmnop")

    registry.register(define_tool(name="failing", description="failing", input_model=_Input, execute=_execute))

    result = await ToolExecutor(registry).execute("failing", {}, CTX)

    assert result.is_error is True
    assert "Bearer" not in result.data
    assert "<redacted-secret>" in result.data


async def test_side_effecting_tool_requires_idempotency_key() -> None:
    calls = 0

    async def _execute(_input: _SideEffectInput, _context: ToolUseContext) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult(data="executed")

    registry = ToolRegistry()
    registry.register(
        define_tool(
            name="charge",
            description="charge",
            input_model=_SideEffectInput,
            execute=_execute,
            side_effecting=True,
        )
    )

    result = await ToolExecutor(registry).execute("charge", {"value": "10"}, CTX)

    assert result.is_error is True
    assert "requires a non-empty idempotency key" in result.data
    assert calls == 0


async def test_completed_side_effect_replays_and_conflicting_input_fails() -> None:
    calls = 0
    seen_keys: list[str | None] = []

    async def _execute(tool_input: _SideEffectInput, context: ToolUseContext) -> ToolResult:
        nonlocal calls
        calls += 1
        seen_keys.append(context.idempotency_key)
        return ToolResult(data=f"charged {tool_input.value}")

    registry = ToolRegistry()
    registry.register(
        define_tool(
            name="charge",
            description="charge",
            input_model=_SideEffectInput,
            execute=_execute,
            side_effecting=True,
        )
    )
    executor = ToolExecutor(registry)

    first = await executor.execute("charge", {"value": "10", "idempotency_key": "order-1"}, CTX)
    replay = await executor.execute("charge", {"value": "10", "idempotency_key": "order-1"}, CTX)
    conflict = await executor.execute("charge", {"value": "20", "idempotency_key": "order-1"}, CTX)

    assert first == replay
    assert conflict.is_error is True
    assert conflict.retry_safe is False
    assert "different input" in conflict.data
    assert calls == 1
    assert seen_keys == ["order-1"]


async def test_concurrent_side_effect_claim_executes_once() -> None:
    calls = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _execute(_input: _SideEffectInput, _context: ToolUseContext) -> ToolResult:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return ToolResult(data="done")

    registry = ToolRegistry()
    registry.register(
        define_tool(
            name="publish",
            description="publish",
            input_model=_SideEffectInput,
            execute=_execute,
            side_effecting=True,
        )
    )
    executor = ToolExecutor(registry)
    first_task = asyncio.create_task(executor.execute("publish", {"value": "post"}, CTX, idempotency_key="publish-1"))
    await entered.wait()
    duplicate = await executor.execute("publish", {"value": "post"}, CTX, idempotency_key="publish-1")
    release.set()
    first = await first_task

    assert first.is_error is not True
    assert duplicate.is_error is True
    assert duplicate.retry_safe is False
    assert "already in progress" in duplicate.data
    assert calls == 1


async def test_sqlite_idempotency_replays_across_executor_restart(tmp_path: Path) -> None:
    calls = 0

    async def _execute(tool_input: _SideEffectInput, _context: ToolUseContext) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult(data=f"sent {tool_input.value}")

    registry = ToolRegistry()
    registry.register(
        define_tool(
            name="send",
            description="send",
            input_model=_SideEffectInput,
            execute=_execute,
            side_effecting=True,
        )
    )
    path = tmp_path / "idempotency.db"
    first_store = SQLiteToolIdempotencyStore(path)
    first = await ToolExecutor(registry, idempotency_store=first_store).execute("send", {"value": "notice"}, CTX, idempotency_key="message-1")
    await first_store.teardown()

    reopened_store = SQLiteToolIdempotencyStore(path)
    replay = await ToolExecutor(registry, idempotency_store=reopened_store).execute("send", {"value": "notice"}, CTX, idempotency_key="message-1")
    pruned = await reopened_store.prune_completed(datetime.now(UTC) + timedelta(seconds=1))
    await reopened_store.teardown()

    assert first == replay
    assert calls == 1
    assert pruned == 1


async def test_sqlite_claim_is_atomic_across_store_connections(tmp_path: Path) -> None:
    path = tmp_path / "shared-idempotency.db"
    first_store = SQLiteToolIdempotencyStore(path)
    second_store = SQLiteToolIdempotencyStore(path)

    first, second = await asyncio.gather(
        first_store.claim("publish", "post-1", "same-input"),
        second_store.claim("publish", "post-1", "same-input"),
    )

    await first_store.teardown()
    await second_store.teardown()
    assert {first.outcome, second.outcome} == {"execute", "in_progress"}


async def test_cancelled_side_effect_remains_indeterminate() -> None:
    entered = asyncio.Event()

    async def _execute(_input: _SideEffectInput, _context: ToolUseContext) -> ToolResult:
        entered.set()
        await asyncio.Future()
        return ToolResult(data="unreachable")

    registry = ToolRegistry()
    registry.register(
        define_tool(
            name="publish",
            description="publish",
            input_model=_SideEffectInput,
            execute=_execute,
            side_effecting=True,
        )
    )
    executor = ToolExecutor(registry)
    task = asyncio.create_task(executor.execute("publish", {"value": "post"}, CTX, idempotency_key="post-1"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    retry = await executor.execute("publish", {"value": "post"}, CTX, idempotency_key="post-1")

    assert retry.is_error is True
    assert retry.retry_safe is False
    assert "already in progress" in retry.data


async def test_side_effect_exception_is_recorded_as_not_retry_safe() -> None:
    calls = 0

    async def _execute(_input: _SideEffectInput, _context: ToolUseContext) -> ToolResult:
        nonlocal calls
        calls += 1
        raise RuntimeError("remote outcome unavailable")

    registry = ToolRegistry()
    registry.register(
        define_tool(
            name="publish",
            description="publish",
            input_model=_SideEffectInput,
            execute=_execute,
            side_effecting=True,
        )
    )
    executor = ToolExecutor(registry)

    first = await executor.execute("publish", {"value": "post"}, CTX, idempotency_key="post-1")
    replay = await executor.execute("publish", {"value": "post"}, CTX, idempotency_key="post-1")

    assert first == replay
    assert first.is_error is True
    assert first.retry_safe is False
    assert calls == 1


async def test_side_effect_can_mark_pre_execution_error_retry_safe() -> None:
    async def _execute(_input: _SideEffectInput, _context: ToolUseContext) -> ToolResult:
        return ToolResult(data="validation failed before request", is_error=True, retry_safe=True)

    registry = ToolRegistry()
    registry.register(
        define_tool(
            name="publish",
            description="publish",
            input_model=_SideEffectInput,
            execute=_execute,
            side_effecting=True,
        )
    )

    result = await ToolExecutor(registry).execute("publish", {"value": "post"}, CTX, idempotency_key="post-1")

    assert result.is_error is True
    assert result.retry_safe is True


async def test_sqlite_pruning_retains_unknown_outcome(tmp_path: Path) -> None:
    store = SQLiteToolIdempotencyStore(tmp_path / "unknown.db")
    claim = await store.claim("publish", "post-1", "same-input")
    await store.complete(
        "publish",
        "post-1",
        ToolResult(data="remote outcome unavailable", is_error=True, retry_safe=False),
    )

    pruned = await store.prune_completed(datetime.now(UTC) + timedelta(seconds=1))
    replay = await store.claim("publish", "post-1", "same-input")
    await store.teardown()

    assert claim.outcome == "execute"
    assert pruned == 0
    assert replay.outcome == "replay"
    assert replay.result is not None and replay.result.retry_safe is False


async def test_sqlite_hashes_keys_and_redacts_persisted_results(tmp_path: Path) -> None:
    path = tmp_path / "protected-claims.db"
    store = SQLiteToolIdempotencyStore(path)
    await store.claim("send", "customer-order-123", "same-input")
    await store.complete(
        "send",
        "customer-order-123",
        ToolResult(data="Bearer abcdefghijklmnop", retry_safe=True),
    )
    replay = await store.claim("send", "customer-order-123", "same-input")
    await store.teardown()

    persisted = path.read_bytes()
    assert b"customer-order-123" not in persisted
    assert b"abcdefghijklmnop" not in persisted
    assert replay.result is not None
    assert "<redacted-secret>" in replay.result.data


async def test_idempotency_store_failure_prevents_execution() -> None:
    calls = 0

    class _UnavailableStore:
        async def claim(self, tool_name: str, key: str, input_fingerprint: str) -> IdempotencyClaim:
            del tool_name, key, input_fingerprint
            raise RuntimeError("database offline")

        async def complete(self, tool_name: str, key: str, result: ToolResult) -> None:
            del tool_name, key, result
            raise AssertionError("complete must not be called")

        async def delete(self, tool_name: str, key: str) -> None:
            del tool_name, key

        async def prune_completed(self, before: datetime) -> int:
            del before
            return 0

    async def _execute(_input: _SideEffectInput, _context: ToolUseContext) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult(data="unsafe")

    registry = ToolRegistry()
    registry.register(
        define_tool(
            name="transfer",
            description="transfer",
            input_model=_SideEffectInput,
            execute=_execute,
            side_effecting=True,
        )
    )

    result = await ToolExecutor(registry, idempotency_store=_UnavailableStore()).execute(
        "transfer", {"value": "10"}, CTX, idempotency_key="transfer-1"
    )

    assert result.is_error is True
    assert "tool was not executed" in result.data
    assert result.retry_safe is True
    assert calls == 0


async def test_idempotency_completion_failure_is_not_retry_safe() -> None:
    class _CompletionFailureStore:
        async def claim(self, tool_name: str, key: str, input_fingerprint: str) -> IdempotencyClaim:
            del tool_name, key, input_fingerprint
            return IdempotencyClaim(outcome="execute")

        async def complete(self, tool_name: str, key: str, result: ToolResult) -> None:
            del tool_name, key, result
            raise RuntimeError("disk full")

        async def delete(self, tool_name: str, key: str) -> None:
            del tool_name, key

        async def prune_completed(self, before: datetime) -> int:
            del before
            return 0

    async def _execute(_input: _SideEffectInput, _context: ToolUseContext) -> ToolResult:
        return ToolResult(data="effect committed")

    registry = ToolRegistry()
    registry.register(
        define_tool(
            name="transfer",
            description="transfer",
            input_model=_SideEffectInput,
            execute=_execute,
            side_effecting=True,
        )
    )

    result = await ToolExecutor(registry, idempotency_store=_CompletionFailureStore()).execute(
        "transfer", {"value": "10"}, CTX, idempotency_key="transfer-1"
    )

    assert result.is_error is True
    assert result.retry_safe is False
    assert "outcome is unknown" in result.data


# -- file tools -------------------------------------------------------------


async def test_file_write_preserves_exact_bytes(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "f.txt"
    content = "line1\nline2\nno-trailing-newline"
    result = await file_write_tool.execute(FileWriteInput(path=str(target), content=content), CTX)
    assert result.is_error is False
    # Byte-exact: LF is preserved, not translated to the host line separator.
    assert target.read_bytes() == content.encode("utf-8")


async def test_file_write_is_atomic_no_temp_leftovers(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    await file_write_tool.execute(FileWriteInput(path=str(target), content="hi"), CTX)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "f.txt"]
    assert leftovers == []


async def test_file_edit_replaces_and_reads_back(tmp_path: Path) -> None:
    target = tmp_path / "f.py"
    target.write_text("a = 1\nb = 2\n", encoding="utf-8")
    result = await file_edit_tool.execute(FileEditInput(path=str(target), old_string="a = 1", new_string="a = 42"), CTX)
    assert result.is_error is False
    read = await file_read_tool.execute(FileReadInput(path=str(target)), CTX)
    assert "a = 42" in read.data


async def test_file_tools_reject_paths_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    context = CTX.model_copy(update={"security_policy": ToolSecurityPolicy(workspace_root=str(workspace))})

    read = await file_read_tool.execute(FileReadInput(path=str(outside)), context)
    write = await file_write_tool.execute(FileWriteInput(path="../escape.txt", content="no"), context)

    assert read.is_error is True and "outside" in read.data
    assert write.is_error is True and "outside" in write.data
    assert not (tmp_path / "escape.txt").exists()


# -- bash --------------------------------------------------------------------


async def test_bash_basic_output() -> None:
    result = await bash_tool.execute(BashInput(command=f"{PY} -c \"print('hi there')\""), CTX)
    assert result.is_error is False
    assert "hi there" in result.data


async def test_bash_output_cap_truncates_with_metadata() -> None:
    result = await bash_tool.execute(
        BashInput(command=f"{PY} -c \"print('A'*5000)\"", max_output_bytes=100),
        CTX,
    )
    assert "[output truncated" in result.data
    assert "showing first 100" in result.data
    assert "dropped]" in result.data


async def test_bash_timeout_reports_and_cleans_up() -> None:
    result = await bash_tool.execute(
        BashInput(command=f'{PY} -c "import time; time.sleep(5)"', timeout=0.5),
        CTX,
    )
    assert result.is_error is True
    assert "timed out" in result.data.lower()


async def test_bash_cancellation_terminates_process_tree(tmp_path: Path) -> None:
    started = tmp_path / "started.txt"
    finished = tmp_path / "finished.txt"
    script = (
        "import pathlib,time; "
        f"pathlib.Path({str(started)!r}).write_text('started'); "
        "time.sleep(1); "
        f"pathlib.Path({str(finished)!r}).write_text('finished')"
    )
    command = subprocess.list2cmdline([sys.executable, "-c", script])
    task = asyncio.ensure_future(bash_tool.execute(BashInput(command=command, timeout=5), CTX))
    for _ in range(100):
        if started.exists():
            break
        await asyncio.sleep(0.01)
    assert started.exists()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(1.1)

    assert not finished.exists()


async def test_bash_can_be_disabled_by_security_policy(tmp_path: Path) -> None:
    context = CTX.model_copy(update={"security_policy": ToolSecurityPolicy(workspace_root=str(tmp_path), allow_shell=False)})
    result = await bash_tool.execute(BashInput(command=f"{PY} -c \"print('no')\""), context)
    assert result.is_error is True
    assert "disabled" in result.data


async def test_bash_filters_parent_environment(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ANYCODE_TEST_SECRET", "must-not-leak")
    context = CTX.model_copy(
        update={
            "security_policy": ToolSecurityPolicy(
                workspace_root=str(tmp_path),
                inherit_environment=False,
            )
        }
    )
    command = f"\"{sys.executable}\" -c \"import os; print(os.getenv('ANYCODE_TEST_SECRET', 'absent'))\""
    result = await bash_tool.execute(BashInput(command=command), context)
    assert result.is_error is False
    assert "absent" in result.data
    assert "must-not-leak" not in result.data


async def test_bash_allowlist_rejects_shell_chaining(tmp_path: Path) -> None:
    context = CTX.model_copy(
        update={
            "security_policy": ToolSecurityPolicy(
                workspace_root=str(tmp_path),
                allowed_shell_commands=("python",),
            )
        }
    )
    result = await bash_tool.execute(BashInput(command="python --version && echo bypass"), context)
    assert result.is_error is True
    assert "control operators" in result.data


# -- grep --------------------------------------------------------------------


async def test_grep_finds_match(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello world\nfoo bar\n", encoding="utf-8")
    result = await grep_tool.execute(GrepInput(pattern="foo", path=str(tmp_path)), CTX)
    assert "foo" in result.data


# -- list_files --------------------------------------------------------------


async def test_list_files_discovers_and_excludes_ignored(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "keep.md").write_text("# doc\n", encoding="utf-8")
    junk = tmp_path / "__pycache__"
    junk.mkdir()
    (junk / "trash.pyc").write_text("noise", encoding="utf-8")

    result = await list_files_tool.execute(ListFilesInput(path=str(tmp_path)), CTX)
    assert "keep.py" in result.data
    assert "keep.md" in result.data
    assert "trash.pyc" not in result.data
    assert "backend:" in result.data


async def test_list_files_glob_filter(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("1", encoding="utf-8")
    (tmp_path / "b.txt").write_text("2", encoding="utf-8")
    result = await list_files_tool.execute(ListFilesInput(path=str(tmp_path), glob="*.py"), CTX)
    assert "a.py" in result.data
    assert "b.txt" not in result.data


# -- parallel tool execution -------------------------------------------------


async def test_runner_executes_independent_tools_concurrently() -> None:
    """Two slow tools in one turn should overlap, not run back to back."""
    from anycode.core.runner import AgentRunner
    from anycode.providers.fake import FakeAdapter, FakeResponse
    from anycode.types import LLMMessage, RunnerOptions, TextBlock

    spans: list[tuple[float, float]] = []

    async def _slow(_input: object, _ctx: ToolUseContext) -> ToolResult:
        start = time.monotonic()
        await asyncio.sleep(0.2)
        spans.append((start, time.monotonic()))
        return ToolResult(data="done")

    class _Empty(BaseModel):
        pass

    slow_tool = define_tool(name="slow", description="sleeps", input_model=_Empty, execute=_slow)
    registry = ToolRegistry()
    registry.register(slow_tool)

    adapter = FakeAdapter(
        responses=[
            FakeResponse(tool_calls=(("slow", {}), ("slow", {}))),
            FakeResponse(text="all done"),
        ]
    )
    runner = AgentRunner(adapter, registry, ToolExecutor(registry), RunnerOptions(model="fake", agent_name="t", max_turns=3))

    began = time.monotonic()
    result = await runner.run([LLMMessage(role="user", content=[TextBlock(text="go")])])
    elapsed = time.monotonic() - began

    assert len(spans) == 2
    # Concurrent execution: total wall time well under the 0.4s serial sum.
    assert elapsed < 0.35
    assert len([c for c in result.tool_calls if c.tool_name == "slow"]) == 2


async def test_runner_rejects_provider_call_for_unallowed_tool() -> None:
    from anycode.core.runner import AgentRunner
    from anycode.providers.fake import FakeAdapter, FakeResponse
    from anycode.types import LLMMessage, RunnerOptions, TextBlock

    executed = False

    async def _restricted(_input: object, _ctx: ToolUseContext) -> ToolResult:
        nonlocal executed
        executed = True
        return ToolResult(data="should not run")

    class _Empty(BaseModel):
        pass

    registry = ToolRegistry()
    registry.register(define_tool(name="restricted", description="restricted", input_model=_Empty, execute=_restricted))
    adapter = FakeAdapter(
        responses=[
            FakeResponse(tool_calls=(("restricted", {}),)),
            FakeResponse(text="done"),
        ]
    )
    runner = AgentRunner(
        adapter,
        registry,
        ToolExecutor(registry),
        RunnerOptions(model="fake", agent_name="t", max_turns=2, allowed_tools=[]),
    )

    result = await runner.run([LLMMessage(role="user", content=[TextBlock(text="go")])])

    assert executed is False
    assert result.tool_calls[0].tool_name == "restricted"
    assert "not allowed" in result.tool_calls[0].output


async def test_runner_cancellation_drains_parallel_tools() -> None:
    from anycode.core.runner import AgentRunner
    from anycode.providers.fake import FakeAdapter, FakeResponse
    from anycode.types import LLMMessage, RunnerOptions, TextBlock

    class _SlowInput(BaseModel):
        name: str

    started: set[str] = set()
    cancelled: set[str] = set()
    all_started = asyncio.Event()

    async def _slow(tool_input: _SlowInput, _ctx: ToolUseContext) -> ToolResult:
        started.add(tool_input.name)
        if len(started) == 2:
            all_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.add(tool_input.name)
            raise
        raise AssertionError("Blocking tool resumed without cancellation.")

    registry = ToolRegistry()
    registry.register(define_tool(name="slow", description="blocks", input_model=_SlowInput, execute=_slow))
    adapter = FakeAdapter(
        responses=[
            FakeResponse(tool_calls=(("slow", {"name": "first"}), ("slow", {"name": "second"}))),
        ]
    )
    runner = AgentRunner(adapter, registry, ToolExecutor(registry), RunnerOptions(model="fake", agent_name="t", max_turns=2))
    task = asyncio.create_task(runner.run([LLMMessage(role="user", content=[TextBlock(text="go")])]))
    await asyncio.wait_for(all_started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled == {"first", "second"}
