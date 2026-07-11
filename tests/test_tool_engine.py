"""Tool execution engine tests: async file IO, bash hardening, discovery, and parallel dispatch."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from anycode.tools.bash import BashInput, bash_tool
from anycode.tools.executor import ToolExecutor
from anycode.tools.file_edit import FileEditInput, file_edit_tool
from anycode.tools.file_read import FileReadInput, file_read_tool
from anycode.tools.file_write import FileWriteInput, file_write_tool
from anycode.tools.grep import GrepInput, grep_tool
from anycode.tools.list_files import ListFilesInput, list_files_tool
from anycode.tools.registry import ToolRegistry, define_tool
from anycode.types import AgentInfo, ToolResult, ToolSecurityPolicy, ToolUseContext

CTX = ToolUseContext(agent=AgentInfo(name="t", role="assistant", model="m"))
PY = "python"


async def test_executor_enforces_security_policy_for_custom_tools() -> None:
    registry = ToolRegistry()

    from pydantic import BaseModel

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

    from pydantic import BaseModel

    class _Input(BaseModel):
        pass

    async def _execute(_input: object, _context: ToolUseContext) -> ToolResult:
        raise RuntimeError("request failed with Bearer abcdefghijklmnop")

    registry.register(define_tool(name="failing", description="failing", input_model=_Input, execute=_execute))

    result = await ToolExecutor(registry).execute("failing", {}, CTX)

    assert result.is_error is True
    assert "Bearer" not in result.data
    assert "<redacted-secret>" in result.data


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

    from pydantic import BaseModel

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
