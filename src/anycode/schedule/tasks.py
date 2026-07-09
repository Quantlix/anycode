"""Scheduled task modes: token spend proportional to reasoning, not total work.

Four execution modes for recurring work:

* `notification` — emit a message. Zero LLM calls.
* `script`       — run a deterministic command. Zero LLM calls.
* `agent`        — run a full agent session (caller supplies the coroutine).
* `hybrid`       — run the script first; invoke the agent only when the
                   script's output demands interpretation (nonzero exit, or a
                   caller-supplied trigger on the output).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict

TaskMode = str  # "notification" | "script" | "agent" | "hybrid"

AgentFn = Callable[[str], Awaitable[str]]
"""Runs an agent session over the given prompt/context; returns its output."""


class ScheduledTask(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    mode: TaskMode
    message: str = ""
    command: str = ""
    prompt: str = ""
    timeout_seconds: float = 300.0


class ScheduledTaskResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    mode: TaskMode
    output: str
    exit_code: int | None = None
    agent_invoked: bool = False


async def run_scheduled_task(
    task: ScheduledTask,
    *,
    agent: AgentFn | None = None,
    hybrid_trigger: Callable[[int, str], bool] | None = None,
) -> ScheduledTaskResult:
    if task.mode == "notification":
        return ScheduledTaskResult(name=task.name, mode=task.mode, output=task.message)

    if task.mode == "script":
        exit_code, output = await _run_command(task.command, task.timeout_seconds)
        return ScheduledTaskResult(name=task.name, mode=task.mode, output=output, exit_code=exit_code)

    if task.mode == "agent":
        if agent is None:
            raise ValueError(f"Scheduled task '{task.name}' has mode 'agent' but no agent callable was provided.")
        output = await agent(task.prompt)
        return ScheduledTaskResult(name=task.name, mode=task.mode, output=output, agent_invoked=True)

    if task.mode == "hybrid":
        exit_code, script_output = await _run_command(task.command, task.timeout_seconds)
        trigger = hybrid_trigger or (lambda code, _out: code != 0)
        if not trigger(exit_code, script_output):
            # The deterministic gather was conclusive: no reasoning needed.
            return ScheduledTaskResult(name=task.name, mode=task.mode, output=script_output, exit_code=exit_code)
        if agent is None:
            raise ValueError(f"Scheduled task '{task.name}' triggered agent interpretation but no agent was provided.")
        prompt = f"{task.prompt}\n\nScript output (exit {exit_code}):\n{script_output}"
        output = await agent(prompt)
        return ScheduledTaskResult(name=task.name, mode=task.mode, output=output, exit_code=exit_code, agent_invoked=True)

    raise ValueError(f"Unknown scheduled task mode: {task.mode!r}")


async def _run_command(command: str, timeout_seconds: float) -> tuple[int, str]:
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
    except TimeoutError:
        process.kill()
        await process.wait()
        return 124, f"Command timed out after {timeout_seconds}s"
    return process.returncode or 0, stdout.decode("utf-8", errors="replace")
