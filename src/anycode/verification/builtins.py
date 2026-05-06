"""Built-in verification sensors.

These sensors shell out to deterministic tooling (`ruff`, `pyright`, `pytest`)
or perform schema validation. They are deliberately conservative: a missing
binary returns a warning rather than crashing the pipeline.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from typing import Final

from pydantic import BaseModel, ValidationError

from anycode.types import VerificationResult, VerificationSensorConfig
from anycode.verification.sensor import Sensor, SensorContext

_DEFAULT_TIMEOUT_SECONDS: Final[int] = 120


async def _run_command(command: list[str], timeout: int = _DEFAULT_TIMEOUT_SECONDS) -> tuple[int, str, str]:
    if not shutil.which(command[0]):
        return -1, "", f"binary '{command[0]}' not found on PATH"
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return -2, "", f"command timed out after {timeout}s"
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


def _result_from_command(
    name: str,
    code: int,
    stdout: str,
    stderr: str,
    *,
    success_message: str,
    failure_severity: str = "error",
) -> VerificationResult:
    passed = code == 0
    if code == -1:
        return VerificationResult(
            sensor_name=name,
            kind="computational",
            passed=False,
            severity="warning",
            message=stderr,
        )
    combined = (stdout + stderr).strip()
    return VerificationResult(
        sensor_name=name,
        kind="computational",
        passed=passed,
        severity="info" if passed else failure_severity,  # type: ignore[arg-type]
        message=success_message if passed else combined[:600] or "command failed",
        evidence={"exit_code": code, "stdout_bytes": len(stdout), "stderr_bytes": len(stderr)},
        feedback_for_agent=None if passed else combined[:1200],
    )


def ruff_sensor(target: str = "src/", *, phases: tuple[str, ...] = ("after_task",)) -> Sensor:
    config = VerificationSensorConfig(name="ruff", kind="computational", phases=phases, options={"target": target})  # type: ignore[arg-type]

    async def _run(_ctx: SensorContext) -> VerificationResult:
        code, out, err = await _run_command(["ruff", "check", target])
        return _result_from_command("ruff", code, out, err, success_message=f"ruff clean for {target}")

    return Sensor(config=config, fn=_run)


def pyright_sensor(target: str = "src/", *, phases: tuple[str, ...] = ("after_task",)) -> Sensor:
    config = VerificationSensorConfig(name="pyright", kind="computational", phases=phases, options={"target": target})  # type: ignore[arg-type]

    async def _run(_ctx: SensorContext) -> VerificationResult:
        code, out, err = await _run_command(["pyright", target])
        return _result_from_command("pyright", code, out, err, success_message=f"pyright clean for {target}")

    return Sensor(config=config, fn=_run)


def pytest_sensor(target: str = "tests/", *, phases: tuple[str, ...] = ("after_task",)) -> Sensor:
    config = VerificationSensorConfig(name="pytest", kind="computational", phases=phases, options={"target": target})  # type: ignore[arg-type]

    async def _run(_ctx: SensorContext) -> VerificationResult:
        code, out, err = await _run_command(["pytest", target, "-q"])
        return _result_from_command(
            "pytest",
            code,
            out,
            err,
            success_message=f"pytest passed for {target}",
            failure_severity="critical",
        )

    return Sensor(config=config, fn=_run)


def schema_sensor(
    schema: type[BaseModel],
    *,
    name: str = "schema",
    phases: tuple[str, ...] = ("after_task",),
) -> Sensor:
    """Validate that the agent's final output is parseable as the given Pydantic model."""
    config = VerificationSensorConfig(name=name, kind="computational", phases=phases)  # type: ignore[arg-type]

    def _run(ctx: SensorContext) -> VerificationResult:
        text = (ctx.output or "").strip()
        if not text:
            return VerificationResult(
                sensor_name=name,
                kind="computational",
                passed=False,
                severity="error",
                message="output is empty",
                feedback_for_agent="Produce JSON conforming to the required schema.",
            )
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return VerificationResult(
                sensor_name=name,
                kind="computational",
                passed=False,
                severity="error",
                message=f"output is not valid JSON: {exc.msg}",
                feedback_for_agent="Re-emit the response as a single JSON object matching the schema.",
            )
        try:
            schema.model_validate(parsed)
        except ValidationError as exc:
            return VerificationResult(
                sensor_name=name,
                kind="computational",
                passed=False,
                severity="error",
                message="schema validation failed",
                evidence={"errors": exc.error_count()},
                feedback_for_agent=str(exc)[:1200],
            )
        return VerificationResult(
            sensor_name=name,
            kind="computational",
            passed=True,
            severity="info",
            message=f"output conforms to {schema.__name__}",
        )

    return Sensor(config=config, fn=_run)
