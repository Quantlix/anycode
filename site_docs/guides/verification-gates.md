---
title: "Gate AnyCode Output with Verification Sensors"
description: "Block bad AnyCode output with quality gates: run ruff, pyright, pytest, regex, and schema sensors at defined phases and decide pass, retry, or block by severity."
keywords: anycode verification, quality gate, QualityGate, ruff_sensor pytest_sensor, schema_sensor, VerificationSensorConfig, decide_gate, block on failure, sensor phases
---

# Verify Output with Quality Gates

A model saying "done" is not the same as the work being correct. Verification sensors run real, computational checks — lint, type-check, tests, schema validation, regex — at defined points in a run, and a gate decides whether to accept the output, make the agent retry, or block the run outright. This guide covers the built-in sensors, how they attach, and how the pass/fail decision is made.

## Sensors and phases

A **sensor** is a check that returns a pass/fail `VerificationResult` with a severity. A **gate** runs the sensors for the current phase and reduces their results to one decision.

| Sensor | Checks | Default failure severity |
| --- | --- | --- |
| `ruff_sensor(target="src/")` | Lint | error |
| `pyright_sensor(target="src/")` | Static types | error |
| `pytest_sensor(target="tests/")` | Tests | critical |
| `schema_sensor(Model)` | Output parses + validates against a Pydantic model | error |
| `regex` (registry) | Output matches (or lacks) a pattern | error (configurable) |

Sensors run at one of four **phases**:

| Phase | Runtime boundary |
| --- | --- |
| `before_tool` | Before each non-empty tool batch; block and escalation prevent invocation. |
| `after_tool` | After the complete tool batch returns and before results enter the next model turn. |
| `after_task` | After a terminal model response and output validation, before task success is committed. This is the default. |
| `after_team` | Once after a successful explicit task workflow or complete coordinator-plus-task run. Evidence remains on `TeamRunResult`. |

An upstream failure skips `after_team`. A team-level `retry` decision returns a recoverable `verification_failed` result instead of automatically rerunning the whole team. See the [runtime contracts](../reference/runtime-contracts.md#verification-boundaries) for lifecycle transitions and exact attachment behavior.

## Build a gate in code

Construct a `QualityGate` from sensors and evaluate it against a `SensorContext`. This is the most flexible path and the only way to use `schema_sensor` or a custom LLM-judge sensor.

```python title="gate.py"
from pydantic import BaseModel

from anycode import QualityGate, SensorContext, ruff_sensor, schema_sensor


class WeatherReport(BaseModel):
    city: str
    temperature_celsius: float
    summary: str


gate = QualityGate([
    ruff_sensor("src/", phases=("after_task",)),
    schema_sensor(WeatherReport, phases=("after_task",)),
])

decision = await gate.evaluate(SensorContext(
    phase="after_task",
    agent_name="reporter",
    run_id="run-1",
    output='{"city": "Paris", "temperature_celsius": 18.5, "summary": "mild"}',
))
print(decision.outcome)   # "pass" | "warn" | "retry" | "block" | "escalate"
print(decision.message)
```

## How the decision is made

`decide_gate` reduces sensor results by **severity**, not by which sensor failed:

| Failing severity | Outcome | Effect at runtime |
| --- | --- | --- |
| `critical` | `block` | Run stops (not recoverable) |
| `error` | `retry` | Agent re-prompted with feedback, up to 3 times |
| `warning` | `warn` | Run continues |
| (all pass) | `pass` | Run continues |

A gate that blocks three times in a row escalates instead. When a sensor's underlying tool is missing (for example `ruff` isn't installed), it returns a **warning**, not an error, and the decision records `warn` while execution continues. Treat missing verification tooling as a deployment error when the check is mandatory.

## Attach sensors declaratively

For registry sensors — `ruff`, `pyright`, `pytest`, `regex` — you can declare verification in config and let the runner build the gate. Attach it at the agent, runner, or engine level.

```python title="declarative.py"
from anycode import AgentConfig
from anycode.types import VerificationSensorConfig

agent = AgentConfig(
    name="builder",
    provider="anthropic",
    model="claude-sonnet-5",
    tools=["file_read", "file_write", "file_edit"],
    verification=(
        VerificationSensorConfig(
            name="pytest",
            kind="computational",
            phases=("after_task",),
            options={"target": "tests/"},
        ),
    ),
)
```

| `VerificationSensorConfig` field | Default | Meaning |
| --- | --- | --- |
| `name` | — | Registry sensor name (`ruff`/`pyright`/`pytest`/`regex`) |
| `phases` | `("after_task",)` | When the sensor runs |
| `block_on_failure` | `False` | Metadata (see the warning below) |
| `options` | `{}` | Sensor-specific options (e.g. `target`, `pattern`) |

!!! warning "Severity drives the outcome, not `block_on_failure`"
    `block_on_failure` and `retry_on_failure` are metadata fields — the decision keys off **severity**. A `pytest` failure is `critical` and blocks; a `ruff` failure is `error` and triggers a retry. Only registry sensors (`ruff`, `pyright`, `pytest`, `regex`) are addressable from config; `schema_sensor` and LLM-judge sensors must be wired in code with `QualityGate(...)`.

## Add a custom sensor

Any callable that takes a `SensorContext` and returns a `VerificationResult` is a sensor — including an async function that asks a model to judge the output.

```python title="custom_sensor.py"
from anycode import Sensor, VerificationResult
from anycode.types import VerificationSensorConfig


async def _judge(ctx) -> VerificationResult:
    ok = "TODO" not in (ctx.output or "")
    return VerificationResult(
        sensor_name="no_todo",
        kind="computational",
        passed=ok,
        severity="error" if not ok else "info",
        message="Output still contains TODO." if not ok else "clean",
    )


judge = Sensor(config=VerificationSensorConfig(name="no_todo", kind="computational"), fn=_judge)
```

## Next steps

- [Build a code-review crew](../tutorials/code-review-crew.md) — verification gates guarding real code changes.
- [Add self-reflection](reflection.md) — model-based critique to pair with computational gates.
- [Evaluate agents with a suite](evaluation.md) — turn verification into a repeatable regression check.
- [Configuration reference](../reference/configuration.md) — every `VerificationSensorConfig` field.
