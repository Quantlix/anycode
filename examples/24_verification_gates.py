"""Phase 6 — declarative quality gate via the verification sensor registry.

Builds a regex sensor through ``build_sensors`` and runs it directly against a
``SensorContext`` so users can see how YAML ``verification:`` blocks
materialise into a Sensor + QualityGate without spinning up an agent.

Run::

    uv run python examples/24_verification_gates.py
"""

from __future__ import annotations

import asyncio

from anycode.types import VerificationSensorConfig
from anycode.verification import QualityGate, build_sensors
from anycode.verification.sensor import SensorContext


async def main() -> None:
    configs = (
        VerificationSensorConfig(
            name="regex",
            kind="computational",
            phases=("after_task",),
            block_on_failure=True,
            options={"pattern": "DONE", "expect": "match"},
        ),
    )

    gate = QualityGate(build_sensors(configs))

    for output in ("the task is DONE.", "still working on it..."):
        ctx = SensorContext(phase="after_task", agent_name="demo", run_id="run-1", output=output)
        decision = await gate.evaluate(ctx)
        print(f"\noutput={output!r}")
        print(f"  outcome: {decision.outcome}")
        for r in decision.results:
            print(f"    - sensor={r.sensor_name} passed={r.passed} severity={r.severity}: {r.message}")


if __name__ == "__main__":
    asyncio.run(main())
