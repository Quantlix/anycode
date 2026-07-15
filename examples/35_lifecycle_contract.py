"""Prove that ``after_team`` evaluates the complete orchestrator result.

Run with::

    uv run python examples/35_lifecycle_contract.py

The example is deterministic and uses a plugin-registered ``FakeAdapter``
factory, so it does not require provider credentials.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

from anycode import (
    AgentConfig,
    AnyCode,
    FakeAdapter,
    FakeResponse,
    OrchestratorConfig,
    ProviderResilienceConfig,
    TeamConfig,
    VerificationSensorConfig,
    register_provider_factory,
)

PROVIDER_NAME = "m0-lifecycle-fixture"


def _response_scripts() -> Iterator[list[FakeResponse]]:
    yield [FakeResponse(text="ASSIGN: worker | implementation | produce the final output\nCOORDINATOR_ONLY")]
    yield [FakeResponse(text="worker output is clean")]


async def main() -> None:
    scripts = _response_scripts()

    async def _provider_factory(**_kwargs: object) -> FakeAdapter:
        return FakeAdapter(responses=next(scripts))

    register_provider_factory(PROVIDER_NAME, _provider_factory)
    verification = VerificationSensorConfig(
        name="regex",
        kind="computational",
        phases=("after_team",),
        options={
            "pattern": "COORDINATOR_ONLY",
            "expect": "no_match",
            "severity": "critical",
        },
    )
    engine = AnyCode(OrchestratorConfig(verification=(verification,)))
    worker = AgentConfig(
        name="worker",
        provider=PROVIDER_NAME,
        model="fake-model",
        provider_resilience=ProviderResilienceConfig(enabled=False),
    )
    team = engine.create_team("contract-team", TeamConfig(name="contract-team", agents=[worker]))

    try:
        result = await engine.run_team(team, "prove the team verification boundary")
    finally:
        await engine.close()

    assert result.success is False
    assert result.stop_reason is not None
    assert result.stop_reason.code == "verification_failed"
    assert [decision.outcome for decision in result.gate_decisions] == ["block"]
    assert [verification.sensor_name for verification in result.verification_results] == ["regex"]

    print(f"success={result.success}")
    print(f"stop_reason={result.stop_reason.code}")
    print(f"gate_outcome={result.gate_decisions[0].outcome}")
    print(f"sensor={result.verification_results[0].sensor_name}")


if __name__ == "__main__":
    asyncio.run(main())
