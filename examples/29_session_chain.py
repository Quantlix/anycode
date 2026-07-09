"""Session chaining: calendar-scale work as fresh sessions over a goal contract.

A months-long task is not one infinite context. It is a durable **goal
contract** (machine-checkable criteria with pass/fail), an append-only
**progress log**, and a chain of bounded fresh-context sessions, each working
the next incomplete criterion. Criteria only flip through the external
verifier — never through the agent's own claim.

Run::

    uv run python examples/29_session_chain.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from anycode import FakeAdapter, FakeResponse, GoalContract, GoalCriterion, SessionChain
from anycode.core.runner import AgentRunner
from anycode.tools.executor import ToolExecutor
from anycode.tools.registry import ToolRegistry
from anycode.types import RunnerOptions, RunResult


async def main() -> None:
    work_dir = Path(tempfile.mkdtemp(prefix="anycode-chain-"))

    contract = GoalContract(
        goal="Ship the data-export feature",
        criteria=(
            GoalCriterion(id="schema", description="Design the export schema"),
            GoalCriterion(id="implement", description="Implement the export endpoint"),
            GoalCriterion(id="tests", description="Cover the endpoint with tests"),
        ),
    )

    def runner_factory() -> AgentRunner:
        # A brand-new adapter per session = a genuinely fresh context window.
        registry = ToolRegistry()
        return AgentRunner(
            FakeAdapter(responses=[FakeResponse(text="criterion work finished and verified locally")]),
            registry,
            ToolExecutor(registry),
            RunnerOptions(model="fake-model", agent_name="chained-worker", max_turns=6),
        )

    async def verifier(criterion: GoalCriterion, result: RunResult) -> str | None:
        # External verification: in real deployments this runs tests, a quality
        # gate, or asks a human. The agent's own claim is never enough.
        if result.stop_reason is not None and result.stop_reason.code == "success":
            return f"verified externally for '{criterion.id}'"
        return None

    chain = SessionChain(
        runner_factory=runner_factory,
        contract=contract,
        work_dir=work_dir,
        verifier=verifier,
        max_sessions=6,
    )
    final = await chain.run()

    print(f"goal complete: {final.complete}")
    for criterion in final.criteria:
        print(f"  [{'x' if criterion.passes else ' '}] {criterion.id}: {criterion.evidence}")
    print(f"\nprogress log ({work_dir / 'progress.md'}):")
    print((work_dir / "progress.md").read_text(encoding="utf-8"))

    assert final.complete


if __name__ == "__main__":
    asyncio.run(main())
