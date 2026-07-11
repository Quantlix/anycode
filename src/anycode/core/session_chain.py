"""Session chaining for calendar-scale work.

Long tasks run as a chain of bounded sessions over a durable goal contract,
not as one infinite context (the initializer/session pattern):

* The **goal contract** is a JSON file of machine-checkable criteria with
  per-item pass/fail. It is the control-flow spine; agents must not redefine
  "done" mid-run, so the chain never edits criterion text — only `passes`
  flips, and only through the caller-supplied verifier.
* The **progress log** is an append-only narrative for orienting the next
  session (what happened, what's next).
* Each session starts a fresh context seeded with the contract status and the
  progress tail, works the highest-priority incomplete criterion, and ends
  within its runner's turn budget.

The verifier is deliberately external: self-grading is the documented failure
mode for long-running agents, so a criterion only passes when independent
verification says so (a quality-gate check, a test run, a human).
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from anycode.core.runner import AgentRunner
from anycode.security.redaction import redact_sensitive, redact_text
from anycode.types import GoalContract, GoalCriterion, LLMMessage, RunResult, TextBlock

Verifier = Callable[[GoalCriterion, RunResult], Awaitable[str | None]]
"""Returns evidence text when the criterion is verified done, else None."""

_CONTRACT_FILE = "contract.json"
_PROGRESS_FILE = "progress.md"
_PROGRESS_TAIL_CHARS = 4000


def load_contract(work_dir: str | Path) -> GoalContract | None:
    path = Path(work_dir) / _CONTRACT_FILE
    if not path.exists():
        return None
    return GoalContract.model_validate_json(path.read_text(encoding="utf-8"))


def save_contract(work_dir: str | Path, contract: GoalContract, *, redact_sensitive_data: bool = True) -> None:
    path = Path(work_dir) / _CONTRACT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    payload = contract.model_dump(mode="json")
    if redact_sensitive_data:
        payload = redact_sensitive(payload)
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


class SessionChain:
    """Drives bounded sessions against a goal contract until done or capped."""

    def __init__(
        self,
        *,
        runner_factory: Callable[[], AgentRunner],
        contract: GoalContract,
        work_dir: str | Path,
        verifier: Verifier,
        max_sessions: int | None = None,
        redact_sensitive_data: bool = True,
    ) -> None:
        self._runner_factory = runner_factory
        self._work_dir = Path(work_dir)
        self._verifier = verifier
        self._max_sessions = max_sessions
        self._redact_sensitive_data = redact_sensitive_data
        self._work_dir.mkdir(parents=True, exist_ok=True)
        existing = load_contract(self._work_dir)
        if existing is not None:
            # A persisted contract is authoritative over the argument: criteria
            # must not silently change between sessions.
            if {c.id for c in existing.criteria} != {c.id for c in contract.criteria}:
                raise ValueError("Goal contract on disk differs from the provided contract — refusing to continue.")
            contract = existing
        self._contract = contract
        save_contract(self._work_dir, self._contract, redact_sensitive_data=self._redact_sensitive_data)

    @property
    def contract(self) -> GoalContract:
        return self._contract

    async def run(self) -> GoalContract:
        """Run sessions until every criterion passes or `max_sessions` is hit."""
        sessions_run = 0
        while not self._contract.complete:
            if self._max_sessions is not None and sessions_run >= self._max_sessions:
                break
            await self.run_session()
            sessions_run += 1
        return self._contract

    async def run_session(self) -> RunResult:
        """Run one fresh-context session against the next incomplete criterion."""
        criterion = self._contract.next_incomplete()
        if criterion is None:
            raise RuntimeError("Goal contract is already complete — nothing to run.")

        runner = self._runner_factory()
        result = await runner.run([LLMMessage(role="user", content=[TextBlock(text=self._session_prompt(criterion))])])

        evidence = None
        if result.stop_reason is not None and result.stop_reason.code == "success":
            evidence = await self._verifier(criterion, result)
        if evidence:
            self._contract = self._contract.mark_passed(criterion.id, evidence)
            save_contract(self._work_dir, self._contract, redact_sensitive_data=self._redact_sensitive_data)

        self._append_progress(criterion, result, verified=bool(evidence))
        return result

    def _session_prompt(self, criterion: GoalCriterion) -> str:
        status_lines = [f"[{'x' if c.passes else ' '}] {c.id}: {c.description}" for c in self._contract.criteria]
        parts = [
            f"GOAL: {self._contract.goal}",
            "",
            "CRITERIA (do not edit or remove; completion is verified externally):",
            *status_lines,
            "",
            f"THIS SESSION: work only on criterion '{criterion.id}': {criterion.description}",
        ]
        if criterion.steps:
            parts.append("Suggested steps:")
            parts.extend(f"  {i + 1}. {step}" for i, step in enumerate(criterion.steps))
        tail = self._progress_tail()
        if tail:
            parts.extend(["", "PROGRESS LOG (most recent):", tail])
        return "\n".join(parts)

    def _progress_tail(self) -> str:
        path = self._work_dir / _PROGRESS_FILE
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")[-_PROGRESS_TAIL_CHARS:]

    def _append_progress(self, criterion: GoalCriterion, result: RunResult, *, verified: bool) -> None:
        path = self._work_dir / _PROGRESS_FILE
        stamp = datetime.now(UTC).isoformat()
        stop = result.stop_reason.code if result.stop_reason else "unknown"
        output = redact_text(result.output) if self._redact_sensitive_data else result.output
        entry = (
            f"\n## {stamp} — criterion `{criterion.id}`\n"
            f"- stop: {stop}, turns: {result.turns}, verified: {'yes' if verified else 'no'}\n"
            f"- output: {output[:500]}\n"
        )
        with path.open("a", encoding="utf-8") as fh:
            fh.write(entry)


def contract_status_summary(contract: GoalContract) -> str:
    """One-line status suitable for progress logs and invariant re-injection."""
    done = sum(1 for c in contract.criteria if c.passes)
    return f"{done}/{len(contract.criteria)} criteria verified for goal: {contract.goal}"


__all__ = [
    "SessionChain",
    "Verifier",
    "contract_status_summary",
    "load_contract",
    "save_contract",
]
