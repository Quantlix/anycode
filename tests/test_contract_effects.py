import asyncio
from datetime import UTC, datetime, timedelta

from anycode.contracts import InMemoryOperationStore, canonical_input_digest

NOW = datetime(2026, 1, 1, tzinfo=UTC)
DIGEST = canonical_input_digest({"account": "A", "amount": 10})


async def test_concurrent_claims_and_fencing_commit_only_one_result() -> None:
    store = InMemoryOperationStore()
    first, second = await asyncio.gather(
        store.claim(
            operation_key="charge:A:10",
            run_id="run-1",
            task_id="task-1",
            input_digest=DIGEST,
            owner_id="worker-1",
            lease_seconds=10,
            now=NOW,
        ),
        store.claim(
            operation_key="charge:A:10",
            run_id="run-1",
            task_id="task-1",
            input_digest=DIGEST,
            owner_id="worker-2",
            lease_seconds=10,
            now=NOW,
        ),
    )
    acquired = first if first.outcome == "acquired" else second
    denied = second if acquired is first else first

    assert acquired.claim is not None
    assert denied.outcome == "busy"
    committed = await store.commit(
        operation_key="charge:A:10",
        owner_id=acquired.claim.owner_id,
        fencing_token=acquired.claim.fencing_token,
        result_artifact_id="receipt-1",
        now=NOW + timedelta(seconds=1),
    )
    replay = await store.commit(
        operation_key="charge:A:10",
        owner_id=acquired.claim.owner_id,
        fencing_token=acquired.claim.fencing_token,
        result_artifact_id="receipt-1",
        now=NOW + timedelta(seconds=2),
    )

    assert committed.outcome == "committed"
    assert replay.outcome == "replay"
    assert replay.result_artifact_id == "receipt-1"


async def test_expired_owner_is_fenced_out_after_reclaim() -> None:
    store = InMemoryOperationStore()
    old = await store.claim(
        operation_key="send:message-1",
        run_id="run-1",
        task_id=None,
        input_digest=DIGEST,
        owner_id="old-worker",
        lease_seconds=1,
        now=NOW,
    )
    current = await store.claim(
        operation_key="send:message-1",
        run_id="run-1",
        task_id=None,
        input_digest=DIGEST,
        owner_id="new-worker",
        lease_seconds=10,
        now=NOW + timedelta(seconds=2),
    )

    assert old.claim is not None and current.claim is not None
    stale = await store.commit(
        operation_key="send:message-1",
        owner_id="old-worker",
        fencing_token=old.claim.fencing_token,
        result_artifact_id="old-result",
        now=NOW + timedelta(seconds=3),
    )
    committed = await store.commit(
        operation_key="send:message-1",
        owner_id="new-worker",
        fencing_token=current.claim.fencing_token,
        result_artifact_id="new-result",
        now=NOW + timedelta(seconds=3),
    )

    assert current.claim.fencing_token == old.claim.fencing_token + 1
    assert stale.outcome == "stale"
    assert committed.outcome == "committed"
    assert (await store.get("send:message-1")).result_artifact_id == "new-result"  # type: ignore[union-attr]


async def test_conflicting_inputs_and_uncertain_effects_fail_closed() -> None:
    store = InMemoryOperationStore()
    claim = await store.claim(
        operation_key="publish:1",
        run_id="run-1",
        task_id=None,
        input_digest=DIGEST,
        owner_id="worker",
        lease_seconds=10,
        now=NOW,
    )
    assert claim.claim is not None

    conflict = await store.claim(
        operation_key="publish:1",
        run_id="run-1",
        task_id=None,
        input_digest=canonical_input_digest({"different": True}),
        owner_id="other",
        lease_seconds=10,
        now=NOW,
    )
    uncertain = await store.mark_uncertain(
        operation_key="publish:1",
        owner_id="worker",
        fencing_token=claim.claim.fencing_token,
        now=NOW,
    )
    retry = await store.claim(
        operation_key="publish:1",
        run_id="run-1",
        task_id=None,
        input_digest=DIGEST,
        owner_id="other",
        lease_seconds=10,
        now=NOW + timedelta(seconds=20),
    )

    assert conflict.outcome == "conflict"
    assert uncertain.outcome == "uncertain"
    assert retry.outcome == "uncertain"
