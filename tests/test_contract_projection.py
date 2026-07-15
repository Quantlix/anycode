import json
from pathlib import Path

from anycode.contracts import Event, IncrementalRunProjector, project_run, validate_event_stream

FIXTURE = Path(__file__).parent / "fixtures" / "contracts" / "v1" / "run-history.json"


def _events() -> list[Event]:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [Event.model_validate(item) for item in raw["events"]]


def test_independent_projectors_build_the_same_terminal_view() -> None:
    events = _events()
    batch = project_run(events)
    incremental = IncrementalRunProjector("run-fixture")
    for event in events:
        outcome = incremental.apply(event)
        assert outcome.ok

    assert batch.ok
    assert batch.projection == incremental.projection
    assert batch.projection is not None
    assert batch.projection.state == "succeeded"
    assert batch.projection.artifact_ids == ("artifact-fixture",)
    assert batch.projection.cursor == 5


def test_event_stream_rejects_gaps_duplicates_and_unknown_causation() -> None:
    events = _events()

    gap = validate_event_stream([events[0], events[2]])
    duplicate = validate_event_stream([events[0], events[0].model_copy(update={"sequence": 2})])
    unknown_cause = validate_event_stream([events[0].model_copy(update={"causation_id": "future-event"})])

    assert not gap.valid and gap.error is not None and gap.error.code == "event_sequence_gap"
    assert not duplicate.valid and duplicate.error is not None and duplicate.error.code == "duplicate_event"
    assert not unknown_cause.valid and unknown_cause.error is not None and unknown_cause.error.code == "unknown_causation"
