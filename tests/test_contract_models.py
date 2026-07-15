import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from anycode import Run
from anycode.contracts import CONTRACT_MODELS, CONTRACT_SCHEMA_VERSION, Result, contract_schema_bundle

FIXTURE = Path(__file__).parent / "fixtures" / "contracts" / "v1" / "run-history.json"


def test_saved_v1_run_fixture_remains_readable_and_frozen() -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    run = Run.model_validate(raw["run"])

    assert run.schema_version == CONTRACT_SCHEMA_VERSION
    assert run.state == "succeeded"
    with pytest.raises(ValidationError):
        run.state = "failed"  # type: ignore[misc]


def test_future_schema_and_unknown_fields_are_rejected() -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))["run"]
    with pytest.raises(ValidationError):
        Run.model_validate({**raw, "schema_version": "2.0"})
    with pytest.raises(ValidationError):
        Run.model_validate({**raw, "surprise": True})


def test_result_values_are_json_only() -> None:
    result = Result(ok=True, value={"nested": [1, True, None, "value"]})
    assert json.loads(result.model_dump_json())["value"]["nested"] == [1, True, None, "value"]

    with pytest.raises(ValidationError):
        Result(ok=True, value=object())


def test_schema_bundle_contains_every_versioned_domain_model() -> None:
    bundle = contract_schema_bundle()
    models = bundle["models"]

    assert bundle["contract_version"] == CONTRACT_SCHEMA_VERSION
    assert isinstance(models, dict)
    assert set(models) == {model.__name__ for model in CONTRACT_MODELS}
    assert set(models) == {
        "Artifact",
        "CapabilityDescriptor",
        "Checkpoint",
        "Event",
        "Message",
        "PolicyDecision",
        "Run",
        "Task",
        "VerificationResult",
    }
