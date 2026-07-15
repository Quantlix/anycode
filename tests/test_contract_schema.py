from pathlib import Path

from anycode.contracts import CONTRACT_MODELS, schema_filename, synchronize_contract_schemas

SCHEMA_ROOT = Path(__file__).parents[1] / "src" / "anycode" / "contracts" / "schemas" / "v1"


def test_checked_in_contract_schemas_are_current() -> None:
    assert synchronize_contract_schemas(SCHEMA_ROOT, check=True) == []
    assert {path.name for path in SCHEMA_ROOT.glob("*.schema.json")} == {schema_filename(model) for model in CONTRACT_MODELS}
