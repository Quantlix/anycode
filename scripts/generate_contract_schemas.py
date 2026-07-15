"""Generate or verify the checked-in semantic-contract JSON Schemas."""

from __future__ import annotations

import argparse
from pathlib import Path

from anycode.contracts.models import CONTRACT_MODELS
from anycode.contracts.schema import SCHEMA_DIRECTORY_NAME, synchronize_contract_schemas

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_ROOT = REPO_ROOT / "src" / "anycode" / "contracts" / "schemas" / SCHEMA_DIRECTORY_NAME


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail when checked-in schemas differ from generated schemas.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    mismatches = synchronize_contract_schemas(SCHEMA_ROOT, check=args.check)
    if args.check and mismatches:
        print(f"Contract schemas are stale: {', '.join(mismatches)}")
        return 1
    action = "updated" if mismatches else "verified"
    print(f"Contract schemas {action}: {len(mismatches) if mismatches else len(CONTRACT_MODELS)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
