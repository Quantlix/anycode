"""Enforce a single source of truth for the package version.

``pyproject.toml`` ``project.version`` is canonical. This script fails when any
other version-bearing surface drifts from it:

- ``README.md`` "Current version" row
- ``CHANGELOG.md`` release section and footer compare links
- ``uv.lock`` pinned version of the project itself
- (optionally) a release tag passed as ``--tag vX.Y.Z``

Usage:
    uv run python scripts/check_versions.py
    uv run python scripts/check_versions.py --tag v0.6.0
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_canonical_version() -> str:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["version"]


def check_readme(version: str, errors: list[str]) -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"^\|\s*Current version\s*\|\s*`([^`]+)`\s*\|", readme, re.MULTILINE)
    if match is None:
        errors.append("README.md: 'Current version' table row not found")
    elif match.group(1) != version:
        errors.append(f"README.md: 'Current version' row says {match.group(1)}, pyproject.toml says {version}")


def check_changelog(version: str, errors: list[str]) -> None:
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if re.search(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$", changelog, re.MULTILINE) is None:
        errors.append(f"CHANGELOG.md: missing dated release section '## [{version}] - YYYY-MM-DD'")
    if re.search(rf"^\[{re.escape(version)}\]: https://", changelog, re.MULTILINE) is None:
        errors.append(f"CHANGELOG.md: missing footer compare link '[{version}]: ...'")
    unreleased = re.search(r"^\[Unreleased\]: \S*/compare/v(\S+)\.\.\.HEAD$", changelog, re.MULTILINE)
    if unreleased is None:
        errors.append("CHANGELOG.md: missing '[Unreleased]: .../compare/vX.Y.Z...HEAD' footer link")
    elif unreleased.group(1) != version:
        errors.append(f"CHANGELOG.md: '[Unreleased]' compares against v{unreleased.group(1)}, expected v{version}")


def check_uv_lock(version: str, errors: list[str]) -> None:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = next((pkg.get("version") for pkg in lock.get("package", []) if pkg.get("name") == "anycode-py"), None)
    if locked is None:
        errors.append("uv.lock: package 'anycode-py' not found")
    elif locked != version:
        errors.append(f"uv.lock: anycode-py pinned at {locked}, pyproject.toml says {version} (run 'uv lock')")


def check_tag(version: str, tag: str, errors: list[str]) -> None:
    if tag != f"v{version}":
        errors.append(f"release tag is '{tag}' but pyproject.toml version is {version}; expected tag 'v{version}'")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag to validate against the canonical version (e.g. v0.6.0)")
    args = parser.parse_args()

    version = read_canonical_version()
    errors: list[str] = []
    check_readme(version, errors)
    check_changelog(version, errors)
    check_uv_lock(version, errors)
    if args.tag:
        check_tag(version, args.tag, errors)

    if errors:
        print(f"Version consistency check FAILED (canonical version: {version})", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"Version consistency check passed: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
