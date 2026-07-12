"""Validate documentation claims that are derived from source code.

Run this after ``mkdocs build --strict`` so generated API HTML is available.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import yaml

import anycode

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "site_docs"
SITE_DIR = REPO_ROOT / "site"
README_PATH = REPO_ROOT / "README.md"
LLMS_PATH = DOCS_DIR / "llms.txt"
EXAMPLES_DIR = REPO_ROOT / "examples"
API_INVENTORY_HTML = SITE_DIR / "reference" / "api-inventory" / "index.html"
DOCS_SITE_PREFIX = "https://quantlix.github.io/anycode/latest/"


def _markdown_section(markdown: str, heading: str) -> str | None:
    match = re.search(rf"^## {re.escape(heading)}\s*$\n(.*?)(?=^## |\Z)", markdown, re.MULTILINE | re.DOTALL)
    return match.group(1) if match else None


def check_api_inventory(errors: list[str]) -> None:
    if not API_INVENTORY_HTML.is_file():
        errors.append(f"{API_INVENTORY_HTML.relative_to(REPO_ROOT)} is missing; run the strict MkDocs build first")
        return

    html = API_INVENTORY_HTML.read_text(encoding="utf-8")
    direct_anchors = set(re.findall(r'id="anycode\.([A-Za-z_][A-Za-z0-9_]*)"', html))

    missing: list[str] = []
    for name in anycode.__all__:
        value = getattr(anycode, name, None)
        if inspect.ismodule(value):
            if f"<code>{name}</code>" not in html:
                missing.append(name)
        elif name not in direct_anchors:
            missing.append(name)

    if missing:
        errors.append(f"generated API inventory is missing exports: {', '.join(sorted(missing))}")


def check_readme_tools(errors: list[str]) -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    section = _markdown_section(readme, "Built-In Tools")
    if section is None:
        errors.append("README.md is missing the 'Built-In Tools' section")
        return

    documented = re.findall(r"^\|\s*`([^`]+)`\s*\|", section, re.MULTILINE)
    expected = [tool.name for tool in anycode.BUILT_IN_TOOLS]
    missing = set(expected) - set(documented)
    stale = set(documented) - set(expected)
    if missing:
        errors.append(f"README.md omits built-in tools: {', '.join(sorted(missing))}")
    if stale:
        errors.append(f"README.md lists unknown built-in tools: {', '.join(sorted(stale))}")
    duplicates = sorted({name for name in documented if documented.count(name) > 1})
    if duplicates:
        errors.append(f"README.md lists built-in tools more than once: {', '.join(duplicates)}")
    if not missing and not stale and not duplicates and documented != expected:
        errors.append(f"README.md built-in tool order is {documented}; expected {expected}")


def check_example_inventory(errors: list[str]) -> None:
    examples = sorted(EXAMPLES_DIR.glob("[0-9][0-9]_*.py"))
    numbers = [int(path.name[:2]) for path in examples]
    expected_numbers = list(range(1, len(examples) + 1))
    if numbers != expected_numbers:
        errors.append(f"numbered examples are not contiguous: found {numbers}, expected {expected_numbers}")

    claim_pattern = re.compile(r"\b(\d+)\s+runnable\s+(?:example\s+)?scripts\b", re.IGNORECASE)
    for path in (README_PATH, LLMS_PATH):
        text = path.read_text(encoding="utf-8")
        claims = [int(value) for value in claim_pattern.findall(text)]
        if not claims:
            errors.append(f"{path.relative_to(REPO_ROOT)} does not state the numbered example count")
            continue
        wrong = sorted({claim for claim in claims if claim != len(examples)})
        if wrong:
            errors.append(f"{path.relative_to(REPO_ROOT)} claims {wrong} runnable examples; found {len(examples)}")


def check_frontmatter(errors: list[str]) -> None:
    for path in sorted(DOCS_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            errors.append(f"{path.relative_to(REPO_ROOT)} is missing YAML frontmatter")
            continue
        closing = text.find("\n---\n", 4)
        if closing == -1:
            errors.append(f"{path.relative_to(REPO_ROOT)} has unterminated YAML frontmatter")
            continue
        frontmatter = text[4:closing]
        try:
            metadata = yaml.safe_load(frontmatter)
        except yaml.YAMLError as exc:
            detail = str(exc).splitlines()[0]
            errors.append(f"{path.relative_to(REPO_ROOT)} has invalid YAML frontmatter: {detail}")
            continue
        if not isinstance(metadata, dict):
            errors.append(f"{path.relative_to(REPO_ROOT)} YAML frontmatter must be a mapping")
            continue
        for field in ("title", "description"):
            value = metadata.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{path.relative_to(REPO_ROOT)} frontmatter is missing '{field}'")


def _docs_source_for_url(url: str) -> Path | None:
    relative_url = url.removeprefix(DOCS_SITE_PREFIX).split("#", 1)[0].split("?", 1)[0].strip("/")
    if not relative_url:
        return DOCS_DIR / "index.md"

    direct = DOCS_DIR / f"{relative_url}.md"
    if direct.is_file():
        return direct
    index = DOCS_DIR / relative_url / "index.md"
    if index.is_file():
        return index
    return None


def check_llms_links(errors: list[str]) -> None:
    llms = LLMS_PATH.read_text(encoding="utf-8")
    urls = set(re.findall(r"https://quantlix\.github\.io/anycode/latest/[^)\s]*", llms))
    for url in sorted(urls):
        if _docs_source_for_url(url) is None:
            errors.append(f"site_docs/llms.txt links to a missing documentation page: {url}")


def main() -> int:
    errors: list[str] = []
    check_api_inventory(errors)
    check_readme_tools(errors)
    check_example_inventory(errors)
    check_frontmatter(errors)
    check_llms_links(errors)

    if errors:
        print("Documentation consistency check FAILED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"Documentation consistency check passed: {len(anycode.__all__)} exports and {len(list(EXAMPLES_DIR.glob('[0-9][0-9]_*.py')))} examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
