"""Tests for source-linked documentation validation."""

from __future__ import annotations

from pathlib import Path

from scripts import check_docs


def test_frontmatter_uses_yaml_and_requires_text_metadata(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "site_docs"
    docs_dir.mkdir()
    (docs_dir / "valid.md").write_text('---\ntitle: "Valid"\ndescription: "Present"\n---\n\n# Valid\n', encoding="utf-8")
    (docs_dir / "invalid.md").write_text('---\ntitle: ["not", "text"]\ndescription: "broken\n---\n', encoding="utf-8")
    monkeypatch.setattr(check_docs, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check_docs, "DOCS_DIR", docs_dir)

    errors: list[str] = []
    check_docs.check_frontmatter(errors)

    assert len(errors) == 1
    assert "invalid YAML frontmatter" in errors[0]


def test_readme_tool_inventory_preserves_registration_order(tmp_path: Path, monkeypatch) -> None:
    expected = [tool.name for tool in check_docs.anycode.BUILT_IN_TOOLS]
    rows = "\n".join(f"| `{name}` | Tool |" for name in reversed(expected))
    readme = tmp_path / "README.md"
    readme.write_text(f"# Project\n\n## Built-In Tools\n\n| Tool | Purpose |\n| --- | --- |\n{rows}\n", encoding="utf-8")
    monkeypatch.setattr(check_docs, "README_PATH", readme)

    errors: list[str] = []
    check_docs.check_readme_tools(errors)

    assert errors == [f"README.md built-in tool order is {list(reversed(expected))}; expected {expected}"]
