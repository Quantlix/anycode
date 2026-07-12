# Pull Request

## Summary

<!-- What problem does this solve, and what observable behavior changes? -->

## Change type

- [ ] Bug fix
- [ ] Backward-compatible feature
- [ ] Breaking change
- [ ] Documentation
- [ ] Test or internal maintenance
- [ ] Security-sensitive change handled through a private advisory

## Compatibility

<!-- Address each affected contract: Python API, CLI, YAML/TOML, checkpoints, durable run data, provider/tool protocols. State "none" when no public contract changes. -->

- Version impact: patch / minor / major / none
- Deprecation or migration path:
- Rollback or persisted-data impact:

## Verification

<!-- List focused tests and exact commands run. -->

- [ ] Regression or behavior tests added or updated
- [ ] `uv run python -m pytest`
- [ ] `uv run python -m ruff check .`
- [ ] `uv run python -m ruff format --check src/`
- [ ] `uv run python -m pyright`
- [ ] `uv run python -m mkdocs build --strict`

## Documentation and release notes

- [ ] Public docstrings and reference pages match the implementation
- [ ] Guides, examples, README, and `site_docs/llms.txt` updated where needed
- [ ] `[Unreleased]` changelog entry added, or no user-visible change
- [ ] Security, configuration, or release documentation updated where needed

## Reviewer notes

<!-- Call out design tradeoffs, follow-up work, deployment concerns, or areas that need especially careful review. -->
<!-- Maintainers may request additional evidence for high-risk changes. -->
