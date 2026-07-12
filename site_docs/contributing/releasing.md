---
title: "Release AnyCode to PyPI with Trusted Publishing"
description: "Prepare, validate, publish, and verify an AnyCode release through TestPyPI, GitHub Releases, PyPI Trusted Publishing, and versioned docs."
keywords: AnyCode release process, PyPI Trusted Publishing, TestPyPI, semantic versioning, GitHub Release, mike versioned docs
---

# Release AnyCode

AnyCode releases use a reviewed release pull request, a TestPyPI preflight, an immutable Git tag, and PyPI Trusted Publishing. The repository-level [`RELEASE.md`](https://github.com/Quantlix/anycode/blob/main/RELEASE.md) is the exact current runbook; this page explains the release sequence and its safety checks.

## Choose the release type

Read every entry under `[Unreleased]` and compare it with the [compatibility policy](../reference/compatibility.md).

| Release | Use |
| --- | --- |
| Patch | Compatible fixes, documentation corrections, packaging fixes, and security fixes |
| Minor before 1.0 | Backward-compatible features and explicitly approved breaking changes with migration guidance |
| Major after 1.0 | Incompatible public-contract changes |
| Pre-release | A candidate that needs ecosystem testing before the final release |

Do not choose a patch version for a new required parameter, removed export, renamed persisted field, or incompatible behavior change.

## Confirm release access

The release manager needs:

- Permission to merge the release pull request and create tags and GitHub Releases.
- Access to run the TestPyPI workflow.
- Approval rights for the protected `pypi` environment, or an available approver.
- PyPI and TestPyPI Trusted Publisher records matching the repository, workflow filename, and environment.

Production publishing uses GitHub OIDC. Do not add a long-lived PyPI token when Trusted Publishing is available.

## Prepare the release pull request

Create a branch from the exact `main` commit intended for release:

```bash
git fetch origin
git switch --create release/X.Y.Z origin/main
```

Update the release surfaces together:

1. Set `project.version` in `pyproject.toml`.
2. Run `uv lock` to synchronize the project entry in `uv.lock`.
3. Update the current-version row in `README.md`.
4. Move `[Unreleased]` entries into `## [X.Y.Z] - YYYY-MM-DD` in `CHANGELOG.md`.
5. Add the release and `[Unreleased]` comparison links at the bottom of the changelog.
6. Confirm migration and rollback guidance for every incompatible or persisted-format change.
7. Review the documentation against the complete release diff.

Editorialize release notes for users. Lead with observable changes, call out deprecations and removals, and avoid copying internal commit messages.

## Run the release gate

Install the locked environment and run all source, test, documentation, and version checks:

```bash
uv sync --locked --group dev
uv run python scripts/check_versions.py
uv run python -m ruff check .
uv run python -m ruff format --check src/
uv run python -m pyright
uv run python -m pytest
uv run python -m mkdocs build --strict
uv run python scripts/check_docs.py
uv run python -m build --no-isolation
uv run python -m twine check --strict dist/*
```

Build from a clean `dist/` and `build/` directory so stale files cannot satisfy an artifact check. CI repeats the release gate and builds fresh artifacts from the tagged source.

Open the pull request and request the approvals required by the [maintainer policy](maintainers.md). After approval and green checks, merge without adding unrelated changes to the release candidate.

## Publish to TestPyPI

Run `.github/workflows/publish-testpypi.yml` against the merged release commit. The workflow rebuilds the wheel and source distribution and publishes through the `testpypi` environment.

Install the candidate into a fresh environment. Use TestPyPI for AnyCode and PyPI as the dependency source:

```bash
uv venv .release-smoke
uv pip install --python .release-smoke \
  --index https://test.pypi.org/simple/ \
  --default-index https://pypi.org/simple/ \
  "anycode-py[cli]==X.Y.Z"
```

Run the environment's Python interpreter and `anycode` executable to verify:

- `from anycode import AnyCode` succeeds.
- `anycode version` reports `X.Y.Z`.
- Core installation and the CLI extra resolve without the repository source tree.

Do not tag the release until this preflight passes.

## Tag and publish

Confirm the local commit matches the merged candidate, then create an annotated tag:

```bash
git switch main
git pull --ff-only origin main
uv run python scripts/check_versions.py --tag vX.Y.Z
git tag -a vX.Y.Z -m "AnyCode X.Y.Z"
git push origin vX.Y.Z
```

Create a GitHub Release from `vX.Y.Z` and use the edited changelog section as its notes. Publishing the GitHub Release triggers `.github/workflows/publish-pypi.yml`; pushing a tag alone does not publish the package.

The production workflow:

1. Validates the tag against `project.version`.
2. Runs lint, format, type, test, build, and metadata checks.
3. Confirms expected wheel and source-distribution filenames.
4. Uploads the exact build artifacts between jobs.
5. Waits for protected `pypi` environment approval.
6. Publishes through OIDC with attestations.

When another release maintainer is available, the environment approver should differ from the release preparer.

## Verify the release

Publication is complete only after external verification:

- The PyPI project shows `X.Y.Z`, one wheel, one source distribution, hashes, and attestations.
- A clean install from PyPI imports `AnyCode` and runs `anycode version`.
- The GitHub Release tag points to the reviewed commit and its notes match the changelog.
- The documentation workflow publishes the `X.Y` site version and moves `latest` to it when appropriate.
- `/latest/`, `/llms.txt`, the API inventory, and release-relevant guides load successfully.

Record a release-process correction in `RELEASE.md` before the next release if any manual recovery was required.

## Patch an older release line

When `main` remains compatible, release the patch from `main`. If `main` contains unreleased incompatible work:

1. Create `maint/X.Y` from the latest affected tag.
2. Land the fix on `main` when it applies there.
3. Cherry-pick the reviewed fix into a `backport/X.Y-description` branch.
4. Open a pull request against `maint/X.Y` and run the full release gate.
5. Prepare and tag the next patch version from the maintenance branch.

Keep feature work and opportunistic refactors out of a patch branch.

## Handle a failed or bad release

- Cancel the workflow if publication has not reached PyPI.
- Never reuse a version after any artifact has been published.
- Fix forward with a new patch version.
- Yank an installable but unsafe or seriously broken release instead of deleting it.
- Add a warning to the affected GitHub Release and changelog that names the safe replacement.
- Use a private GitHub Security Advisory and coordinated disclosure for vulnerabilities.

PyPI files and Git tags are immutable release records. Treat a published artifact as permanent evidence, even when it is later yanked.

## Next steps

- [Maintainer governance and change policy](maintainers.md)
- [Compatibility and versioning](../reference/compatibility.md)
- [Documentation contributor guide](docs-guide.md)
- [Production readiness](../guides/production-readiness.md)
