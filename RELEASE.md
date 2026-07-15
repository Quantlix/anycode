# AnyCode Release Runbook

This is the authoritative procedure for publishing the `anycode-py` distribution, its GitHub Release, and versioned documentation. Maintainer authority, branch rules, compatibility review, and incident ownership are defined in [MAINTAINERS.md](MAINTAINERS.md).

AnyCode publishes through GitHub Actions, protected GitHub environments, and PyPI Trusted Publishing. Do not upload production artifacts from a developer workstation while this path is available.

## Release architecture

| Workflow | Trigger | Purpose | Protected environment |
| --- | --- | --- | --- |
| `.github/workflows/package-validation.yml` | Pull request, `main`, or manual | Build wheel and source distribution, check metadata, and smoke-test installed core and CLI packages | None |
| `.github/workflows/publish-testpypi.yml` | Manual | Rebuild and publish a candidate to TestPyPI | `testpypi` |
| `.github/workflows/publish-pypi.yml` | Published GitHub Release | Validate the tag, rebuild, transfer exact artifacts, and publish to PyPI | `pypi` |
| `.github/workflows/docs.yml` | Pull request, `main`, manual dispatch, or `v*` tag | Validate development docs; publish versioned docs only from a release-tag push | None |

Trusted Publishing grants the publish job a short-lived OIDC identity. Only publish jobs receive `id-token: write`; build and validation jobs receive read-only repository access.

## One-time repository setup

### GitHub environments

Create `testpypi` and `pypi` in **Settings > Environments**.

- Require approval for `pypi`.
- Restrict environment deployment branches or tags when repository policy supports it.
- Do not store a long-lived PyPI API token.
- When another release maintainer is available, require an approver other than the release preparer.

### Trusted Publisher records

Register these publishers in the package settings on TestPyPI and PyPI:

| Field | TestPyPI | PyPI |
| --- | --- | --- |
| Owner | `Quantlix` | `Quantlix` |
| Repository | `anycode` | `anycode` |
| Workflow | `publish-testpypi.yml` | `publish-pypi.yml` |
| Environment | `testpypi` | `pypi` |
| Project | `anycode-py` | `anycode-py` |

Publisher configuration lives in the hosting services and cannot be proven by repository CI alone. Verify it manually after a repository transfer, workflow rename, environment rename, or ownership change.

### Branch protection

Protect `main` according to [MAINTAINERS.md](MAINTAINERS.md): pull requests, passing required checks, resolved conversations, no force pushes, and no deletion. Require the CI, documentation, and package-validation jobs by their current GitHub check names.

## Support policy

The latest published minor line receives compatible bug and security fixes. Critical fixes may be backported to the previous minor line when the change is safe and practical. Older lines require an upgrade unless a security advisory states otherwise.

AnyCode is pre-1.0:

- Patch releases contain compatible fixes, documentation, packaging, and security updates.
- Minor releases contain backward-compatible features and may contain explicitly approved breaking changes.
- Breaking changes are never silent. They require migration and rollback guidance, compatibility tests, documentation, and prominent release notes.

## Select the version

AnyCode follows Semantic Versioning and uses PEP 440-compatible package versions.

| Version | Use |
| --- | --- |
| Next patch version | Compatible fix to the current pre-1.0 minor line |
| Next minor version, patch reset to zero | Feature release or approved pre-1.0 breaking change |
| `X.Y.ZrcN` | Release candidate |
| Next major version after 1.0 | Incompatible public-contract change |

Review the complete `[Unreleased]` section and the diff from the previous tag. Treat these as public contracts:

- Names in `anycode.__all__`.
- Documented signatures, defaults, exceptions, and behavior.
- CLI commands and output intended for automation.
- Declarative YAML and TOML configuration.
- Checkpoint and durable run formats.

The [compatibility policy](site_docs/reference/compatibility.md) defines the required evidence for each contract.

## Canonical version and synchronized surfaces

`project.version` in `pyproject.toml` is the single source of truth. A release preparation pull request synchronizes:

1. `pyproject.toml`.
2. The project package entry in `uv.lock` by running `uv lock`.
3. The `Current version` row in `README.md`.
4. A dated `## [X.Y.Z] - YYYY-MM-DD` section in `CHANGELOG.md`.
5. The release and `[Unreleased]` comparison links at the bottom of `CHANGELOG.md`.
6. The Git tag `vX.Y.Z` or `vX.Y.ZrcN`.

`scripts/check_versions.py` fails when these surfaces drift. A published package version and its tag are immutable; correct an error with a new version.

## Changelog rules

`CHANGELOG.md` follows Keep a Changelog. Move entries from `[Unreleased]` into the dated release section and organize them under the headings that apply:

- `Added`
- `Changed`
- `Deprecated`
- `Removed`
- `Fixed`
- `Security`

Write entries for users, not as a commit log. State the affected API and observable outcome. Begin incompatible entries with **Breaking:** and link to concrete migration instructions. Include deprecation and removal versions where applicable.

Before releasing, confirm the notes include:

- Major features and fixes.
- Breaking changes, removals, and deprecations.
- Supported Python or dependency changes.
- Persisted-format reader or writer changes.
- Security fixes that are ready for disclosure.
- Upgrade, migration, and rollback instructions.

## Prepare the release pull request

Create a branch from the exact `main` commit intended for release:

```bash
git fetch origin
git switch --create release/X.Y.Z origin/main
```

Update the synchronized version surfaces and run `uv lock`. Limit this branch to release notes, metadata, documentation, and release-blocking fixes. Do not add opportunistic features or refactors after candidate validation begins.

The pull request must state:

- Why the selected version is correct.
- Which public or persisted contracts changed.
- Whether deprecations and migration windows were honored.
- The local gate results.
- TestPyPI and rollback plans.

Request the approvals required by [MAINTAINERS.md](MAINTAINERS.md).

## Local release gate

Install the locked environment:

```bash
uv sync --locked --group dev
```

Run source, version, test, and documentation checks:

```bash
uv run python scripts/check_versions.py
uv run python -m ruff check .
uv run python -m ruff format --check src/
uv run python -m pyright
uv run python -m pytest
uv run python -m mkdocs build --strict
uv run python scripts/check_docs.py
```

Delete stale build output before packaging.

PowerShell:

```powershell
Remove-Item -Recurse -Force dist, build, .release-smoke -ErrorAction SilentlyContinue
```

POSIX shell:

```bash
rm -rf dist build .release-smoke
```

Build and check metadata:

```bash
uv run python -m build --no-isolation
uv run python -m twine check --strict dist/*
```

Inspect `dist/`. It must contain exactly the candidate wheel and source distribution. CI remains authoritative for the supported Python and operating-system matrix and repeats the build from a clean checkout.

## Merge and identify the candidate

After approval and green required checks:

1. Merge the release pull request.
2. Record the merge commit SHA.
3. Do not merge another change into the candidate. If `main` advances, create the tag from the recorded release commit rather than an unreviewed later commit.
4. Run TestPyPI against the release commit. The manual workflow checks out the selected branch or ref.

## TestPyPI preflight

Run **Publish To TestPyPI** from GitHub Actions against the candidate ref. The workflow rebuilds artifacts, runs the release gate, and publishes through the protected `testpypi` environment.

Create a fresh virtual environment and install AnyCode from TestPyPI while resolving dependencies from PyPI:

```bash
uv venv .release-smoke
uv pip install --python .release-smoke \
  --index https://test.pypi.org/simple/ \
  --default-index https://pypi.org/simple/ \
  "anycode-py[cli]==X.Y.Z"
```

On Windows, run:

```powershell
.release-smoke\Scripts\python -c "from anycode import AnyCode; print(AnyCode.__name__)"
.release-smoke\Scripts\anycode version
```

On POSIX systems, run:

```bash
.release-smoke/bin/python -c "from anycode import AnyCode; print(AnyCode.__name__)"
.release-smoke/bin/anycode version
```

Verify the reported version, core import, CLI command, wheel metadata, and expected optional dependencies. Do not create the production tag until this preflight passes.

## Create the immutable tag

Fetch the reviewed candidate commit and validate the intended tag locally:

```bash
git fetch origin
git switch --detach <release-commit-sha>
uv run python scripts/check_versions.py --tag vX.Y.Z
git tag -a vX.Y.Z -m "AnyCode X.Y.Z"
git push origin vX.Y.Z
```

For a pre-release, use matching PEP 440 metadata and tag, such as `1.0.0rc1` and `v1.0.0rc1`.

Never move or replace a published tag. If the wrong commit was tagged and no release or package was published, stop and coordinate correction among maintainers. Once any public release artifact exists, issue a new version.

## Publish the GitHub Release and PyPI package

Create a GitHub Release from the tag:

- Title: `AnyCode X.Y.Z`.
- Body: the edited `CHANGELOG.md` release section, including migration links.
- Mark pre-releases correctly.
- Confirm the target tag and commit before publishing.

Publishing the GitHub Release triggers `.github/workflows/publish-pypi.yml`. A tag alone does not publish to PyPI.

The workflow validates the release tag against `pyproject.toml`, runs the complete release gate, rebuilds fresh distributions, verifies expected filenames, transfers those exact artifacts to the publish job, waits for `pypi` environment approval, and uploads through Trusted Publishing with attestations.

Do not enable `skip-existing`. A duplicate package version must fail visibly.

## Post-release verification

The release manager verifies all external surfaces before declaring success:

- PyPI lists `X.Y.Z`, the expected wheel and source distribution, hashes, and attestations.
- A clean environment installs `anycode-py==X.Y.Z` from PyPI.
- `from anycode import AnyCode` and `anycode version` work outside the source tree.
- The GitHub Release tag resolves to the reviewed release commit.
- GitHub release notes match the changelog and migration guidance.
- The docs workflow publishes the `X.Y` version and updates the `latest` alias when appropriate.
- The documentation home, `/llms.txt`, compatibility page, release guide, and complete API inventory load under `latest`.

Check the package page after index propagation rather than assuming a green workflow means every user-facing surface is correct.

## Maintenance and hotfix releases

Release a compatible patch from `main` while `main` remains suitable. When `main` contains unreleased incompatible work:

1. Create `maint/X.Y` from the latest tag in the affected line.
2. Fix `main` first when the defect exists there.
3. Cherry-pick the reviewed fix into `backport/X.Y-description`.
4. Open a pull request against `maint/X.Y`.
5. Keep the backport minimal and compatible.
6. Run the full release gate and prepare the next patch version from the maintenance branch.
7. Tag the maintenance-branch release commit and follow the same TestPyPI and production publication steps.

Security patches follow [SECURITY.md](SECURITY.md) and may remain in a private advisory fork until coordinated publication.

## Failed publication and bad releases

### Failure before PyPI upload

Cancel or let the workflow fail. Fix the cause in a reviewed pull request. If no artifact has been published and no public release exists, maintainers may retry the same workflow from the unchanged tag after correcting only external configuration. A source change requires a new version and tag once the original release is public.

### Defective published release

- Never overwrite files or reuse the version.
- Yank an installable but unsafe or seriously broken PyPI version rather than deleting it.
- Add a warning to the affected GitHub Release that identifies impact and the safe replacement.
- Prepare and publish a corrected patch through the complete process.
- Add the incident and resolution to the changelog.

### Security release

Coordinate the package, GitHub Security Advisory, CVE request when applicable, changelog, upgrade guidance, and disclosure timing. Publish exploit details only after fixed artifacts are available unless active exploitation requires an earlier warning.

## Release completion record

Record these facts in the release issue, pull request, or GitHub Release discussion:

- Release version, tag, and commit SHA.
- Release manager and production approver.
- Required-check and TestPyPI workflow URLs.
- PyPI and documentation deployment workflow URLs.
- Post-release smoke-test result.
- Any recovery action or runbook correction.

Update this runbook after a failed release, manual recovery, workflow rename, publisher change, or newly discovered verification gap.
