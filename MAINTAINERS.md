# AnyCode Maintainer Handbook

This handbook defines how maintainers govern, change, and release AnyCode. It is the internal operating policy for repository collaborators with triage, write, or release access. Public contribution instructions belong in `CONTRIBUTING.md`; exact publishing commands belong in `RELEASE.md`.

The words **must**, **must not**, **should**, and **may** describe required, prohibited, recommended, and optional practices.

## Operating principles

1. `main` stays releasable. A merged change has passed its required checks, includes its tests, and carries the documentation and changelog updates needed to describe it.
2. Public contracts change deliberately. Top-level imports, configuration files, persisted artifacts, CLI behavior, and documented defaults are reviewed for compatibility before merge.
3. Releases are reproducible. Tagged source, built distributions, release notes, and published documentation identify the same version.
4. Publishing uses reviewed automation. Maintainers do not upload production artifacts from a workstation while Trusted Publishing is available.
5. Security reports remain private until a coordinated fix and disclosure are ready.

## Roles and authority

GitHub repository permissions are the source of truth for current access. AnyCode currently uses these functional roles:

| Role | Responsibilities | Required access |
| --- | --- | --- |
| Contributor | Propose focused changes and respond to review | Read |
| Triage maintainer | Reproduce reports, label issues, close duplicates, and identify release impact | Triage |
| Core maintainer | Review and merge changes, enforce compatibility policy, and manage project settings | Write or maintain |
| Release manager | Prepare a release PR, run release gates, create the tag and GitHub Release, approve publication, and verify the result | Maintain or admin plus protected environment approval |
| Security maintainer | Receive private reports, coordinate fixes and advisories, and approve disclosure | Security advisory access |

A maintainer must not approve their own access elevation. Changes to branch protection, trusted publishers, protected environments, or repository ownership require approval from another core maintainer when more than one is active.

## Decision making

Routine fixes and compatible enhancements use pull request review. The reviewing core maintainer decides whether the change meets the documented contract and quality bar.

The following changes require an issue or design discussion before implementation and explicit core-maintainer approval in the pull request:

- A new public subsystem or provider integration.
- A removal, rename, or behavior change in a public contract.
- A persisted schema or declarative configuration format change.
- A dependency that materially changes install size, licensing, security exposure, or supported platforms.
- A change to release automation, credentials, branch protection, or security boundaries.
- A substantial architecture change that crosses existing module ownership boundaries.

When maintainers disagree, record the options, compatibility impact, operational cost, and decision in the issue or pull request. Prefer the smallest reversible choice. If consensus is unavailable, the repository owner makes the final decision and records the rationale.

## Branch model

AnyCode uses a trunk-based workflow. There is no permanent `develop` branch.

### Protected branch

`main` is the integration branch and must be protected in GitHub:

- Changes enter through pull requests, except an emergency repository-recovery action by an administrator.
- Required CI and documentation checks must pass before merge.
- Conversations must be resolved before merge.
- Force pushes and branch deletion must be disabled.
- Stale approvals should be dismissed when code changes materially after review.
- Administrators should follow the same protections during normal operation.

The repository settings, not this file, enforce protection. Maintainers review those settings after workflow or ownership changes.

### Topic branches

Create one short-lived branch per coherent change from the current `origin/main`:

```bash
git fetch origin
git switch --create feat/short-description origin/main
```

Use lowercase, hyphenated names with one of these prefixes:

| Prefix | Use |
| --- | --- |
| `feat/` | Backward-compatible user-facing capability |
| `fix/` | Bug fix |
| `docs/` | Documentation-only change |
| `refactor/` | Internal behavior-preserving change |
| `test/` | Test or harness change |
| `chore/` | Build, dependency, CI, or repository maintenance |
| `release/` | Release preparation |
| `backport/` | A reviewed change applied to a maintained release line |
| `security/` | Private security-fix branch only |

Delete topic branches after merge. History may be rewritten on a contributor-owned topic branch with `git push --force-with-lease`; never rewrite `main`, release tags, or shared maintenance branches.

### Maintenance branches

Do not create a maintenance branch for every release. Create `maint/X.Y` only when the project commits to ship another patch for that release line while `main` has moved on to incompatible or unreleased work.

- Branch `maint/X.Y` from tag `vX.Y.0` or the latest patch tag in that line.
- Accept compatible bug fixes, security fixes, and documentation corrections only.
- Never merge the maintenance branch back into `main` as a bulk merge.
- Land the fix on `main` first when it applies there, then cherry-pick it in a dedicated `backport/X.Y-description` pull request.
- Tag patch releases from the maintenance branch and run the same release gate used for `main`.
- Do not force-push or delete a maintenance branch while its release line is supported.

## Development workflow

### Before implementation

1. Confirm the problem, intended behavior, and affected public contracts.
2. Search for the owning implementation, nearby tests, and existing documentation.
3. For design-sensitive work, obtain agreement in an issue before investing in implementation.
4. Choose the version impact and changelog category early: added, changed, deprecated, removed, fixed, or security.

### During implementation

- Keep each pull request focused on one outcome.
- Match the architecture and public import boundaries in `AGENTS.md`.
- Add or update tests that fail without the behavior change.
- Update docstrings when signatures, defaults, exceptions, or semantics change.
- Update public documentation in the same pull request as user-visible behavior.
- Never commit secrets, generated `site/` output, build artifacts, local evidence, or environment-specific files.
- Use Conventional Commit subjects: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, or `chore:`.

Commits should be understandable without reading the diff. Review branches may contain incremental commits, but the final merged commit must have a clear subject and a coherent change description. Squash merge is the default for ordinary pull requests.

### Local validation

Run the smallest relevant test while iterating. Before requesting final review, run the full repository gate:

```bash
uv sync --group dev
uv run python -m ruff check .
uv run python -m ruff format --check src/
uv run python -m pyright
uv run python -m pytest
uv run python -m mkdocs build --strict
```

Packaging, dependency, CLI, or release changes must also pass:

```bash
uv run python scripts/check_versions.py
uv run python -m build --no-isolation
uv run python -m twine check --strict dist/*
```

CI remains authoritative across the supported operating-system and Python-version matrix.

## Pull request policy

Every pull request must explain:

- The user or maintainer problem being solved.
- The chosen implementation and any meaningful tradeoff.
- The compatibility and versioning impact.
- The tests and commands used to verify the change.
- The documentation and changelog impact.
- Any rollout, migration, rollback, security, or persistence concern.

Draft pull requests may be incomplete. A pull request marked ready for review must be scoped, documented, tested, and free of unrelated changes.

### Review requirements

At least one core maintainer must approve a change before merge. Require a second qualified approval when a change:

- Intentionally breaks a public contract.
- Alters persisted data, migration, security, authentication, release, or publishing behavior.
- Changes branch protection or required CI policy.
- Adds a runtime dependency with a broad transitive or licensing impact.

When only one qualified maintainer is active, that maintainer must document the risk analysis in the pull request and rely on all automated gates. They must not bypass failed checks.

Reviewers verify behavior, tests, public API boundaries, failure modes, typing, documentation, changelog classification, and release impact. Approval is for the current material diff; request a fresh review after a substantial rewrite.

### Merge rules

- Prefer squash merge for a linear, readable `main` history.
- Use rebase merge only when the individual commits are intentionally curated and independently useful.
- Do not merge with failing or cancelled required checks.
- Do not use merge commits for ordinary topic branches.
- Delete the source branch after merge unless it is a protected maintenance branch.

## Change and compatibility management

The supported Python API is the set of names in `anycode.__all__`. Documented CLI behavior, accepted configuration, and readable persisted formats are also public contracts. Internal modules and private names may change without a compatibility guarantee, but maintainers should still avoid needless churn.

### Change classification

| Change | Compatible release | Required evidence |
| --- | --- | --- |
| Bug fix that restores documented behavior | Patch | Regression test and changelog entry when users are affected |
| Optional parameter or model field with a safe default | Patch before 1.0; minor after 1.0 when materially user-facing | Tests, API docs, and serialization review |
| New public feature or export | Minor | Tests, docs, changelog, and public API baseline review |
| Deprecation | Minor | Runtime warning, replacement, migration note, and planned removal version |
| Public removal, required parameter, renamed field, or incompatible behavior | Minor before 1.0; major after 1.0 | Approved design, migration and rollback notes, tests, docs, and prominent changelog entry |
| Persisted format incompatibility | Minor before 1.0; major after 1.0 | New format version, old-reader policy, migration tooling or procedure, fixtures, and rollback plan |
| Security fix | Patch when practical | Private advisory workflow, regression test that does not expose the vulnerability before disclosure |

Pre-1.0 status permits breaking changes in a minor release, but it does not make silent breakage acceptable.

### Deprecation policy

A practical deprecation must remain available for at least one released minor version before removal. Longer windows are preferred for heavily used APIs or persisted formats.

A deprecation must:

1. Emit `DeprecationWarning` from the caller's stack frame.
2. Name the deprecated behavior, replacement, version first deprecated, and earliest removal version.
3. Preserve existing behavior throughout the announced window.
4. Appear in API docs, migration guidance, and the changelog.
5. Include tests for both the warning and the compatibility path.

Immediate removal requires explicit approval and is reserved for a security issue, data-loss risk, legally unusable behavior, or an implementation that cannot safely coexist with its replacement. The release notes must explain why the ordinary window was impossible.

### Breaking-change review

The pull request author must include a compatibility statement answering:

1. Which public contract changes?
2. Which users, files, or stored data are affected?
3. Can an additive API or compatibility shim avoid the break?
4. What warning and migration path is available?
5. What is the rollback procedure?
6. Which version increment communicates the change?

The reviewer must verify these claims against tests and documentation, not only the implementation diff.

## Version management

AnyCode follows Semantic Versioning. While the project is in `0.y.z` development:

- Increment the patch component for compatible bug, documentation, packaging, and security fixes.
- Increment the minor component and reset the patch component to zero for backward-compatible features or an explicitly documented breaking change.
- `1.0.0` declares the documented public contracts stable enough for standard SemVer major-version guarantees.

`project.version` in `pyproject.toml` is the single source of truth. The release preparation change must synchronize:

- `pyproject.toml`
- `uv.lock`
- The current-version statement in `README.md`
- The dated section and comparison links in `CHANGELOG.md`
- The Git tag `vX.Y.Z`

Run `uv run python scripts/check_versions.py` to verify those surfaces. Never edit a published distribution or move a release tag. Correct a bad release with a new version; yank the affected PyPI version when installation should be discouraged.

Pre-release identifiers use PEP 440 forms in package metadata, such as `1.0.0rc1`. The corresponding Git tag is `v1.0.0rc1`.

## Changelog policy

`CHANGELOG.md` follows Keep a Changelog and is written for users. Every pull request with a user-visible effect updates `[Unreleased]` unless the pull request is explicitly labeled or documented as not requiring a changelog entry.

- **Added** for new capabilities.
- **Changed** for compatible behavior changes.
- **Deprecated** for supported behavior scheduled for removal.
- **Removed** for removed behavior.
- **Fixed** for bug fixes.
- **Security** for disclosed vulnerability fixes.

Entries state the observable outcome and affected API. Avoid commit-log language, internal filenames, and claims that are not verified. Breaking changes begin with **Breaking:** and link to migration instructions.

## Release management

The release manager follows `RELEASE.md`. A normal release uses a reviewed `release/X.Y.Z` pull request and separates preparation from publication:

1. Confirm the intended scope and semantic version.
2. Freeze the candidate to fixes for release blockers, documentation, metadata, and release automation.
3. Synchronize version surfaces and editorialize `[Unreleased]` into dated release notes.
4. Run the complete quality, test, docs, version, build, metadata, and installed-wheel checks.
5. Merge the approved release pull request.
6. Publish and smoke-test on TestPyPI.
7. Create the immutable tag from the verified release commit.
8. Publish the GitHub Release, approve the protected PyPI environment, and let Trusted Publishing rebuild and upload artifacts.
9. Verify PyPI metadata, hashes, attestations, installation, CLI behavior, GitHub release notes, and versioned documentation.
10. Record any procedural correction in `RELEASE.md` before the next release.

The person approving the protected production environment should be different from the person who prepared the release when another release maintainer is available.

### Hotfixes and bad releases

If `main` is safe for the patch, prepare the hotfix there. If `main` contains unreleased incompatible work, branch from the affected release tag into `maint/X.Y`, apply the minimal fix, and publish the next patch from that branch. Forward-port the fix to `main` in a separate pull request.

If a release is defective:

- Stop or cancel publication if artifacts have not reached PyPI.
- Never reuse the version after any artifact has been published.
- Yank, rather than delete, an installable but unsafe or seriously broken PyPI release.
- Publish a corrected patch as soon as its full gate passes.
- Add a changelog notice and GitHub Release warning that identify impact and the safe replacement.
- Use a GitHub Security Advisory for vulnerabilities and coordinate disclosure timing.

## Documentation ownership

Documentation is part of the definition of done. The code owner and reviewer share responsibility for keeping it accurate.

When behavior changes, update all affected surfaces in the same pull request:

- Public docstrings and `site_docs/reference/public-api.md` coverage.
- Task guidance under `site_docs/guides/`.
- Concepts when architecture or guarantees change.
- Runnable examples under `examples/`.
- `README.md` when installation, positioning, support, or the first-use path changes.
- `site_docs/llms.txt` and `mkdocs.yml` when important pages move or are added.
- Compatibility, security, configuration, CLI, and release references when their contracts change.

Examples and prose must use public imports, current signatures, supported model identifiers, and real defaults from source. Prefer generated API reference for signatures and hand-written guides for workflows. Run the strict docs build on every documentation change.

Before each release, the release manager checks documentation against the release diff, confirms every navigation target exists, and verifies the published `latest` alias points to the released minor version.

## Security maintenance

Security reports use the private process in `SECURITY.md`. Do not discuss an undisclosed vulnerability in public issues, pull requests, branches, CI logs, or changelog entries.

Grant repository and publishing permissions by least privilege. Require multifactor authentication for maintainers with write or release access. Review GitHub Actions permissions when workflow dependencies change, and keep `id-token: write` limited to Trusted Publishing jobs.

## Policy maintenance

Review this handbook and `RELEASE.md` after a failed release, compatibility incident, security event, ownership change, or material CI change. Otherwise review them before the first minor release of each calendar year.

Policy changes use a pull request and the same review rules as code. A policy must describe current practice or include the automation needed to enforce the new practice.

## Practice sources

This policy adapts established practices to AnyCode's size and Python packaging model:

- [Django's Git workflow](https://docs.djangoproject.com/en/dev/internals/contributing/writing-code/working-with-git/) distinguishes rewritable topic branches from immutable public branches and requires tests and warning-free docs.
- [Django's release process](https://docs.djangoproject.com/en/dev/internals/release-process/) separates feature development from stable patch lines and gives deprecations a defined release window.
- [NumPy's development workflow](https://numpy.org/doc/stable/dev/development_workflow.html) uses one focused branch per change, explicit backport pull requests, and user-facing release notes for API changes.
- [NumPy's release guide](https://numpy.org/doc/stable/dev/releasing.html) treats release preparation, artifact validation, documentation, and post-release verification as separate responsibilities.
- [Ruff's contributing guide](https://docs.astral.sh/ruff/contributing/) requires design agreement for high-cost features, a written test plan, generated-documentation updates, and reviewed automated publishing.
- [Semantic Versioning 2.0.0](https://semver.org/) defines version intent and requires a declared public API.
- [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) provides the user-centered changelog structure.

These sources inform the policy; this handbook and the repository's enforced checks govern AnyCode.
