---
title: "AnyCode Maintainer Governance and Change Policy"
description: "Understand AnyCode maintainer roles, branch protection, change approval, compatibility, deprecation windows, backports, release ownership, and support policy."
keywords: AnyCode maintainers, open source governance, branch policy, semantic versioning, deprecation policy, backport policy
---

# Maintainer Governance and Change Policy

AnyCode maintainers keep `main` releasable, review public contracts before they change, and publish through protected automation. The repository-level [`MAINTAINERS.md`](https://github.com/Quantlix/anycode/blob/main/MAINTAINERS.md) is the authoritative internal policy; this page makes its operating model visible in the versioned site.

## Roles and responsibility

GitHub permissions are the source of truth for current access.

| Role | Responsibility |
| --- | --- |
| Triage maintainer | Reproduce reports, label issues, close duplicates, and identify release impact |
| Core maintainer | Review and merge changes, enforce architecture and compatibility policy, and manage repository settings |
| Release manager | Prepare the release pull request, run gates, create the tag and GitHub Release, and verify publication |
| Security maintainer | Coordinate private reports, fixes, advisories, CVEs, and disclosure |

Maintainers use least-privilege access and multifactor authentication. Access elevation, branch protection, trusted publishers, and protected release environments should receive review from another core maintainer when more than one is active.

## How decisions are made

Routine fixes and compatible enhancements use pull request review. The following changes need an issue or design discussion before implementation:

- New public subsystems and provider integrations.
- Public removals, renames, required parameters, or incompatible behavior.
- Declarative config and persisted schema changes.
- Dependencies with material install, licensing, security, or platform impact.
- Release, credential, branch-protection, or security-boundary changes.
- Architecture changes that cross established ownership boundaries.

Record alternatives, compatibility impact, operational cost, and the selected decision in the issue or pull request. Prefer a smaller reversible contract. If maintainers cannot reach consensus, the repository owner records the final decision and rationale.

## Branch policy

AnyCode uses trunk-based development with no permanent `develop` branch.

### `main`

`main` is the protected integration branch:

- Changes enter through pull requests during normal operation.
- Required code, test, package, and documentation checks pass before merge.
- Review conversations are resolved.
- Force pushes and deletion are disabled.
- Material changes after approval require fresh review.

Repository settings enforce these rules. Maintainers review the settings after workflow, ownership, or required-check changes.

### Topic branches

Topic branches start from the current `main`, cover one outcome, and are deleted after merge. Contributors may rewrite their own branch with `--force-with-lease`; public branches and tags remain immutable.

### Maintenance branches

Create `maint/X.Y` only when the project intends to release another patch from that line after `main` has moved to incompatible or unreleased work.

- Branch from the latest release tag in the line.
- Accept compatible fixes, security fixes, and documentation corrections only.
- Land an applicable fix on `main` first, then cherry-pick it in a `backport/X.Y-description` pull request.
- Run the complete release gate before tagging a maintenance release.
- Do not bulk-merge a maintenance branch back into `main`.

This keeps backport intent visible and prevents old implementation choices from replacing newer code.

## Public compatibility boundaries

Maintainers review five contracts independently:

1. Names exported by `anycode.__all__`.
2. Documented signatures, defaults, exceptions, and behavior.
3. CLI commands and output intended for automation.
4. Declarative YAML and TOML files.
5. Checkpoints, durable run records, transcripts, and turn checkpoints.

Internal modules and private names can change without a compatibility guarantee, but avoid churn that creates unnecessary downstream work. The [compatibility reference](../reference/compatibility.md) defines current reader and writer versions for persisted formats.

## Version policy

AnyCode follows Semantic Versioning. During `0.y.z` development:

| Version change | Contents |
| --- | --- |
| Patch component increment | Compatible bug, documentation, packaging, and security fixes |
| Minor component increment, patch reset to zero | Backward-compatible features and any explicitly approved, documented breaking changes |
| `1.0.0` | Declares the documented public contracts stable under standard SemVer major-version rules |

Pre-1.0 status permits a breaking change in a minor release. It does not permit silent breakage. Every incompatible change needs design approval, tests, a version decision, migration and rollback instructions, documentation, and a prominent changelog entry.

`project.version` in `pyproject.toml` is canonical. `scripts/check_versions.py` verifies the README, changelog release section and comparison links, lockfile, and optional release tag against it.

## Deprecation and removal

When a practical transition exists, keep deprecated behavior for at least one released minor version before removal. Longer windows are appropriate for heavily used APIs and stored data.

A deprecation must:

1. Emit `DeprecationWarning` from the caller's frame.
2. Name the replacement, first deprecated version, and earliest removal version.
3. Preserve behavior during the announced window.
4. Appear in reference docs, migration guidance, and the changelog.
5. Include warning and compatibility-path tests.

Immediate removal is reserved for a security issue, data-loss risk, legal constraint, or behavior that cannot safely coexist with its replacement. Release notes explain why a normal window was impossible.

## Review requirements

One core-maintainer approval is required for every merge. A second qualified approval is required, when another maintainer is available, for:

- Breaking public-contract changes.
- Persisted data, migration, security, authentication, release, or publishing changes.
- Branch-protection and required-check changes.
- Runtime dependencies with broad transitive or licensing impact.

When only one qualified maintainer is active, the pull request records the risk analysis and all automated gates must pass. Failed checks are not bypassed.

## Changelog and documentation ownership

`CHANGELOG.md` follows Keep a Changelog and describes observable user outcomes. User-visible pull requests update `[Unreleased]` under Added, Changed, Deprecated, Removed, Fixed, or Security. Breaking entries begin with **Breaking:** and link to migration guidance.

Documentation is part of the change. The author and reviewer verify affected docstrings, guides, concepts, examples, README content, configuration and CLI references, compatibility tables, `mkdocs.yml`, and `site_docs/llms.txt`. The generated [complete API inventory](../reference/api-inventory.md) and `scripts/check_docs.py` turn key source-linked claims into CI failures instead of review-time guesswork.

## Support and incident handling

The latest minor line receives compatible bug and security fixes. Critical fixes may be backported one minor line when the patch is safe and practical; older lines require an upgrade. A security advisory can define a narrower support decision for a specific vulnerability.

After a failed release, compatibility incident, security event, ownership change, or material CI change, maintainers update the policy and runbook with the lesson that changes future action. Policy changes use a reviewed pull request.

## Next steps

- [Development workflow](development.md)
- [Release process](releasing.md)
- [Compatibility and versioning](../reference/compatibility.md)
- [Security and threat model](../reference/security.md)
