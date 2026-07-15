---
title: "AnyCode Compatibility and Versioning Policy"
description: "Understand AnyCode public API stability, declarative config versions, checkpoint compatibility, durable run formats, and safe upgrade or rollback procedures."
keywords: AnyCode compatibility, semantic versioning, public API stability, config format version, checkpoint migration, run store schema
---

# Compatibility and Versioning

AnyCode treats top-level Python imports, declarative config files, workflow checkpoints, and durable run artifacts as separate compatibility contracts. Each contract has an explicit boundary and upgrade rule so a newer writer cannot be mistaken for data an older runtime understands.

## Python API contract

Imports declared by `anycode.__all__` are the supported Python API. Prefer:

```python
from anycode import AgentConfig, AnyCode, ToolRegistry
```

Internal paths such as `anycode.core.runner` and private names beginning with `_` are implementation details. They can change without a compatibility guarantee.

The v0.6 top-level API is protected by an additive compatibility baseline in CI. Patch releases may add exports, optional parameters, model fields with defaults, and new enum or literal values. They do not remove a baseline export or make an existing call require a new argument.

Before 1.0, an intentional breaking API change requires a minor version bump, explicit maintainer approval, migration and rollback notes, and compatibility tests. After 1.0, it requires a major version bump. Pre-1.0 status permits documented iteration; it does not permit silent breakage.

When a practical transition exists, deprecated behavior remains available for at least one released minor version. It emits `DeprecationWarning`, names its replacement and earliest removal version, preserves behavior during the window, and has warning and compatibility-path tests. Immediate removal is reserved for security, data-loss, legal, or coexistence constraints and must be explained in the release notes.

!!! note "Typed models can grow"
    Result and configuration models may gain fields with defaults in a compatible release. Match fields by name, avoid positional construction, and ignore fields your application does not consume when serializing models across service boundaries.

## Declarative config contract

YAML and TOML team files use config format v1. Add the marker to newly maintained files:

```yaml
format_version: 1
name: production-team
agents:
  - name: worker
    model: gpt-4o-mini
    provider: openai
```

Files without `format_version` are treated as legacy v1 files. A runtime rejects versions outside its supported range with `UnsupportedConfigVersionError`. It also rejects unknown root, agent, task, and nested configuration fields with `UnknownConfigFieldError` instead of silently discarding a typo or a field written for a newer runtime.

Adding an optional field does not require a format bump. A bump is required when the same document could otherwise receive a different meaning, when a required field cannot be defaulted safely, or when a field is removed or renamed. AnyCode does not guess at those migrations.

## Persisted schema contract

| Artifact | Writes | Reads | Missing version | Future version |
| --- | --- | --- | --- | --- |
| Declarative YAML/TOML | v1 | v1 | Treated as v1 | `UnsupportedConfigVersionError` |
| Workflow checkpoint | v2 | v1-v2 | Treated as v1 | `UnsupportedCheckpointVersionError` |
| Durable run record, transcript, and turn checkpoint | v1 | v1 | Treated as v1 | `UnsupportedRunStoreVersionError` |
| Preview semantic JSON contract | 1.0 | 1.0 | Model default is 1.0; writers always emit it | Validation fails closed |

Workflow checkpoint v2 added lifecycle, verification, context, retry, and terminal outcome fields. The v1 reader path supplies safe defaults for those absent fields.

Durable run artifacts use one format version across run metadata, transcript events, and turn checkpoints. Protected payload envelopes have their own protector version; changing encryption or key metadata does not silently change the run schema.

The preview semantic contract has checked-in JSON Schemas and immutable history fixtures. Additive fields must have safe defaults within `1.0`; removing, renaming, or reinterpreting a field requires a new contract version. Run `uv run python scripts/generate_contract_schemas.py --check` during upgrade validation. See [semantic contracts](semantic-contracts.md) for the wire boundary.

## Upgrade procedure

1. Stop new work or route it away from the instance being upgraded.
2. Back up config, checkpoint, run-store, idempotency, and memory data.
3. Install the target AnyCode version in a staging environment.
4. Load representative configs and every persisted artifact type before resuming a run.
5. Run one deterministic workflow and verify terminal status, accounting, and telemetry.
6. Deploy readers before enabling any feature that writes a newer format.

If a reader raises an unsupported-version exception, stop and use a controlled migration. Do not edit the version number alone; the payload shape and semantics may differ.

## Rollback procedure

A code rollback is safe only while the older runtime can read every artifact written since the upgrade. If the writer format changed, restore the pre-upgrade data snapshot or run a tested downgrade migration first. Keep old binaries and data backups for the full rollback window.

## Release checklist

A release that changes a public or persisted contract must include:

- A compatible additive API update or the required semantic-version bump.
- An updated public API baseline when a new supported export is intentionally added.
- Reader tests for the oldest supported persisted format.
- A test that rejects the next unsupported format version.
- Migration and rollback notes for any incompatible schema change.
- Strict loading tests for new declarative config fields.

## Change approval

Public removals, required parameters, incompatible behavior, and persisted-format changes start with an issue or design discussion. The proposal identifies the affected users and data, additive alternatives, warning window, migration, rollback, and semantic version impact.

The [maintainer governance policy](../contributing/maintainers.md) defines approvals, branch handling, and backports. The [release process](../contributing/releasing.md) turns the approved compatibility decision into version metadata, release notes, artifact checks, and published documentation.
