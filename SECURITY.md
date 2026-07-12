# Security Policy

AnyCode coordinates model providers, local tools, external MCP servers, files, commands, and persisted workflow state. Security reports need a private path so maintainers can validate and fix an issue before operational details are public.

## Supported versions

AnyCode is pre-1.0 software. The latest published minor release is the supported line for security fixes.

| Version | Security support |
| --- | --- |
| Latest `0.Y.z` line | Supported |
| Previous minor line | Critical fixes may be backported when a safe, low-risk patch is practical |
| Older lines | Not supported |

Users should upgrade to the newest patch in the supported line. A GitHub Security Advisory may name a narrower affected range or a required minimum version.

## Report a vulnerability

Do not open a public issue, discussion, or pull request for a suspected vulnerability.

Use GitHub's private vulnerability reporting form:

<https://github.com/Quantlix/anycode/security/advisories/new>

Include:

- The affected version, commit, and configuration.
- The vulnerability class and realistic impact.
- Reproduction steps or a minimal proof of concept.
- Whether exploitation requires untrusted prompts, tool access, credentials, network access, or a specific backend.
- Any mitigation or patch you have tested.
- Your preferred name and disclosure credit, or a request to remain anonymous.

Remove real credentials, personal data, and third-party confidential data from the report. If private vulnerability reporting is unavailable, contact a repository owner through GitHub to request a private channel and do not include vulnerability details in that initial message.

## Response process

The response targets below guide maintainers but are not a service-level agreement:

1. Acknowledge a complete report within three business days.
2. Confirm the affected surface and initial severity within seven business days when reproduction is possible.
3. Keep the reporter informed when scope, severity, or release timing changes.
4. Develop the fix in the private advisory fork or another access-controlled branch.
5. Add a regression test that avoids publishing an immediately reusable exploit when disclosure is still embargoed.
6. Prepare the patched release, advisory, changelog entry, upgrade guidance, and any CVE request together.
7. Publish details after fixed packages are available, unless active exploitation or ecosystem coordination requires a different timeline.

Maintainers may request more information, lower the severity when required preconditions materially limit impact, or close reports that describe expected trusted-code behavior without crossing a documented security boundary.

## Security boundaries

The public [security and threat model](https://quantlix.github.io/anycode/latest/reference/security/) defines framework controls and operator responsibilities. In particular:

- Built-in policy is not an operating-system sandbox.
- Custom Python tools and plugins execute as trusted application code.
- Provider and MCP traffic needs application-owned network and identity controls.
- Credential redaction is defense in depth, not a data-loss-prevention system.
- Approval, idempotency, and verification controls must match the consequence of the action.

Reports are especially useful when they demonstrate a bypass of a documented boundary, including tool allowlists, path restrictions, MCP ownership, protected persistence, redaction, schema-version rejection, idempotency, or release integrity.

## Disclosure and release policy

- Security fixes use the smallest compatible patch release when practical.
- A fix may break compatibility only when preserving behavior would leave users exposed; release notes must provide migration instructions.
- Published versions and release tags are immutable. A defective security release is replaced with a new version.
- Vulnerable releases may be yanked from PyPI when continued installation is unsafe.
- Credit is offered to reporters who request it and follow coordinated disclosure.

Operational release steps are in [RELEASE.md](RELEASE.md), and maintainer responsibilities are in [MAINTAINERS.md](MAINTAINERS.md).
