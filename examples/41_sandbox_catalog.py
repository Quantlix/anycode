# Demo 41 — Sandbox Provider Catalog
# Execute: uv run python examples/41_sandbox_catalog.py
#
# Demonstrates the expanded sandbox provider catalog behind one protocol:
#   1. create_sandbox_provider(name) builds any of the six remote backends
#      (daytona, e2b, modal, runloop, vercel, langsmith) without importing
#      the SDK until first use
#   2. capabilities() reports each backend's real guarantees — isolation,
#      networking, snapshots, streaming — plus a limitations tuple
#   3. Fail-closed guards: unsupported network modes, foreign secret-reference
#      prefixes, and snapshot restores are rejected with typed errors before
#      any sandbox is created
#   4. health() degrades to "unavailable" when a provider's SDK is missing
#
# Runs fully offline: no credentials and no optional SDKs are required.

import asyncio

from anycode import ExecutionContext, SandboxSpec, create_sandbox_provider
from anycode.sandbox import SANDBOX_PROVIDER_EXTRAS

SEPARATOR = "-" * 72

CONTEXT = ExecutionContext(
    principal="user:demo",
    tenant_scope="tenant:examples",
    classification="internal",
)


def make_spec(**overrides) -> SandboxSpec:
    base = {
        "run_id": "run-catalog",
        "correlation_id": "corr-catalog",
        "context": CONTEXT,
        "network": "unrestricted",
    }
    base.update(overrides)
    return SandboxSpec(**base)


async def main() -> None:
    print("=== Sandbox Provider Catalog Demo ===\n")

    # --- Section A: the catalog and its capability reports ---
    print(SEPARATOR)
    print("Section A: create_sandbox_provider + capabilities\n")
    header = f"  {'provider':<10} {'extra':<18} {'isolation':<10} {'network':<13} {'snapshots':<10} streaming"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, extra in sorted(SANDBOX_PROVIDER_EXTRAS.items()):
        provider = create_sandbox_provider(name)  # cheap; SDK loads lazily on first use
        caps = provider.capabilities()
        streaming = "live" if caps.command_streaming else "buffered"
        print(f"  {caps.provider:<10} {extra:<18} {caps.isolation:<10} {caps.networking:<13} {str(caps.snapshots):<10} {streaming}")

    print("\n  limitations are part of the contract, e.g. vercel reports:")
    for line in create_sandbox_provider("vercel").capabilities().limitations[:3]:
        print(f"    - {line}")

    # --- Section B: fail-closed guards run before any SDK or network call ---
    print(f"\n{SEPARATOR}")
    print("Section B: fail-closed guards\n")

    def error_code(result) -> str:
        return result.error.code if result.error else "<none>"

    # 1. Network mode the backend cannot enforce -> typed error, no sandbox.
    vercel = create_sandbox_provider("vercel")
    result = await vercel.create(make_spec(network="none"))
    print(f"  vercel + network='none'      -> ok={result.ok} code={error_code(result)}")

    # 2. Secret reference scoped to another provider -> rejected at create time.
    modal = create_sandbox_provider("modal")
    result = await modal.create(make_spec(secret_references={"TOKEN": "daytona:package-token"}))
    print(f"  modal + 'daytona:...' secret -> ok={result.ok} code={error_code(result)}")

    # 3. Snapshot restore on a backend without snapshot support.
    e2b = create_sandbox_provider("e2b")
    result = await e2b.create(make_spec(snapshot="snap-1234"))
    print(f"  e2b + snapshot restore       -> ok={result.ok} code={error_code(result)}")

    # 4. Plaintext secrets never make it into a spec at all.
    try:
        make_spec(secret_references={"TOKEN": "sk-plaintext-value"})
    except ValueError:
        print("  plaintext secret value       -> rejected by SandboxSpec validation")

    # 5. Unknown provider names fail loudly with the available catalog.
    try:
        create_sandbox_provider("firecracker")
    except ValueError as error:
        print(f"  unknown provider             -> {error}")

    # --- Section C: health reflects SDK availability ---
    print(f"\n{SEPARATOR}")
    print("Section C: health checks\n")
    for name in sorted(SANDBOX_PROVIDER_EXTRAS):
        health = await create_sandbox_provider(name).health()
        hint = "" if health.status == "healthy" else f'  (pip install "anycode-py[{SANDBOX_PROVIDER_EXTRAS[name]}]")'
        print(f"  {name:<10} {health.status}{hint}")

    print(f"\n{SEPARATOR}\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
