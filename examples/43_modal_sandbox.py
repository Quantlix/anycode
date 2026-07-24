# Demo 43 — Modal Sandbox (live)
# Execute: uv run python examples/43_modal_sandbox.py
#
# Runs the sandbox lifecycle against real Modal gVisor sandboxes, including the
# two capabilities Modal adds over most catalog peers:
#   1. Enforced network isolation: network='none' maps to block_network, and
#      network='allowlist' takes CIDR ranges (domain allowlists fail closed)
#   2. Filesystem snapshots: snapshot() captures the filesystem as a Modal
#      image reference you can store as provenance
#
# Requires the sandbox-modal extra and Modal credentials (`modal token new`,
# stored in ~/.modal.toml or MODAL_TOKEN_ID/MODAL_TOKEN_SECRET).
# Skips gracefully when no credentials are configured.

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from anycode import ExecutionContext, SandboxCommand, SandboxSpec, create_sandbox_provider

load_dotenv()

SEPARATOR = "-" * 60


def has_credentials() -> bool:
    return bool(os.environ.get("MODAL_TOKEN_ID")) or (Path.home() / ".modal.toml").exists()


async def main() -> None:
    print("=== Modal Sandbox Live Demo ===\n")

    provider = create_sandbox_provider("modal")
    caps = provider.capabilities()
    print(f"provider={caps.provider} isolation={caps.isolation} snapshots={caps.snapshots}")

    if not has_credentials():
        print("\nskipped: run `modal token new` (or set MODAL_TOKEN_ID/MODAL_TOKEN_SECRET) to run live")
        return

    # network='none' is enforced for real here: Modal blocks all egress.
    spec = SandboxSpec(
        run_id="run-modal-demo",
        task_id="task-lifecycle",
        correlation_id="corr-modal-demo",
        context=ExecutionContext(
            principal="user:examples",
            tenant_scope="tenant:examples",
            classification="internal",
        ),
        network="none",
        labels={"workload": "example-43"},
    )

    print(f"\n{SEPARATOR}\ncreating gVisor sandbox (network blocked)...")
    created = await provider.create(spec)
    if not created.ok or created.handle is None:
        raise RuntimeError(created.error.message if created.error else "create failed")
    handle = created.handle
    print(f"sandbox id: {handle.id}")

    try:
        # --- prove the network policy is real, not advisory ---
        offline = await provider.execute(
            handle,
            SandboxCommand(command="python3 -c 'import urllib.request; urllib.request.urlopen(\"https://pypi.org\", timeout=5)'", timeout_seconds=30),
        )
        print(f"\negress attempt under network='none': ok={offline.ok} exit={offline.exit_code} (failure expected)")

        # --- normal command execution still works ---
        result = await provider.execute(
            handle,
            SandboxCommand(command="python3 --version && echo stage=$STAGE", environment={"STAGE": "example-43"}, timeout_seconds=60),
        )
        print(f"execute ok={result.ok} exit={result.exit_code}")
        print(f"stdout:\n{result.stdout.strip()}")

        # --- files round-trip through sandbox.open ---
        payload = b"state-worth-keeping\n"
        write = await provider.write_file(handle, "/tmp/state.txt", payload)
        read = await provider.read_file(handle, "/tmp/state.txt")
        print(f"\nwrite_file ok={write.ok}  read_file ok={read.ok} round_trip={read.data == payload}")

        # --- snapshot: filesystem state becomes a Modal image reference ---
        snap = await provider.snapshot(handle, "example-43-checkpoint")
        if snap.ok:
            print(f"snapshot ok reference={snap.reference}")
        else:
            print(f"snapshot failed: {snap.error.code if snap.error else 'unknown'}")
    finally:
        destroyed = await provider.destroy(handle)
        print(f"\ndestroy ok={destroyed.ok}")

    print(f"\n{SEPARATOR}\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
