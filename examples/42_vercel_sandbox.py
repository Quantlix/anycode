# Demo 42 — Vercel Sandbox (live)
# Execute: uv run python examples/42_vercel_sandbox.py
#
# Runs the full sandbox lifecycle against a real Vercel Sandbox microVM:
#   create -> execute -> write_file -> read_file -> stream -> destroy
#
# Requires the sandbox-vercel extra and Vercel credentials in the environment:
#   uv add "anycode-py[sandbox-vercel]"
#   VERCEL_OIDC_TOKEN=...            (or VERCEL_TOKEN + VERCEL_TEAM_ID + VERCEL_PROJECT_ID)
# Skips gracefully when no credentials are configured.

import asyncio
import os

from dotenv import load_dotenv

from anycode import ExecutionContext, SandboxCommand, SandboxSpec, create_sandbox_provider

load_dotenv()

SEPARATOR = "-" * 60


def has_credentials() -> bool:
    return bool(os.environ.get("VERCEL_OIDC_TOKEN") or os.environ.get("VERCEL_TOKEN"))


async def main() -> None:
    print("=== Vercel Sandbox Live Demo ===\n")

    provider = create_sandbox_provider("vercel")
    caps = provider.capabilities()
    print(f"provider={caps.provider} isolation={caps.isolation} streaming={'live' if caps.command_streaming else 'buffered'}")

    if not has_credentials():
        print("\nskipped: set VERCEL_OIDC_TOKEN (or VERCEL_TOKEN + team/project ids) to run live")
        return

    # Vercel enforces no egress restrictions through the stable adapter, so the
    # spec must ask for what the backend can actually deliver: unrestricted.
    spec = SandboxSpec(
        run_id="run-vercel-demo",
        task_id="task-lifecycle",
        correlation_id="corr-vercel-demo",
        context=ExecutionContext(
            principal="user:examples",
            tenant_scope="tenant:examples",
            classification="internal",
        ),
        network="unrestricted",
        labels={"workload": "example-42"},
    )

    print(f"\n{SEPARATOR}\ncreating microVM...")
    created = await provider.create(spec)
    if not created.ok or created.handle is None:
        raise RuntimeError(created.error.message if created.error else "create failed")
    handle = created.handle
    print(f"sandbox id: {handle.id}")

    try:
        # --- execute: cwd + environment fold into one shell invocation ---
        result = await provider.execute(
            handle,
            SandboxCommand(
                command="uname -a && python3 --version && echo workload=$WORKLOAD",
                environment={"WORKLOAD": "example-42"},
                timeout_seconds=60,
            ),
        )
        print(f"\nexecute ok={result.ok} exit={result.exit_code}")
        print(f"stdout:\n{result.stdout.strip()}")
        if result.evidence:
            print(f"evidence: {result.evidence.digest[:24]}... exit_code={result.evidence.metadata['exit_code']}")

        # --- files: round-trip bytes through the sandbox filesystem ---
        payload = b"reviewed-by-anycode\n"
        write = await provider.write_file(handle, "/tmp/input.txt", payload)
        print(f"\nwrite_file ok={write.ok}")
        read = await provider.read_file(handle, "/tmp/input.txt")
        print(f"read_file  ok={read.ok} round_trip={read.data == payload}")

        # --- stream: buffered on vercel, but the contract shape is identical ---
        print("\nstream chunks:")
        async for chunk in provider.stream(handle, SandboxCommand(command="seq 1 3; echo boom >&2")):
            print(f"  [{chunk.sequence}] {chunk.stream}: {chunk.data.strip()}")
    finally:
        destroyed = await provider.destroy(handle)
        print(f"\ndestroy ok={destroyed.ok}")

    print(f"\n{SEPARATOR}\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
