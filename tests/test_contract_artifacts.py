import base64

from anycode.contracts import (
    ArtifactAccessContext,
    ArtifactStore,
    ArtifactWriteRequest,
    LocalArtifactStore,
)


def _request(artifact_id: str, data: bytes, *, inline: bool | None = None) -> ArtifactWriteRequest:
    return ArtifactWriteRequest(
        artifact_id=artifact_id,
        run_id="run-1",
        task_id="task-1",
        name=f"{artifact_id}.txt",
        media_type="text/plain",
        data=base64.b64encode(data).decode("ascii"),
        producer="test-worker",
        correlation_id="corr-1",
        inline=inline,
    )


async def test_local_store_round_trips_inline_and_reference_artifacts(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path, max_inline_bytes=4)
    context = ArtifactAccessContext(principal="tester")

    inline = await store.put(_request("small", b"tiny"), context)
    referenced = await store.put(_request("large", b"larger payload"), context)
    inline_read = await store.get("small", context)
    referenced_read = await store.get("large", context)

    assert isinstance(store, ArtifactStore)
    assert inline.ok and inline.artifact is not None and inline.artifact.content.form == "inline"
    assert referenced.ok and referenced.artifact is not None and referenced.artifact.content.form == "reference"
    assert inline_read.read_bytes() == b"tiny"
    assert referenced_read.read_bytes() == b"larger payload"


async def test_store_detects_tampering_and_artifact_id_conflicts(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path, max_inline_bytes=0)
    context = ArtifactAccessContext(principal="tester")
    written = await store.put(_request("artifact-1", b"original"), context)
    assert written.artifact is not None

    blob = tmp_path / "blobs" / written.artifact.digest.removeprefix("sha256:")
    blob.write_bytes(b"tampered")
    tampered = await store.get("artifact-1", context)
    conflict = await store.put(_request("artifact-1", b"different"), context)

    assert not tampered.ok and tampered.error is not None and tampered.error.code == "artifact_integrity_failed"
    assert not conflict.ok and conflict.error is not None and conflict.error.code == "artifact_conflict"


async def test_access_hook_denies_reads_without_leaking_metadata(tmp_path) -> None:
    def authorize(action, artifact, context) -> bool:  # type: ignore[no-untyped-def]
        return action == "write"

    store = LocalArtifactStore(tmp_path, access_hook=authorize)
    context = ArtifactAccessContext(principal="untrusted")
    assert (await store.put(_request("secret", b"classified"), context)).ok

    denied = await store.get("secret", context)
    assert not denied.ok
    assert denied.artifact is None
    assert denied.data_base64 is None
    assert denied.error is not None and denied.error.code == "artifact_access_denied"
