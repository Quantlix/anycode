---
title: "AnyCode TypeScript Service Client"
description: Use the AnyCode TypeScript service client from Node.js or browsers with lifecycle, artifact, cancellation, and resumable event-stream APIs for hosted runs.
keywords: anycode typescript, AnyCodeClient, Node agent client, browser agent client, resumable SSE
---

# AnyCode TypeScript service client

The TypeScript package in `clients/typescript` is a thin client for the versioned AnyCode service contract. It does not embed the Python runtime or expose Python objects. It supports Node.js 20+ and modern browsers through standard `fetch`, `ReadableStream`, and `AbortSignal` APIs, with no runtime dependencies.

## Operate a run

```typescript
import { AnyCodeClient } from "@quantlix/anycode-client";

const client = new AnyCodeClient({
  baseUrl: "https://agents.example.com",
  accessToken: async () => acquireShortLivedToken(),
});

const accepted = await client.submit({
  task: {
    title: "Review the release",
    description: "Check compatibility, migration notes, and rollback guidance.",
  },
}, crypto.randomUUID());

for await (const event of client.stream(accepted.run.id)) {
  console.log(event.type, event.sequence);
}

const artifacts = await client.listArtifacts(accepted.run.id);
```

The client also exposes `get`, `list`, `cancel`, `resume`, `message`, `subscribe`, and single-artifact retrieval. Service errors become `ApiError`; incompatible contract versions become `CompatibilityError`. Stream reconnection sends the last semantic cursor and suppresses duplicate events.

## Authentication and compatibility

Supply authorization per request with `getAccessToken` or a custom header callback. The SDK deliberately does not persist credentials. A browser deployment must permit the service origin and streaming headers through CORS; Node applications should keep credential acquisition server-side.

The package sends its supported contract version on every request and checks the server response. Run `npm test` in `clients/typescript` to build the strict TypeScript sources and execute lifecycle, error, version, and reconnect tests.

The SDK is a service client, not an in-process TypeScript agent runtime. Python remains the embedded-runtime surface.
