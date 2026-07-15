# AnyCode TypeScript client (preview)

This thin client operates the AnyCode service lifecycle with no Python-specific payloads and no runtime dependencies. It uses standard `fetch`, Web Streams, and `AbortSignal`, so the same ESM build supports current browsers and Node.js 20+.

```ts
import { AnyCodeClient } from "@quantlix/anycode-client";

const client = new AnyCodeClient({
  baseUrl: "https://agents.example.com",
  accessToken: () => identityProvider.getAccessToken(),
});

const submitted = await client.submit({ task: { title: "Research the incident" } }, crypto.randomUUID());
for await (const event of client.stream(submitted.run.id, { after: 0 })) {
  console.log(event.type, event.sequence);
}
```

Use short-lived user tokens in browsers. Never embed service credentials in a browser bundle. Node hosts may provide `accessToken` or asynchronous headers from their workload-identity provider. Streams reconnect from the last semantic sequence and discard duplicates. The client sends contract version `1.0` and fails explicitly if a service advertises an incompatible version.
