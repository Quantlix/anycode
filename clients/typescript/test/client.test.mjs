import assert from "node:assert/strict";
import test from "node:test";

import { AnyCodeClient, ApiError, CompatibilityError, parseEventStream } from "../dist/index.js";

const headers = { "content-type": "application/json", "x-anycode-contract-version": "1.0" };

function jsonResponse(value, init = {}) {
  return new Response(JSON.stringify(value), { status: 200, headers, ...init });
}

test("submit sends auth, idempotency, and contract headers", async () => {
  let captured;
  const client = new AnyCodeClient({
    baseUrl: "https://service.example/",
    accessToken: async () => "token-value",
    fetch: async (url, init) => {
      captured = { url, init };
      return jsonResponse({ run: { id: "run-1" }, task: { id: "task-1" }, duplicate: false });
    },
  });

  const result = await client.submit({ task: { title: "demo" } }, "admit-1");

  assert.equal(result.run.id, "run-1");
  assert.equal(captured.url, "https://service.example/v1/runs");
  const sent = new Headers(captured.init.headers);
  assert.equal(sent.get("authorization"), "Bearer token-value");
  assert.equal(sent.get("idempotency-key"), "admit-1");
  assert.equal(sent.get("x-anycode-contract-version"), "1.0");
});

test("all lifecycle and artifact operations use the service schema", async () => {
  const calls = [];
  const client = new AnyCodeClient({
    baseUrl: "https://service.example",
    fetch: async (url, init) => {
      calls.push([url, init.method]);
      if (url.includes("/messages")) return jsonResponse({ id: "message-1" });
      if (url.endsWith("/artifacts")) return jsonResponse({ items: [] });
      if (url.includes("/artifacts/")) return jsonResponse({ artifact: { id: "artifact-1" } });
      if (url.includes("?cursor=")) return jsonResponse({ items: [], next_cursor: null });
      return jsonResponse({ id: "run-1" });
    },
  });

  await client.get("run-1");
  await client.list({ cursor: "next", limit: 10, state: "running" });
  await client.cancel("run-1", "operator");
  await client.resume("run-1", { checkpoint_id: "checkpoint-1" });
  await client.message("run-1", { parts: [{ type: "text", text: "continue" }] });
  await client.listArtifacts("run-1");
  await client.artifact("run-1", "artifact-1");

  assert.deepEqual(calls, [
    ["https://service.example/v1/runs/run-1", "GET"],
    ["https://service.example/v1/runs?cursor=next&limit=10&state=running", "GET"],
    ["https://service.example/v1/runs/run-1:cancel", "POST"],
    ["https://service.example/v1/runs/run-1:resume", "POST"],
    ["https://service.example/v1/runs/run-1/messages", "POST"],
    ["https://service.example/v1/runs/run-1/artifacts", "GET"],
    ["https://service.example/v1/runs/run-1/artifacts/artifact-1", "GET"],
  ]);
});

test("typed API and compatibility errors retain retry guidance", async () => {
  const api = new AnyCodeClient({
    baseUrl: "https://service.example",
    fetch: async () =>
      jsonResponse(
        { error: { code: "capacity", message: "busy", retryable: true, details: {} } },
        { status: 503, headers: { ...headers, "retry-after": "2", "x-request-id": "request-1" } },
      ),
  });
  await assert.rejects(api.get("run-1"), (error) => {
    assert(error instanceof ApiError);
    assert.equal(error.code, "capacity");
    assert.equal(error.retryAfterSeconds, 2);
    assert.equal(error.requestId, "request-1");
    return true;
  });

  const incompatible = new AnyCodeClient({
    baseUrl: "https://service.example",
    fetch: async () => jsonResponse({}, { headers: { "x-anycode-contract-version": "2.0" } }),
  });
  await assert.rejects(incompatible.get("run-1"), CompatibilityError);
});

test("SSE parser handles chunk boundaries and stream drops duplicate cursors", async () => {
  const encoder = new TextEncoder();
  const raw = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode("id: 1\ndata: {\"sequence\":1}\n"));
      controller.enqueue(encoder.encode("\nid: 2\ndata: {\"sequence\":2}\n\n"));
      controller.close();
    },
  });
  const parsed = [];
  for await (const event of parseEventStream(raw)) parsed.push(event);
  assert.deepEqual(parsed.map((event) => event.id), ["1", "2"]);

  const frames = [
    { sequence: 1, type: "run.accepted", payload: {} },
    { sequence: 2, type: "run.transitioned", payload: { to: "running" } },
    { sequence: 2, type: "run.transitioned", payload: { to: "running" } },
    { sequence: 3, type: "run.transitioned", payload: { to: "succeeded" } },
  ];
  const client = new AnyCodeClient({
    baseUrl: "https://service.example",
    fetch: async () =>
      new Response(frames.map((frame) => `id: ${frame.sequence}\ndata: ${JSON.stringify(frame)}\n\n`).join(""), {
        headers: { "content-type": "text/event-stream", "x-anycode-contract-version": "1.0" },
      }),
  });
  const sequences = [];
  for await (const event of client.stream("run-1")) sequences.push(event.sequence);
  assert.deepEqual(sequences, [1, 2, 3]);
});
