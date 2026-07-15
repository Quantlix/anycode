import { ApiError, CompatibilityError } from "./errors.js";
import { parseEventStream } from "./sse.js";
import {
  CONTRACT_VERSION,
  type Artifact,
  type ArtifactEnvelope,
  type ContractErrorPayload,
  type JsonObject,
  type ListRunsRequest,
  type Message,
  type Page,
  type RequestOptions,
  type ResumeRequest,
  type Run,
  type SemanticEvent,
  type SendMessageRequest,
  type StreamOptions,
  type SubmitRequest,
  type SubmitResult,
} from "./types.js";

const DEFAULT_MAX_RECONNECTS = 8;
const DEFAULT_RECONNECT_MS = 500;
const MAX_RECONNECT_MS = 30_000;

export interface AnyCodeClientOptions {
  baseUrl: string;
  fetch?: typeof globalThis.fetch;
  headers?: HeadersInit | (() => HeadersInit | Promise<HeadersInit>);
  accessToken?: string | (() => string | Promise<string>);
  userAgent?: string;
}

function encodePath(value: string): string {
  return encodeURIComponent(value);
}

function isTerminal(event: SemanticEvent): boolean {
  if (event.type !== "run.transitioned") return false;
  const target = event.payload["to"];
  return target === "succeeded" || target === "failed" || target === "canceled" || target === "rejected";
}

export class AnyCodeClient {
  readonly baseUrl: string;
  private readonly fetcher: typeof globalThis.fetch;
  private readonly configuredHeaders: AnyCodeClientOptions["headers"] | undefined;
  private readonly accessToken: AnyCodeClientOptions["accessToken"] | undefined;
  private readonly userAgent: string | undefined;

  constructor(options: AnyCodeClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.fetcher = options.fetch ?? globalThis.fetch;
    if (!this.fetcher) throw new Error("A standards-compatible fetch implementation is required.");
    this.configuredHeaders = options.headers;
    this.accessToken = options.accessToken;
    this.userAgent = options.userAgent;
  }

  async submit(request: SubmitRequest, admissionKey: string, options: RequestOptions = {}): Promise<SubmitResult> {
    if (!admissionKey) throw new TypeError("admissionKey must not be empty");
    return this.request<SubmitResult>("POST", "/v1/runs", request as unknown as JsonObject, options, {
      "idempotency-key": admissionKey,
    });
  }

  async get(runId: string, options: RequestOptions = {}): Promise<Run> {
    return this.request<Run>("GET", `/v1/runs/${encodePath(runId)}`, undefined, options);
  }

  async list(query: ListRunsRequest = {}, options: RequestOptions = {}): Promise<Page<Run>> {
    const parameters = new URLSearchParams();
    if (query.cursor) parameters.set("cursor", query.cursor);
    if (query.limit !== undefined) parameters.set("limit", String(query.limit));
    if (query.state) parameters.set("state", query.state);
    const suffix = parameters.size ? `?${parameters}` : "";
    return this.request<Page<Run>>("GET", `/v1/runs${suffix}`, undefined, options);
  }

  async cancel(runId: string, reason: string, options: RequestOptions = {}): Promise<Run> {
    return this.request<Run>("POST", `/v1/runs/${encodePath(runId)}:cancel`, { reason }, options);
  }

  async resume(runId: string, request: ResumeRequest = {}, options: RequestOptions = {}): Promise<Run> {
    return this.request<Run>("POST", `/v1/runs/${encodePath(runId)}:resume`, request as unknown as JsonObject, options);
  }

  async message(runId: string, request: SendMessageRequest, options: RequestOptions = {}): Promise<Message> {
    const extra = request.admission_key ? { "idempotency-key": request.admission_key } : undefined;
    return this.request<Message>(
      "POST",
      `/v1/runs/${encodePath(runId)}/messages`,
      request as unknown as JsonObject,
      options,
      extra,
    );
  }

  async listArtifacts(runId: string, options: RequestOptions = {}): Promise<Page<Artifact>> {
    return this.request<Page<Artifact>>("GET", `/v1/runs/${encodePath(runId)}/artifacts`, undefined, options);
  }

  async artifact(runId: string, artifactId: string, options: RequestOptions = {}): Promise<ArtifactEnvelope> {
    return this.request<ArtifactEnvelope>(
      "GET",
      `/v1/runs/${encodePath(runId)}/artifacts/${encodePath(artifactId)}`,
      undefined,
      options,
    );
  }

  async *stream(runId: string, options: StreamOptions = {}): AsyncGenerator<SemanticEvent> {
    let cursor = options.after ?? 0;
    let reconnectDelay = DEFAULT_RECONNECT_MS;
    let reconnects = 0;
    const reconnect = options.reconnect ?? true;
    const maxReconnects = options.max_reconnects ?? DEFAULT_MAX_RECONNECTS;
    while (true) {
      const response = await this.rawRequest(
        "GET",
        `/v1/runs/${encodePath(runId)}/events?after=${cursor}`,
        undefined,
        options,
        { accept: "text/event-stream", "last-event-id": String(cursor) },
      );
      if (!response.body) throw new ApiError(response.status, { code: "empty_stream", message: "Event stream has no body.", retryable: true });
      for await (const frame of parseEventStream(response.body)) {
        if (frame.retry !== undefined) reconnectDelay = Math.min(frame.retry, MAX_RECONNECT_MS);
        const event = JSON.parse(frame.data) as SemanticEvent;
        if (event.sequence <= cursor) continue;
        cursor = event.sequence;
        reconnects = 0;
        yield event;
        if (isTerminal(event)) return;
      }
      if (!reconnect || reconnects >= maxReconnects || options.signal?.aborted) return;
      reconnects += 1;
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(resolve, reconnectDelay);
        options.signal?.addEventListener(
          "abort",
          () => {
            clearTimeout(timeout);
            reject(options.signal?.reason ?? new DOMException("Aborted", "AbortError"));
          },
          { once: true },
        );
      });
      reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_MS);
    }
  }

  subscribe(runId: string, options: StreamOptions = {}): AsyncGenerator<SemanticEvent> {
    return this.stream(runId, options);
  }

  private async request<T>(
    method: string,
    path: string,
    body: JsonObject | undefined,
    options: RequestOptions,
    extraHeaders?: HeadersInit,
  ): Promise<T> {
    const response = await this.rawRequest(method, path, body, options, extraHeaders);
    if (response.status === 204) return undefined as T;
    return (await response.json()) as T;
  }

  private async rawRequest(
    method: string,
    path: string,
    body: JsonObject | undefined,
    options: RequestOptions,
    extraHeaders?: HeadersInit,
  ): Promise<Response> {
    const configured = typeof this.configuredHeaders === "function" ? await this.configuredHeaders() : this.configuredHeaders;
    const headers = new Headers(configured);
    headers.set("accept", headers.get("accept") ?? "application/json");
    headers.set("x-anycode-contract-version", CONTRACT_VERSION);
    if (body !== undefined) headers.set("content-type", "application/json");
    if (this.userAgent && typeof window === "undefined") headers.set("user-agent", this.userAgent);
    const token = typeof this.accessToken === "function" ? await this.accessToken() : this.accessToken;
    if (token) headers.set("authorization", `Bearer ${token}`);
    new Headers(extraHeaders).forEach((value, key) => headers.set(key, value));
    const request: RequestInit = { method, headers };
    if (body !== undefined) request.body = JSON.stringify(body);
    if (options.signal !== undefined) request.signal = options.signal;
    const response = await this.fetcher(`${this.baseUrl}${path}`, request);
    const serviceVersion = response.headers.get("x-anycode-contract-version");
    if (serviceVersion && serviceVersion !== CONTRACT_VERSION) throw new CompatibilityError(serviceVersion);
    if (!response.ok) throw await this.apiError(response);
    return response;
  }

  private async apiError(response: Response): Promise<ApiError> {
    let payload: ContractErrorPayload;
    try {
      const parsed = (await response.json()) as ContractErrorPayload | { error: ContractErrorPayload };
      payload = "error" in parsed ? parsed.error : parsed;
    } catch {
      payload = { code: "http_error", message: response.statusText || `HTTP ${response.status}`, retryable: response.status >= 500 };
    }
    const retryAfter = response.headers.get("retry-after");
    return new ApiError(response.status, payload, {
      ...(response.headers.get("x-request-id") ? { requestId: response.headers.get("x-request-id")! } : {}),
      ...(retryAfter && /^\d+(\.\d+)?$/.test(retryAfter) ? { retryAfterSeconds: Number(retryAfter) } : {}),
    });
  }
}
