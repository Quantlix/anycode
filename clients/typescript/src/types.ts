export const CONTRACT_VERSION = "1.0" as const;

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

export type RunState = "accepted" | "queued" | "running" | "waiting" | "succeeded" | "failed" | "canceled" | "rejected";
export type WaitingReason =
  | "dependency"
  | "schedule"
  | "input_required"
  | "authorization_required"
  | "approval_required"
  | "retry_backoff"
  | "capacity"
  | "external_signal";

export interface ContractErrorPayload {
  code: string;
  message: string;
  retryable: boolean;
  details?: JsonObject;
}

export interface Cancellation {
  status: "none" | "requested" | "acknowledged" | "lost_to_completion";
  requested_at?: string | null;
  acknowledged_at?: string | null;
  reason?: string | null;
}

export interface Run {
  schema_version: typeof CONTRACT_VERSION;
  id: string;
  state: RunState;
  root_task_id?: string | null;
  correlation_id: string;
  causation_id?: string | null;
  generation: number;
  attempt: number;
  waiting_reason?: WaitingReason | null;
  cancellation: Cancellation;
  error?: ContractErrorPayload | null;
  created_at: string;
  updated_at: string;
  last_event_sequence: number;
  metadata: JsonObject;
}

export interface Task {
  schema_version: typeof CONTRACT_VERSION;
  id: string;
  run_id: string;
  state: RunState;
  title: string;
  description: string;
  correlation_id: string;
  causation_id?: string | null;
  generation: number;
  attempt: number;
  dependencies: string[];
  waiting_reason?: WaitingReason | null;
  produced_artifact_ids: string[];
  allow_partial_dependency_artifacts: boolean;
  error?: ContractErrorPayload | null;
  created_at: string;
  updated_at: string;
  metadata: JsonObject;
}

export interface TextPart {
  type: "text";
  text: string;
}

export interface DataPart {
  type: "data";
  data: JsonValue;
}

export interface ArtifactPart {
  type: "artifact";
  artifact_id: string;
}

export type MessagePart = TextPart | DataPart | ArtifactPart;

export interface Message {
  schema_version: typeof CONTRACT_VERSION;
  id: string;
  run_id: string;
  task_id?: string | null;
  role: "system" | "user" | "agent" | "tool";
  parts: MessagePart[];
  correlation_id: string;
  causation_id?: string | null;
  generation: number;
  attempt: number;
  created_at: string;
  metadata: JsonObject;
}

export interface InlineArtifactContent {
  form: "inline";
  data: string;
  encoding: "utf-8" | "base64";
}

export interface ArtifactReference {
  form: "reference";
  uri: string;
  provider: string;
  expires_at?: string | null;
}

export interface Artifact {
  schema_version: typeof CONTRACT_VERSION;
  id: string;
  run_id: string;
  task_id?: string | null;
  name: string;
  media_type: string;
  size: number;
  digest: `sha256:${string}`;
  content: InlineArtifactContent | ArtifactReference;
  provenance: {
    producer: string;
    source_artifact_ids: string[];
    operation_key?: string | null;
    created_at: string;
  };
  classification: "public" | "internal" | "confidential" | "restricted";
  correlation_id: string;
  generation: number;
  attempt: number;
  finalized: boolean;
  created_at: string;
  metadata: JsonObject;
}

export interface BaseEvent<TType extends string, TPayload extends JsonObject> {
  schema_version: typeof CONTRACT_VERSION;
  payload_version: number;
  id: string;
  run_id: string;
  task_id?: string | null;
  sequence: number;
  type: TType;
  payload: TPayload;
  correlation_id: string;
  causation_id?: string | null;
  generation: number;
  attempt: number;
  emitted_at: string;
}

export type RunTransitionEvent = BaseEvent<
  "run.transitioned",
  { from: RunState; to: RunState; waiting_reason: WaitingReason | null; error: JsonValue; cancellation_status: string }
>;
export type TaskTransitionEvent = BaseEvent<
  "task.transitioned",
  { from: RunState; to: RunState; waiting_reason: WaitingReason | null; error: JsonValue }
>;
export type CancellationRequestedEvent = BaseEvent<"cancellation.requested", { reason: string; status: "requested" }>;
export type CancellationAcknowledgedEvent = BaseEvent<
  "cancellation.acknowledged",
  { from: RunState; to: "canceled"; status: "acknowledged" }
>;
export type MessageEvent = BaseEvent<"message.created", { message_id: string }>;
export type ArtifactEvent = BaseEvent<"artifact.finalized", { artifact_id: string; digest: string }>;

export type KnownEvent =
  | RunTransitionEvent
  | TaskTransitionEvent
  | CancellationRequestedEvent
  | CancellationAcknowledgedEvent
  | MessageEvent
  | ArtifactEvent;
export type SemanticEvent = KnownEvent | BaseEvent<string, JsonObject>;

export interface SubmitRequest {
  task: {
    title: string;
    description?: string;
    input?: JsonObject;
    metadata?: JsonObject;
  };
  correlation_id?: string;
  execution_context?: JsonObject;
}

export interface SubmitResult {
  run: Run;
  task: Task;
  duplicate: boolean;
}

export interface ListRunsRequest {
  cursor?: string;
  limit?: number;
  state?: RunState;
}

export interface Page<T> {
  items: T[];
  next_cursor?: string | null;
}

export interface ResumeRequest {
  checkpoint_id?: string;
  signal?: JsonValue;
}

export interface SendMessageRequest {
  role?: "user" | "agent";
  parts: MessagePart[];
  admission_key?: string;
}

export interface ArtifactEnvelope {
  artifact: Artifact;
  data_base64?: string | null;
}

export interface RequestOptions {
  signal?: AbortSignal;
}

export interface StreamOptions extends RequestOptions {
  after?: number;
  reconnect?: boolean;
  max_reconnects?: number;
}
