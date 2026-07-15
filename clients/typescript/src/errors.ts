import type { ContractErrorPayload, JsonObject } from "./types.js";

export class AnyCodeError extends Error {
  readonly code: string;
  readonly retryable: boolean;
  readonly details: JsonObject;

  constructor(payload: ContractErrorPayload, options?: ErrorOptions) {
    super(payload.message, options);
    this.name = "AnyCodeError";
    this.code = payload.code;
    this.retryable = payload.retryable;
    this.details = payload.details ?? {};
  }
}

export class ApiError extends AnyCodeError {
  readonly status: number;
  readonly requestId?: string;
  readonly retryAfterSeconds?: number;

  constructor(
    status: number,
    payload: ContractErrorPayload,
    metadata: { requestId?: string; retryAfterSeconds?: number } = {},
    options?: ErrorOptions,
  ) {
    super(payload, options);
    this.name = "ApiError";
    this.status = status;
    if (metadata.requestId !== undefined) this.requestId = metadata.requestId;
    if (metadata.retryAfterSeconds !== undefined) this.retryAfterSeconds = metadata.retryAfterSeconds;
  }
}

export class CompatibilityError extends AnyCodeError {
  readonly serviceVersion: string;

  constructor(serviceVersion: string) {
    super({
      code: "unsupported_service_contract",
      message: `Service contract ${serviceVersion} is not supported by this client.`,
      retryable: false,
      details: { service_version: serviceVersion },
    });
    this.name = "CompatibilityError";
    this.serviceVersion = serviceVersion;
  }
}
