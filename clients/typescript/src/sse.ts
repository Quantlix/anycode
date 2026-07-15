export interface ServerSentEvent {
  id?: string;
  event?: string;
  data: string;
  retry?: number;
}

function decodeEvent(lines: string[]): ServerSentEvent | undefined {
  let id: string | undefined;
  let event: string | undefined;
  let retry: number | undefined;
  const data: string[] = [];
  for (const line of lines) {
    if (!line || line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    const field = separator < 0 ? line : line.slice(0, separator);
    let value = separator < 0 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "id") id = value;
    else if (field === "event") event = value;
    else if (field === "data") data.push(value);
    else if (field === "retry" && /^\d+$/.test(value)) retry = Number(value);
  }
  if (data.length === 0) return undefined;
  const decoded: ServerSentEvent = { data: data.join("\n") };
  if (id !== undefined) decoded.id = id;
  if (event !== undefined) decoded.event = event;
  if (retry !== undefined) decoded.retry = retry;
  return decoded;
}

export async function* parseEventStream(stream: ReadableStream<Uint8Array>): AsyncGenerator<ServerSentEvent> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done }).replaceAll("\r\n", "\n").replaceAll("\r", "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        const decoded = decodeEvent(buffer.slice(0, boundary).split("\n"));
        buffer = buffer.slice(boundary + 2);
        if (decoded) yield decoded;
        boundary = buffer.indexOf("\n\n");
      }
      if (done) break;
    }
    if (buffer.trim()) {
      const decoded = decodeEvent(buffer.split("\n"));
      if (decoded) yield decoded;
    }
  } finally {
    reader.releaseLock();
  }
}
