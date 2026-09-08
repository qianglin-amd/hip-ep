// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Thin client for the oga-server API. Streaming helpers here rather than in the views,
// so every view parses SSE the same way.

const BASE = "";

async function request(path, options = {}) {
  const response = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body.detail || body.error?.message || detail;
    } catch { /* non-JSON error body; keep the status line */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

export const api = {
  status: () => request("/v1/status"),
  hardware: () => request("/v1/hardware"),
  catalog: (params = {}) => {
    const query = new URLSearchParams(Object.entries(params).filter(([, v]) => v));
    return request(`/v1/catalog${query.toString() ? "?" + query : ""}`);
  },
  model: (id) => request(`/v1/catalog/${encodeURIComponent(id)}`),
  localModels: () => request("/v1/local"),
  deleteLocal: (id) => request(`/v1/local/${encodeURIComponent(id)}`, { method: "DELETE" }),
  pull: (model) => request("/v1/pull", { method: "POST", body: JSON.stringify({ model }) }),
  port: (spec) => request("/v1/port", { method: "POST", body: JSON.stringify(spec) }),
  portPreflight: (source) =>
    request("/v1/port/preflight", { method: "POST", body: JSON.stringify({ source }) }),
  recipes: () => request("/v1/recipes"),
  architectures: () => request("/v1/architectures"),
  jobs: (kind) => request(`/v1/jobs${kind ? "?kind=" + kind : ""}`),
  job: (id) => request(`/v1/jobs/${id}`),
  cancelJob: (id) => request(`/v1/jobs/${id}/cancel`, { method: "POST" }),
  published: () => request("/v1/benchmarks/published"),
  localBenchmarks: () => request("/v1/benchmarks/local"),
  runBenchmark: (spec) =>
    request("/v1/benchmarks/run", { method: "POST", body: JSON.stringify(spec) }),
};

/**
 * Stream a chat completion.
 *
 * Returns a handle with abort(), so the composer's stop button can cancel a
 * generation without waiting for the model to finish.
 */
export function streamChat({ model, messages, params = {}, onDelta, onDone, onError }) {
  const controller = new AbortController();

  (async () => {
    try {
      const response = await fetch(BASE + "/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({ model, messages, stream: true, ...params }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || body.error?.message || `HTTP ${response.status}`);
      }

      let usage = null;
      for await (const event of readSSE(response, controller.signal)) {
        if (event === "[DONE]") break;
        const chunk = JSON.parse(event);
        if (chunk.error) throw new Error(chunk.error.message || "generation failed");
        if (chunk.usage) usage = chunk.usage;
        const text = chunk.choices?.[0]?.delta?.content;
        if (text) onDelta?.(text);
      }
      onDone?.(usage);
    } catch (error) {
      if (error.name === "AbortError") onDone?.(null, true);
      else onError?.(error);
    }
  })();

  return { abort: () => controller.abort() };
}

/** Subscribe to a Server-Sent Events endpoint. Returns a handle with close(). */
export function subscribe(path, { onMessage, onError } = {}) {
  const controller = new AbortController();

  (async () => {
    try {
      const response = await fetch(BASE + path, { signal: controller.signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      for await (const event of readSSE(response, controller.signal)) {
        if (event === "[DONE]") break;
        try {
          onMessage?.(JSON.parse(event));
        } catch { /* a partially flushed frame; the next one supersedes it */ }
      }
    } catch (error) {
      if (error.name !== "AbortError") onError?.(error);
    }
  })();

  return { close: () => controller.abort() };
}

/** Decode a `text/event-stream` body into successive `data:` payloads. */
async function* readSSE(response, signal) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (!signal.aborted) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // Frames are separated by a blank line; anything after the last one is partial.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        const data = frame
          .split("\n")
          .filter((line) => line.startsWith("data: "))
          .map((line) => line.slice(6))
          .join("\n");
        if (data) yield data;
      }
    }
  } finally {
    reader.cancel().catch(() => {});
  }
}

/** Read a File into a base64 data URI for the chat image path. */
export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error(`could not read ${file.name}`));
    reader.readAsDataURL(file);
  });
}
