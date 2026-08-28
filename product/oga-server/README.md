<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# oga-server

An OpenAI- and Ollama-compatible inference server for AMD GPUs, backed by HIP EP
through ONNX Runtime GenAI.

```bash
python product/serve.py           # http://127.0.0.1:11434
```

The default port is 11434, which is Ollama's, so existing Ollama clients work with no
configuration at all.

## API surface

### OpenAI

| Endpoint | Notes |
|---|---|
| `GET /v1/models` | Certified models in OpenAI list shape |
| `GET /v1/models/{id}` | One model |
| `POST /v1/chat/completions` | Streaming and non-streaming; multimodal content parts |
| `POST /v1/completions` | Legacy text completion |
| `POST /v1/embeddings` | Returns 501 — see below |

Chat responses carry two extra fields inside `usage`: `ttft_s` and `tps`. They are
additive, so a strict OpenAI client ignores them, but they mean a local server can
answer "how fast was that" without a separate benchmark run.

Embeddings return `501 Not Implemented` on purpose. No embedding model is certified for
HIP EP yet, and returning zero vectors would let a retrieval pipeline silently index
garbage.

### Ollama

| Endpoint | Notes |
|---|---|
| `GET /api/version` | |
| `GET /api/tags` | Certified models, with `hipep.installed` marking what is local |
| `GET /api/ps` | Models resident in memory |
| `POST /api/show` | Model details and capabilities |
| `POST /api/chat` | Newline-delimited JSON stream |
| `POST /api/generate` | Prompt completion, with `images` for VLMs |
| `POST /api/pull` | Starts a pull job, streams Ollama-shaped progress |
| `DELETE /api/delete` | Remove a local model |

### Product APIs

| Endpoint | Notes |
|---|---|
| `GET /v1/catalog` | Catalog joined with local state, device fit and benchmarks |
| `GET /v1/local` · `DELETE /v1/local/{id}` | The local model store |
| `POST /v1/pull` · `POST /v1/port` | Start a pull or a bring-your-own-model job |
| `GET /v1/recipes` | Quantization recipes |
| `GET /v1/jobs` · `/v1/jobs/{id}` · `/v1/jobs/{id}/events` | Jobs and their SSE streams |
| `POST /v1/jobs/{id}/cancel` | Cancel |
| `GET /v1/benchmarks/published` | Certified results shipped in the repo |
| `POST /v1/benchmarks/run` | Measure this machine |
| `GET /v1/benchmarks/local` | Results measured here |
| `GET /v1/status` · `/v1/status/events` | Engine, load, sessions, queue, hardware |
| `GET /v1/hardware` | Live GPU detection |
| `GET /health` | Liveness |

Interactive API reference: `/api-docs`. Console: `/`. Documentation: `/docs`.

## Engines

| Engine | When it is used | Responses |
|---|---|---|
| `oga` | onnxruntime-genai is importable **and** an AMD GPU is present | Real model output |
| `replay` | Anything else, or `HIPEP_ENGINE=replay` | Placeholder text, real certified timings |

The replay engine is what makes the rest of the product developable and demonstrable
without hardware. It streams at the model's published TTFT and TPS for the prompt
length it was given, so the console, the benchmark harness and the client samples all
behave exactly as they will on a GPU host. It labels every response and
`/v1/status` reports `engine.real = false`.

`/v1/status` also carries `engine_reason`, which explains *why* the current engine was
chosen — so a user who expected GPU inference and got replay is told which precondition
failed instead of having to guess.

## Configuration

Every setting is an environment variable; the common ones also have flags.

| Variable | Default | Meaning |
|---|---|---|
| `HIPEP_HOST` | `127.0.0.1` | Bind address |
| `HIPEP_PORT` | `11434` | Bind port |
| `HIPEP_ENGINE` | `auto` | `auto`, `oga` or `replay` |
| `HIPEP_HOME` | `~/.hipep` | Model store and job workspace |
| `HIPEP_MAX_LOADED` | `2` | Resident models before LRU eviction |
| `HIPEP_MAX_CONCURRENT` | `4` | In-flight requests per model |
| `HIPEP_MAX_QUEUE` | `64` | Queue depth before requests are rejected with 503 |
| `HIPEP_IDLE_UNLOAD` | `900` | Seconds before an unused model is unloaded |
| `HIPEP_MAX_TOKENS` | `512` | Default completion length |
| `HIPEP_MAX_JOBS` | `1` | Concurrent port/quantize jobs |
| `HIPEP_CORS` | `1` | Allow cross-origin browser clients |

## Concurrency

Three limits are enforced and all three are visible in `/v1/status`:

1. **Resident models** — beyond `HIPEP_MAX_LOADED`, the least recently used model is
   evicted. A model that is mid-generation is never evicted.
2. **Requests per model** — `HIPEP_MAX_CONCURRENT` in flight; the rest queue.
3. **Queue depth** — past `HIPEP_MAX_QUEUE`, new requests get `503` with a message
   saying so, rather than joining an unbounded queue that will time out anyway.

This is the case that fails quietly otherwise: without admission control the tenth
concurrent request does not error, it just makes the other nine slow and eventually
exhausts GPU memory mid-generation.

## Testing

```bash
python product/serve.py --port 11500 --engine replay   # terminal 1
python product/oga-server/tests/smoke.py --base http://127.0.0.1:11500
```

The smoke test asserts the compatibility shapes real clients depend on: OpenAI list and
chunk objects, Ollama ndjson frames and nanosecond durations, SSE termination, image
routing, error messages, job streaming and session accounting.
