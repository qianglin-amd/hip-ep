---
title: Server mode
subtitle: The console, the endpoints and what the server guarantees
section: Instructions
order: 10
---

`hipep serve` starts one process that is both the inference server and the web console.
There is no separate front end to deploy and no build step: the console is static files
served from the same wheel.

```bash
hipep serve                      # 127.0.0.1:11434
hipep serve --port 8080          # somewhere else
hipep serve --host 0.0.0.0       # reachable on the network
hipep serve --engine replay      # no GPU required
```

## The console

Open the server's address in a browser. Seven pages, each of which maps to endpoints you
can call yourself:

| Page | What it is for |
|---|---|
| **Chat** | Talk to a model. TTFT and tokens/s per turn, markdown rendering, cancellation. |
| **Vision** | Same, with images: drop, paste or browse. Only vision-capable models are listed. |
| **Models** | The zoo joined with what is installed here and what fits this machine. |
| **Port a model** | Bring a Hugging Face repo, watch the eight pipeline stages, get a usable model. |
| **Benchmarks** | Certified results, a local run against them, and your run history. |
| **Server** | Live load, in-flight requests, every connected session, hardware. |
| **Develop** | Client samples generated against this server's address and your chosen model. |

## Configuration

Every setting is an environment variable, so a container or a service unit needs no config
file. The flags that `hipep serve` accepts override the corresponding variable.

| Variable | Default | Effect |
|---|---|---|
| `HIPEP_HOST`, `HIPEP_PORT` | `127.0.0.1`, `11434` | Listen address |
| `HIPEP_HOME` | `~/.hipep` | Models, job records, local benchmark results |
| `HIPEP_ENGINE` | `auto` | `oga` or `replay`; auto falls back to replay with a reason |
| `HIPEP_MAX_LOADED` | `2` | Resident models before the least recently used is evicted |
| `HIPEP_MAX_CONCURRENT` | `4` | Concurrent generations per model |
| `HIPEP_MAX_QUEUE` | `64` | Queued requests before new ones are rejected with 503 |
| `HIPEP_IDLE_UNLOAD` | `900` | Seconds before a model nobody is using is unloaded |
| `HIPEP_MAX_TOKENS` | `512` | Used when a request omits `max_tokens` |
| `HIPEP_MAX_JOBS` | `1` | Concurrent port, pull and benchmark jobs |
| `HIPEP_CORS` | `1` | Allow cross-origin browser clients |

On a machine whose memory fits exactly one model, set `HIPEP_MAX_LOADED=1` and
`HIPEP_MAX_CONCURRENT=1`. The defaults suit a card with room to spare.

## Admission control

Three limits, all reported at `/v1/status` and on the Server page:

**Resident models.** Loading a second model on a GPU that only fits one does not degrade,
it fails. The scheduler evicts the least recently used model instead, and never evicts a
model that is mid-generation.

**Per-model concurrency.** Two concurrent generations on one model share the GPU and both
get slower; twenty of them exhaust memory. Requests past the limit queue.

**Queue depth.** Past the queue limit the server returns `503` with a message saying so,
which is a better answer than a request that waits five minutes and then fails.

## Multiple users

Sessions are tracked per client: address plus user agent, or an explicit
`X-HIPEP-Session` header if your application sets one. The Server page lists every session
with its request count, tokens generated, age and idle time, plus the requests in flight
right now and the last twelve completions.

For a live feed rather than a page, subscribe:

```bash
curl -N http://127.0.0.1:11434/v1/status/events
```

This is the endpoint the console uses. It is also how you would drive a lab dashboard: it
carries load, limits, sessions, in-flight requests and recent completions.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat, streaming or not |
| `POST /v1/completions` | Legacy OpenAI completion |
| `GET /v1/models` | OpenAI model list |
| `GET /v1/catalog` | Full catalog: quantization, footprint, benchmarks, installed, fits |
| `POST /v1/pull` | Download a certified model; returns a job |
| `POST /v1/port` | Port your own model; returns a job |
| `POST /v1/import` | Install a model a peer server is sharing; returns a job |
| `GET /v1/artifacts/{id}` | Manifest of an installed model, for a peer to import |
| `GET /v1/local`, `DELETE /v1/local/{id}` | What is installed here; remove one |
| `GET /v1/recipes` | Export and quantization recipes |
| `GET /v1/jobs`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/events` | Job list, detail, live stream |
| `POST /v1/benchmarks/run`, `GET /v1/benchmarks/published` | Measure, and the certified numbers |
| `GET /v1/status`, `GET /v1/status/events` | Load, sessions, hardware; live version |
| `GET /health` | Liveness, version and engine name |
| `POST /api/chat`, `/api/generate`, `GET /api/tags`, `POST /api/show`, `/api/pull` | Ollama compatibility |

Details and copy-paste clients are in [API and client usage](api.html).

## Logs

The server logs one line per lifecycle event and one per completed request. For more:

```bash
hipep serve --log-level debug
```

Do not benchmark with debug logging on. Per-request tracing changes the numbers it is
printing.
