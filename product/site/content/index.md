---
title: Overview
subtitle: What HIP EP is and how the pieces fit
path: docs/index.html
order: 0
---

HIP EP runs ONNX models on AMD GPUs. It is an execution provider for ONNX Runtime: a
compiler lowers ONNX operations through an MLIR HIP dialect to LLVM IR, and the generated
code dispatches to hipDNN, MIOpen, hipBLASLt and hand-written HIP kernels.

Around that compiler sits everything needed to actually use it for local AI:

| Piece | What it does |
|---|---|
| **oga-server** | Local HTTP server. OpenAI and Ollama compatible, model management, benchmarking, porting jobs. |
| **HIP EP Studio** | The console the server serves at `/`: chat, vision, tool-calling agent, model zoo, benchmarks, live server load. |
| **Model zoo** | Certified models with published benchmarks, quantization schemas and reproducible recipes. |
| **Toolkit** | The export and quantization recipes the porting pipeline runs, using AMD Quark. |
| **Client examples** | Streaming clients in Python, C++ and C#/.NET, plus curl and Ollama. |

If you want to get a model answering questions on your own machine, start with
[Install on Windows](install-windows.html) or [Install on Linux](install-linux.html).
If you want to call it from an application, go to
[API and client usage](api.html).

## The shape of a request

```text
your application
      │  POST /v1/chat/completions        OpenAI or Ollama wire format
      ▼
oga-server ──► scheduler ──► engine (ONNX Runtime GenAI)
      │            │                 │
      │            │                 └─► HIP EP ──► hipDNN / MIOpen / hipBLASLt / HIP kernels
      │            └─ admission control: resident models, per-model concurrency, queue depth
      └─► streamed deltas, then a usage block with ttft_s, decode_s and tps
```

Two properties of this path are worth knowing before you build on it.

**The server is the measurement path.** Benchmarks are taken over the same HTTP endpoint
your application calls, not through a private fast path, so a published number is one you
can reproduce. The harness separates the engine's own measurement from the end-to-end
measurement, so you can see how much of a latency figure is transport.

**Admission control is not optional.** A local GPU has one pool of memory, and the way an
unbounded server fails is not an error message but an out-of-memory error part-way through
a generation. The server bounds resident models, per-model concurrency and queue depth,
and reports all three on the Server page and at `/v1/status`.

## Models

The zoo is data, not code: a YAML entry plus a benchmark run. Each model names the
artifact it was built from, the recipe that produced it, the exact quantization schema, its
footprint on disk and in VRAM, and the measurement that certified it.

See [Models](models/index.html) for the catalog and
[Benchmarks](benchmarks/index.html) for published results at 128, 2K and 16K prompts.

Your own models are first-class. The [porting pipeline](port-a-model.html) fetches a
Hugging Face repo, quantizes it with Quark, converts it, validates it, benchmarks it and
registers it in your local zoo, where it behaves exactly like a certified model.

## Running on your own hardware

HIP EP targets the architecture of the GPU that will execute the model. This is not a
preference: a kernel built for one architecture and launched on another crashes at launch
rather than falling back. The installers and `build.py` detect the architecture; if you
override it, override it correctly.

| Hardware | Architecture | Typical memory |
|---|---|---|
| Ryzen AI Max+ (Strix Halo) | gfx1151 | up to 96 GB shared |
| Ryzen AI 300 (Strix, Krackan) | gfx1150, gfx1151 | shared with system |
| Radeon RX 7000 | gfx1100–gfx1102 | 8–24 GB |
| Radeon RX 6000 | gfx1030 | 8–16 GB |
| Instinct MI300 | gfx942 | 128–192 GB |

The model zoo records a minimum VRAM figure per model, and the console marks models that
do not fit the machine you are on rather than letting you discover it mid-load.

## Developing without a GPU {#replay}

The server has a replay engine. It streams placeholder text paced to each model's
*published* TTFT and decode rate, so the console, the API, the benchmark harness, the
porting pipeline and the client samples all work on a machine with no AMD GPU and no
ONNX Runtime GenAI install.

```bash
HIPEP_ENGINE=replay python -m hipep_server
```

It is explicit about what it is. Every response is labelled, the console shows a replay
badge, and `/v1/status` reports `engine.real: false`. A replayed response is never
mistaken for model output, and a replayed benchmark is never mistaken for a measurement of
your machine.

This matters more than it sounds. Without it, everything above the engine would be
undevelopable and undemonstrable until hardware and wheels line up, and the documentation
could not carry runnable examples.

## Where things live

| Path | Contents |
|---|---|
| `product/oga-server/` | The server, the scheduler, the engines and the console |
| `product/modelzoo/` | Catalog entries and published benchmark results |
| `product/toolkit/hipep_toolkit/` | The `hipep` CLI and the export/quantization recipes |
| `product/examples/` | Client examples |
| `product/site/` | This site and the generator that builds it |
| `lib/`, `include/`, `backend-mlir-compiler/` | The compiler and the execution provider |
