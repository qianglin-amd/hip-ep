<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIP EP Product

Everything above the engine: the server, the console, the model zoo, the toolkit, the
benchmarks, the docs site and the client samples.

The rest of this repository builds **HIP EP**, an ONNX Runtime Execution Provider that
compiles ONNX graphs through a custom MLIR HIP dialect and runs them on AMD GPUs. This
directory turns that engine into something a developer can install, point an app at,
and benchmark.

## Quick start

```bash
python -m pip install -r product/requirements.txt
python -m hipep_server            # http://127.0.0.1:11434
```

Open <http://127.0.0.1:11434> for HIP EP Studio, or point any OpenAI or Ollama client
at the same port. Without an AMD GPU the server starts on the replay engine, so the
console, the APIs, the docs and the tests all work on any machine.

See [`docs/quickstart-windows.md`](docs/quickstart-windows.md) and
[`docs/quickstart-linux.md`](docs/quickstart-linux.md) for the GPU-backed setup.

## Layout

| Path | What it is |
|---|---|
| [`plan/`](plan/) | Product plan, roadmap and the generated review deck |
| [`oga-server/`](oga-server/) | `hipep_server` — OpenAI- and Ollama-compatible server, plus HIP EP Studio |
| [`modelzoo/`](modelzoo/) | Certified model registry, published benchmark results, model cards |
| [`toolkit/`](toolkit/) | `hipep` CLI — fetch, export, quantize, convert, validate, benchmark, publish |
| [`bench/`](bench/) | Benchmark suites and result archive, all driven through the server |
| [`examples/`](examples/) | C++, Python and WinML client samples |
| [`site/`](site/) | Public landing page and documentation site |
| [`docs/`](docs/) | Product documentation source |

## Principles

1. **The server is the product boundary.** The console, the benchmarks and the samples
   all speak HTTP to `oga-server`. We never publish a number from a path a user cannot
   reproduce.
2. **It works without a GPU.** Every surface degrades to the replay engine, which
   reproduces published certified timings. Development, CI, docs and demos never block
   on hardware availability.
3. **No build step in the front end.** HIP EP Studio is served as static assets. No
   Node.js, no bundler, no network at runtime.
4. **Models are data.** Adding a certified model is a registry entry plus a benchmark
   run, not a code change.
5. **Every long operation is a job.** Import, export, quantize, benchmark and certify
   share one job model, one status stream and one artifact layout.

## Status

Tracked in [`MILESTONES.md`](MILESTONES.md). Each milestone is a git tag so any
checkpoint can be checked out, reviewed and re-tested.
