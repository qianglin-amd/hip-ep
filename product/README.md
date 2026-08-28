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
./product/install.sh        # Linux
.\product\install.ps1       # Windows
```

Either one creates a virtualenv, installs the server and the `hipep` CLI, builds the
documentation, and finishes by running `hipep doctor` so you can see what the machine
can actually do before starting anything.

```bash
hipep doctor                # drivers, wheels, GPU, store, catalog, port
hipep serve                 # http://127.0.0.1:11434
```

Open <http://127.0.0.1:11434> for HIP EP Studio, or point any OpenAI or Ollama client
at the same port. Without an AMD GPU the server starts on the replay engine, so the
console, the APIs, the docs and the tests all work on any machine.

See [`site/content/install-windows.md`](site/content/install-windows.md) and
[`site/content/install-linux.md`](site/content/install-linux.md) for the GPU-backed
setup, and [`docker/`](docker/) to run the server in a container on Linux.

## Layout

| Path | What it is |
|---|---|
| [`plan/`](plan/) | Product plan, roadmap and the generated review deck |
| [`oga-server/`](oga-server/) | `hipep_server` — OpenAI- and Ollama-compatible server, plus HIP EP Studio |
| [`modelzoo/`](modelzoo/) | Certified model registry, published benchmark results, model cards |
| [`toolkit/`](toolkit/) | `hipep` CLI — fetch, export, quantize, convert, validate, benchmark, publish |
| [`examples/`](examples/) | C++, Python and C# client samples |
| [`site/`](site/) | Landing page and documentation site: prose in `content/`, generator in `build.py` |
| [`docker/`](docker/) | Container image and compose file for serving on Linux |
| [`tests/`](tests/) | Test suite for all of the above |

## Building and testing

Both work the same way on Windows and Linux, and neither needs a GPU.

```bash
cd product
pip install -e ".[dev,site]"

pytest                  # the full suite, about 30 seconds
python site/build.py    # writes the documentation site to site/dist/
```

The suite runs against the replay engine with its clock disabled, so it needs no AMD
GPU, no `onnxruntime-genai` and no network. See [`tests/README.md`](tests/README.md)
for what each module covers.

When an install misbehaves, `hipep doctor` is the faster answer than the test suite: it
runs without a server and reports each precondition separately, so the failure names a
layer instead of a symptom. `hipep doctor --json` is the form to use in a CI step.

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
