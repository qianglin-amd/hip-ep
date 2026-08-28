---
title: Troubleshooting
subtitle: What goes wrong, and what it means
section: Instructions
order: 60
---

Start here:

```bash
hipep doctor          # nothing starts, or you are on a new machine
hipep status          # it started, but something is slow or unexpected
```

`doctor` needs no server. It walks the preconditions in dependency order — Python,
packages, GPU and device profile, engine, porting toolchain, model store, catalog, docs,
port — and reports each separately, so the answer is a named layer rather than a guess.
Warnings are not failures; only a hard failure exits non-zero.

`status` asks a running server what it decided: the engine, *why* that engine was
selected, the GPU it found, the admission limits and what is loaded. Most surprises on a
new machine are answered by its first two lines.

## It says replay, and I have a GPU

The replay engine streams placeholder text at each model's published speed. It is
correct behaviour when it is chosen deliberately and a nuisance when it is not. The
`engine reason` field says which of these applies:

| Reason | Fix |
|---|---|
| `HIPEP_ENGINE=replay` | Unset it, or pass `--engine oga`. |
| `onnxruntime-genai is not installed` | `pip install onnxruntime-genai`, matched to your platform. |
| `no AMD GPU detected` | See the next section. |

To make the failure loud instead of silent, ask for the real engine explicitly:

```bash
hipep serve --engine oga
```

That refuses to start rather than falling back, which is what you want in CI and on a
machine that is supposed to have working hardware.

## No AMD GPU detected

Check the driver stack first, outside HIP EP:

```bash
rocm-smi              # Linux
hipInfo               # either platform, from the ROCm/TheRock SDK
```

If those do not see the GPU, nothing above them will.

On Linux the usual cause is group membership: your user must be in `render` and `video`,
and a fresh `usermod` does not apply until you log out.

```bash
sudo usermod -aG render,video "$USER"
```

On Windows, install the current Adrenalin driver rather than the one Windows Update
supplied.

## Access violation, or a crash the moment a kernel launches

The execution provider was built for a different GPU architecture than the one running
it. HIP kernels are compiled per architecture and a mismatch does not degrade — it
crashes at launch.

```bash
amdgpu-arch                    # e.g. gfx1151
python build.py --hip_arch gfx1151
```

`build.py` detects the architecture; you only need this when you overrode it.

## Correct answers, but far slower than published

Almost always CPU fallback. ONNX Runtime falls back to the CPU provider for any graph the
execution provider declines, so a compilation failure looks like a slow GPU rather than an
error. Accuracy proves nothing here.

```bash
HIPDNN_EP_STRICT=1 hipep serve
```

With that set, a graph that was expected to be offloaded fails to compile loudly instead
of quietly running on the CPU.

Second most common cause: measuring with tracing on. `HIPDNN_EP_PERF=1` and
`HIPDNN_EP_DEBUG=1` change the numbers they print. Never benchmark with either enabled.

## Slower than published, and it really is on the GPU

In order of likelihood:

**Not warmed up.** The first inference includes compilation. `hipep bench` discards a
warmup run for exactly this reason; a hand-rolled measurement should too.

**Thermal drift.** A thin laptop chassis at 40 W sustained is a different machine from
one at 65 W burst. Compute-bound decode numbers drift over minutes.

**Concurrent runs.** GPU benchmarks are not parallelisable. Two at once invalidates both.
`hipep ps` shows what else is in flight.

**A different quantization.** Compare the schemas on the two model cards. An fp16 KV
cache, an unpruned LM head or a smaller group size each cost throughput.

## Out of memory

`hipep status` reports the limits that are meant to prevent this:

```bash
HIPEP_MAX_LOADED=1 HIPEP_MAX_CONCURRENT=1 hipep serve
```

One resident model and one generation at a time is the safe configuration on a machine
whose memory fits exactly one model. The defaults (2 and 4) suit a larger card.

Out of memory during a *port* rather than during serving is the quantization stage.
Lower `--chunk-size`; it trades port time for peak memory and does not change the result.

## 503, server is at capacity

The queue is at `HIPEP_MAX_QUEUE` (64 by default). This is admission control working: the
alternative to rejecting the request is accepting it and running out of memory part-way
through someone else's generation. Raise the limit if your workload is bursty and you
would rather wait than be refused.

## Streaming arrives all at once

The client is buffering, not the server.

- `curl` needs `-N`.
- .NET `HttpClient` needs `HttpCompletionOption.ResponseHeadersRead`.
- A reverse proxy in front of the server needs response buffering disabled. The server
  sets `X-Accel-Buffering: no`, which nginx honours.

## An image is rejected

The server does not fetch remote `image_url` values. An inference server that follows
URLs its clients supply is an outbound HTTP client, which is a security problem rather
than a convenience. Inline the image as a base64 data URI; `hipep run --image` and the
Python sample both do this.

A 400 naming the vision models means the model you chose is text-only.

## My ported model produces nonsense

The recipe does not suit the architecture — most often a group size the model's shapes do
not divide evenly. Try the family's recipe from the zoo, or `int4-rtn-g128`, which needs
no calibration and works on most decoder architectures.

If the *fetch* or *inspect* stage failed instead, the message names what it found. A new
architecture usually needs a recipe and sometimes an operation in the compiler.

## Port 11434 is already in use

Ollama. Either stop it, or run beside it:

```bash
hipep serve --port 11500
```

The port is 11434 by default precisely so that existing Ollama clients need no
configuration, which is also why the two cannot share it.

On Windows, find the owner:

```powershell
Get-NetTCPConnection -LocalPort 11434 | Select-Object OwningProcess
```

## The console is blank, or the docs 404

The console ships with the server and needs no build. The documentation site does:

```bash
python product/site/build.py
```

The server serves the generated site at `/docs` when it exists and skips it otherwise.

## Something changed in the runtime and the results did not

Compiled model artifacts are cached. After changing runtime code or a custom kernel,
invalidate the cache before drawing conclusions, or you are measuring the previous build.
