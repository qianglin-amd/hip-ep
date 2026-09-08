---
title: Install on Windows
subtitle: From nothing to a streaming model
order: 1
---

Five minutes, most of it download time.

## Before you start

| Requirement | Notes |
|---|---|
| AMD GPU | Ryzen AI, Ryzen AI Max+, Radeon RX 6000 or newer, or Instinct |
| Adrenalin driver | Current release. An old driver is the most common cause of a failed load. |
| Windows 11 or Windows 10 22H2 | Earlier builds are untested |
| Python 3.10 or newer | `python --version`. The Microsoft Store build works. |
| Disk space | 6–20 GB per model, depending on the model |

Check the GPU the driver reports before installing anything:

```powershell
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion
```

## Install

```powershell
.\product\install.ps1
```

That creates a virtualenv under `product\.venv`, installs the server and the CLI, builds
this documentation and finishes by running the preflight below. Add `-Dev` to install and
run the test suite, `-Quant` to add the porting toolchain.

It installs the server, the console and the model zoo metadata. It does not pull model
weights; you choose those next.

There is no release on PyPI yet, so installing from a checkout is the only path today.
When one is published this page will carry the `pip install` line instead.

## Check the machine first

```powershell
hipep doctor
```

```text
hipep doctor on Windows 11 (AMD64)

  ok    python
        3.12.10 at C:\Python3\Python312\python.exe
  ok    gpu 0
        AMD Radeon(TM) 8060S Graphics (gfx1151), 33.9 GB, profile strix-halo
  warn  engine
        onnxruntime-genai is not installed; the server will replay published timings
        try: install onnxruntime-genai to run real models
  ok    model store
        C:\Users\you\.hipep, 259 GB free
```

Each line is one precondition, so a problem names the layer that owns it rather than
surfacing later as a slow model or a silent CPU fallback. Warnings are not failures: a
machine with no GPU still serves the replay engine and still runs the test suite. Add
`--json` to consume it from a script.

## Start the server

```powershell
hipep serve
```

```text
hip-ep oga-server 0.1.0
  host    WINAMD-2QAMU6OE
  gpu     AMD Radeon(TM) 8060S Graphics (gfx1151)
  engine  hip-ep on gfx1151
  catalog 23 models (release_0.4)
  models  C:\Users\you\.hipep\models
  listen  http://127.0.0.1:11434
```

Read that block. `engine` is the line that matters: if it says `replay`, the server did not
find a usable GPU stack and is streaming placeholder text. The reason is printed next to
it, and the console shows a replay badge.

Port 11434 is Ollama's port, chosen so that tools already configured for Ollama find this
server with no configuration at all.

## Get a model

Open <http://127.0.0.1:11434> and pick one from the Models page, or from the command line:

```powershell
hipep pull llama3.1:8b
```

Sizes and minimum VRAM are listed per model in [Models](models/index.html). On a 16 GB
machine, `llama3.1:8b` or `mistral:7b` is a good first choice; on Strix Halo with 48 GB or
more, `gpt-oss:20b` and the larger Qwen models are comfortable.

## Talk to it

From the console: open <http://127.0.0.1:11434>, choose the model, type. Time to first
token and tokens per second are shown for every turn.

From an application, change one line:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="not-needed")
print(
    client.chat.completions.create(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": "Hello"}],
    )
    .choices[0]
    .message.content
)
```

More, including C++ and C#, in [API and client usage](api.html).

## Running the tests

The product's own test suite runs against the replay engine, so it needs no GPU, no
`onnxruntime-genai` and no network. Use it to check an install, or before sending a
change.

```powershell
cd product
python -m pip install -e ".[dev,site]"
pytest
```

About thirty seconds, and identical on Linux. It covers the catalog, both API surfaces,
admission control, the job pipeline, artifact sharing, the benchmark harness, the CLI
and the documentation build. `pytest -m "not slow"` skips the one test that runs at the
certified replay pace.

## Building from source

Building the execution provider is only necessary if you are changing it. The
architecture must match the GPU that will run the model.

```powershell
# Detect the architecture; do not guess it
& "$env:THEROCK_DIST\lib\llvm\bin\amdgpu-arch.exe"

# Configure and build (Visual Studio 2022 Developer PowerShell, Ninja generator)
python build.py
```

`build.py` detects the architecture, resolves dependencies, builds, installs and runs the
default test suite. Pass `--hip_arch gfx1151` to override detection, `--mock` to build
without ROCm hardware, or `--skip_tests` to skip the post-install run. Full details are in
the repository's Windows quick start.

## Making the server reachable from other machines

The server binds to localhost by default. To serve a workstation to a lab:

```powershell
hipep serve --host 0.0.0.0
New-NetFirewallRule -DisplayName "HIP EP" -Direction Inbound -LocalPort 11434 `
  -Protocol TCP -Action Allow
```

There is no authentication: it is a local server. If it is reachable beyond a network you
control, put it behind a reverse proxy that authenticates.

## When something is wrong

- **`engine replay` when you expected the GPU.** The reason is printed next to it, and
  again at `/v1/status`. Usually a missing `onnxruntime-genai` wheel or an unsupported
  driver.
- **A model loads, then the process dies.** Almost always an architecture mismatch in a
  self-built EP. Check `gcnArchName` from `hipInfo` against the build.
- **Slower than the published numbers.** Confirm the graph ran on the GPU rather than
  falling back to CPU: `HIPDNN_EP_STRICT=1` turns a silent fallback into an error.

More in [Troubleshooting](troubleshooting.html).
