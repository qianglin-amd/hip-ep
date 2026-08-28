<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Client examples

Runnable clients for oga-server, one per language, each small enough to read in a sitting
and copy into an application. Every one of them streams, reports time to first token and
tokens per second, and takes `--base` so it can point at a server on another machine.

Start the server first:

```bash
python -m hipep_server          # from product/oga-server/
```

Then pick a language. `--base http://HOST:11434` reaches a server on another machine on
your network; the default is localhost.

| Example | Language | Needs |
|---|---|---|
| [`python/openai_chat.py`](python/openai_chat.py) | Python | `pip install openai` |
| [`python/vision_chat.py`](python/vision_chat.py) | Python | `pip install openai` |
| [`cpp/chat_stream.cpp`](cpp/chat_stream.cpp) | C++17 | CMake, any compiler |
| [`winml/ChatClient.cs`](winml/ChatClient.cs) | C# / .NET | .NET 8 SDK or newer |
| [`curl/requests.sh`](curl/requests.sh) | Bash | curl, jq |
| [`ollama/README.md`](ollama/README.md) | any Ollama client | — |

## Python

```bash
pip install -r python/requirements.txt
python python/openai_chat.py --prompt "Explain what an execution provider does."
python python/vision_chat.py --image photo.jpg --prompt "What is in this image?"
```

The server implements the OpenAI chat API, so the official SDK works unchanged; only
`base_url` differs from a cloud deployment. `usage` additionally carries `ttft_s`,
`decode_s` and `tps`, which the SDK preserves as attributes.

## C++

```bash
cd cpp
cmake -B build -S . && cmake --build build --config Release
./build/chat_stream --model llama3.1:8b --prompt "Write a haiku about GPUs."
```

No package manager and no SDK: one POST and a line-oriented read of the response.
[`cpp/http_stream.hpp`](cpp/http_stream.hpp) holds a ~150-line HTTP transport built on
the platform sockets so the sample compiles on Windows and Linux out of the box.
Replace it with whatever HTTP client your application already links — libcurl,
cpp-httplib, WinHTTP, Boost.Beast — and keep `chat_stream.cpp` as it is.

## C# / .NET

```bash
cd winml
dotnet run -- --model llama3.1:8b --prompt "Summarise DirectML in one sentence."
```

No package reference: `HttpClient` and `System.Text.Json` ship with the framework. The
one detail that matters for streaming is `HttpCompletionOption.ResponseHeadersRead`;
without it `HttpClient` buffers the whole response and every token arrives at once.

## Shell

```bash
./curl/requests.sh                       # tour of the API against localhost
./curl/requests.sh http://gpubox:11434   # against another machine
```

## Running a model in your own process instead

These examples talk to a server, which is the right default: weights load once, several
applications share one GPU, and the server handles admission control. If you need HIP EP
inside your own process instead, register the execution provider with ONNX Runtime
directly:

```python
import onnxruntime as ort

# HIP EP is a plugin execution provider: register the library, then select the devices
# it reports rather than naming a provider string.
ort.register_execution_provider_library("MorphiZenEP", "hipgpu.dll")
devices = [d for d in ort.get_ep_devices() if d.ep_name == "MorphiZenEP"]

options = ort.SessionOptions()
options.add_provider_for_devices(devices, {})
# Without this, a graph HIP EP declines falls back to CPU and still returns correct
# answers, which makes a compilation failure look like a slow GPU.
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

session = ort.InferenceSession("model.onnx", sess_options=options)
```

A complete version of that script, including the DLL search-path setup Windows needs, is
[`python/examples/run_onnx.py`](../../python/examples/run_onnx.py). The C++ equivalent,
using `RegisterExecutionProviderLibrary` and
`SessionOptionsAppendExecutionProvider_V2`, is in
[`backend-mlir-compiler/test/test_e2e_mlir.cpp`](../../backend-mlir-compiler/test/test_e2e_mlir.cpp).
Both need a HIP EP build for your GPU architecture; see the
[Windows](../../docs/quick_start.md) or [Linux](../../docs/quick_start_linux.md) quick
start.

## Working offline

Every example works against the replay engine, which streams placeholder text at each
model's published TTFT and tokens/s without a GPU or any model weights. Responses are
labelled and `/v1/status` reports `engine.real = false`, so a replayed run is never
mistaken for a real one. That makes these samples useful for wiring up an integration
before hardware is available:

```bash
HIPEP_ENGINE=replay python -m hipep_server
```
