---
title: API and client usage
subtitle: OpenAI, Ollama, and clients in Python, C++ and C#
section: Instructions
order: 11
---

The server implements two wire protocols on one port. Most applications need a base URL
change and nothing else.

```text
OpenAI    http://127.0.0.1:11434/v1
Ollama    http://127.0.0.1:11434
```

There is no authentication. Any string works as an API key, because a local server has
nobody to authenticate.

## Chat completions

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Explain quantization in two sentences."}],
    "max_tokens": 256
  }'
```

The response is standard OpenAI, with three extra fields in `usage`:

```json
{
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 256,
    "total_tokens": 270,
    "ttft_s": 0.187,
    "decode_s": 6.0021,
    "total_s": 6.1891,
    "tps": 42.65
  }
}
```

`ttft_s` is time to first token, `decode_s` is the decode window excluding TTFT, and `tps`
is `completion_tokens / decode_s`. These are the same definitions the benchmark harness and
the published numbers use, so an application can report speed that matches the model card.

## Streaming

Set `"stream": true`. Frames are server-sent events; the last frame before `[DONE]` carries
`usage`.

```text
data: {"choices":[{"delta":{"role":"assistant","content":""}}]}
data: {"choices":[{"delta":{"content":"Quantization "}}]}
...
data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{...}}
data: [DONE]
```

Two mistakes account for most "streaming does not work" reports, and both are in the
client:

- **curl** buffers by default. Use `-N`.
- **.NET `HttpClient`** buffers the whole response unless you pass
  `HttpCompletionOption.ResponseHeadersRead`. Without it every token arrives at once and
  your measured TTFT is the total generation time.

## Images

Images travel inline as base64 data URIs, to any model whose `capabilities.vision` is true.

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="not-needed")
data_uri = "data:image/jpeg;base64," + base64.b64encode(open("photo.jpg", "rb").read()).decode()

response = client.chat.completions.create(
    model="gemma3-vl:4b",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]}],
)
```

The server does not fetch remote `image_url` values. An inference server that follows URLs
supplied by its clients is an outbound HTTP client, which is a security problem rather than
a convenience. Sending an image to a text-only model returns a 400 naming the vision models
you could use instead.

## Ollama compatibility

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
ollama list
ollama run llama3.1:8b "Why is quantization useful?"
```

`/api/chat` and `/api/generate` stream newline-delimited JSON, and the final frame carries
`total_duration`, `load_duration`, `prompt_eval_duration` and `eval_duration` in
nanoseconds, as Ollama clients expect.

Applications that accept an OpenAI base URL — Open WebUI, Continue, LangChain, LlamaIndex —
should use `/v1` instead and skip the compatibility layer.

## Client examples {#examples}

Runnable versions of all of these live in `product/examples/`.

| Example | Language | Needs |
|---|---|---|
| `python/openai_chat.py` | Python | `pip install openai` |
| `python/vision_chat.py` | Python | `pip install openai` |
| `cpp/chat_stream.cpp` | C++17 | CMake and a compiler; no package manager |
| `winml/ChatClient.cs` | C# / .NET | .NET 8 SDK or newer; no package reference |
| `curl/requests.sh` | Bash | curl, jq |

### Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "Explain what an execution provider does."}],
    max_tokens=256,
    stream=True,
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.usage:
        print(f"\n\nTTFT {chunk.usage.ttft_s:.3f}s, {chunk.usage.tps:.1f} tok/s")
```

### C++

The shipped sample carries a small sockets transport in `http_stream.hpp`, so it builds
with CMake on Windows and Linux without a package manager. Swap in libcurl or cpp-httplib
and the rest of the file is unchanged.

```cpp
#include "http_stream.hpp"

const hipep::Session session;
session.post_lines("http://127.0.0.1:11434", "/v1/chat/completions", body,
                   [](const std::string &line) {
                     if (line.rfind("data: ", 0) != 0) return;
                     // parse the delta out of line.substr(6)
                   });
```

```bash
cd product/examples/cpp
cmake -B build -S . && cmake --build build --config Release
./build/chat_stream --model llama3.1:8b --prompt "Write a haiku about GPUs."
```

### C# / .NET

```csharp
using var http = new HttpClient { BaseAddress = new Uri("http://127.0.0.1:11434/") };
using var message = new HttpRequestMessage(HttpMethod.Post, "v1/chat/completions")
{
    Content = JsonContent.Create(new
    {
        model = "llama3.1:8b",
        messages = new[] { new { role = "user", content = "Hello" } },
        stream = true,
    }),
};

// Without ResponseHeadersRead this is not a stream.
using var response = await http.SendAsync(message, HttpCompletionOption.ResponseHeadersRead);
using var reader = new StreamReader(await response.Content.ReadAsStreamAsync());

while (await reader.ReadLineAsync() is { } line)
{
    if (!line.StartsWith("data: ")) continue;
    var body = line["data: ".Length..];
    if (body is "[DONE]") break;
    // parse and write the delta
}
```

## Running a model in your own process

If you do not want a server, HIP EP is an ONNX Runtime execution provider and can be used
directly. It is a plugin EP: register the library, then select the devices it reports.

```python
import onnxruntime as ort

ort.register_execution_provider_library("MorphiZenEP", "hipgpu.dll")
devices = [d for d in ort.get_ep_devices() if d.ep_name == "MorphiZenEP"]

options = ort.SessionOptions()
options.add_provider_for_devices(devices, {})
# A graph the EP declines otherwise falls back to CPU and still returns correct answers,
# which makes a compilation failure look like a slow GPU.
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

session = ort.InferenceSession("model.onnx", sess_options=options)
```

The C++ equivalent uses `RegisterExecutionProviderLibrary` and
`SessionOptionsAppendExecutionProvider_V2`. Both paths need an EP built for your GPU
architecture.

## Errors

| Status | Meaning |
|---|---|
| `400` | Bad request. An image sent to a text-only model names the vision models available. |
| `404` | Unknown model. The body suggests close matches from the catalog. |
| `503` | Queue is full. Retry, or raise `HIPEP_MAX_QUEUE`. |
| `500` | The engine failed. The message is the engine's, unedited. |
