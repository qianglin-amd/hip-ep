// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Develop: copy-paste client samples, generated against this server's address and the
// model the user picks, so a sample runs without editing.

import { state } from "../app.js";
import { copyButton, el, fill, toast } from "../ui.js";

const SAMPLES = [
  {
    id: "python-openai",
    label: "Python · OpenAI SDK",
    language: "python",
    file: "examples/python/openai_chat.py",
    note:
      "The server implements the OpenAI chat API, so the official SDK works unchanged. " +
      "Point base_url at oga-server and use any string as the key.",
    code: (ctx) => `# pip install openai
from openai import OpenAI

client = OpenAI(base_url="${ctx.base}/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="${ctx.model}",
    messages=[{"role": "user", "content": "Explain what an execution provider does."}],
    max_tokens=256,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
    # The final chunk carries HIP EP's timing extensions.
    if chunk.usage:
        print(f"\\n\\nTTFT {chunk.usage.ttft_s:.3f}s, {chunk.usage.tps:.1f} tok/s")
`,
  },
  {
    id: "python-vision",
    label: "Python · Vision",
    language: "python",
    file: "examples/python/vision_chat.py",
    note:
      "Images travel as base64 data URIs. The server never fetches remote URLs, which " +
      "keeps an inference server from becoming an outbound HTTP client.",
    code: (ctx) => `import base64
from openai import OpenAI

client = OpenAI(base_url="${ctx.base}/v1", api_key="not-needed")

with open("photo.jpg", "rb") as handle:
    data_uri = "data:image/jpeg;base64," + base64.b64encode(handle.read()).decode()

response = client.chat.completions.create(
    model="${ctx.vlm}",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ],
    }],
    max_tokens=256,
)
print(response.choices[0].message.content)
`,
  },
  {
    id: "cpp",
    label: "C++ · libcurl",
    language: "cpp",
    file: "examples/cpp/chat_stream.cpp",
    note:
      "No SDK needed: one POST and a line-oriented read of the SSE body. Build with " +
      "cmake --build examples/cpp/build after configuring with libcurl available.",
    code: (ctx) => `// cmake -B build -S . && cmake --build build
#include <curl/curl.h>

#include <iostream>
#include <string>

namespace {

// Each SSE frame arrives as "data: {json}\\n\\n". Extracting the delta with a find is
// enough for a sample; production code should use a JSON parser.
size_t on_data(char *ptr, size_t size, size_t nmemb, void *userdata) {
  auto &buffer = *static_cast<std::string *>(userdata);
  buffer.append(ptr, size * nmemb);

  size_t frame_end;
  while ((frame_end = buffer.find("\\n\\n")) != std::string::npos) {
    const std::string frame = buffer.substr(0, frame_end);
    buffer.erase(0, frame_end + 2);
    if (frame.rfind("data: ", 0) != 0) continue;

    const std::string payload = frame.substr(6);
    if (payload == "[DONE]") return size * nmemb;

    const size_t key = payload.find("\\"content\\":\\"");
    if (key == std::string::npos) continue;
    size_t i = key + 11;
    for (; i < payload.size() && payload[i] != '"'; ++i) {
      if (payload[i] == '\\\\' && i + 1 < payload.size()) {
        const char escaped = payload[++i];
        std::cout << (escaped == 'n' ? '\\n' : escaped);
      } else {
        std::cout << payload[i];
      }
    }
    std::cout.flush();
  }
  return size * nmemb;
}

}  // namespace

int main() {
  const std::string body = R"({
    "model": "${ctx.model}",
    "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
    "max_tokens": 128,
    "stream": true
  })";

  curl_global_init(CURL_GLOBAL_DEFAULT);
  CURL *curl = curl_easy_init();
  if (curl == nullptr) return 1;

  std::string buffer;
  curl_slist *headers = curl_slist_append(nullptr, "Content-Type: application/json");
  curl_easy_setopt(curl, CURLOPT_URL, "${ctx.base}/v1/chat/completions");
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, on_data);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);

  const CURLcode result = curl_easy_perform(curl);
  if (result != CURLE_OK) {
    std::cerr << "request failed: " << curl_easy_strerror(result) << "\\n";
  }

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  curl_global_cleanup();
  return result == CURLE_OK ? 0 : 1;
}
`,
  },
  {
    id: "winml",
    label: "C# · WinML / .NET",
    language: "csharp",
    file: "examples/winml/ChatClient.cs",
    note:
      "A Windows app usually already has HttpClient. This is the same pattern the " +
      "WinML sample app uses to talk to a locally running oga-server.",
    code: (ctx) => `// dotnet add package System.Net.Http.Json
using System.Net.Http.Json;
using System.Text.Json;

using var http = new HttpClient { BaseAddress = new Uri("${ctx.base}/") };

var request = new
{
    model = "${ctx.model}",
    messages = new[] { new { role = "user", content = "Summarise DirectML in one sentence." } },
    max_tokens = 200,
    stream = true,
};

using var response = await http.PostAsJsonAsync("v1/chat/completions", request);
response.EnsureSuccessStatusCode();

using var stream = await response.Content.ReadAsStreamAsync();
using var reader = new StreamReader(stream);

while (await reader.ReadLineAsync() is { } line)
{
    if (!line.StartsWith("data: ")) continue;

    var payload = line["data: ".Length..];
    if (payload == "[DONE]") break;

    using var document = JsonDocument.Parse(payload);
    var choice = document.RootElement.GetProperty("choices")[0];
    if (choice.GetProperty("delta").TryGetProperty("content", out var content))
    {
        Console.Write(content.GetString());
    }
}
Console.WriteLine();
`,
  },
  {
    id: "ollama",
    label: "Ollama clients",
    language: "bash",
    file: "examples/ollama/README.md",
    note:
      "The server also answers the Ollama API on the same port, so tools written for " +
      "Ollama work without a code change. Point OLLAMA_HOST at this server.",
    code: (ctx) => `# Any Ollama client, unchanged
export OLLAMA_HOST=${ctx.base}
ollama list
ollama run ${ctx.model} "Why is quantization useful?"

# Raw API
curl ${ctx.base}/api/chat -d '{
  "model": "${ctx.model}",
  "messages": [{"role": "user", "content": "Hello"}]
}'

# Open WebUI, LangChain, LlamaIndex and Continue all accept an OpenAI base URL:
#   ${ctx.base}/v1
`,
  },
  {
    id: "curl",
    label: "curl",
    language: "bash",
    file: "examples/curl/requests.sh",
    note: "Everything the console does is reachable from a shell.",
    code: (ctx) => `# Certified catalog, joined with what is installed here
curl -s ${ctx.base}/v1/catalog | jq '.models[] | {id, tier, installed, fits}'

# One completion
curl -s ${ctx.base}/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "${ctx.model}",
  "messages": [{"role": "user", "content": "Hello"}]
}' | jq '.choices[0].message.content, .usage'

# Download a certified model, following the job to completion
JOB=$(curl -s ${ctx.base}/v1/pull -H "Content-Type: application/json" \\
  -d '{"model":"${ctx.model}"}' | jq -r .job.id)
curl -N ${ctx.base}/v1/jobs/$JOB/events

# Benchmark this machine and compare against the certified numbers
curl -s ${ctx.base}/v1/benchmarks/run -H "Content-Type: application/json" \\
  -d '{"model":"${ctx.model}","contexts":[128,2048],"repeats":3}'

# Live server load, including other users
curl -N ${ctx.base}/v1/status/events
`,
  },
];

export async function renderDevelop(root) {
  const models = state.catalog.filter((m) => m.tier === "llm" || m.tier === "vlm");
  const llm = models.find((m) => m.tier === "llm") || models[0];
  const vlm = models.find((m) => m.capabilities?.vision);

  const ctx = {
    base: location.origin,
    model: llm?.id || "llama3.1:8b",
    vlm: vlm?.id || "gemma3-vl:4b",
  };

  const modelSelect = el("select", {
    style: "max-width:280px",
    onchange: (event) => { ctx.model = event.target.value; paint(); },
  }, models.map((m) => el("option", { value: m.id, selected: m.id === ctx.model }, m.name)));

  const tabs = el("div", { class: "tabs" });
  const host = el("div", {});
  let current = SAMPLES[0].id;

  for (const sample of SAMPLES) {
    tabs.append(
      el("button", {
        class: sample.id === current ? "active" : "",
        onclick: (event) => {
          current = sample.id;
          for (const b of tabs.children) b.classList.remove("active");
          event.currentTarget.classList.add("active");
          paint();
        },
      }, sample.label),
    );
  }

  root.append(
    el("div", { class: "card notice-card" }, [
      el("h3", { text: "One endpoint, any client" }),
      el("p", { class: "muted" },
        "oga-server speaks the OpenAI and Ollama APIs on the same port, so most existing " +
        "code needs only a base URL change. The samples below are generated against this " +
        "server's address and the model you select, and ship in the repository as runnable files."),
      el("div", { class: "row wrap", style: "margin-top:12px" }, [
        el("label", { class: "field", style: "margin:0" }, ["Model used in samples", modelSelect]),
        el("span", { class: "grow" }),
        el("div", { class: "share" }, [
          el("code", { text: `${ctx.base}/v1` }),
          copyButton(() => `${ctx.base}/v1`, "Copy base URL"),
        ]),
      ]),
    ]),
    tabs,
    host,
  );

  function paint() {
    const sample = SAMPLES.find((s) => s.id === current);
    const code = sample.code(ctx);
    fill(host, 
      el("div", { class: "card" }, [
        el("div", { class: "row" }, [
          el("div", {}, [
            el("h3", { text: sample.label }),
            el("div", { class: "faint small mono", text: sample.file }),
          ]),
          el("span", { class: "grow" }),
          copyButton(() => code, "Copy"),
          el("button", {
            class: "btn btn-sm",
            text: "Download",
            onclick: () => {
              const blob = new Blob([code], { type: "text/plain" });
              const link = el("a", {
                href: URL.createObjectURL(blob),
                download: sample.file.split("/").pop(),
              });
              link.click();
              URL.revokeObjectURL(link.href);
              toast(`Saved ${link.download}`, "ok");
            },
          }),
        ]),
        el("p", { class: "muted small", text: sample.note }),
        el("div", { class: "code-block" }, el("pre", {}, el("code", {
          class: `lang-${sample.language}`,
          text: code,
        }))),
      ]),
    );
  }

  paint();
}
