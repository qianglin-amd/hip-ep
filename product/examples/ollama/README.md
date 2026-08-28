<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Using Ollama clients with oga-server

oga-server answers the Ollama API on the same port as the OpenAI API, so a tool written
for Ollama works against HIP EP without a code change. Point it at the server and the
models in your zoo appear as tags.

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
ollama list
ollama run llama3.1:8b "Why is quantization useful?"
```

## Raw API

```bash
# Chat, streamed as newline-delimited JSON
curl http://127.0.0.1:11434/api/chat -d '{
  "model": "llama3.1:8b",
  "messages": [{"role": "user", "content": "Hello"}]
}'

# Single-turn generate
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Write a haiku about GPUs.",
  "stream": false
}'

# Installed and certified models
curl http://127.0.0.1:11434/api/tags

# Download a certified model
curl http://127.0.0.1:11434/api/pull -d '{"model": "llama3.1:8b"}'
```

The final frame of a streamed response carries `total_duration`, `load_duration`,
`prompt_eval_duration` and `eval_duration` in nanoseconds, as Ollama clients expect. The
`/v1` endpoints report the same timings as seconds in the `usage` block.

## Applications that accept an OpenAI base URL

Most tools need only the base URL, which avoids the Ollama compatibility layer
altogether:

| Application | Setting |
|---|---|
| Open WebUI | OpenAI API base URL: `http://127.0.0.1:11434/v1` |
| Continue (VS Code / JetBrains) | `apiBase: http://127.0.0.1:11434/v1`, provider `openai` |
| LangChain | `ChatOpenAI(base_url="http://127.0.0.1:11434/v1", api_key="not-needed")` |
| LlamaIndex | `OpenAILike(api_base="http://127.0.0.1:11434/v1", is_chat_model=True)` |
| Obsidian / Raycast AI plugins | any field asking for an OpenAI-compatible endpoint |

Any string works as the API key: oga-server is a local server and does not
authenticate. If you expose it beyond localhost, put it behind a reverse proxy that
does.

## Differences from Ollama worth knowing

- **Model names are ours.** Tags come from the HIP EP model zoo, so
  `ollama pull` fetches a certified HIP EP artifact rather than a GGUF file.
- **No Modelfiles.** Model behaviour comes from the catalog entry and the recipe it was
  built with. Use the port page, or the `/v1/port` endpoint, to add your own model.
- **`/api/embeddings` is only available for models that publish an embedding head.**
  Ask for `/v1/catalog` and look for the `embedding` capability.
