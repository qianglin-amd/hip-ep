---
title: Port a model
subtitle: From a Hugging Face repo to a model in your zoo
section: Instructions
order: 12
---

The certified zoo is a starting point, not a limit. If a model runs on your machine, it
belongs in your zoo, and porting it is a job you watch rather than a script you debug.

Open **Port a model** in the console, or call `POST /v1/port`.

## What you provide

| Field | Meaning | Default |
|---|---|---|
| `source` | Hugging Face repo (`org/model`), a URL, or a local path | required |
| `name` | Model id to publish as | derived from the source |
| `quantization` | Export and quantization recipe id | `int4-rtn-g128` |
| `chunk_size` | Quark chunk size | `512` |
| `prune_logit` | Prune the LM head | `true` |
| `calibration_samples` | Used only by calibrating recipes such as AWQ | `128` |
| `benchmark` | Benchmark before publishing | `true` |

The two defaults worth explaining:

**`chunk_size=512`** bounds how much of the model Quark holds at once. Quantizing a large
model in one piece is the fastest way to exhaust memory on a machine that could otherwise
run the result. 512 is the value the certified recipes use.

**`prune_logit=true`** drops the unused portion of the LM head. It reduces the artifact
noticeably on models with large vocabularies and does not change generated output.

If you do not know which recipe to choose, leave it: `int4-rtn-g128` is round-to-nearest
INT4 with group size 128, needs no calibration data, and works on most decoder
architectures. Move to an AWQ recipe when you care more about quality than about port time.
The full list is in [Export recipes](recipes.html).

## What the pipeline does

Eight stages, streamed to the console as they run:

| Stage | What happens |
|---|---|
| **Fetch weights** | Download the repo, verify checksums. Resumable and cached. |
| **Inspect** | Read the config: architecture, vocabulary, context, tensor layout. |
| **Export to ONNX** | Build the ONNX graph, including the vision tower for a VLM. |
| **Quantize (Quark)** | Apply the recipe. Calibration runs here if the recipe needs it. |
| **Convert for OGA** | Produce `genai_config.json` and the pipeline the runtime loads. |
| **Validate** | Load it, generate, check the output is coherent and the shapes are right. |
| **Benchmark** | Measure TTFT and tokens/s at the same contexts the zoo publishes. |
| **Publish** | Register it in your catalog and write the metadata sidecar. |

Stages are cached: a failed run resumes rather than starting over, which matters when the
fetch stage was twenty minutes of download.

## When it finishes

The model is usable immediately — no restart. It appears in the Models page tagged **your
import**, is selectable in Chat and Vision, and is a valid `model` for every API endpoint,
including benchmarks. On the next server start, your imports are discovered from disk and
re-registered, so the catalog is identical.

Ported models live under `$HIPEP_HOME/models/<id>/` with a `hipep.json` sidecar recording
the source, the recipe and the parameters used.

## Sharing it

A port is expensive enough that nobody should do it twice. The finished job prints a share
command, and any machine that can reach your server can install the result:

```bash
# on the machine that ported it
hipep serve --host 0.0.0.0

# on the machine that wants it
hipep pull qwen2.5:7b-mine --from http://porting-box:11434
```

The receiving server reads a manifest from the peer, transfers the files one at a time and
registers the model locally. The transfer is resumable — re-running after an interruption
fetches only what is missing — and the peer's quantization schema and capabilities travel
with the artifact, so the imported model's card matches the original.

## From the API

```bash
JOB=$(curl -s http://127.0.0.1:11434/v1/port \
  -H "Content-Type: application/json" \
  -d '{
        "source": "Qwen/Qwen2.5-7B-Instruct",
        "name": "qwen2.5:7b-mine",
        "quantization": "int4-awq-g128",
        "chunk_size": 512,
        "prune_logit": true
      }' | jq -r .job.id)

# Follow it; the stream ends when the job does
curl -N http://127.0.0.1:11434/v1/jobs/$JOB/events
```

Cancel with `POST /v1/jobs/{id}/cancel`. Cancellation is checked between stages and inside
the long ones, so a cancelled job stops promptly and leaves the cache intact.

## What can go wrong

**Unsupported architecture.** The inspect stage fails with the architecture it found. A new
architecture usually needs a recipe, sometimes an operation in the compiler; both are
tracked as ordinary work rather than being your problem to patch.

**Out of memory during quantization.** Lower `chunk_size`. It trades port time for peak
memory and changes nothing about the result.

**Validation produces nonsense.** Usually a wrong recipe for the architecture, most often a
group size the model's shapes do not divide evenly. Try the family's recipe from the zoo,
or `int4-rtn-g128`.

**It works but is slower than a certified model of the same size.** Compare the
quantization schemas on the model cards. An fp16 KV cache, an unpruned LM head or a smaller
group size all cost throughput.
