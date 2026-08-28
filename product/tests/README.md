<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Product test suite

Covers `oga-server`, the `hipep` CLI, the job pipeline and the documentation site.

The whole suite runs against the **replay engine** with its clock disabled, so it needs
no AMD GPU, no `onnxruntime-genai` and no network access. It runs the same way on
Windows and Linux and finishes in about half a minute.

## Running

```bash
cd product
pip install -e ".[dev,site]"
pytest
```

Useful selections:

```bash
pytest -m "not slow"          # skip the certified-pace timing test
pytest tests/test_openai_api.py
pytest -k admission           # by name
```

## What each module covers

| Module | Covers |
|---|---|
| `test_catalog.py` | Catalog loading, completeness of every shipped model card, name resolution, published-benchmark lookup and context snapping |
| `test_openai_api.py` | `/v1` chat, streaming, images, model listing, the `usage` timing extension |
| `test_ollama_api.py` | `/api` tags, show, chat, generate, pull/delete and the nanosecond duration fields |
| `test_scheduler.py` | Admission control: queue depth, per-model concurrency, LRU eviction, idle sweep, session accounting |
| `test_pipeline.py` | Pull and port jobs, stage sequencing, Quark defaults, cancellation, immediate usability of a ported model |
| `test_sharing.py` | Artifact manifests, per-file transfer, path-traversal refusal and peer import between two servers |
| `test_bench.py` | Prompt/estimator agreement, transport cost accounting, measurement against the published curve |
| `test_status.py` | Health, status, hardware, multi-user session reporting, stream serialization |
| `test_toolkit.py` | CLI table alignment and formatting, model-name resolution, argument parsing |
| `test_site.py` | Documentation build, internal link check, model-card content |

## Clock compression

Two settings exist so the suite does not have to wait out timings it is only checking
the shape of:

| Setting | Environment variable | Effect |
|---|---|---|
| `replay_speed` | `HIPEP_REPLAY_SPEED` | Divides the replayed TTFT and decode pacing. `0` removes it entirely. |
| `simulation_speed` | `HIPEP_SIM_SPEED` | Divides the sleeps in simulated pipeline stages. |

Compression scales a measurement but not the published numbers it is compared against,
so any test asserting on a delta against certified values has to run at
`replay_speed=1.0`. Those are marked `slow`.

## Fixtures

Defined in `conftest.py`:

| Fixture | What it gives you |
|---|---|
| `catalog` | The shipped catalog, loaded once per session |
| `client` | A started in-process server (`TestClient`), replay clock off |
| `paced_client` | The same, with the replay clock running 100x fast |
| `live_server` | Starts real servers on real ports; needed wherever something crosses a socket |
| `finished` | Polls a job until it reaches a terminal state |
| `chat_model`, `vision_model` | Ids resolved from the catalog |
| `png_data_url` | A 1x1 PNG for the image paths |

`live_server` is not a convenience. The benchmark harness measures over HTTP by design,
and the artifact importer builds its own client to reach a peer; neither goes through
the in-process ASGI transport.
