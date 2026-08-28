<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Milestones

Every milestone below is a git tag on the `feat/hipep-product` branch. To review or
re-test any checkpoint:

```bash
git checkout <tag>
cd product && pip install -e ".[dev,site]"
pytest              # verify the checkpoint still passes
python -m hipep_server
```

Return to the tip with `git checkout feat/hipep-product`.

Nothing here has been pushed to a remote. All tags and commits are local.

---

## M0 — Foundations · Aug 28 – Sep 11, 2026

Goal: the product exists as a runnable thing, and the plan is reviewable.

| Checkpoint | Tag | Status |
|---|---|---|
| M0.1 Plan and review deck | `hipep-product-m0.1-plan` | Done |
| M0.2 Model registry seeded from release_0.4 | `hipep-product-m0.2-registry` | Done |
| M0.3 `oga-server` with OpenAI + Ollama APIs on the replay engine | `hipep-product-m0.3-server` | Done |
| M0.4 HIP EP Studio console | `hipep-product-m0.4-studio` | Done |
| M0.5 Model pipeline jobs and the `hipep` toolkit | `hipep-product-m0.5-toolkit` | Done |
| M0.6 Landing page, docs site and client samples | `hipep-product-m0.6-site` | Done |
| M0.7 Test suite for Windows and Linux | `hipep-product-m0.7-tests` | Done |
| M0 complete | `hipep-product-m0` | Done |

### M0.1 — Plan and review deck

- `product/plan/PRODUCT_PLAN.md` — problem, vision, architecture, roadmap, resourcing,
  risks, success metrics, open questions.
- `product/plan/make_deck.py` — generates the 14-slide review deck from the plan.
- `product/plan/preview_deck.py` — renders the deck to PNG for layout review on hosts
  with no PowerPoint or LibreOffice.
- `product/plan/HIP-EP-Product-Plan.pptx` — the deck.

Review with: open the `.pptx`, or `python product/plan/preview_deck.py` and look at
`product/plan/preview/contact-sheet.png`.

### M0.2 — Model registry

- 16 certified LLM and VLM entries plus Whisper and vision tiers, seeded from the
  internal ROCm HIP EP dashboard, release_0.4, run date 2026-08-27.
- Published TTFT and TPS at 128 / 2048 / 16384 context, accuracy where measured, and
  the recipe that reproduces each model.

Review with: `hipep models` and `hipep show qwen2.5:14b`.

### M0.3 — Server

- OpenAI: `/v1/models`, `/v1/chat/completions` (streaming and non-streaming),
  `/v1/completions`, `/v1/embeddings`.
- Ollama: `/api/tags`, `/api/show`, `/api/chat`, `/api/generate`, `/api/pull`,
  `/api/ps`, `/api/version`.
- Scheduler with per-model workers, queue depth, admission control and sessions.
- Replay engine reproducing certified timings; OGA engine for GPU hosts.

Review with: `python -m hipep_server` then `curl http://127.0.0.1:11434/v1/models`.

### M0.4 — HIP EP Studio

- Chat with live TTFT, TPS and token counts.
- VLM image + text with drag-and-drop and paste.
- Model manager with pull progress.
- Benchmark runner and published results.
- Multi-user server status: workers, queue, sessions, GPU memory.

Review with: `python -m hipep_server` and open <http://127.0.0.1:11434>.

### M0.5 — Toolkit and jobs

- `hipep` CLI: `serve`, `models`, `show`, `ls`, `rm`, `recipes`, `run`, `pull`, `port`,
  `bench`, `status`, `ps`, `jobs`, `cancel`.
- Job pipeline: fetch → inspect → export → quantize → convert → validate → benchmark →
  publish, each stage resumable and cached, with a live status stream.
- Quark quantization stages ported from the Quark AI Agent Skills, defaulting to
  `chunk_size=512` and the `prune_logit` method.

Review with: `hipep status`, then `hipep port <hf-id>` and watch the stages.

### M0.6 — Site and samples

- Landing page and documentation site under `product/site/`, served by the server at
  `/docs` and publishable as static files.
- Model cards generated from the registry.
- Client samples in C++, Python and C# under `product/examples/`.

Review with: `python site/build.py`, then `python -m hipep_server` and open
<http://127.0.0.1:11434/docs>.

### M0.7 — Test suite

- 153 tests over the catalog, both API surfaces, admission control, the job pipeline,
  peer artifact sharing, the benchmark harness, the CLI and the documentation build.
- Runs on Windows and Linux with no GPU, no `onnxruntime-genai` and no network, in
  about 30 seconds.
- Clock compression (`HIPEP_REPLAY_SPEED`, `HIPEP_SIM_SPEED`) so replayed timings and
  simulated pipeline stages do not have to be waited out.

Found and fixed while writing it: a simulated port produced a share URL that served an
empty manifest, `load_duration` was hardcoded to zero on the Ollama surface, and seven
VLM entries were missing their parameter counts.

Review with: `cd product && pytest`. See [`tests/README.md`](tests/README.md).

---

## M1 — Serving Alpha · Sep 12 – Oct 2, 2026

Planned. Real OGA engine streaming on Strix Halo, OpenAI + Ollama parity verified
against the reference clients, VLM image path on hardware, console chat with live
TTFT/TPS from real runs, multi-user status under load.

## M2 — Model Pipeline · Oct 3 – Oct 23, 2026

Planned. Hugging Face URL to a benchmarked local model, unattended, from the console.

## M3 — Benchmark & Certification · Oct 24 – Nov 13, 2026

Planned. Nightly certification matrix; published numbers reconciled with the internal
dashboard.

## M4 — Cross-platform · Nov 14 – Dec 4, 2026

Planned. Linux server and container, Windows installer, four device profiles, remote
and bring-your-own-machine mode. Gated on a Linux GPU host, which we do not have yet.

## M5 — Docs & SDK · Dec 5 – Dec 25, 2026

Planned. Full docs, a card for every certified model, recipes for six or more
architectures.

## M6 — GA 1.0 · Jan 5 – Jan 30, 2027

Planned. Public launch.
