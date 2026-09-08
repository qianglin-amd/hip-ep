<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Milestones

The schedule runs in three phases to GA 1.0 on 31 October 2026. M0 below is the
completed foundation work that the phases build on; see
[`plan/PRODUCT_PLAN.md`](plan/PRODUCT_PLAN.md) for the roadmap.

Every phase and checkpoint below is a git tag on the `feat/hipep-product` branch. To
review or re-test any checkpoint:

```bash
git checkout <tag>
cd product && pip install -e ".[dev,site]"
pytest              # verify the checkpoint still passes
python -m hipep_server
```

Return to the tip with `git checkout feat/hipep-product`.

Nothing here has been pushed to a remote. All tags and commits are local.

---

## M0 — Foundations · Aug 28 – Sep 1, 2026

Goal: the product exists as a runnable thing, and the plan is reviewable.

| Checkpoint | Tag | Status |
|---|---|---|
| M0.1 Plan and review deck | `hipep-product-m0.1-plan` | Done |
| M0.2 Model registry seeded from release_0.4 | `hipep-product-m0.2-registry` | Done |
| M0.3 `oga-server` with OpenAI + Ollama APIs on the replay engine | `hipep-product-m0.3-server` | Done |
| M0.4 HIP EP Studio console and port recipes | `hipep-product-m0.4-studio` | Done |
| M0.5 Client samples in C++, Python and C# | `hipep-product-m0.5-examples` | Done |
| M0.6 `hipep` CLI, docs site and peer model sharing | `hipep-product-m0.6-cli-site` | Done |
| M0.7 Test suite for Windows and Linux | `hipep-product-m0.7-tests` | Done |
| M0.8 Install path: preflight, installers, container | `hipep-product-m0.8-platform` | Done |
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

### M0.4 — HIP EP Studio and recipes

- Chat with live TTFT, TPS and token counts.
- VLM image + text with drag-and-drop and paste.
- Model manager with pull progress.
- Benchmark runner and published results.
- Multi-user server status: workers, queue, sessions, GPU memory.
- Job pipeline: fetch → inspect → export → quantize → convert → validate → benchmark →
  publish, each stage resumable and cached, with a live status stream.
- Quark quantization stages ported from the Quark AI Agent Skills, defaulting to
  `chunk_size=512` and the `prune_logit` method, with a recipe per architecture.

Review with: `python -m hipep_server` and open <http://127.0.0.1:11434>.

### M0.5 — Client samples

- C++, Python and C# clients under `product/examples/`, each streaming chat and
  reporting TTFT and TPS against a running server.
- The C# sample carries the WinML path for Windows applications.

Review with: `python -m hipep_server`, then follow
[`examples/README.md`](examples/README.md).

### M0.6 — CLI, site and peer sharing

- `hipep` CLI: `serve`, `models`, `show`, `ls`, `rm`, `recipes`, `run`, `pull`, `port`,
  `bench`, `status`, `ps`, `jobs`, `cancel`. HTTP only, so it drives a remote server.
- Landing page and documentation site under `product/site/`, served by the server at
  `/docs` and publishable as static files. Model cards generate from the registry.
- Peer artifact sharing so a second machine copies a ported model instead of repeating
  the quantization.

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

### M0.8 — Install path

- `hipep doctor`: preflight over Python, packages, GPU and device profile, engine
  selection, porting toolchain, model store, catalog, docs and port. Runs without a
  server, because the install it diagnoses is often the one that cannot start one.
  Exits non-zero only on a hard failure, so a GPU-free CI runner stays green.
- `install.sh` and `install.ps1`: virtualenv, install, build the docs, run the preflight.
  Both refuse an interpreter older than 3.10 rather than failing later in pip.
- `docker/`: image and compose file for serving on Linux, with the device passthrough
  spelled out. The image deliberately ships without `onnxruntime-genai`, which is built
  per ROCm version and GPU architecture; supply one with `OGA_WHEEL`.

Review with: `hipep doctor`, then `docker compose -f product/docker/compose.yaml up`
on a Linux host.

---

## P1 — Serving on hardware · Sep 2 – Sep 30, 2026

Goal: the certified model set runs through HIP EP on real hardware, and the numbers the
console reports are the numbers the hardware produced.

| Checkpoint | Tag | Status |
|---|---|---|
| P1.1 AMD GPU EP wheels installed; real OGA engine streaming on Strix Halo | `hipep-product-p1.1-engine` | Done |
| P1.2 Per-family chat templates and stop markers | `hipep-product-p1.2-templates` | Done |
| P1.3 VLM image path on hardware | `hipep-product-p1.3-vision` | Planned |
| P1.4 OpenAI + Ollama parity verified against reference clients | `hipep-product-p1.4-parity` | Planned |
| P1.5 Multi-user scheduling under load | `hipep-product-p1.5-load` | Planned |
| P1 complete | `hipep-product-p1` | Planned |

### P1.1 — Real engine on HIP EP

The AMD GPU umbrella EP ships as its own wheels rather than compiled into
onnxruntime-genai, which imposes a load order the server now handles in
`engines/oga.py`: preload the wheel's `onnxruntime.dll` ahead of the Windows ML copy in
`system32`, set `AMDGPU_EP_PATH` before importing genai, and do *not* also call
`register_execution_provider_library` — registering the EP twice leaves genai reporting
no supported AMD GPU devices.

Review with: `python -m hipep_server --engine oga`, then a chat completion against a
downloaded model.

### P1.2 — Chat templates

onnxruntime-genai applies no chat template, so the server supplies one per family,
selected from `model.type` in the artifact's `genai_config.json`. Sending one family's
markup to another does not raise; the model reads the markers as text, continues the
transcript it appears to have been handed, and answers on the user's behalf until it
reaches `max_tokens`.

Review with: `pytest tests/test_chat_format.py`.

## P2 — Pipeline & certification · Oct 1 – Oct 16, 2026

Planned. Hugging Face URL to a benchmarked local model, unattended, from the console.
Bench harness runs through `oga-server`; nightly certification matrix; published numbers
reconciled with the internal dashboard.

Certification records the artifact identity alongside its throughput. A benchmark that
measures speed without printing output cannot tell a base model from an instruction-tuned
one, and that gap has already cost debugging time.

## P3 — Cross-platform, docs & GA 1.0 · Oct 17 – Oct 31, 2026

Planned. Linux server and container, Windows installer, four device profiles, remote and
bring-your-own-machine mode. A card for every certified model, recipes for six or more
architectures, C++/Python/WinML samples. Security review, opt-in telemetry, perf
hardening, support rotation. Public launch.

Gated on a Linux GPU host, which we do not have yet. This is the largest single schedule
risk in the plan and it now sits in the final two weeks.
