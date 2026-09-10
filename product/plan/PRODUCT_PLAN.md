<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIP EP Product Plan

Status: **Draft for review** · Author: HIP EP team · Date: 2026-09-02 · Plan ID: `HIPEP-PROD-2026H2`

Revision 2 replaces the six-milestone schedule that ran to January 2027 with three
phases and GA 1.0 on 31 October 2026.

This document is the single source of truth for the HIP EP productization effort. The
slide deck at [`HIP-EP-Product-Plan.pptx`](HIP-EP-Product-Plan.pptx) is generated from
this plan by [`make_deck.py`](make_deck.py); edit the plan, then regenerate the deck.

---

## 1. Problem statement

HIP EP today is an excellent *engine* and a poor *product*.

We ship an ONNX Runtime Execution Provider that compiles ONNX graphs through a custom
MLIR HIP dialect and runs them on AMD GPUs with competitive performance. But everything
above the engine is missing or tribal knowledge:

| What a user wants | What exists today |
|---|---|
| "Give me a server I can point my app at" | No server. ORT/OGA C++ CLIs and pytest files. |
| "Which models actually work?" | An internal dashboard behind the corporate network. |
| "Run this Hugging Face model" | A multi-day manual export/quantize/split/merge expedition. |
| "How fast is it?" | Numbers live in an internal tracker, not in public docs. |
| "Show me the code for my app" | `hip-onnx-runner -m model.onnx`. |
| "Try it in a browser" | Nothing. |

Competing local-inference stacks (Ollama, LM Studio, FastFlowLM) win on exactly this
layer. Their engines are not better than ours; their *product* is. This plan closes
that gap.

## 2. Product vision

> **Ship HIP EP as a complete local-AI product for AMD GPUs: one server, one console,
> one model zoo, one benchmark, on Windows and Linux.**

A developer with an AMD GPU should go from a cold machine to a streaming chat response
in under ten minutes, and from a Hugging Face URL to a certified, benchmarked,
quantized local model without writing a script.

## 3. Product surfaces

| # | Surface | Name | What it is |
|---|---|---|---|
| 1 | Server | `oga-server` | OpenAI- and Ollama-compatible HTTP server over ONNX Runtime GenAI + HIP EP |
| 2 | Console | HIP EP Studio | Zero-install web UI: chat, VLM, model manager, benchmarks, server status |
| 3 | Model zoo | HIP EP Model Zoo | Certified model registry, model cards, pull-from-known-source |
| 4 | Toolkit | `hipep` CLI | Export, quantize (Quark), convert, validate, benchmark, publish |
| 5 | Recipes | Recipe collection | Per-architecture YAML recipes so users can bring their own models |
| 6 | Site | Landing page + docs | Public front door, quick starts, model cards, benchmark numbers |
| 7 | SDK samples | Examples | C++, Python, WinML clients users can paste into their project |

## 4. Architecture

```text
                        ┌───────────────────────────────────────────┐
   Browser ───────────► │  HIP EP Studio (static SPA, no build step) │
   curl / SDK ────────► │  Chat · VLM · Models · Bench · Status      │
   Ollama client ─────► └────────────────────┬──────────────────────┘
   OpenAI SDK ─────────►                     │ HTTP / SSE
                                             ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │                            oga-server                                │
   │  ┌────────────┬────────────┬────────────┬────────────┬────────────┐  │
   │  │ OpenAI API │ Ollama API │  Jobs API  │  Bench API │ Status API │  │
   │  └────────────┴────────────┴────────────┴────────────┴────────────┘  │
   │  Scheduler (per-model workers, queue, admission control, sessions)   │
   │  Registry (certified models) · Store (local models) · Job runner     │
   └──────────┬────────────────────────────────┬──────────────────────────┘
              │ engine ABI                     │ subprocess
              ▼                                ▼
   ┌────────────────────────┐      ┌──────────────────────────────────────┐
   │ OGA engine             │      │ hipep toolkit                        │
   │ onnxruntime_genai      │      │ fetch → export → quantize → convert  │
   │   + AMDGPUExecutionEP  │      │      → validate → benchmark → publish│
   │   profile = hip        │      │ (Quark PTQ, chunk_size=512,          │
   └───────────┬────────────┘      │  prune_logit)                        │
               │                   └──────────────────────────────────────┘
               ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ HIP EP  (hipgpu / amdgpu-ep)  →  MLIR HIP dialect → LLVM → HIP        │
   │ Radeon 8060S · Strix · Krackan · Strix Halo · Radeon dGPU · Instinct  │
   └──────────────────────────────────────────────────────────────────────┘
```

Key architectural decisions:

- **The server is the product boundary.** Benchmarks, the console, and the examples all
  go through `oga-server`. We never benchmark a path that a user cannot reproduce.
- **Engine abstraction with a mock.** `oga-server` runs against a real OGA engine on a
  GPU host, and against a deterministic mock engine anywhere else. The console, the
  docs, the tests, and the demo work on a laptop with no AMD GPU.
- **Zero-build front end.** The console is served as static assets. No Node.js, no npm,
  no bundler in the release path. It works offline and inside air-gapped labs.
- **Jobs are first class.** Import, export, quantize, benchmark, and certify are the
  same job abstraction with the same live status stream and the same artifact layout.
- **Registry is data, not code.** Certified models are YAML/JSON. Adding a model is a
  data change plus a benchmark run, not a code change.

## 5. Certified model scope (initial)

Sourced from the internal ROCm HIP EP dashboard, release_0.4 (run date 2026-08-27),
focusing on LLM and VLM as directed.

**LLM (8):** Llama-3.1-8B (AWQ INT4 asym), Mistral-7B-Instruct-v0.3 (AWQ INT4),
Qwen2.5-14B-Instruct (RTN INT4), Qwen2.5-Coder-14B-Instruct (RTN INT4),
Phi-4-14B (RTN INT4), gpt-oss-20b (RTN INT4), gpt-oss-120b (AWQ UINT4),
DeepSeek-R1-Distill-Llama-70B (AWQ INT4).

**VLM (8):** Qwen3.5-9B, Qwen3.5-35B-A3B, Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.8-27B,
Gemma-3-4B-IT, Gemma-4-12B-IT, Gemma-4-26B-A4B-IT.

Whisper (7) and Vision (14) stay in the registry as secondary tiers; the console and
docs foreground LLM and VLM.

## 6. Roadmap

Three phases, GA 1.0 at the end of October 2026. Foundations are already complete, which
is what makes the compressed schedule credible: the product tree, registry, server,
console, CLI, docs site, installers and test suite all exist and run on Strix Halo today.
Every phase ends with a git tag and a `MILESTONES.md` entry so any checkpoint can be
re-reviewed and re-tested.

| ID | Phase | Window | Exit criteria | Tag |
|---|---|---|---|---|
| **M0** | Foundations | Aug 28 – Sep 1, 2026 | **Complete.** Product tree, registry, `oga-server`, console, CLI, client samples, docs site, installers and test suite live locally on Strix Halo. | `hipep-product-m0` |
| **P1** | Serving on hardware | Sep 2 – Sep 30, 2026 | Real OGA engine streaming through HIP EP for the certified LLM and VLM set. OpenAI + Ollama API parity. VLM image path. Console chat with live TTFT/TPS. Multi-user scheduling under load. | `hipep-product-p1` |
| **P2** | Pipeline & certification | Oct 1 – Oct 16, 2026 | HF URL → export → Quark quantize → validate → publish, driven from the console with live status and a shareable artifact link. Bench harness runs through `oga-server`; nightly certification matrix; published numbers match the internal dashboard within tolerance. | `hipep-product-p2` |
| **P3** | Cross-platform, docs & GA 1.0 | Oct 17 – Oct 31, 2026 | Linux server + container, Windows installer, device profiles for Strix, Krackan, Strix Halo and Radeon dGPU. Model cards for every certified model, C++/Python/WinML samples. Security review, opt-in telemetry, perf hardening, support rotation. Public launch. | `hipep-product-v1.0` |

Two phases carry what were previously two milestones each. That is deliberate rather than
optimistic: the pipeline, bench harness, installers, container and docs site landed early
as part of foundations, so P2 and P3 are hardening and validating existing surfaces on
hardware, not building them from nothing.

### Gantt (weeks from 2026-09-01)

```text
Week            1     2     3     4     5     6     7     8     9
P1 Serving      ████████████████████████
P2 Pipeline+Cert                        ████████████
P3 Platform/GA                                      ████████████
GA 1.0                                                        ▲
```

## 7. Resourcing

Peak 9.5 FTE. The compressed schedule front-loads the team: every role is at full load
from P1 onward, because there is no longer a quiet milestone to absorb a slip.

| Role | FTE | M0 | P1 | P2 | P3 | Owns |
|---|---|---|---|---|---|---|
| Tech lead / architect | 1.0 | ● | ● | ● | ● | Architecture, reviews, roadmap |
| Server & API engineer | 2.0 | ● | ● | ● | ● | `oga-server`, scheduler, APIs |
| Front-end engineer | 1.5 | ○ | ● | ● | ● | HIP EP Studio, landing page |
| Model enablement engineer | 2.0 | ○ | ● | ● | ● | Export, quantization, recipes |
| Performance engineer | 1.0 | ○ | ● | ● | ● | Bench harness, tuning, regressions |
| Docs / DevRel | 1.0 | ○ | ○ | ● | ● | Docs site, model cards, samples |
| QA / release engineer | 1.0 | ○ | ● | ● | ● | CI, certification, packaging |
| **Total** | **9.5** | 4.0 | 8.5 | 9.5 | 9.5 | |

● primary load · ○ partial load

### Hardware and infrastructure

| Item | Qty | Purpose | Status |
|---|---|---|---|
| Strix Halo (Radeon 8060S, gfx1151, 48 GB) | 1 | Primary dev + demo server | **Available now** (this machine) |
| Strix / Krackan laptops | 2 | Device-profile validation | Needed by P1 |
| Radeon dGPU workstation (gfx1100/gfx1201) | 1 | Discrete GPU coverage | Needed by P1 |
| Linux GPU server | 1 | Linux `oga-server` hosting | **Not yet available** — needed by P3; code paths exist now and are validated in Docker |
| Nightly CI runner with GPU | 1 | Certification matrix | Needed by P2 |
| Public docs hosting | 1 | Landing page + docs | Needed by P3 |

The compressed schedule turns hardware from a planning item into a gate. Every machine
above is now needed within eight weeks rather than five months, and the Linux host in
particular has no slack left in front of it.

## 8. Risks

| # | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | No Linux GPU server available | Linux support slips or ships unvalidated | High | Build and test the Linux path in the existing `docker/` flow now; the hosted Linux server is a P3 gate with no slack in front of it; gate the Linux claim on real hardware |
| R2 | OGA + HIP EP wheel availability lags | Server cannot run real models on fresh machines | Medium | Mock engine keeps the whole product testable; pin known-good wheel set; ship a preflight doctor command |
| R3 | Model export/quantize is slow and fragile | P2 slips into the GA window | High | Recipes are data-driven and resumable; every stage caches artifacts; failures produce an actionable diagnosis, not a stack trace |
| R4 | Certified numbers drift from the internal dashboard | Loss of trust in published numbers | Medium | One harness, run through the server, same metric definitions; publish run metadata with every number |
| R5 | Multi-user concurrency exhausts GPU memory | Demo failures under load | Medium | Admission control, per-model worker cap, queue with visible depth, memory headroom checks |
| R6 | Scope creep into a general model hub | GA slips | Medium | Certified tiers are fixed per phase; BYO models are user-owned, never certified |
| R7 | Two months absorbs six milestones of scope | GA slips or ships thin | High | Foundations are already complete, so each phase hardens an existing surface rather than building one; scope is fixed per phase and cut rather than extended if a gate is missed |
| R8 | A wrong model artifact looks like an engine bug | Days lost to misdirected debugging | Medium | Certification records the artifact identity, not just its throughput; a throughput benchmark that never prints output cannot detect a mislabelled or base-versus-instruct model |

## 9. Success metrics

| Metric | Baseline | GA target |
|---|---|---|
| Cold machine → first streamed token | n/a (no server) | < 10 min |
| HF URL → benchmarked local model | ~2 days, manual | < 60 min, unattended |
| Certified LLM + VLM models | 0 published | 16 published with cards |
| API compatibility | none | OpenAI chat/completions/models + Ollama tags/chat/generate/pull/show |
| Platforms | Windows dev builds | Windows + Linux, 4 device profiles |
| Published benchmark coverage | 0 | 16 models × 3 context lengths |

## 10. Out of scope for 1.0

Training and fine-tuning; multi-node serving; a hosted public inference service;
model hosting/redistribution of third-party weights; Windows ARM; video understanding
in the VLM path.

## 11. Open questions for review

1. Do we publish benchmark numbers publicly at GA, or keep the dashboard internal and
   publish relative numbers only?
2. Which Linux GPU server do we get, and when? This is the single largest schedule risk.
3. Is `oga-server` the shipping name, or do we want a product name for the server?
4. Do we redistribute converted/quantized certified models, or only recipes that
   reproduce them from the original Hugging Face weights?
5. Telemetry: opt-in from GA, or defer to 1.1?
