#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""End-to-end smoke test against a running oga-server.

Exercises the OpenAI surface, the Ollama surface, the catalog, the job pipeline and
the status API, and asserts the compatibility shapes that real clients depend on.

    python -m hipep_server            # in one terminal
    python tests/smoke.py             # in another

Exits non-zero on the first failure, so it is usable as a CI gate.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time

import httpx

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    results.append((PASS if condition else FAIL, name, detail))
    print(f"  [{PASS if condition else FAIL}] {name}{f' — {detail}' if detail else ''}")
    return condition


# A 1x1 transparent PNG, small enough to inline and valid enough to decode.
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default=None, help="defaults to the first certified LLM")
    ap.add_argument("--skip-jobs", action="store_true")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    client = httpx.Client(base_url=base, timeout=120.0)

    print("\n== health and status ==")
    health = client.get("/health").json()
    check("GET /health returns ok", health.get("status") == "ok", health.get("engine", ""))

    status = client.get("/v1/status").json()
    engine = status.get("engine", {})
    check("GET /v1/status reports an engine", bool(engine.get("name")), engine.get("name", ""))
    check("status carries limits", "limits" in status)
    check("status carries host info", bool(status.get("host", {}).get("hostname")))
    gpus = status.get("host", {}).get("gpus", [])
    check("hardware detected", True, gpus[0]["name"] if gpus else "no AMD GPU (fine)")

    print("\n== catalog ==")
    models = client.get("/v1/models").json()
    ids = [m["id"] for m in models.get("data", [])]
    check("GET /v1/models returns an OpenAI list", models.get("object") == "list", f"{len(ids)} models")
    check("entries have the OpenAI model shape",
          all({"id", "object", "owned_by"} <= set(m) for m in models["data"]))

    catalog = client.get("/v1/catalog").json()
    check("GET /v1/catalog joins local state",
          all("installed" in m and "fits" in m for m in catalog["models"]),
          f"{len(catalog['models'])} models, release {catalog['benchmark'].get('release')}")

    llms = [m for m in catalog["models"] if m["tier"] == "llm"]
    vlms = [m for m in catalog["models"] if m["tier"] == "vlm"]
    check("LLM tier populated", len(llms) >= 8, f"{len(llms)} LLM")
    check("VLM tier populated", len(vlms) >= 8, f"{len(vlms)} VLM")
    check("published benchmarks attached",
          all(m.get("benchmark") for m in llms),
          "every certified LLM has TTFT/TPS")

    model = args.model or llms[0]["id"]
    print(f"\n== chat: {model} ==")

    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Explain HIP EP in one sentence."}],
        "max_tokens": 40,
    }
    t0 = time.perf_counter()
    completion = client.post("/v1/chat/completions", json=body).json()
    elapsed = time.perf_counter() - t0
    check("POST /v1/chat/completions returns a completion",
          completion.get("object") == "chat.completion")
    content = completion["choices"][0]["message"]["content"]
    check("response has content", len(content) > 0, f"{len(content)} chars in {elapsed:.1f}s")
    usage = completion.get("usage", {})
    check("usage carries ttft_s and tps",
          "ttft_s" in usage and "tps" in usage,
          f"TTFT {usage.get('ttft_s')}s, {usage.get('tps')} tok/s")

    print("\n== streaming ==")
    chunks, first_at, done = 0, None, False
    start = time.perf_counter()
    with client.stream("POST", "/v1/chat/completions",
                       json={**body, "stream": True, "max_tokens": 25}) as response:
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                done = True
                break
            chunk = json.loads(payload)
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content"):
                chunks += 1
                if first_at is None:
                    first_at = time.perf_counter() - start
    check("SSE stream delivers deltas", chunks > 0, f"{chunks} chunks")
    check("SSE stream terminates with [DONE]", done)
    check("streamed TTFT is measurable", first_at is not None, f"{first_at:.2f}s" if first_at else "")

    print("\n== ollama compatibility ==")
    version = client.get("/api/version").json()
    check("GET /api/version", "version" in version, version.get("version", ""))
    tags = client.get("/api/tags").json()
    check("GET /api/tags returns models",
          isinstance(tags.get("models"), list) and tags["models"],
          f"{len(tags.get('models', []))} tags")
    check("tags carry Ollama details",
          all({"name", "model", "digest", "details"} <= set(t) for t in tags["models"]))

    show = client.post("/api/show", json={"model": model}).json()
    check("POST /api/show", "details" in show and "capabilities" in show)

    lines = []
    with client.stream("POST", "/api/chat",
                       json={"model": model,
                             "messages": [{"role": "user", "content": "hi"}],
                             "options": {"num_predict": 15}}) as response:
        for line in response.iter_lines():
            if line.strip():
                lines.append(json.loads(line))
    check("POST /api/chat streams ndjson", len(lines) > 1, f"{len(lines)} frames")
    check("final frame is done with durations",
          lines[-1].get("done") is True and "eval_duration" in lines[-1],
          f"eval_count={lines[-1].get('eval_count')}")

    print("\n== vision routing ==")
    vlm = vlms[0]["id"]
    image_message = {
        "model": vlm,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + base64.b64encode(TINY_PNG).decode()}},
            ],
        }],
        "max_tokens": 20,
    }
    vision = client.post("/v1/chat/completions", json=image_message)
    check("VLM accepts an image", vision.status_code == 200, f"{vlm} -> {vision.status_code}")

    text_only = client.post("/v1/chat/completions", json={**image_message, "model": model})
    check("text-only model rejects images with a useful error",
          text_only.status_code == 400 and "vision" in text_only.json()["detail"],
          text_only.json().get("detail", "")[:60])

    print("\n== errors ==")
    missing = client.post("/v1/chat/completions",
                          json={"model": "nope:1b", "messages": [{"role": "user", "content": "x"}]})
    check("unknown model returns 404 with suggestions",
          missing.status_code == 404 and "catalog" in missing.json()["detail"])

    print("\n== benchmarks ==")
    published = client.get("/v1/benchmarks/published").json()
    check("published results available",
          len(published.get("results", [])) > 0,
          f"{len(published.get('results', []))} models, {published['meta'].get('release')}")

    if not args.skip_jobs:
        print("\n== jobs ==")
        job = client.post("/v1/benchmarks/run",
                          json={"model": model, "contexts": [128], "repeats": 1,
                                "warmup": 0, "generate_tokens": 8}).json()["job"]
        check("benchmark job starts", job["state"] in {"pending", "running"}, job["id"])

        final = None
        with client.stream("GET", f"/v1/jobs/{job['id']}/events") as response:
            for line in response.iter_lines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    final = json.loads(line[6:])
                    if final["state"] in {"succeeded", "failed", "cancelled"}:
                        break
        check("job streams to completion",
              final and final["state"] == "succeeded",
              (final or {}).get("error") or f"{final['progress'] * 100:.0f}%" if final else "")
        if final and final["state"] == "succeeded":
            summary = final["result"]["summary"]
            measured = summary["contexts"]["128"]
            check("benchmark produced measurements",
                  measured["tps"] > 0,
                  f"TTFT {measured['ttft_s']:.2f}s, {measured['tps']:.1f} tok/s")
            check("measurements compared against published numbers",
                  "128" in summary["delta_vs_published"],
                  json.dumps(summary["delta_vs_published"].get("128", {})))

    print("\n== multi-user status ==")
    status = client.get("/v1/status").json()
    check("sessions tracked", status["load"]["sessions"] >= 1,
          f"{status['load']['sessions']} session(s)")
    check("totals accumulated", status["totals"]["requests"] > 0,
          f"{status['totals']['requests']} requests, "
          f"{status['totals']['completion_tokens']} tokens generated")

    failed = [r for r in results if r[0] == FAIL]
    print(f"\n{'=' * 60}")
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("\nFailures:")
        for _, name, detail in failed:
            print(f"  - {name} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
