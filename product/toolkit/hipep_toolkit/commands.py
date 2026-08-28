# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Implementations of the ``hipep`` subcommands.

Each handler takes the parsed arguments, talks to the server over HTTP and returns a
process exit status. Nothing here reaches into the server's Python objects, so the CLI
keeps working against a server on another machine.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import time
from pathlib import Path
from typing import Any

from . import term
from .client import Client, ServerError

TIER_LABEL = {"llm": "LLM", "vlm": "vision", "speech": "speech", "vision": "vision"}


def _client(args: argparse.Namespace) -> Client:
    return Client(getattr(args, "url", None))


def _out(text: str = "") -> None:
    print(text)


def _readable(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


# --------------------------------------------------------------------------- serve


def serve(args: argparse.Namespace) -> int:
    """Start the server in this process by delegating to its own entry point.

    Re-implementing the argument handling here would give two ways to configure one
    server, so the flags are forwarded verbatim instead.
    """
    try:
        from hipep_server.__main__ import main as server_main
    except ImportError as exc:
        _out(
            "the server package is not installed in this environment.\n"
            "Install it with: pip install -e product"
        )
        _out(f"  ({exc})")
        return 1
    return server_main(args.server_args)


# -------------------------------------------------------------------------- models


def models(args: argparse.Namespace) -> int:
    with _client(args) as client:
        data = client.catalog(tier=args.tier, q=args.search)
        entries = data.get("models") or []
        if not entries:
            _out("no models matched.")
            return 1

        if args.json:
            _out(json.dumps(entries, indent=2))
            return 0

        rows = []
        for entry in entries:
            bench = entry.get("benchmark") or {}
            tps = (bench.get("tps") or {}).get("2048")
            marks = []
            if entry.get("installed"):
                marks.append("installed")
            if not entry.get("fits"):
                marks.append("too large")
            if entry.get("status") == "imported":
                marks.append("imported")
            rows.append(
                [
                    entry["id"],
                    TIER_LABEL.get(entry["tier"], entry["tier"]),
                    entry.get("parameters") or "-",
                    entry.get("quant_label") or "-",
                    term.gigabytes((entry.get("footprint") or {}).get("min_vram_gb")),
                    f"{tps:.0f}" if tps else "-",
                    ", ".join(marks),
                ]
            )
        _out(
            term.table(
                ["MODEL", "KIND", "PARAMS", "QUANTIZATION", "VRAM", "TOK/S", "NOTES"],
                rows,
                right=(4, 5),
            )
        )
        meta = data.get("benchmark") or {}
        release = meta.get("release")
        _out()
        _out(
            term.paint(
                f"{len(entries)} models"
                + (f", tokens/s from {release} at a 2K prompt" if release else ""),
                "dim",
            )
        )
        return 0


def show(args: argparse.Namespace) -> int:
    with _client(args) as client:
        entry = client.resolve_model(args.model)
        if args.json:
            _out(json.dumps(entry, indent=2))
            return 0

        quant = entry.get("quantization") or {}
        footprint = entry.get("footprint") or {}
        context = entry.get("context") or {}
        caps = [k for k, v in (entry.get("capabilities") or {}).items() if v]

        _out(term.paint(entry["name"], "bold"))
        if entry.get("description"):
            _out(entry["description"])
        _out()
        _out(
            term.keyvalues(
                [
                    ("model", entry["id"]),
                    ("kind", TIER_LABEL.get(entry["tier"], entry["tier"])),
                    ("base model", entry.get("base_model") or ""),
                    ("parameters", entry.get("parameters") or ""),
                    ("quantization", entry.get("quant_label") or ""),
                    ("context", f"{context.get('max'):,}" if context.get("max") else ""),
                    ("weights", term.gigabytes(footprint.get("weights_gb"))),
                    ("min VRAM", term.gigabytes(footprint.get("min_vram_gb"))),
                    ("license", entry.get("license") or ""),
                    ("capabilities", ", ".join(caps)),
                    ("recipe", entry.get("recipe") or ""),
                    ("status", entry.get("status") or ""),
                    ("installed", "yes" if entry.get("installed") else "no"),
                    ("fits this machine", "yes" if entry.get("fits") else "no"),
                ]
            )
        )

        detail = [
            (key.replace("_", " "), _readable(value))
            for key, value in quant.items()
            if value is not None
        ]
        if detail:
            _out()
            _out(term.paint("  quantization schema", "dim"))
            _out(term.keyvalues(detail, indent="    "))

        bench = entry.get("benchmark") or {}
        ttft, tps = bench.get("ttft_s") or {}, bench.get("tps") or {}
        if ttft:
            _out()
            _out(term.paint(f"  published performance ({bench.get('release', '')})", "dim"))
            rows = [
                [
                    f"{int(ctx):,}",
                    f"{ttft[ctx]:.2f} s" if ttft.get(ctx) else "-",
                    f"{tps.get(ctx):.1f}" if tps.get(ctx) else "-",
                ]
                for ctx in sorted(ttft, key=int)
            ]
            _out(term.table(["PROMPT", "TTFT", "TOK/S"], rows, right=(1, 2), indent="    "))
        return 0


def local(args: argparse.Namespace) -> int:
    with _client(args) as client:
        data = client.get("/v1/local")
        entries = data.get("models") or []
        if not entries:
            _out("no models installed. Pull one with: hipep pull <model>")
            return 0
        if args.json:
            _out(json.dumps(entries, indent=2))
            return 0
        now = time.time()
        rows = [
            [
                entry["id"],
                term.bytes_human(entry.get("size_bytes")),
                entry.get("recipe") or "-",
                entry.get("origin") or "-",
                term.ago(entry.get("installed_at"), now),
            ]
            for entry in entries
        ]
        _out(term.table(["MODEL", "SIZE", "RECIPE", "ORIGIN", "INSTALLED"], rows, right=(1,)))
        _out()
        _out(
            term.paint(
                f"{len(entries)} installed, {term.bytes_human(data.get('total_bytes'))} "
                f"in {data.get('root')}",
                "dim",
            )
        )
        return 0


def remove(args: argparse.Namespace) -> int:
    with _client(args) as client:
        if not args.yes:
            answer = input(f"remove {args.model} and its files? [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                _out("cancelled.")
                return 1
        result = client.delete(f"/v1/local/{args.model}")
        _out(f"removed {result['deleted']}")
        return 0


def recipes(args: argparse.Namespace) -> int:
    with _client(args) as client:
        entries = (client.get("/v1/recipes") or {}).get("recipes") or []
        if args.json:
            _out(json.dumps(entries, indent=2))
            return 0
        rows = []
        for entry in entries:
            quant = entry.get("quantization") or {}
            summary = " ".join(
                str(v)
                for v in (
                    quant.get("weight_dtype"),
                    quant.get("scheme"),
                    f"g{quant['group_size']}" if quant.get("group_size") else None,
                )
                if v
            )
            rows.append(
                [
                    entry.get("id", "-"),
                    summary or "fp16",
                    entry.get("recommended_for") or entry.get("name") or "-",
                ]
            )
        _out(term.table(["RECIPE", "QUANTIZATION", "RECOMMENDED FOR"], rows))
        return 0


# ----------------------------------------------------------------------------- run


def _image_part(path: str) -> dict[str, Any]:
    """Inline a local image as a data URI.

    The server deliberately does not fetch remote URLs, so the client is the only
    place a file can be turned into something the API accepts.
    """
    file = Path(path).expanduser()
    if not file.is_file():
        raise SystemExit(f"no such image: {file}")
    mime = mimetypes.guess_type(file.name)[0] or "image/png"
    encoded = base64.b64encode(file.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def _user_message(text: str, images: list[str]) -> dict[str, Any]:
    if not images:
        return {"role": "user", "content": text}
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    content.extend(_image_part(path) for path in images)
    return {"role": "user", "content": content}


def _generate(
    client: Client,
    model_id: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int | None,
    temperature: float | None,
    quiet: bool,
) -> tuple[str, dict[str, Any]]:
    """Stream one turn to stdout and return the text plus the usage block."""
    payload: dict[str, Any] = {"model": model_id, "messages": messages, "stream": True}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    started = time.perf_counter()
    first_at: float | None = None
    parts: list[str] = []
    usage: dict[str, Any] = {}

    for frame in client.stream_events("/v1/chat/completions", payload, method="POST"):
        if isinstance(frame, dict) and frame.get("error"):
            raise ServerError(500, str(frame["error"].get("message") or frame["error"]))
        for choice in frame.get("choices") or []:
            piece = (choice.get("delta") or {}).get("content")
            if piece:
                if first_at is None:
                    first_at = time.perf_counter()
                parts.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
        if frame.get("usage"):
            usage = frame["usage"]

    if parts and not parts[-1].endswith("\n"):
        _out()
    if not quiet:
        _out()
        _out(term.paint("  " + _timing_line(usage, len(parts), started, first_at), "dim"))
    return "".join(parts), usage


def _timing_line(
    usage: dict[str, Any], pieces: int, started: float, first_at: float | None
) -> str:
    """One line accounting for the whole turn.

    Model load is called out separately because it dominates a first request and is
    otherwise indistinguishable, from the client's side, from a slow prefill.
    """
    observed = (first_at or time.perf_counter()) - started
    load = usage.get("load_s") or 0.0
    parts = [
        f"{usage.get('completion_tokens', pieces)} tokens",
        f"TTFT {usage['ttft_s']:.3f}s" if usage.get("ttft_s") else "TTFT -",
        f"{usage['tps']:.1f} tok/s" if usage.get("tps") else "- tok/s",
    ]
    if load >= 0.05:
        parts.append(f"model load {load:.2f}s (first request only)")
    parts.append(f"client first token {observed:.3f}s")
    return " - ".join(parts)


def run(args: argparse.Namespace) -> int:
    with _client(args) as client:
        entry = client.resolve_model(args.model)
        model_id = entry["id"]
        if args.image and not (entry.get("capabilities") or {}).get("vision"):
            _out(f"{model_id} does not accept images. Try: hipep models --tier vlm")
            return 1
        if not entry.get("installed") and not args.no_pull:
            _out(term.paint(f"{model_id} is not installed; pulling it first.", "dim"))
            if _follow_job(client, client.post("/v1/pull", {"model": model_id})["job"]) != 0:
                return 1

        messages: list[dict[str, Any]] = []
        if args.system:
            messages.append({"role": "system", "content": args.system})

        prompt = " ".join(args.prompt).strip() if args.prompt else ""
        if not prompt and not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()

        if prompt and not args.chat:
            messages.append(_user_message(prompt, args.image))
            _generate(
                client,
                model_id,
                messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                quiet=args.quiet,
            )
            return 0

        engine = client.health().get("engine", "?")
        _out(term.paint(f"{model_id} on {engine}. Blank line or Ctrl-D to exit.", "dim"))

        # Images attach to the first turn only; a follow-up question refers to the
        # image already in the conversation rather than re-uploading it.
        pending = _user_message(prompt, args.image) if prompt else None
        while True:
            if pending is None:
                try:
                    typed = input(term.paint(">>> ", "cyan")).strip()
                except (EOFError, KeyboardInterrupt):
                    _out()
                    return 0
                if not typed:
                    return 0
                pending = _user_message(typed, [])

            messages.append(pending)
            pending = None
            try:
                text, _ = _generate(
                    client,
                    model_id,
                    messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    quiet=args.quiet,
                )
            except KeyboardInterrupt:
                _out()
                _out(term.paint("  cancelled", "yellow"))
                messages.pop()
                continue
            messages.append({"role": "assistant", "content": text})


# ---------------------------------------------------------------------------- jobs


def _follow_job(client: Client, job: dict[str, Any], *, verbose: bool = False) -> int:
    """Render a job's progress until it reaches a terminal state.

    Returns 0 when the job succeeded, so callers can chain on it.
    """
    job_id = job["id"]
    _out(term.paint(f"{job['title']}  ({job_id})", "bold"))
    line = term.Line()
    seen_logs = 0
    final: dict[str, Any] = job

    for update in client.stream_events(f"/v1/jobs/{job_id}/events"):
        final = update
        if verbose:
            logs = update.get("logs") or []
            for entry in logs[seen_logs:]:
                line.done()
                _out(f"  {entry['text']}")
            seen_logs = max(seen_logs, len(logs))

        stage = next((s for s in update.get("stages") or [] if s["state"] == "running"), None)
        label = stage["label"] if stage else update["state"]
        detail = (stage or {}).get("detail") or ""
        line.update(
            f"  {term.bar(update.get('progress') or 0.0)} "
            f"{int((update.get('progress') or 0) * 100):3d}%  {label}"
            + (f" - {detail}" if detail else "")
        )
        if update["state"] in {"succeeded", "failed", "cancelled"}:
            break

    line.done()
    state = final.get("state", "unknown")
    duration = term.duration(final.get("duration_s"))
    if state == "succeeded":
        _out(f"  {term.state_label(state)} in {duration}")
        _print_job_result(final)
        return 0
    _out(f"  {term.state_label(state)} after {duration}")
    if final.get("error"):
        _out(f"  {term.paint(final['error'], 'red')}")
    if not verbose:
        _out(term.paint(f"  full log: hipep jobs {job_id} --logs", "dim"))
    return 1


# Keys that only make sense while a job is running, or that duplicate a nicer field.
_RESULT_NOISE = {"completed_bytes", "total_bytes", "engine", "device", "simulated"}
_BYTE_KEYS = {"size_bytes"}


def _print_job_result(job: dict[str, Any]) -> None:
    result = job.get("result") or {}
    if not result:
        return
    if result.get("summary"):
        _print_bench_result(result["summary"])
        return

    pairs = []
    for key, value in result.items():
        if key in _RESULT_NOISE or isinstance(value, dict | list):
            continue
        text = term.bytes_human(value) if key in _BYTE_KEYS else _readable(value)
        pairs.append((key.replace("_", " "), text))
    if pairs:
        _out(term.keyvalues(pairs, indent="  "))

    if result.get("simulated"):
        _out()
        _out(
            term.paint(
                "  simulated: this server runs the replay engine, so no artifact was "
                "produced and nothing was transferred.",
                "yellow",
            )
        )
    if result.get("model"):
        command = term.paint("hipep run " + str(result["model"]), "bold")
        _out()
        _out(f"  run it: {command}")
    if result.get("share_command"):
        _out(f"  share:  {term.paint(str(result['share_command']), 'bold')}")


def _print_bench_result(summary: dict[str, Any]) -> None:
    """Measured numbers beside the published ones, with the transport cost separated.

    The delta compares engine measurements against certified engine measurements. HTTP
    and scheduling cost is shown in its own column so it is not read as a regression.
    """
    contexts = summary.get("contexts") or {}
    if not contexts:
        return
    published = summary.get("published") or {}
    deltas = summary.get("delta_vs_published") or {}

    rows = []
    for key in sorted(contexts, key=int):
        row = contexts[key]
        reference = published.get(key) or {}
        delta = deltas.get(key) or {}
        rows.append(
            [
                f"{int(key):,}",
                f"{row.get('ttft_s', 0):.3f} s",
                f"{row.get('tps', 0):.1f}",
                f"+{row.get('transport_s', 0):.3f} s",
                f"{reference['ttft_s']:.3f} s" if reference.get("ttft_s") else "-",
                f"{reference['tps']:.1f}" if reference.get("tps") else "-",
                _delta_text(delta),
            ]
        )
    _out()
    _out(
        term.table(
            ["PROMPT", "TTFT", "TOK/S", "HTTP", "CERT TTFT", "CERT TOK/S", "VS CERTIFIED"],
            rows,
            right=(1, 2, 3, 4, 5),
            indent="  ",
        )
    )
    if summary.get("model"):
        _out()
        _out(
            term.paint(
                f"  {summary['model']} - {summary.get('repeats', '?')} runs per context, "
                f"{summary.get('generate_tokens', '?')} tokens generated, "
                f"median reported",
                "dim",
            )
        )


def _delta_text(delta: dict[str, Any]) -> str:
    if not delta:
        return "-"
    ttft, tps = delta.get("ttft_pct"), delta.get("tps_pct")
    if ttft is None or tps is None:
        return "-"
    # Faster means lower TTFT and higher tokens/s, so the two colour rules differ.
    parts = [
        term.paint(f"TTFT {ttft:+.1f}%", "green" if ttft <= 3 else "yellow"),
        term.paint(f"tok/s {tps:+.1f}%", "green" if tps >= -3 else "yellow"),
    ]
    return "  ".join(parts)


def jobs(args: argparse.Namespace) -> int:
    with _client(args) as client:
        if not args.job:
            entries = (client.get("/v1/jobs", limit=args.limit) or {}).get("jobs") or []
            if not entries:
                _out("no jobs yet.")
                return 0
            now = time.time()
            rows = [
                [
                    entry["id"],
                    entry["kind"],
                    term.state_label(entry["state"]),
                    f"{int((entry.get('progress') or 0) * 100)}%",
                    term.duration(entry.get("duration_s")),
                    term.ago(entry.get("created_at"), now),
                    entry["title"],
                ]
                for entry in entries
            ]
            _out(
                term.table(
                    ["JOB", "KIND", "STATE", "DONE", "TOOK", "STARTED", "TITLE"],
                    rows,
                    right=(3, 4),
                )
            )
            return 0

        job = client.get(f"/v1/jobs/{args.job}")
        if args.json:
            _out(json.dumps(job, indent=2))
            return 0
        if args.follow and job["state"] in {"pending", "running"}:
            return _follow_job(client, job, verbose=args.logs)

        _out(term.paint(f"{job['title']}  ({job['id']})", "bold"))
        _out(
            term.keyvalues(
                [
                    ("state", term.state_label(job["state"])),
                    ("progress", f"{int((job.get('progress') or 0) * 100)}%"),
                    ("duration", term.duration(job.get("duration_s"))),
                    ("error", job.get("error") or ""),
                ]
            )
        )
        _out()
        rows = [
            [
                stage["label"],
                term.state_label(stage["state"]),
                f"{int((stage.get('progress') or 0) * 100)}%",
                term.duration(stage.get("duration_s")),
                stage.get("detail") or "",
            ]
            for stage in job.get("stages") or []
        ]
        _out(term.table(["STAGE", "STATE", "DONE", "TOOK", "DETAIL"], rows, right=(2, 3)))
        _print_job_result(job)
        if args.logs:
            _out()
            for entry in job.get("logs") or []:
                _out(f"  {entry['text']}")
        return 0


def cancel(args: argparse.Namespace) -> int:
    with _client(args) as client:
        client.post(f"/v1/jobs/{args.job}/cancel")
        _out(f"cancelled {args.job}")
        return 0


def pull(args: argparse.Namespace) -> int:
    with _client(args) as client:
        if args.source:
            # Importing from a peer: the model id is the peer's, so it must not be
            # resolved against this server's catalog, which has never heard of it.
            job = client.post(
                "/v1/import",
                {"url": args.source, "model": args.model, "name": args.name},
            )["job"]
        else:
            entry = client.resolve_model(args.model)
            job = client.post("/v1/pull", {"model": entry["id"]})["job"]
        if args.detach:
            _out(f"{job['id']} started. Follow it with: hipep jobs {job['id']} --follow")
            return 0
        return _follow_job(client, job, verbose=args.verbose)


def port(args: argparse.Namespace) -> int:
    payload = {
        "source": args.source,
        "name": args.name,
        "quantization": args.recipe,
        "chunk_size": args.chunk_size,
        "prune_logit": args.prune_logit,
        "calibration_samples": args.calibration_samples,
        "benchmark": not args.no_benchmark,
    }
    with _client(args) as client:
        job = client.post("/v1/port", {k: v for k, v in payload.items() if v is not None})[
            "job"
        ]
        _out(
            term.paint(
                f"  recipe {args.recipe}, chunk_size {args.chunk_size}, "
                f"prune_logit {'on' if args.prune_logit else 'off'}",
                "dim",
            )
        )
        if args.detach:
            _out(f"{job['id']} started. Follow it with: hipep jobs {job['id']} --follow")
            return 0
        return _follow_job(client, job, verbose=args.verbose)


def bench(args: argparse.Namespace) -> int:
    with _client(args) as client:
        entry = client.resolve_model(args.model)
        payload = {
            "model": entry["id"],
            "contexts": args.contexts,
            "generate_tokens": args.tokens,
            "repeats": args.repeats,
            "warmup": args.warmup,
        }
        job = client.post("/v1/benchmarks/run", payload)["job"]
        if args.detach:
            _out(f"{job['id']} started. Follow it with: hipep jobs {job['id']} --follow")
            return 0
        return _follow_job(client, job, verbose=args.verbose)


# -------------------------------------------------------------------------- status


def status(args: argparse.Namespace) -> int:
    with _client(args) as client:
        data = client.status()
        if args.json:
            _out(json.dumps(data, indent=2))
            return 0

        engine = data.get("engine") or {}
        host = data.get("host") or {}
        load = data.get("load") or {}
        limits = data.get("limits") or {}
        totals = data.get("totals") or {}
        storage = data.get("storage") or {}
        catalog = data.get("catalog") or {}

        _out(term.paint(f"oga-server {data.get('version', '')} at {client.url}", "bold"))
        _out()
        rows = [
            (
                "engine",
                f"{engine.get('name')}"
                + ("" if engine.get("real") else term.paint("  (replay)", "yellow")),
            ),
            ("engine reason", data.get("engine_reason") or ""),
            ("uptime", term.duration(data.get("uptime_s"))),
            (
                "host",
                f"{host.get('hostname', '')} "
                f"({host.get('os', '')} {host.get('arch', '')})".strip(),
            ),
        ]
        gpus = host.get("gpus") or []
        for gpu in gpus:
            rows.append(
                (
                    f"gpu {gpu.get('index', 0)}",
                    f"{gpu.get('name')} ({gpu.get('gfx_arch')}), "
                    f"{term.gigabytes(gpu.get('memory_gb'))}, "
                    f"profile {gpu.get('profile')}",
                )
            )
        if not gpus:
            rows.append(("gpu", term.paint("none detected", "yellow")))
        rows.extend(
            [
                ("system memory", term.gigabytes(host.get("memory_gb"))),
                ("catalog", f"{catalog.get('models', 0)} models, {catalog.get('release')}"),
                (
                    "installed",
                    f"{storage.get('models', 0)} models, "
                    f"{term.bytes_human(storage.get('bytes'))}",
                ),
                ("store", storage.get("root") or ""),
            ]
        )
        _out(term.keyvalues(rows))
        _out()
        _out(term.paint("  load", "dim"))
        _out(
            term.keyvalues(
                [
                    (
                        "in flight",
                        f"{load.get('active_requests', 0)} "
                        f"(queued {load.get('queued_requests', 0)} "
                        f"of {limits.get('max_queue_depth', '-')})",
                    ),
                    (
                        "loaded models",
                        f"{load.get('loaded_models', 0)} "
                        f"of {limits.get('max_loaded_models', '-')}",
                    ),
                    ("sessions", str(load.get("sessions", 0))),
                    (
                        "served",
                        f"{totals.get('requests', 0)} requests, "
                        f"{totals.get('completion_tokens', 0):,} tokens generated, "
                        f"{totals.get('errors', 0)} errors, "
                        f"{totals.get('rejected', 0)} rejected",
                    ),
                ],
                indent="    ",
            )
        )

        sessions = data.get("sessions_detail") or []
        if sessions:
            _out()
            _out(term.paint("  sessions", "dim"))
            rows = [
                [
                    session["id"],
                    session.get("client") or "-",
                    session.get("api") or "-",
                    str(session.get("requests", 0)),
                    f"{session.get('completion_tokens', 0):,}",
                    term.duration(session.get("idle_s")),
                ]
                for session in sessions
            ]
            _out(
                term.table(
                    ["SESSION", "CLIENT", "API", "REQS", "TOKENS", "IDLE"],
                    rows,
                    right=(3, 4, 5),
                    indent="    ",
                )
            )
        return 0


def ps(args: argparse.Namespace) -> int:
    with _client(args) as client:
        data = client.status()
        loaded = (data.get("engine") or {}).get("loaded") or []
        active = data.get("active") or []

        if args.watch:
            return _watch_ps(client)

        _out(
            term.paint(
                f"{len(loaded)} model(s) resident: {', '.join(loaded) or 'none'}", "dim"
            )
        )
        if not active:
            _out("nothing in flight.")
            return 0
        _out()
        _out(_ps_table(active))
        return 0


def _ps_table(active: list[dict[str, Any]]) -> str:
    rows = [
        [
            entry["id"],
            entry["model"],
            entry.get("api") or "-",
            entry.get("state") or "-",
            f"{entry.get('wait_s', 0):.1f}s",
            f"{entry.get('ttft_s'):.3f}s" if entry.get("ttft_s") else "-",
            str(entry.get("tokens", 0)),
            f"{entry.get('tps'):.1f}" if entry.get("tps") else "-",
        ]
        for entry in active
    ]
    return term.table(
        ["REQUEST", "MODEL", "API", "STATE", "WAIT", "TTFT", "TOKENS", "TOK/S"],
        rows,
        right=(4, 5, 6, 7),
    )


def _watch_ps(client: Client) -> int:
    """Live view of in-flight requests, driven by the status event stream."""
    _out(term.paint("watching in-flight requests. Ctrl-C to stop.", "dim"))
    line = term.Line()
    try:
        for update in client.stream_events("/v1/status/events"):
            load = update.get("load") or {}
            active = update.get("active") or []
            summary = (
                f"  {load.get('active_requests', 0)} active, "
                f"{load.get('queued_requests', 0)} queued, "
                f"{load.get('sessions', 0)} sessions, "
                f"{load.get('loaded_models', 0)} models resident"
            )
            if active:
                busiest = active[0]
                summary += f"  |  {busiest['model']} {busiest.get('state', '')}"
                if busiest.get("tps"):
                    summary += f" {busiest['tps']:.1f} tok/s"
            line.update(summary)
    except KeyboardInterrupt:
        pass
    line.done()
    return 0
