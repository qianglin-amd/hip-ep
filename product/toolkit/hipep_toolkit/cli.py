# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Argument parsing and dispatch for the ``hipep`` command.

The verbs follow Ollama's where they mean the same thing (``serve``, ``pull``, ``run``,
``ls``, ``rm``, ``ps``) so that knowledge transfers, and add the ones a compiler-backed
stack needs (``port``, ``bench``, ``recipes``, ``jobs``).
"""

from __future__ import annotations

import argparse
import contextlib
import sys

from . import __version__, commands
from .client import DEFAULT_URL, ServerError, ServerUnreachable

EPILOG = """\
examples:
  hipep doctor                        check drivers, wheels, GPU and store
  hipep serve --port 11434            start the server and the web console
  hipep models --tier vlm             certified vision models
  hipep pull llama3.1:8b              download a certified model
  hipep run llama3.1:8b "why HIP?"    one-shot generation
  hipep run gemma3-vl:4b --image cat.jpg "what is this?"
  hipep run                           interactive chat with an installed model
  hipep port Qwen/Qwen2.5-7B-Instruct --name qwen2.5:7b-mine
  hipep bench llama3.1:8b --contexts 128 2048
  hipep status                        engine, hardware, load and sessions
"""


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--url",
        default=None,
        metavar="URL",
        help=f"server address (default: $HIPEP_URL or {DEFAULT_URL})",
    )


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="print the raw JSON response")


def _add_job_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--detach", action="store_true", help="return as soon as the job is queued"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print the job log as it runs"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hipep",
        description="Run and manage local LLMs and VLMs on AMD GPUs with HIP EP.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"hipep {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # --- serve ---------------------------------------------------------------
    p = sub.add_parser(
        "serve",
        help="start oga-server and the web console",
        description=(
            "Start the server. Remaining arguments are passed to oga-server, so "
            "'hipep serve --help' after '--' shows its own options."
        ),
    )
    p.add_argument(
        "server_args",
        nargs=argparse.REMAINDER,
        metavar="[server options]",
        help="forwarded to oga-server, e.g. --port 8080 --engine replay",
    )
    p.set_defaults(handler=commands.serve)

    # --- catalog -------------------------------------------------------------
    p = sub.add_parser("models", help="list the certified model zoo")
    _add_common(p)
    _add_json(p)
    p.add_argument("--tier", choices=["llm", "vlm", "speech", "vision"], help="filter by kind")
    p.add_argument("-s", "--search", metavar="TEXT", help="filter by name, family or id")
    p.set_defaults(handler=commands.models)

    p = sub.add_parser("show", help="model card: quantization, footprint, published numbers")
    _add_common(p)
    _add_json(p)
    p.add_argument("model", help="model id, or a prefix of one")
    p.set_defaults(handler=commands.show)

    p = sub.add_parser("ls", aliases=["list"], help="models installed on this machine")
    _add_common(p)
    _add_json(p)
    p.set_defaults(handler=commands.local)

    p = sub.add_parser("rm", help="delete an installed model and its files")
    _add_common(p)
    p.add_argument("model")
    p.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    p.set_defaults(handler=commands.remove)

    p = sub.add_parser("recipes", help="export and quantization recipes available for porting")
    _add_common(p)
    _add_json(p)
    p.set_defaults(handler=commands.recipes)

    # --- inference -----------------------------------------------------------
    p = sub.add_parser(
        "run",
        help="generate from a model, or chat with it",
        description=(
            "With a prompt, generates once and exits. Without one, opens an "
            "interactive chat. Reads a prompt from stdin when it is not a terminal."
        ),
    )
    _add_common(p)
    p.add_argument("model", nargs="?", help="model id; defaults to an installed one")
    p.add_argument("prompt", nargs="*", help="the prompt; omit for interactive chat")
    p.add_argument(
        "--image",
        action="append",
        default=[],
        metavar="PATH",
        help="attach a local image (repeatable); requires a vision model",
    )
    p.add_argument("--system", metavar="TEXT", help="system message")
    p.add_argument("-n", "--max-tokens", type=int, help="cap the generated length")
    p.add_argument("--temperature", type=float, help="sampling temperature")
    p.add_argument(
        "-c", "--chat", action="store_true", help="stay interactive after the first prompt"
    )
    p.add_argument("-q", "--quiet", action="store_true", help="output only the generated text")
    p.add_argument(
        "--no-pull", action="store_true", help="fail instead of installing a missing model"
    )
    p.set_defaults(handler=commands.run)

    # --- model lifecycle -----------------------------------------------------
    p = sub.add_parser(
        "pull",
        help="download a certified model, or import one another server is sharing",
        description=(
            "Without --from, downloads the certified artifact named in the catalog. "
            "With --from, copies the model from a peer oga-server, which is how a "
            "colleague installs a model you ported without repeating the port."
        ),
    )
    _add_common(p)
    _add_job_flags(p)
    p.add_argument("model")
    p.add_argument(
        "--from",
        dest="source",
        metavar="URL",
        help="base address of a peer oga-server sharing this model",
    )
    p.add_argument("--name", metavar="ID", help="install under a different id")
    p.set_defaults(handler=commands.pull)

    p = sub.add_parser(
        "port",
        help="bring your own model: fetch, export, quantize, convert, validate, benchmark",
        description=(
            "Port a Hugging Face repo or a local checkpoint into a model this server "
            "can serve. Quark defaults are chunk_size 512 and logit pruning on."
        ),
    )
    _add_common(p)
    _add_job_flags(p)
    p.add_argument("source", help="Hugging Face repo id, URL, or local path")
    p.add_argument("--name", metavar="ID", help="model id to publish as")
    p.add_argument(
        "--recipe",
        default="int4-rtn-g128",
        metavar="ID",
        help="export and quantization recipe (see 'hipep recipes')",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Quark chunk size; lower it if quantization runs out of memory",
    )
    p.add_argument(
        "--no-prune-logit",
        dest="prune_logit",
        action="store_false",
        help="keep the full LM head instead of pruning it",
    )
    p.add_argument(
        "--calibration-samples",
        type=int,
        default=128,
        help="samples used by calibrating recipes such as AWQ",
    )
    p.add_argument(
        "--no-benchmark", action="store_true", help="publish without measuring first"
    )
    p.set_defaults(handler=commands.port, prune_logit=True)

    # --- measurement ---------------------------------------------------------
    p = sub.add_parser(
        "bench",
        help="measure TTFT and tokens/s on this machine",
        description=(
            "Runs the same harness that produced the published numbers, over the same "
            "HTTP path, and prints your result beside the certified one."
        ),
    )
    _add_common(p)
    _add_job_flags(p)
    p.add_argument("model", nargs="?", help="model id; defaults to an installed one")
    p.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[128, 2048],
        metavar="N",
        help="prompt lengths to measure",
    )
    p.add_argument(
        "-n", "--tokens", type=int, default=64, help="tokens to generate per measurement"
    )
    p.add_argument("--repeats", type=int, default=3, help="measured runs per context")
    p.add_argument("--warmup", type=int, default=1, help="unmeasured runs per context")
    p.set_defaults(handler=commands.bench)

    # --- diagnostics ---------------------------------------------------------
    p = sub.add_parser(
        "doctor",
        help="check this machine: drivers, wheels, GPU, store, catalog, port",
        description=(
            "Preflight the install. Runs without a server and reports each "
            "precondition separately, so a failure names the layer that owns it. "
            "Exits non-zero only on a hard failure; a GPU-free machine warns and "
            "still passes, because it can serve the replay engine and run the tests."
        ),
    )
    _add_common(p)
    _add_json(p)
    p.add_argument(
        "--no-server", action="store_true", help="skip the check for a running server"
    )
    p.set_defaults(handler=commands.doctor)

    # --- server introspection ------------------------------------------------
    p = sub.add_parser("status", help="engine, hardware, limits, load and sessions")
    _add_common(p)
    _add_json(p)
    p.set_defaults(handler=commands.status)

    p = sub.add_parser("ps", help="requests in flight and models resident")
    _add_common(p)
    p.add_argument("-w", "--watch", action="store_true", help="follow the live status stream")
    p.set_defaults(handler=commands.ps)

    p = sub.add_parser("jobs", help="list jobs, or inspect and follow one")
    _add_common(p)
    _add_json(p)
    p.add_argument("job", nargs="?", help="job id; omit to list recent jobs")
    p.add_argument("-f", "--follow", action="store_true", help="stream progress until it ends")
    p.add_argument("--logs", action="store_true", help="include the job log")
    p.add_argument("--limit", type=int, default=20, help="how many jobs to list")
    p.set_defaults(handler=commands.jobs)

    p = sub.add_parser("cancel", help="cancel a running job")
    _add_common(p)
    p.add_argument("job")
    p.set_defaults(handler=commands.cancel)

    return parser


def _use_utf8_output() -> None:
    """Write generated text as UTF-8 regardless of the console's code page.

    Model output is arbitrary Unicode and a Windows console still defaults to a legacy
    code page, where an em dash raises UnicodeEncodeError. Python's replacement path is
    also slow enough to distort the client-side latency the CLI reports, so the first
    token appears to arrive seconds late.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(OSError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _use_utf8_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 1

    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print()
        return 130
    except ServerUnreachable as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except ServerError as exc:
        print(f"server: {exc.detail}", file=sys.stderr)
        return 4
