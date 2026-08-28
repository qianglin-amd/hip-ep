# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Command line entry point: ``python -m hipep_server``."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .config import DEFAULT_PORT, Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oga-server",
        description="OpenAI- and Ollama-compatible inference server for AMD GPUs (HIP EP).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=None, help="bind address")
    parser.add_argument("--port", type=int, default=None, help=f"bind port (Ollama uses {DEFAULT_PORT})")
    parser.add_argument(
        "--engine",
        choices=["auto", "oga", "replay"],
        default=None,
        help="inference engine; auto uses OGA when a GPU and onnxruntime-genai are present",
    )
    parser.add_argument("--data-dir", default=None, help="model store and job workspace")
    parser.add_argument(
        "--max-loaded", type=int, default=None, help="how many models may be resident at once"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=None, help="in-flight requests per model"
    )
    parser.add_argument("--reload", action="store_true", help="auto-reload on source changes")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )
    parser.add_argument("--version", action="version", version=f"oga-server {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    settings = Settings()
    for attr, value in (
        ("host", args.host),
        ("port", args.port),
        ("engine", args.engine),
        ("max_loaded_models", args.max_loaded),
        ("max_concurrent_per_model", args.max_concurrent),
    ):
        if value is not None:
            setattr(settings, attr, value)
    if args.data_dir:
        from pathlib import Path

        settings.data_dir = Path(args.data_dir).expanduser().resolve()

    import uvicorn

    from .app import create_app

    if args.reload:
        # uvicorn's reloader needs an import string, so settings travel by environment.
        import os

        os.environ["HIPEP_HOST"] = settings.host
        os.environ["HIPEP_PORT"] = str(settings.port)
        os.environ["HIPEP_ENGINE"] = settings.engine
        os.environ["HIPEP_HOME"] = str(settings.data_dir)
        uvicorn.run(
            "hipep_server.app:create_app",
            factory=True,
            host=settings.host,
            port=settings.port,
            reload=True,
            log_level=args.log_level,
        )
        return 0

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=args.log_level,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
