# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import ROUTERS
from .catalog import load_catalog, merge_local_models
from .config import Settings
from .config import settings as default_settings
from .engines import select_engine
from .hardware import detect_host
from .jobs import JobManager
from .pipeline import Pipeline
from .scheduler import Scheduler
from .store import ModelStore

log = logging.getLogger(__name__)

DESCRIPTION = """
An OpenAI- and Ollama-compatible inference server for AMD GPUs, backed by HIP EP.

* **OpenAI** — point any OpenAI SDK at `/v1` and change nothing else.
* **Ollama** — the default port is 11434, so existing Ollama clients work as-is.
* **Console** — HIP EP Studio is served at `/`.
* **Docs** — the generated documentation site is served at `/docs`, when it has been built.

Chat responses carry `ttft_s`, `decode_s` and `tps` in the usage block, because a local
inference server that cannot tell you its latency is not much use.
"""


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    settings.ensure_dirs()

    app.state.host = detect_host()
    app.state.catalog = load_catalog(settings.catalog_dir, settings.benchmarks_dir)
    app.state.store = ModelStore(settings.models_dir)
    imported = merge_local_models(app.state.catalog, app.state.store.list())

    engine, reason = select_engine(settings, app.state.host)
    app.state.engine_reason = reason
    app.state.scheduler = Scheduler(engine, app.state.catalog, settings)
    app.state.jobs = JobManager(settings.jobs_dir, settings.max_concurrent_jobs)
    app.state.pipeline = Pipeline(
        jobs=app.state.jobs,
        store=app.state.store,
        catalog=app.state.catalog,
        settings=settings,
        simulate=not engine.real,
    )

    gpu = app.state.host.primary_gpu
    log.info("hip-ep oga-server %s", __version__)
    log.info("  host    %s", app.state.host.hostname)
    log.info("  gpu     %s", f"{gpu.name} ({gpu.gfx_arch})" if gpu else "none detected")
    # Plain ASCII: the Windows console defaults to cp1252 and mangles anything else.
    log.info("  engine  %s - %s", engine.name, reason)
    log.info(
        "  catalog %d models (%s)",
        len(app.state.catalog),
        app.state.catalog.benchmark_meta.get("release", "no benchmarks"),
    )
    if imported:
        log.info(
            "  local   %d imported model(s): %s",
            len(imported),
            ", ".join(m.id for m in imported),
        )
    log.info("  models  %s", settings.models_dir)
    log.info("  listen  %s", settings.base_url)

    sweeper = asyncio.create_task(_idle_sweeper(app), name="idle-sweeper")
    try:
        yield
    finally:
        sweeper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sweeper
        for model_id in list(engine.loaded_models()):
            with contextlib.suppress(Exception):
                await engine.unload(model_id)


async def _idle_sweeper(app: FastAPI) -> None:
    """Periodically release models nobody is using, freeing GPU memory."""
    while True:
        await asyncio.sleep(60)
        with contextlib.suppress(Exception):
            await app.state.scheduler.unload_idle()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or default_settings
    app = FastAPI(
        title="oga-server",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        # /docs is the documentation site; the API reference moves aside.
        docs_url="/api-docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.settings = settings

    if settings.cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-HIPEP-Session"],
        )

    for router in ROUTERS:
        app.include_router(router)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "type": type(exc).__name__,
                    "path": request.url.path,
                }
            },
        )

    _mount_static(app, settings)
    return app


def _mount_static(app: FastAPI, settings: Settings) -> None:
    """Serve the console at / and the documentation site at /docs.

    Both are plain static assets with no build step, so they work offline and can be
    published to any static host unchanged.
    """
    site = settings.site_dir
    if site.is_dir():
        docs = site / "docs"
        if docs.is_dir():
            # A mount serves its index at "/docs/" only; the bare "/docs" is a miss.
            # Registered before the mount so it matches first. Without it the console's
            # own Docs link 404s, as does the path a reader would type.
            @app.get("/docs", include_in_schema=False)
            async def docs_index() -> RedirectResponse:
                return RedirectResponse("/docs/")

            app.mount("/docs", StaticFiles(directory=str(docs), html=True), name="docs")
        assets = site / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

        # response_model=None: the return type is a union of Response subclasses, and
        # FastAPI would otherwise try to derive a Pydantic response model from it.
        @app.get("/landing", include_in_schema=False, response_model=None)
        async def landing() -> FileResponse | RedirectResponse:
            index = site / "index.html"
            if index.is_file():
                return FileResponse(index)
            return RedirectResponse("/")

    web = settings.web_dir
    if web.is_dir():
        # Mounted last and at the root so every unmatched path falls through to the
        # console's index.html, which is what a single-page app needs.
        app.mount("/", StaticFiles(directory=str(web), html=True), name="console")
    else:

        @app.get("/", include_in_schema=False)
        async def root() -> JSONResponse:
            return JSONResponse(
                {
                    "name": "oga-server",
                    "version": __version__,
                    "console": "not built",
                    "api": {"openai": "/v1", "ollama": "/api", "reference": "/api-docs"},
                }
            )
