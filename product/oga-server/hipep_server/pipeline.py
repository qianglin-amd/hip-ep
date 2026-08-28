# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Job pipelines: pull a certified model, port your own model, run a benchmark.

Each pipeline is a list of stages wired onto the generic job runner. The stage bodies
delegate to `hipep_toolkit` when the real toolchain is installed, and otherwise run in
simulation, which is what lets the console demonstrate the full workflow on a machine
with no GPU and no exporter.

Simulation is never silent. Every simulated stage says so in the job log, the job
result carries `simulated: true`, and the console renders a badge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from .catalog import Catalog, Model, model_from_local
from .config import Settings
from .jobs import Job, JobContext, JobManager, Stage
from .store import ModelStore

log = logging.getLogger(__name__)


def _toolkit():
    """Import the toolkit lazily; it is optional at runtime."""
    try:
        import hipep_toolkit

        return hipep_toolkit
    except ImportError:
        return None


PORT_STAGES = [
    ("fetch", "Fetch weights"),
    ("inspect", "Inspect architecture"),
    ("export", "Export to ONNX"),
    ("quantize", "Quantize"),
    ("convert", "Build OGA pipeline"),
    ("validate", "Validate on device"),
    ("benchmark", "Benchmark"),
    ("publish", "Publish to local zoo"),
]

PULL_STAGES = [
    ("resolve", "Resolve source"),
    ("download", "Download artifact"),
    ("verify", "Verify artifact"),
    ("register", "Register in local zoo"),
]

BENCH_STAGES = [
    ("prepare", "Load model"),
    ("warmup", "Warm up"),
    ("measure", "Measure TTFT and TPS"),
    ("summarize", "Summarize"),
]


class Pipeline:
    def __init__(
        self,
        *,
        jobs: JobManager,
        store: ModelStore,
        catalog: Catalog,
        settings: Settings,
        simulate: bool,
    ):
        self.jobs = jobs
        self.store = store
        self.catalog = catalog
        self.settings = settings
        # Simulation follows the engine: a replay-engine server has no real toolchain
        # behind it, so pretending otherwise would only produce confusing failures.
        self.simulate = simulate

    # --- pull -----------------------------------------------------------------

    def start_pull(self, model: Model) -> Job:
        job = self.jobs.create(
            kind="pull",
            title=f"Pull {model.name}",
            stages=[Stage(name=n, label=label) for n, label in PULL_STAGES],
            params={"model": model.id, "source": (model.artifact or {}).get("source", {})},
        )
        return self.jobs.start(
            job,
            {
                "resolve": self._pull_resolve(model),
                "download": self._pull_download(model),
                "verify": self._pull_verify(model),
                "register": self._pull_register(model),
            },
        )

    def _pull_resolve(self, model: Model):
        async def stage(ctx: JobContext) -> None:
            source = (model.artifact or {}).get("source") or {}
            repo = source.get("repo")
            if not repo:
                raise RuntimeError(
                    f"{model.id} has no download source in the catalog. "
                    "Build it locally with `hipep port` instead."
                )
            ctx.log(f"source: {source.get('kind', 'huggingface')}:{repo}")
            target = self.store.path_for(model.id)
            ctx.log(f"target: {target}")
            ctx.result(repo=repo, target=str(target))
            if self.store.is_installed(model.id):
                ctx.log("already installed; the download stage will verify and skip")
            ctx.progress(1.0)

        return stage

    def _pull_download(self, model: Model):
        async def stage(ctx: JobContext) -> None:
            target = self.store.path_for(model.id)
            if self.store.is_installed(model.id):
                ctx.log("artifact already present, skipping download")
                ctx.progress(1.0, "cached")
                return
            repo = ctx.job.result.get("repo")
            total = int(float((model.footprint or {}).get("weights_gb") or 4.0) * 1024**3)
            ctx.result(total_bytes=total, completed_bytes=0)

            if self.simulate:
                ctx.log(
                    "SIMULATED download — no network transfer. The server is running "
                    "the replay engine, so there is no runtime that could use the "
                    "weights.",
                    level="warning",
                )
                await _simulated_transfer(ctx, total, seconds=6.0)
                target.mkdir(parents=True, exist_ok=True)
                (target / "SIMULATED").write_text(
                    f"Simulated pull of {model.id} from {repo}.\n"
                    "No weights were downloaded.\n",
                    encoding="utf-8",
                )
                return

            await self._download_from_hub(ctx, repo, target, total)

        return stage

    async def _download_from_hub(
        self, ctx: JobContext, repo: str, target: Path, total: int
    ) -> None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is not installed; "
                "run `pip install -r product/requirements.txt`"
            ) from exc

        ctx.log(f"downloading {repo} -> {target}")
        loop = asyncio.get_running_loop()

        def work() -> str:
            return snapshot_download(
                repo_id=repo,
                local_dir=str(target),
                max_workers=4,
            )

        # snapshot_download is blocking and has no async progress hook, so the stage
        # reports coarse progress by polling the target directory while it runs.
        task = loop.run_in_executor(None, work)
        while not task.done():
            await asyncio.sleep(1.0)
            got = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            ctx.result(completed_bytes=got)
            ctx.progress(min(0.99, got / total) if total else 0.5, _human(got))
            ctx.check_cancelled()
        await task
        ctx.progress(1.0, "complete")

    def _pull_verify(self, model: Model):
        async def stage(ctx: JobContext) -> None:
            target = self.store.path_for(model.id)
            config = target / "genai_config.json"
            if config.is_file():
                ctx.log("genai_config.json present; artifact is runnable")
                ctx.result(ready=True)
            else:
                ctx.log(
                    "genai_config.json missing; the artifact is registered but not "
                    "runnable. Convert it with `hipep port` before serving.",
                    level="warning",
                )
                ctx.result(ready=False)
            ctx.progress(1.0)

        return stage

    def _pull_register(self, model: Model):
        async def stage(ctx: JobContext) -> None:
            source = (model.artifact or {}).get("source") or {}
            local = self.store.register(
                model.id,
                origin="certified",
                source=f"{source.get('kind', 'huggingface')}:{source.get('repo', '')}",
                recipe=model.recipe,
                note="simulated pull" if self.simulate else None,
                extra={"simulated": self.simulate},
            )
            ctx.result(
                model=model.id,
                path=local.path,
                size_gb=local.size_gb,
                simulated=self.simulate,
            )
            ctx.log(f"registered {model.id} ({local.size_gb} GB)")
            ctx.progress(1.0)

        return stage

    # --- port (bring your own model) -----------------------------------------

    def start_port(self, spec: dict[str, Any]) -> Job:
        source = spec.get("source", "")
        name = spec.get("name") or _derive_id(source)
        job = self.jobs.create(
            kind="port",
            title=f"Port {source}",
            stages=[Stage(name=n, label=label) for n, label in PORT_STAGES],
            params={
                "source": source,
                "name": name,
                "quantization": spec.get("quantization", "int4-rtn-g128"),
                "chunk_size": spec.get("chunk_size", 512),
                "prune_logit": spec.get("prune_logit", True),
                "calibration_samples": spec.get("calibration_samples", 128),
                "device": spec.get("device", "auto"),
                "benchmark": spec.get("benchmark", True),
            },
        )
        return self.jobs.start(job, self._port_stages(job))

    def _port_stages(self, job: Job) -> dict[str, Any]:
        toolkit = _toolkit()
        params = job.params

        async def run(stage_name: str, ctx: JobContext) -> None:
            ctx.check_cancelled()
            if toolkit is not None and not self.simulate:
                fn = getattr(toolkit.stages, stage_name, None)
                if fn is not None:
                    await fn(ctx, params, store=self.store)
                    return
                ctx.log(f"toolkit has no '{stage_name}' stage; simulating", level="warning")
            await _simulate_port_stage(ctx, stage_name, params)

        def make(stage_name: str):
            async def stage(ctx: JobContext) -> None:
                await run(stage_name, ctx)

            return stage

        stages = {name: make(name) for name, _ in PORT_STAGES}

        async def publish(ctx: JobContext) -> None:
            await run("publish", ctx)
            model_id = params["name"]
            local = self.store.register(
                model_id,
                origin="imported",
                source=params["source"],
                recipe=params.get("quantization"),
                note="simulated port" if self.simulate else None,
                extra={
                    "simulated": self.simulate,
                    "quantization": params.get("quantization"),
                    "chunk_size": params.get("chunk_size"),
                    "prune_logit": params.get("prune_logit"),
                },
            )
            # Registering in the catalog is what makes the model usable immediately:
            # chat, /v1/models and the benchmark harness all resolve through it, so
            # without this the user would have to restart the server to use their port.
            entry = self.catalog.register(model_from_local(local))

            share_url = f"{self.settings.base_url}/v1/catalog/{model_id}"
            download_url = f"{self.settings.base_url}/api/artifacts/{model_id}"
            ctx.result(
                model=model_id,
                path=local.path,
                share_url=share_url,
                download_url=download_url,
                share_command=f"hipep pull {model_id} --from {self.settings.base_url}",
                openai_base_url=f"{self.settings.base_url}/v1",
                simulated=self.simulate,
            )
            ctx.log(f"published {model_id} as a {entry.status} model")
            ctx.log(f"share:    {share_url}")
            ctx.log(f"download: {download_url}")
            ctx.log(
                f"chat:     POST {self.settings.base_url}/v1/chat/completions "
                f'with {{"model": "{model_id}"}}'
            )

        stages["publish"] = publish
        return stages

    # --- benchmark ------------------------------------------------------------

    def start_bench(self, model: Model, spec: dict[str, Any]) -> Job:
        contexts = spec.get("contexts") or [128, 2048, 16384]
        job = self.jobs.create(
            kind="benchmark",
            title=f"Benchmark {model.name}",
            stages=[Stage(name=n, label=label) for n, label in BENCH_STAGES],
            params={
                "model": model.id,
                "contexts": contexts,
                "generate_tokens": spec.get("generate_tokens", 64),
                "repeats": spec.get("repeats", 3),
                "warmup": spec.get("warmup", 1),
            },
        )
        return self.jobs.start(job, self._bench_stages(model, job))

    def _bench_stages(self, model: Model, job: Job) -> dict[str, Any]:
        from .bench import BenchRunner

        runner = BenchRunner(self.settings, model, job.params)

        async def prepare(ctx: JobContext) -> None:
            ctx.log(f"target: {self.settings.base_url}")
            ctx.log(
                f"contexts: {job.params['contexts']}, "
                f"generate {job.params['generate_tokens']} tokens, "
                f"{job.params['repeats']} repeats after {job.params['warmup']} warmup"
            )
            await runner.prepare(ctx)

        async def warmup(ctx: JobContext) -> None:
            await runner.warmup(ctx)

        async def measure(ctx: JobContext) -> None:
            await runner.measure(ctx)

        async def summarize(ctx: JobContext) -> None:
            summary = await runner.summarize(ctx)
            path = self.settings.results_dir / f"{job.id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            ctx.result(summary=summary, path=str(path))
            ctx.log(f"results written to {path}")

        return {
            "prepare": prepare,
            "warmup": warmup,
            "measure": measure,
            "summarize": summarize,
        }


# --- simulation ---------------------------------------------------------------


async def _simulated_transfer(ctx: JobContext, total: int, seconds: float) -> None:
    steps = 40
    for i in range(1, steps + 1):
        ctx.check_cancelled()
        await asyncio.sleep(seconds / steps)
        got = int(total * i / steps)
        ctx.result(completed_bytes=got)
        ctx.progress(i / steps, f"{_human(got)} / {_human(total)}")


_PORT_STAGE_SCRIPT: dict[str, tuple[float, list[str]]] = {
    "fetch": (
        5.0,
        ["resolving revision", "downloading safetensors shards", "verifying checksums"],
    ),
    "inspect": (
        1.5,
        [
            "reading config.json",
            "architecture detected",
            "counting quantizable linear layers",
            "lm_head excluded from quantization",
        ],
    ),
    "export": (
        6.0,
        [
            "tracing model to ONNX",
            "folding constants",
            "externalizing weights above 1 KB",
            "writing model.onnx + model.onnx.data",
        ],
    ),
    "quantize": (
        8.0,
        [
            "building calibration set",
            "running Quark PTQ",
            "inserting QDQ nodes",
            "pruning logits",
            "validating initializers are byte-identical",
        ],
    ),
    "convert": (
        3.0,
        ["splitting embedding / decoder / lm_head", "writing genai_config.json"],
    ),
    "validate": (
        3.0,
        [
            "creating an ORT session with AMDGPUExecutionProvider",
            "HIPDNN_EP_STRICT=1: refusing silent CPU fallback",
            "comparing logits against the fp16 reference",
        ],
    ),
    "benchmark": (4.0, ["measuring TTFT at 128 / 2048 / 16384", "measuring decode rate"]),
    "publish": (1.0, ["writing model card", "registering in the local zoo"]),
}


async def _simulate_port_stage(
    ctx: JobContext, stage_name: str, params: dict[str, Any]
) -> None:
    seconds, steps = _PORT_STAGE_SCRIPT.get(stage_name, (2.0, ["working"]))
    ctx.log(
        f"SIMULATED '{stage_name}' — the export and quantization toolchain is not "
        "installed on this host, so no artifact is produced.",
        level="warning",
    )
    if stage_name == "quantize":
        ctx.log(
            f"plan: scheme={params.get('quantization')} "
            f"chunk_size={params.get('chunk_size')} "
            f"prune_logit={params.get('prune_logit')} "
            f"calibration_samples={params.get('calibration_samples')}"
        )
    per = seconds / max(1, len(steps))
    for i, step in enumerate(steps, 1):
        ctx.check_cancelled()
        ctx.log(step)
        await asyncio.sleep(per)
        ctx.progress(i / len(steps), step)
    ctx.result(simulated=True)


def _derive_id(source: str) -> str:
    """Turn a Hugging Face id or path into a catalog-style local model id."""
    tail = source.rstrip("/").split("/")[-1]
    slug = "".join(c if c.isalnum() or c in "-._" else "-" for c in tail).strip("-").lower()
    return f"local/{slug}" if slug else f"local/model-{int(time.time())}"


def _human(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024 or unit == "TB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024
    return f"{num:.1f} TB"
