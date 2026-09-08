# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Real implementations of the port pipeline's stages.

The pipeline looks these up by stage name and falls back to simulation for any it
cannot find, so this module is written to disappear cleanly: when no toolchain is
reachable, `__getattr__` refuses every name and the port simulates and says so, exactly
as it did before this module existed.

Why the heavy work runs in a subprocess
---------------------------------------
The server and the exporter cannot share an interpreter. AMD Quark and the ONNX model
builder cap Python below 3.14 and pull in torch; the server has no such ceiling and no
reason to carry a multi-gigabyte import. So the toolchain lives in its own environment
and is driven over the command line, which also keeps an exporter segfault from taking
the server with it. `HIPEP_PORT_PYTHON` names that interpreter; if the current one can
already do the work, it is used directly.

What produces the artifact
--------------------------
`onnxruntime_genai.models.builder` fuses graph export and weight-only quantization into
one pass, so the two named stages do not map one-to-one onto commands: `export` runs the
builder, and `quantize` reports what quantization actually landed in the graph it
produced. That is a verification the simulated pipeline could not do, and it is what
catches a builder that silently produced a float model.

The builder has no HIP execution provider. It emits a generic INT4 graph and a
`genai_config.json` naming whichever provider it was asked for, so `convert` retargets
that config onto the AMDGPU provider the runtime looks for. That retarget is the step
that makes the artifact a HIP EP model rather than a CPU one.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# What the runtime looks for in a genai_config to route a model at the AMD GPU.
AMDGPU_PROVIDER = {"AMDGPU": {"profile": "hip"}}

# The builder has no HIP target, so one of its targets has to stand in, and the choice
# decides the shape of the graph rather than what it will run on: `int4` combined with
# `cpu` is the one case the builder gives an FP32 interface and an `accuracy_level` of 4,
# meaning int8 compute. Feeding that to the AMD GPU produced a model that loaded, ran at
# a plausible speed and emitted only punctuation. Every non-CPU target yields the FP16
# interface and unset accuracy level that the certified artifacts have, and `convert`
# retargets the config afterwards -- so this names a graph layout, not a dependency, and
# nothing DirectML is installed or needed to run the result.
BUILD_TARGET = os.environ.get("HIPEP_PORT_BUILD_TARGET", "dml")

# Files an OGA artifact needs beside the graph. The builder writes the tokenizer itself,
# but only these are required for the model to load.
REQUIRED_ARTIFACT_FILES = ("genai_config.json", "tokenizer.json")

# Quantization schemes the builder can produce on its own. AWQ is deliberately absent:
# it needs a Quark calibration pass, and quietly substituting round-to-nearest would
# hand back a model that is not the one the recipe asked for.
BUILDER_ALGORITHMS = {"rtn": "rtn", "k_quant": "k_quant", "default": "default"}

# Block sizes the builder accepts.
BUILDER_BLOCK_SIZES = (16, 32, 64, 128, 256)

BENCHMARK_TOKENS = 64

# Fewer decode steps than this is not a slow model, it is a broken one, and averaging
# over them produces a throughput figure that is nonsense rather than merely noisy.
MIN_BENCHMARK_TOKENS = 8

# Characters between spaces, averaged over a continuation. Working ports sit at 4-6 and
# the broken one sat at 23, so this is placed to need no precision.
MAX_PROSE_WORD_LENGTH = 12


# How often an unfinished progress line is worth repeating into the log, in seconds.
REDRAW_LOG_INTERVAL_S = 15.0


class ToolchainMissing(RuntimeError):
    """No interpreter on this machine can run the exporter."""


@functools.lru_cache(maxsize=4)
def _probe(explicit: str | None) -> str | None:
    """Resolve the toolchain interpreter once per distinct environment setting.

    Cached because the fallback costs a subprocess, and the pipeline asks once per
    stage. Keying on the environment variable keeps a test that sets it from reading a
    previous answer.
    """
    if explicit and Path(explicit).is_file():
        return explicit
    if _can_build(sys.executable):
        return sys.executable
    managed = managed_toolchain()
    if managed is not None and _can_build(str(managed)):
        return str(managed)
    return None


def managed_toolchain() -> Path | None:
    """The interpreter in the toolchain environment the installer builds, if present.

    Looked for by convention so that installing the toolchain is enough to enable real
    ports. Requiring `HIPEP_PORT_PYTHON` as well would mean the install silently did
    nothing until the operator also set a variable nobody told them about.
    """
    root = Path(os.environ.get("HIPEP_HOME") or (Path.home() / ".hipep")) / "toolchain"
    for relative in ("Scripts/python.exe", "bin/python3", "bin/python"):
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def toolchain_python() -> str | None:
    """The interpreter that can run the exporter, or None if there is not one.

    Checked in order of how explicit the answer is: an operator's choice first, then
    this interpreter if it happens to be capable, then the managed environment.
    """
    return _probe(os.environ.get("HIPEP_PORT_PYTHON"))


def _can_build(python: str) -> bool:
    """Whether `python` can import the model builder.

    Asked by running it rather than by importing here, because importing the builder
    pulls in torch and this is called from the server process.
    """
    try:
        done = subprocess.run(
            [python, "-c", "import onnxruntime_genai.models.builder"],
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


# --- stage bodies -------------------------------------------------------------


async def _fetch(ctx, params: dict[str, Any], *, store) -> None:
    """Bring the source weights onto this machine."""
    source = params["source"]
    target = _workdir(store, params) / "source"

    if Path(source).is_dir():
        ctx.log(f"using local weights at {source}")
        ctx.result(source_path=str(Path(source).resolve()))
        ctx.progress(1.0, "local weights")
        return

    ctx.log(f"downloading {source} from Hugging Face")
    ctx.progress(0.05, "resolving repository")

    def download() -> str:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=source,
            local_dir=str(target),
            # The exporter reads the PyTorch weights and the tokenizer; anything else in
            # the repository is weight duplicates in other formats, which on a large
            # model is more download than the port itself.
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.model",
                "*.txt",
                "tokenizer*",
            ],
        )

    ctx.check_cancelled()
    path = await asyncio.to_thread(download)
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    ctx.log(f"downloaded {_human(size)} to {path}")
    ctx.result(source_path=str(path), source_bytes=size)
    ctx.progress(1.0, _human(size))


async def _inspect(ctx, params: dict[str, Any], *, store) -> None:
    """Read the model's own config and report what is about to be ported."""
    source = Path(ctx.job.result.get("source_path") or params["source"])
    config_path = source / "config.json"
    if not config_path.is_file():
        raise RuntimeError(
            f"no config.json under {source}; this does not look like a Hugging Face model"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    architecture = (config.get("architectures") or ["unknown"])[0]
    layers = config.get("num_hidden_layers")
    hidden = config.get("hidden_size")
    vocab = config.get("vocab_size")
    context = config.get("max_position_embeddings")

    ctx.log(f"architecture      {architecture}")
    ctx.log(f"layers            {layers}, hidden size {hidden}")
    ctx.log(f"vocabulary        {vocab}")
    ctx.log(f"trained context   {context}")

    ctx.result(
        architecture=architecture,
        num_hidden_layers=layers,
        hidden_size=hidden,
        vocab_size=vocab,
        max_position_embeddings=context,
    )
    ctx.progress(1.0, architecture)


async def _export(ctx, params: dict[str, Any], *, store) -> None:
    """Run the builder, which exports the graph and quantizes its weights together."""
    source = Path(ctx.job.result.get("source_path") or params["source"])
    build_dir = _workdir(store, params) / "build"
    cache_dir = _workdir(store, params) / "cache"
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    plan = _quantization_plan(params)

    requested = str(params.get("quantization") or "int4-rtn-g128")
    hidden = ctx.job.result.get("hidden_size")
    fitted = _fit_block_size(plan["block_size"], hidden)
    if fitted != plan["block_size"]:
        ctx.log(
            f"block size {plan['block_size']} does not divide a hidden size of {hidden}; "
            f"quantizing in blocks of {fitted} instead",
            level="warning",
        )
        plan["block_size"] = fitted
        # The model has to be labelled with the scheme it was built at. Publishing the
        # requested one instead puts an artifact in the zoo whose card does not describe
        # it, and the difference is exactly the one that decides whether it works.
        ctx.result(quantization_applied=_scheme_with_block(requested, fitted))

    ctx.log(
        f"plan: scheme={params.get('quantization')} block_size={plan['block_size']} "
        f"symmetric={plan['is_symmetric']} algorithm={plan['algo_config']}"
    )
    ctx.log("the builder exports and quantizes in one pass; both stages report on it")

    extra = [
        f"block_size={plan['block_size']}",
        f"is_symmetric={str(plan['is_symmetric']).lower()}",
        f"algo_config={plan['algo_config']}",
    ]
    if params.get("prune_logit"):
        # Leaving the logit head at full precision is what the certified recipes do; the
        # quality cost of quantizing it is out of proportion to the size it saves.
        extra.append("nodes_to_exclude=/lm_head/MatMul")

    command = [
        _require_toolchain(),
        "-m",
        "onnxruntime_genai.models.builder",
        "-i",
        str(source),
        "-o",
        str(build_dir),
        "-p",
        plan["precision"],
        "-e",
        BUILD_TARGET,
        "-c",
        str(cache_dir),
        "--extra_options",
        *extra,
    ]
    ctx.log("running: " + " ".join(command[1:]))
    await _run(ctx, command, "builder")

    produced = sorted(p.name for p in build_dir.glob("*.onnx"))
    if not produced:
        raise RuntimeError(f"the builder wrote no ONNX graph into {build_dir}")
    ctx.log(f"built {', '.join(produced)}")
    ctx.result(build_path=str(build_dir), onnx_files=produced)
    ctx.progress(1.0, ", ".join(produced))


async def _quantize(ctx, params: dict[str, Any], *, store) -> None:
    """Report the quantization actually present in the graph the builder produced.

    The plan is an argument; this is the outcome. A builder that ignored the request,
    or fell back to float because it did not recognise an operator, produces a model
    that loads and runs and is several times the expected size, which is otherwise not
    noticed until someone reads a benchmark.
    """
    build_dir = Path(ctx.job.result["build_path"])
    graph = next(iter(sorted(build_dir.glob("*.onnx"))), None)
    if graph is None:
        raise RuntimeError(f"no ONNX graph under {build_dir} to inspect")

    counts = await asyncio.to_thread(_count_quantized_ops, graph)
    quantized = counts.get("MatMulNBits", 0)
    plain = counts.get("MatMul", 0)

    ctx.log(f"quantized matmuls   {quantized}")
    ctx.log(f"unquantized matmuls {plain}")
    if not quantized:
        raise RuntimeError(
            "the builder produced no quantized matmuls, so the weights are still "
            "float; the requested scheme was not applied"
        )

    weights = sum(f.stat().st_size for f in build_dir.rglob("*") if f.is_file())
    ctx.log(f"artifact size       {_human(weights)}")
    ctx.result(quantized_matmuls=quantized, unquantized_matmuls=plain, artifact_bytes=weights)
    ctx.progress(1.0, f"{quantized} quantized matmuls")


async def _convert(ctx, params: dict[str, Any], *, store) -> None:
    """Assemble the artifact and point its config at the AMD GPU."""
    build_dir = Path(ctx.job.result["build_path"])
    target = store.path_for(params["name"])
    target.mkdir(parents=True, exist_ok=True)

    ctx.log(f"assembling artifact in {target}")
    moved = 0
    for item in build_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, target / item.name)
            moved += 1
    ctx.progress(0.6, f"{moved} files")

    config_path = target / "genai_config.json"
    if not config_path.is_file():
        raise RuntimeError("the builder wrote no genai_config.json")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    session = (
        config.setdefault("model", {})
        .setdefault("decoder", {})
        .setdefault("session_options", {})
    )
    previous = session.get("provider_options")
    session["provider_options"] = [dict(AMDGPU_PROVIDER)]
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    ctx.log(f"retargeted provider {previous} -> {session['provider_options']}")

    missing = [name for name in REQUIRED_ARTIFACT_FILES if not (target / name).is_file()]
    if missing:
        raise RuntimeError(f"the artifact is missing {', '.join(missing)}")

    ctx.result(path=str(target), provider_options=session["provider_options"])
    ctx.progress(1.0, "ready to load")


def _degenerate(continuation: str) -> str | None:
    """Say why a continuation shows a broken artifact, or None if it reads as language.

    A wrongly quantized model does not fail to load and does not run slowly, so a port
    that only checks for a non-empty string calls it good and publishes it. Each of the
    three signals below is one that got past the previous version of this check on a
    real port, and all three separate broken from working by a wide margin rather than
    a delicate threshold.
    """
    if "\ufffd" in continuation:
        # Real text decodes. A replacement character means the token ids coming out do
        # not correspond to anything the tokenizer knows.
        return "the continuation contained undecodable bytes"

    words = re.findall(r"[^\W\d_]+", continuation, flags=re.UNICODE)
    if not words:
        return "the continuation held no words, only punctuation or whitespace"

    chunks = continuation.split()
    mean_length = sum(len(chunk) for chunk in chunks) / len(chunks)
    # Prose runs 4-6 characters between spaces in every sample taken from a working
    # port; a model quantized in blocks that do not divide its rows ran to 23, because
    # it emits word fragments glued together rather than words.
    if len(chunks) >= 5 and mean_length > MAX_PROSE_WORD_LENGTH:
        return (
            f"the continuation averaged {mean_length:.0f} characters between spaces, "
            "which is word salad rather than prose"
        )

    distinct = {word.casefold() for word in words}
    # Short answers are legitimately repetitive, so only judge once there is enough text
    # for repetition to mean something.
    if len(words) >= 8 and len(distinct) * 4 <= len(words):
        common = max(distinct, key=lambda w: sum(x.casefold() == w for x in words))
        return (
            f"the continuation repeated itself, using {len(distinct)} distinct words "
            f"across {len(words)} (most often {common!r})"
        )
    return None


async def _validate(ctx, params: dict[str, Any], *, store) -> None:
    """Load the artifact on the GPU and generate from it.

    This runs in the server's own interpreter on purpose. It is the one that carries
    the AMD GPU execution provider, so it is the only place where "it loaded" means the
    thing that will serve the model can load it.
    """
    target = store.path_for(params["name"])
    prompt = "The capital of France is"

    def generate() -> tuple[str, str]:
        import onnxruntime_genai as og

        model = og.Model(str(target))
        tokenizer = og.Tokenizer(model)
        gen_params = og.GeneratorParams(model)
        gen_params.set_search_options(max_length=48, do_sample=False)
        generator = og.Generator(model, gen_params)
        generator.append_tokens(tokenizer.encode(prompt))
        while not generator.is_done():
            generator.generate_next_token()
        produced = tokenizer.decode(generator.get_sequence(0))
        return produced, str(getattr(model, "device_type", "unknown"))

    ctx.log("loading the artifact with onnxruntime-genai")
    ctx.check_cancelled()
    try:
        text, device = await asyncio.to_thread(generate)
    except ImportError as exc:
        # Not a failed port: the artifact exists and this machine cannot try it.
        ctx.log(f"cannot validate here: {exc}", level="warning")
        ctx.result(validated=False, validation_skipped="onnxruntime-genai not installed")
        ctx.progress(1.0, "skipped")
        return

    ctx.log(f"device      {device}")
    ctx.log(f"generated   {text[:160]}")

    continuation = text[len(prompt) :] if text.startswith(prompt) else text
    complaint = _degenerate(continuation)
    if complaint:
        raise RuntimeError(
            f"the artifact loaded and ran on {device} but {complaint}. The weights are "
            f"numerically wrong rather than absent. Generated: {continuation[:120]!r}"
        )

    ctx.result(validated=True, validation_device=device, validation_sample=text[:400])
    ctx.progress(1.0, "generated on device")


async def _publish(ctx, params: dict[str, Any], *, store) -> None:
    """Tidy up before the pipeline registers the artifact.

    The registration itself belongs to the pipeline, which owns the catalog. This stage
    exists so that a port which did every other stage for real is not recorded as
    simulated on the strength of the one stage this module did not implement.
    """
    work = _workdir(store, params)
    if work.exists():
        # The source snapshot and the pre-retarget build are together larger than the
        # artifact, and nothing reads them once the model is assembled.
        freed = sum(f.stat().st_size for f in work.rglob("*") if f.is_file())
        shutil.rmtree(work, ignore_errors=True)
        ctx.log(f"removed {_human(freed)} of intermediate files")
    ctx.progress(1.0, "ready")


async def _benchmark(ctx, params: dict[str, Any], *, store) -> None:
    """Time a short generation so the port ends with a number from this machine."""
    if not params.get("benchmark", True):
        ctx.log("benchmark not requested")
        ctx.progress(1.0, "skipped")
        return
    if not ctx.job.result.get("validated"):
        ctx.log("nothing was validated on this machine, so there is nothing to time")
        ctx.progress(1.0, "skipped")
        return

    target = store.path_for(params["name"])

    def measure() -> dict[str, float]:
        import time

        import onnxruntime_genai as og

        model = og.Model(str(target))
        tokenizer = og.Tokenizer(model)
        prompt = tokenizer.encode("Explain what an execution provider does.")

        def once() -> tuple[float, float, int]:
            gen_params = og.GeneratorParams(model)
            # A throughput figure has to come from a fixed number of decode steps, so
            # the run is pinned at both ends. Without the floor an instruction-tuned
            # model can treat the prompt as complete and stop at the first token, and
            # what gets timed is one token rather than sixty-four.
            gen_params.set_search_options(
                max_length=len(prompt) + BENCHMARK_TOKENS,
                min_length=len(prompt) + BENCHMARK_TOKENS,
                do_sample=False,
            )
            generator = og.Generator(model, gen_params)
            start = time.perf_counter()
            generator.append_tokens(prompt)
            generator.generate_next_token()
            first = time.perf_counter()
            decoded = 0
            while not generator.is_done():
                generator.generate_next_token()
                decoded += 1
            return first - start, time.perf_counter() - first, decoded

        once()  # warm up; the first pass pays for allocation and autotuning
        ttft, decode, decoded = once()

        # The first token is prefill, and it is already reported as TTFT; counting it
        # again in the decode rate divides a token that took no measured time by a
        # window that excludes it. A model that stops at the first token then produces
        # a rate in the hundreds of thousands, which is how this was found.
        if decoded < MIN_BENCHMARK_TOKENS or decode <= 0:
            raise RuntimeError(
                f"the model stopped after {decoded + 1} token(s), which is too few to "
                "time. That is a sign the artifact is broken rather than fast."
            )

        return {
            "ttft_s": round(ttft, 4),
            "decode_s": round(decode, 4),
            "tokens": decoded + 1,
            "tps": round(decoded / decode, 2),
        }

    ctx.log(f"timing a {BENCHMARK_TOKENS}-token generation")
    ctx.check_cancelled()
    try:
        numbers = await asyncio.to_thread(measure)
    except Exception as exc:
        # The artifact was already generated from and checked in `validate`. Failing the
        # port here would throw away a model that works because a measurement of it did
        # not, so this reports the number as missing instead of the port as broken.
        ctx.log(f"could not time this model: {exc}", level="warning")
        ctx.result(benchmark_skipped=str(exc))
        ctx.progress(1.0, "not timed")
        return

    ctx.log(f"ttft {numbers['ttft_s']}s, {numbers['tps']} tok/s over {numbers['tokens']}")
    ctx.result(benchmark=numbers)
    ctx.progress(1.0, f"{numbers['tps']} tok/s")


# --- helpers ------------------------------------------------------------------


def _require_toolchain() -> str:
    python = toolchain_python()
    if python is None:
        raise ToolchainMissing(
            "no interpreter on this machine can run the ONNX model builder. "
            "Install it into a Python 3.11-3.13 environment and point "
            "HIPEP_PORT_PYTHON at that interpreter."
        )
    return python


def _workdir(store, params: dict[str, Any]) -> Path:
    """Scratch space for one port, beside the model it is building."""
    work = store.root / ".ports" / params["name"].replace(":", "--").replace("/", "__")
    work.mkdir(parents=True, exist_ok=True)
    return work


def _quantization_plan(params: dict[str, Any]) -> dict[str, Any]:
    """Turn a recipe id such as `int4-rtn-g128` into builder options."""
    scheme = str(params.get("quantization") or "int4-rtn-g128").lower()
    parts = scheme.split("-")
    precision = parts[0] if parts and parts[0] in {"int4", "int8", "fp16"} else "int4"

    algorithm = "rtn"
    for part in parts:
        if part in {"rtn", "default"}:
            algorithm = part
        elif part in {"kquant", "k_quant"}:
            algorithm = "k_quant"
        elif part == "awq":
            raise ValueError(
                "AWQ needs a Quark calibration pass, which this pipeline does not run "
                "yet. Use int4-rtn-g128, or pull a certified AWQ artifact instead of "
                "porting one."
            )

    group = next(
        (int(part[1:]) for part in parts if part.startswith("g") and part[1:].isdigit()),
        int(params.get("chunk_size") or 128),
    )
    # The builder takes a power-of-two block size; a recipe group size is the same idea.
    block = group if group in BUILDER_BLOCK_SIZES else 128

    return {
        "precision": precision,
        "algo_config": BUILDER_ALGORITHMS[algorithm],
        "block_size": block,
        "is_symmetric": "asym" not in parts,
    }


def _fit_block_size(block: int, hidden: int | None) -> int:
    """The largest supported block size that divides `hidden`, at most `block`.

    A quantization block that does not divide the row it splits leaves a short final
    block, and the weights that land in it come back wrong. The model still exports,
    loads and runs at full speed -- it just stops producing language, which is why this
    is corrected here rather than left for someone to discover in a chat window. Two of
    the first three models tried hit it: a hidden size of 960 is not a multiple of 128.
    """
    if not hidden:
        return block
    for candidate in sorted(BUILDER_BLOCK_SIZES, reverse=True):
        if candidate <= block and hidden % candidate == 0:
            return candidate
    return min(BUILDER_BLOCK_SIZES)


def _scheme_with_block(scheme: str, block: int) -> str:
    """The recipe id rewritten to name the block size actually used."""
    parts = [
        f"g{block}" if part.startswith("g") and part[1:].isdigit() else part
        for part in scheme.split("-")
    ]
    if not any(part == f"g{block}" for part in parts):
        parts.append(f"g{block}")
    return "-".join(parts)


def _count_quantized_ops(graph: Path) -> dict[str, int]:
    """Count the operators that decide whether quantization actually happened."""
    import onnx

    # The weights live in external data next to the graph and are not needed to read
    # the node list, which on a large model is the difference between megabytes and
    # tens of gigabytes.
    model = onnx.load(str(graph), load_external_data=False)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        if node.op_type in {"MatMul", "MatMulNBits"}:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


async def _run(ctx, command: list[str], label: str) -> None:
    """Run a toolchain command, streaming its output into the job log."""
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert process.stdout is not None
    pending = ""
    last_redraw = 0.0
    try:
        while True:
            # Read by chunk rather than by line. The exporter draws progress bars with
            # carriage returns and no newline, so to a line reader the whole export is
            # one unterminated line that overruns the stream buffer and kills the job.
            chunk = await process.stdout.read(8192)
            if not chunk:
                break
            # Normalize first: on Windows every newline arrives as CRLF, and the
            # carriage return in it is a line ending rather than the redraw marker the
            # loop below treats a lone one as.
            pending += chunk.decode("utf-8", "replace").replace("\r\n", "\n")

            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                # A carriage return rewrites the line rather than ending it, so only
                # what follows the last one is what the line finally said.
                text = line.rsplit("\r", 1)[-1].strip()
                if text:
                    ctx.log(f"{label}: {text}")

            if "\r" in pending:
                # Collapse the unfinished line to its latest state. Keeping every
                # redraw both grows this buffer without bound and buries every real
                # message under a few hundred progress frames.
                pending = pending.rsplit("\r", 1)[-1]
                now = time.monotonic()
                if now - last_redraw >= REDRAW_LOG_INTERVAL_S and pending.strip():
                    # An export can run for a long time between newlines, and a job
                    # that logs nothing for minutes is indistinguishable from a hung one.
                    ctx.log(f"{label}: {pending.strip()}")
                    last_redraw = now

            # Cancelling the job has to stop the exporter, which otherwise keeps a
            # multi-gigabyte conversion running with nothing waiting for it.
            try:
                ctx.check_cancelled()
            except BaseException:
                process.kill()
                raise
        if pending.strip():
            ctx.log(f"{label}: {pending.strip()}")
    finally:
        await process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"{label} exited with status {process.returncode}")


def _human(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024 or unit == "TB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024
    return f"{num:.1f} TB"


_STAGES = {
    "fetch": _fetch,
    "inspect": _inspect,
    "export": _export,
    "quantize": _quantize,
    "convert": _convert,
    "validate": _validate,
    "benchmark": _benchmark,
    "publish": _publish,
}


def __getattr__(name: str):
    """Expose the stages only when something on this machine can run them.

    The pipeline treats a missing stage as "simulate this one and say so", which is the
    right behaviour on a machine with no exporter. Refusing the lookup here is what lets
    that path keep working now that this module exists.
    """
    if name in _STAGES:
        if toolchain_python() is None:
            raise AttributeError(
                f"{name} needs an exporter; set HIPEP_PORT_PYTHON to an interpreter "
                "that has onnxruntime-genai's model builder"
            )
        return _STAGES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
