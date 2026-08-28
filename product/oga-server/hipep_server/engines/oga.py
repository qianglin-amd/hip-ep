# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""ONNX Runtime GenAI engine driving HIP EP.

This is the real path: onnxruntime-genai loads a converted model directory whose
genai_config.json selects the AMDGPU execution provider with the `hip` profile, which
routes the graph through HIP EP's MLIR compiler onto the GPU.

Two properties matter for correctness of the product, not just of the code:

* Generation runs on a worker thread. onnxruntime-genai is blocking and holds the GIL
  in places; running it inline would stall the whole event loop and make the server
  appear hung to every other user.
* CPU fallback is refused by default. ONNX Runtime will happily run a graph on the CPU
  when HIP EP declines it, which produces correct answers at a tenth of the speed and
  silently invalidates any benchmark. `HIPDNN_EP_STRICT=1` turns that into an error.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ..catalog import Model
from .base import (
    Delta,
    Engine,
    EngineError,
    GenerationRequest,
    Stopwatch,
    estimate_tokens,
)

log = logging.getLogger(__name__)

_SENTINEL = object()


def oga_available() -> tuple[bool, str]:
    """Whether onnxruntime-genai can be imported, and why not if it cannot."""
    try:
        import onnxruntime_genai  # noqa: F401
    except ImportError as exc:
        return False, f"onnxruntime-genai is not installed ({exc})"
    return True, ""


class _LoadedModel:
    def __init__(self, model: Model, path: Path, og: Any):
        self.model = model
        self.path = path
        self.og_model = og.Model(str(path))
        # Multimodal models expose a processor; text-only models expose a tokenizer.
        self.processor = None
        self.tokenizer = None
        if model.supports_vision and hasattr(og, "MultiModalProcessor"):
            try:
                self.processor = og.MultiModalProcessor(self.og_model)
            except Exception as exc:  # noqa: BLE001
                log.warning("no multimodal processor for %s: %s", model.id, exc)
        if self.processor is None:
            self.tokenizer = og.Tokenizer(self.og_model)
        self.lock = threading.Lock()

    def close(self) -> None:
        self.processor = None
        self.tokenizer = None
        self.og_model = None


class OgaEngine(Engine):
    name = "oga"
    real = True

    def __init__(self, *, models_dir: Path, strict: bool = True):
        self._models_dir = models_dir
        self._loaded: dict[str, _LoadedModel] = {}
        self._load_lock = asyncio.Lock()
        if strict:
            # Set before the first session is created; ONNX Runtime reads it at
            # compile time, so flipping it later has no effect.
            os.environ.setdefault("HIPDNN_EP_STRICT", "1")

    # --- lifecycle ------------------------------------------------------------

    def model_path(self, model: Model) -> Path:
        return self._models_dir / model.id.replace(":", "--")

    async def load(self, model: Model) -> None:
        if model.id in self._loaded:
            return
        async with self._load_lock:
            if model.id in self._loaded:
                return
            ok, why = oga_available()
            if not ok:
                raise EngineError(why)
            path = self.model_path(model)
            if not (path / "genai_config.json").is_file():
                raise EngineError(
                    f"model '{model.id}' is not downloaded. "
                    f"Pull it first: POST /api/pull or `hipep pull {model.id}`."
                )
            import onnxruntime_genai as og

            loop = asyncio.get_running_loop()
            loaded = await loop.run_in_executor(None, _LoadedModel, model, path, og)
            self._loaded[model.id] = loaded
            log.info("loaded %s from %s", model.id, path)

    async def unload(self, model_id: str) -> None:
        loaded = self._loaded.pop(model_id, None)
        if loaded:
            await asyncio.get_running_loop().run_in_executor(None, loaded.close)
            log.info("unloaded %s", model_id)

    def loaded_models(self) -> list[str]:
        return list(self._loaded)

    # --- generation -----------------------------------------------------------

    async def stream(self, request: GenerationRequest) -> AsyncIterator[Delta]:
        await self.load(request.model)
        loaded = self._loaded[request.model.id]

        # A bounded queue applies backpressure: if a client stops reading, the
        # generator thread blocks instead of buffering an entire response in memory.
        out: queue.Queue = queue.Queue(maxsize=256)
        cancel = threading.Event()
        loop = asyncio.get_running_loop()

        thread = threading.Thread(
            target=self._generate_blocking,
            args=(loaded, request, out, cancel),
            name=f"oga-{request.model.id}",
            daemon=True,
        )
        thread.start()

        watch = Stopwatch()
        completion_tokens = 0
        prompt_tokens = estimate_tokens(request.prompt_text)
        try:
            while True:
                item = await loop.run_in_executor(None, out.get)
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise EngineError(str(item)) from item
                if isinstance(item, dict):  # metadata from the worker
                    prompt_tokens = item.get("prompt_tokens", prompt_tokens)
                    continue
                watch.mark_first_token()
                completion_tokens += 1
                yield Delta(text=item)
        finally:
            cancel.set()

        watch.finish()
        usage = watch.usage(prompt_tokens, completion_tokens)
        yield Delta(done=True, finish_reason="stop", usage=usage.to_dict())

    def _generate_blocking(
        self,
        loaded: _LoadedModel,
        request: GenerationRequest,
        out: queue.Queue,
        cancel: threading.Event,
    ) -> None:
        """Run onnxruntime-genai on a worker thread, pushing decoded text to `out`."""
        try:
            import onnxruntime_genai as og

            params = og.GeneratorParams(loaded.og_model)
            search = {
                "max_length": min(
                    request.model.max_context,
                    estimate_tokens(request.prompt_text) + request.params.max_tokens,
                ),
                "temperature": request.params.temperature,
                "top_p": request.params.top_p,
                "top_k": request.params.top_k,
                "do_sample": request.params.temperature > 0,
            }
            if request.params.repetition_penalty != 1.0:
                search["repetition_penalty"] = request.params.repetition_penalty
            params.set_search_options(**search)

            prompt = _render_prompt(request)

            # The model holds one generator at a time; concurrent requests for the same
            # model are serialised here and queued by the scheduler above.
            with loaded.lock:
                if loaded.processor is not None:
                    images = _load_images(og, request)
                    inputs = loaded.processor(prompt, images=images)
                    params.set_inputs(inputs)
                    detok = loaded.processor.create_stream()
                    prompt_len = 0
                else:
                    tokens = loaded.tokenizer.encode(prompt)
                    params.input_ids = tokens
                    detok = og.TokenizerStream(loaded.tokenizer)
                    prompt_len = len(tokens)

                out.put({"prompt_tokens": prompt_len or estimate_tokens(prompt)})

                generator = og.Generator(loaded.og_model, params)
                stops = request.params.stop
                buffered = ""
                while not generator.is_done():
                    if cancel.is_set():
                        break
                    generator.compute_logits()
                    generator.generate_next_token()
                    token = generator.get_next_tokens()[0]
                    piece = detok.decode(token)
                    if not piece:
                        continue
                    if stops:
                        buffered += piece
                        hit = next((s for s in stops if s in buffered), None)
                        if hit:
                            head = buffered.split(hit, 1)[0]
                            if head:
                                out.put(head)
                            break
                        # Hold back only as much as the longest stop string needs.
                        keep = max(len(s) for s in stops) - 1
                        if len(buffered) > keep:
                            out.put(buffered[:-keep] if keep else buffered)
                            buffered = buffered[-keep:] if keep else ""
                    else:
                        out.put(piece)
                if stops and buffered and not cancel.is_set():
                    out.put(buffered)
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an error
            log.exception("generation failed for %s", request.model.id)
            out.put(exc)
        finally:
            out.put(_SENTINEL)


def _render_prompt(request: GenerationRequest) -> str:
    """Build the prompt string for the model's chat template.

    onnxruntime-genai does not apply a chat template for every architecture, so we
    render a generic one. Models whose converted artifact carries a template will have
    it applied by the processor; this is the fallback that keeps unknown models usable.
    """
    parts: list[str] = []
    for message in request.messages:
        tag = {"system": "system", "user": "user", "assistant": "assistant"}.get(
            message.role, "user"
        )
        content = message.content
        if message.images:
            # Image placeholders are what the multimodal processors expect to find.
            content = "".join(f"<|image_{i + 1}|>" for i in range(len(message.images))) + content
        parts.append(f"<|{tag}|>\n{content}<|end|>")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _load_images(og: Any, request: GenerationRequest) -> Any:
    """Wrap request image bytes in the GenAI Images container."""
    if not request.images:
        return None
    import tempfile

    paths = []
    for i, data in enumerate(request.images):
        tmp = Path(tempfile.gettempdir()) / f"hipep-img-{os.getpid()}-{id(request)}-{i}.png"
        tmp.write_bytes(data)
        paths.append(str(tmp))
    return og.Images.open(*paths)
