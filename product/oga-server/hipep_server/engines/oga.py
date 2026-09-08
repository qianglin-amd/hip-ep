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
import json
import logging
import os
import queue
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
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


def _drain(out: queue.Queue) -> list[Any]:
    """Wait for one item from the worker, then take whatever else is already queued."""
    items = [out.get()]
    try:
        while True:
            items.append(out.get_nowait())
    except queue.Empty:
        pass
    return items

_ep_ready = False


def _prepare_amdgpu_ep() -> None:
    """Point onnxruntime-genai at the AMD GPU execution provider before it loads.

    The plugin EP ships as its own wheel rather than being compiled into
    onnxruntime-genai, which imposes three ordering constraints. Each one fails either
    silently or with a message that points somewhere other than the cause, so they are
    handled here once instead of at the call sites:

    * Windows resolves a bare `onnxruntime.dll` out of system32 ahead of PATH, and the
      Windows ML copy that lives there is several versions behind the wheel. Loading
      the wheel's copy first leaves the right module resident, so the EP binds to it.
    * onnxruntime-genai loads the EP itself from `AMDGPU_EP_PATH`, so that variable has
      to exist before the module is imported.
    * The EP must *not* also be handed to `register_execution_provider_library`.
      Registering it twice leaves genai reporting no supported AMD GPU devices at all.
    """
    global _ep_ready
    if _ep_ready:
        return
    _ep_ready = True
    try:
        import onnxruntime_ep_amdgpu
    except ImportError:
        # No plugin wheel. A genai build with the provider compiled in still works,
        # so leave it to fail later with its own message if it cannot.
        return

    try:
        import ctypes

        import onnxruntime

        bundled = Path(onnxruntime.__file__).parent / "capi" / "onnxruntime.dll"
        if bundled.is_file():
            ctypes.CDLL(str(bundled))
        os.environ.setdefault("AMDGPU_EP_PATH", onnxruntime_ep_amdgpu.get_library_path())
    except Exception as exc:  # noqa: BLE001
        log.warning("could not prepare the AMD GPU execution provider: %s", exc)


def oga_available() -> tuple[bool, str]:
    """Whether onnxruntime-genai can be imported, and why not if it cannot."""
    _prepare_amdgpu_ep()
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
        self.model_type = _model_type(path)
        self.chat = chat_format(self.model_type)
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

        # Started before the worker thread, not after. onnxruntime-genai holds the GIL
        # through prefill, so a `Thread.start()` that returns only once the worker
        # yields it lands the clock at roughly the first token: TTFT collapses to
        # microseconds and prefill disappears from the total instead of being reported.
        watch = Stopwatch()

        thread = threading.Thread(
            target=self._generate_blocking,
            args=(loaded, request, out, cancel),
            name=f"oga-{request.model.id}",
            daemon=True,
        )
        thread.start()
        completion_tokens = 0
        prompt_tokens = estimate_tokens(request.prompt_text)
        finished = False
        try:
            while not finished:
                # One thread hand-off per burst rather than per token. The worker holds
                # the GIL between decodes, so a round trip for every token leaves the
                # consumer waiting on the interpreter switch interval each time and the
                # stream falls steadily further behind the generation it is reporting.
                for item in await loop.run_in_executor(None, _drain, out):
                    if item is _SENTINEL:
                        finished = True
                        break
                    if isinstance(item, Exception):
                        raise EngineError(str(item)) from item
                    if isinstance(item, dict):  # metadata from the worker
                        prompt_tokens = item.get("prompt_tokens", prompt_tokens)
                        if "first_token_at" in item:
                            watch.mark_first_token_at(item["first_token_at"])
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
        first_token_at: float | None = None
        try:
            import onnxruntime_genai as og

            params = og.GeneratorParams(loaded.og_model)
            search = {
                "temperature": request.params.temperature,
                "top_p": request.params.top_p,
                "top_k": request.params.top_k,
                "do_sample": request.params.temperature > 0,
            }
            if request.params.repetition_penalty != 1.0:
                search["repetition_penalty"] = request.params.repetition_penalty

            prompt = loaded.chat.render(request.messages)

            # The model holds one generator at a time; concurrent requests for the same
            # model are serialised here and queued by the scheduler above.
            with loaded.lock:
                if loaded.processor is not None:
                    images = _load_images(og, request)
                    inputs = loaded.processor(prompt, images=images)
                    tokens = None
                    detok = loaded.processor.create_stream()
                    prompt_len = 0
                else:
                    inputs = None
                    tokens = loaded.tokenizer.encode(prompt)
                    detok = loaded.tokenizer.create_stream()
                    prompt_len = len(tokens)

                # Reported before the generator is built, because building it is where
                # the prompt is actually processed and the caller times that as TTFT.
                counted = prompt_len or estimate_tokens(prompt)
                out.put({"prompt_tokens": counted})

                # max_length has to cover the prompt actually sent, so it can only be
                # settled once that prompt exists. Sizing it from the raw request text
                # misses everything the chat markup adds, and onnxruntime-genai rejects
                # a budget smaller than the input outright rather than clamping it.
                search["max_length"] = min(
                    request.model.max_context, counted + request.params.max_tokens
                )
                params.set_search_options(**search)

                # The prompt attaches to the generator rather than to the params, and
                # logits are computed by generate_next_token. Both moved when
                # onnxruntime-genai gained continuous batching.
                generator = og.Generator(loaded.og_model, params)
                if inputs is not None:
                    generator.set_inputs(inputs)
                else:
                    generator.append_tokens(tokens)
                # The family's own end-of-turn markers, on top of anything the caller
                # asked for, so a model whose artifact under-declares its EOS ids still
                # stops at the end of its answer instead of continuing the transcript.
                stops = list(dict.fromkeys([*(request.params.stop or ()), *loaded.chat.stops]))
                buffered = ""
                while not generator.is_done():
                    if cancel.is_set():
                        break
                    generator.generate_next_token()
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                        out.put({"first_token_at": first_token_at})
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
                        # Hold back only a tail that could still grow into a stop
                        # marker. Reserving the longest marker's width unconditionally
                        # delays every chunk by several tokens, which on the console
                        # reads as the model pausing before it answers.
                        keep = _stop_prefix_len(buffered, stops)
                        if len(buffered) > keep:
                            out.put(buffered[: len(buffered) - keep])
                            buffered = buffered[len(buffered) - keep :]
                    else:
                        out.put(piece)
                if stops and buffered and not cancel.is_set():
                    out.put(buffered)
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an error
            log.exception("generation failed for %s", request.model.id)
            out.put(exc)
        finally:
            out.put(_SENTINEL)


@dataclass(frozen=True)
class ChatFormat:
    """How one model family marks up a conversation.

    `turns` maps a role to a template taking `{content}`; a role absent from the mapping
    reuses the user template. `reply` opens the assistant turn the model is expected to
    complete, and `stops` are the markers it closes that turn with.

    The stops matter as much as the markup. Several converted artifacts declare only a
    bare end-of-text id in genai_config.json -- Llama 3.1 lists `<|end_of_text|>` but
    not the `<|eot_id|>` it actually emits after a chat turn -- so a generator left to
    rely on that id alone never reports itself done and runs to max_tokens.
    """

    turns: dict[str, str]
    reply: str
    prefix: str = ""
    stops: tuple[str, ...] = ()

    def render(self, messages: Any) -> str:
        parts = [self.prefix]
        for message in messages:
            content = message.content
            if message.images:
                # Image placeholders are what the multimodal processors look for.
                placeholders = "".join(
                    f"<|image_{i + 1}|>" for i in range(len(message.images))
                )
                content = placeholders + content
            template = self.turns.get(message.role, self.turns["user"])
            parts.append(template.format(content=content))
        parts.append(self.reply)
        return "".join(parts)


def _symmetric(
    turn: str, reply: str, *, prefix: str = "", stops: tuple[str, ...] = ()
) -> ChatFormat:
    """A format whose three roles differ only by name."""
    return ChatFormat(
        turns={
            role: turn.format(role=role, content="{content}")
            for role in ("system", "user", "assistant")
        },
        reply=reply,
        prefix=prefix,
        stops=stops,
    )


# ChatML, used by the Qwen families and the default for anything unrecognised, since it
# is by far the most common markup among current converted artifacts.
_CHATML = _symmetric(
    "<|im_start|>{role}\n{content}<|im_end|>\n",
    "<|im_start|>assistant\n",
    stops=("<|im_end|>",),
)

_CHAT_FORMATS: tuple[tuple[str, ChatFormat], ...] = (
    ("qwen3", _CHATML),
    ("qwen2", _CHATML),
    ("qwen", _CHATML),
    (
        "llama",
        _symmetric(
            "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            prefix="<|begin_of_text|>",
            stops=("<|eot_id|>", "<|end_of_text|>"),
        ),
    ),
    (
        "phi4",
        _symmetric(
            "<|im_start|>{role}<|im_sep|>{content}<|im_end|>",
            "<|im_start|>assistant<|im_sep|>",
            stops=("<|im_end|>",),
        ),
    ),
    (
        "phi",
        _symmetric(
            "<|{role}|>\n{content}<|end|>\n",
            "<|assistant|>\n",
            stops=("<|end|>", "<|user|>"),
        ),
    ),
    (
        "gemma",
        # Gemma has no system role and calls the assistant "model".
        ChatFormat(
            turns={
                "system": "<start_of_turn>user\n{content}<end_of_turn>\n",
                "user": "<start_of_turn>user\n{content}<end_of_turn>\n",
                "assistant": "<start_of_turn>model\n{content}<end_of_turn>\n",
            },
            reply="<start_of_turn>model\n",
            stops=("<end_of_turn>",),
        ),
    ),
    (
        "mistral",
        # Mistral wraps only the user turn, so this one is genuinely asymmetric.
        ChatFormat(
            turns={
                "system": "[INST] {content} [/INST]",
                "user": "[INST] {content} [/INST]",
                "assistant": "{content}</s>",
            },
            reply="",
            prefix="<s>",
            stops=("</s>", "[INST]"),
        ),
    ),
)


def chat_format(model_type: str | None) -> ChatFormat:
    """Pick the chat markup for an artifact's `model.type`.

    onnxruntime-genai applies no chat template of its own, so the server has to supply
    one. Matching on the type recorded in genai_config.json is what the reference
    benchmark scripts do, and it is why a model can behave correctly under
    model_benchmark.exe while looking broken here: sending one family's markup to
    another leaves the markers as ordinary text, so the model continues the transcript
    it appears to have been given and answers on the user's behalf.
    """
    name = (model_type or "").lower()
    for prefix, fmt in _CHAT_FORMATS:
        if name.startswith(prefix):
            return fmt
    return _CHATML


def _stop_prefix_len(text: str, stops: list[str]) -> int:
    """Length of the shortest tail of `text` that must be withheld from the client.

    A stop marker that lies wholly inside `text` is detected by the caller. The only
    other way one can appear is by straddling the boundary, so the sole text that
    cannot be released yet is a suffix which is also a proper prefix of some marker --
    usually nothing at all, which is what lets tokens stream out as they are decoded.
    """
    longest = 0
    for stop in stops:
        # A full match is the caller's case; only shorter overlaps are undecided.
        for size in range(min(len(stop) - 1, len(text)), longest, -1):
            if text.endswith(stop[:size]):
                longest = size
                break
    return longest


def _model_type(path: Path) -> str | None:
    """Read `model.type` out of an artifact's genai_config.json."""
    try:
        config = json.loads((path / "genai_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("could not read genai_config.json in %s: %s", path, exc)
        return None
    model = config.get("model")
    return model.get("type") if isinstance(model, dict) else None


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
