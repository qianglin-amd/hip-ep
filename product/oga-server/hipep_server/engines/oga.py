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
import contextlib
import json
import logging
import os
import queue
import shutil
import tempfile
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
    ModelNotAvailable,
    Stopwatch,
    estimate_tokens,
    image_prefill_tokens,
)
from .tools import describe_tools, extract_tool_calls

log = logging.getLogger(__name__)

_SENTINEL = object()

# How long the generating thread stands aside each token so the event loop can write
# what has been produced. Short next to a decode, which is tens of milliseconds on the
# models served here; shorter values leave the stream arriving in visible bursts.
_YIELD_S = 0.001

# Carrying a generator between turns keeps its KV cache, so a follow-up question only
# has to read the words that were added rather than the whole conversation again.
# Set HIPEP_KV_REUSE=0 to prefill every turn from scratch, which costs time but makes a
# reply depend only on the conversation and not on what the server happens to hold.
_KV_REUSE = os.environ.get("HIPEP_KV_REUSE", "1").lower() not in ("0", "false", "no")

# A kept generator can only grow to the budget it was built with, so that budget is
# rounded up to leave room for turns to come. Sizing it to the current turn exactly
# would force a rebuild, and so a full prefill, on the very next question.
_KV_GROWTH = max(1, int(os.environ.get("HIPEP_KV_GROWTH", "2048")))


@dataclass
class _Session:
    """A generator kept alive between turns, and what its cache is known to hold.

    `fed` is the token sequence the cache was built from, which is the prompt plus
    everything generated from it. The next turn is only allowed onto this cache for as
    far as it agrees with `fed`; past that point the cache describes a different
    conversation and has to be rewound.
    """

    generator: Any
    fed: list[int]
    budget: int
    search: tuple[tuple[str, Any], ...]


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _resume(loaded: _LoadedModel, tokens: list[int], key: tuple, needed: int) -> Any | None:
    """Continue the kept generator on `tokens`, or None if it cannot serve them.

    The saved cache is usable up to the point where this prompt still matches what
    built it. That is normally the whole of the previous turn, so only the new question
    is read; when the caller edits or drops earlier messages the match ends sooner and
    the cache is rewound to there, which still spares everything before the change.
    """
    session = loaded.session
    if session is None or session.search != key or session.budget < needed:
        return None
    # One token has to be left to append: a generator is driven by feeding it something
    # to answer, so a prompt already held in full still has to re-read its last token.
    common = min(_common_prefix(session.fed, tokens), len(tokens) - 1)
    if common <= 0:
        return None
    try:
        if common < len(session.fed):
            session.generator.rewind_to(common)
        session.generator.append_tokens(tokens[common:])
    except Exception as exc:  # noqa: BLE001 - a refused cache is recoverable
        log.warning("could not continue the kept generator, prefilling instead: %s", exc)
        loaded.session = None
        return None
    log.debug("reused %d of %d prompt tokens for %s", common, len(tokens), loaded.model.id)
    return session.generator


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
        self.chat = chat_format(self.model_type, _declared_template(path))
        # Multimodal models expose a processor; text-only models expose a tokenizer.
        self.processor = None
        self.tokenizer = None
        if model.supports_vision:
            # The processor is built by the model, not constructed from it: the class is
            # exposed for its methods but has no constructor bound, so calling it raises
            # and leaves a vision model reading its prompt as plain text.
            try:
                self.processor = self.og_model.create_multimodal_processor()
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "no multimodal processor for %s, so images cannot be read: %s",
                    model.id,
                    exc,
                )
        if self.processor is None:
            self.tokenizer = og.Tokenizer(self.og_model)
        self.lock = threading.Lock()
        # At most one conversation is carried at a time. A second one would double the
        # cache held on the GPU for a case the console cannot reach, so a request that
        # does not continue the kept conversation replaces it instead of joining it.
        self.session: _Session | None = None

    def close(self) -> None:
        self.session = None
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
                # A simulated port leaves a marker where the artifact would be. It is
                # listed and selectable like any other local model, so without this the
                # user is told to download something they appear to already have.
                if (path / "SIMULATED").is_file():
                    raise ModelNotAvailable(
                        f"'{model.id}' came from a simulated port and has no weights. "
                        "Install the porting toolchain with: "
                        'pip install -e "product[quant]" , then port it again.'
                    )
                raise ModelNotAvailable(
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
        # A tool call is markup the user must never see, and it is only recognisable
        # once complete, so a request carrying tools is accumulated and split at the
        # end rather than streamed through. Requests without tools stream unchanged.
        collecting = bool(request.tools)
        collected: list[str] = []
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
                    if collecting:
                        collected.append(item)
                    else:
                        yield Delta(text=item)
        finally:
            cancel.set()

        calls = []
        if collecting:
            text, calls = extract_tool_calls("".join(collected))
            if text:
                yield Delta(text=text)

        watch.finish()
        usage = watch.usage(prompt_tokens, completion_tokens)
        yield Delta(
            done=True,
            finish_reason="tool_calls" if calls else "stop",
            usage=usage.to_dict(),
            tool_calls=calls,
        )

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

            prompt = loaded.chat.render(request.messages, request.tools)

            # A model with no working processor cannot see an image, but it will answer
            # anyway and what it invents reads exactly like a description of the picture
            # that was sent. Refusing is the only way the caller finds out.
            if request.images and loaded.processor is None:
                raise EngineError(
                    f"'{request.model.id}' cannot read images on this build: its "
                    "multimodal processor failed to load, so the server would be "
                    "answering from the text alone. The server log records why."
                )

            # The model holds one generator at a time; concurrent requests for the same
            # model are serialised here and queued by the scheduler above.
            with loaded.lock:
                if loaded.processor is not None:
                    # The staged files only have to outlive the processor call: what it
                    # returns is the pixels already read, not a reference to the path.
                    with _staged_images(og, request) as images:
                        inputs = loaded.processor(prompt, images=images)
                    tokens = None
                    detok = loaded.processor.create_stream()
                    prompt_len = _input_length(inputs, prompt, len(request.images))
                else:
                    inputs = None
                    # Listed rather than left as the tokenizer's array, because these
                    # tokens are compared against a kept cache token by token.
                    tokens = list(loaded.tokenizer.encode(prompt))
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
                needed = counted + request.params.max_tokens
                # Sampling settings are fixed when the generator is built, so a turn
                # that changes them cannot join a cache built under the old ones.
                key = tuple(sorted(search.items()))
                reusable = _KV_REUSE and inputs is None

                generator = None
                if reusable:
                    generator = _resume(loaded, tokens, key, needed)
                if generator is None:
                    budget = needed
                    if reusable:
                        # Room for the turns after this one, so the cache survives them.
                        budget = -(-needed // _KV_GROWTH) * _KV_GROWTH
                    search["max_length"] = min(request.model.max_context, budget)
                    params.set_search_options(**search)
                    # The prompt attaches to the generator rather than to the params,
                    # and logits are computed by generate_next_token. Both moved when
                    # onnxruntime-genai gained continuous batching.
                    loaded.session = None
                    generator = og.Generator(loaded.og_model, params)
                    if inputs is not None:
                        generator.set_inputs(inputs)
                    else:
                        generator.append_tokens(tokens)
                    if reusable:
                        loaded.session = _Session(
                            generator, list(tokens), search["max_length"], key
                        )
                # The family's own end-of-turn markers, on top of anything the caller
                # asked for, so a model whose artifact under-declares its EOS ids still
                # stops at the end of its answer instead of continuing the transcript.
                stops = list(dict.fromkeys([*(request.params.stop or ()), *loaded.chat.stops]))
                buffered = ""
                while not generator.is_done():
                    if cancel.is_set():
                        break
                    # onnxruntime-genai holds the GIL for the whole of a decode, and
                    # this thread reacquires it before the event loop is scheduled, so
                    # the frames the loop has to write pile up until generation ends
                    # and the answer arrives in one burst. Standing aside for a moment
                    # each token leaves the loop a window to drain the queue and write,
                    # which is what makes the stream arrive as it is produced.
                    time.sleep(_YIELD_S)
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

                # What the cache holds now is the prompt plus whatever was generated
                # from it, including a part-written answer from a cancelled turn. Read
                # it back from the generator rather than reconstructing it, so the next
                # turn compares against the tokens the model actually saw.
                if loaded.session is not None:
                    loaded.session.fed = list(generator.get_sequence(0))
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an error
            log.exception("generation failed for %s", request.model.id)
            # The generator failed part way, so what its cache holds is unknown.
            loaded.session = None
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
    # How an attached image is marked inside a turn, with `{n}` the image's position in
    # the conversation. The processor matches the pictures it was handed against these
    # markers and refuses the prompt if the two disagree, and the marker is per family:
    # the Phi wording in a Qwen prompt is read as ordinary text, which leaves the model
    # describing a picture it was never shown.
    image: str = "Picture {n}: <|vision_start|><|image_pad|><|vision_end|>\n"

    def render(self, messages: Any, tools: list[dict[str, Any]] | None = None) -> str:
        parts = [self.prefix]
        # Images are numbered across the conversation rather than within a turn, which
        # is what the families' own templates do and what keeps the count matching the
        # pictures handed to the processor.
        seen_images = 0
        if tools:
            # The tool list is a system turn because that is where these models were
            # trained to look for it, and because it must not read as something the
            # user said.
            description = describe_tools(tools)
            if description:
                parts.append(
                    self.turns.get("system", self.turns["user"]).format(content=description)
                )
        for message in messages:
            content = message.content
            if message.tool_calls:
                # Replay the model's own earlier call, or a later turn has a result
                # arriving in answer to nothing.
                content = "\n".join(
                    filter(
                        None,
                        [
                            content,
                            *(
                                f'<tool_call>{{"name": "{call.name}", '
                                f'"arguments": {call.arguments}}}</tool_call>'
                                for call in message.tool_calls
                            ),
                        ],
                    )
                )
            role = message.role
            if role == "tool":
                # No family has a role for this, so the result is handed back in the
                # user turn, tagged with the tool it came from.
                label = message.name or message.tool_call_id or "tool"
                content = (
                    f'<tool_response>{{"name": "{label}", '
                    f'"content": {json.dumps(content, ensure_ascii=False)}}}</tool_response>'
                )
                role = "user"
            if message.images:
                # Image placeholders are what the multimodal processors look for.
                placeholders = "".join(
                    self.image.format(n=seen_images + i + 1)
                    for i in range(len(message.images))
                )
                seen_images += len(message.images)
                content = placeholders + content
            template = self.turns.get(role, self.turns["user"])
            parts.append(template.format(content=content))
        parts.append(self.reply)
        return "".join(parts)


def _symmetric(
    turn: str,
    reply: str,
    *,
    prefix: str = "",
    stops: tuple[str, ...] = (),
    image: str | None = None,
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
        **({"image": image} if image else {}),
    )


# Phi marks an image by its number alone; the Qwen wording carries no meaning here.
_PHI_IMAGE = "<|image_{n}|>\n"


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
            image=_PHI_IMAGE,
        ),
    ),
    (
        "phi",
        _symmetric(
            "<|{role}|>\n{content}<|end|>\n",
            "<|assistant|>\n",
            stops=("<|end|>", "<|user|>"),
            image=_PHI_IMAGE,
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
            image="<start_of_image>",
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


# The marker that identifies each family's markup, most specific first. Phi-4 has to be
# tested before ChatML because it uses ChatML's turn markers with its own separator.
_FORMAT_MARKERS: tuple[tuple[str, str], ...] = (
    ("<|im_sep|>", "phi4"),
    ("<|im_start|>", "qwen"),
    ("<|start_header_id|>", "llama"),
    ("<start_of_turn>", "gemma"),
    ("<|end|>", "phi"),
    ("[INST]", "mistral"),
)


def chat_format(model_type: str | None, declared: str | None = None) -> ChatFormat:
    """Pick the chat markup for an artifact.

    onnxruntime-genai applies no chat template of its own, so the server has to supply
    one, and sending one family's markup to another leaves the markers as ordinary text:
    the model then continues the transcript it appears to have been given and answers on
    the user's behalf.

    `declared` is the model's own chat template, and it decides when it is there, because
    architecture does not imply markup. SmolLM2 is `LlamaForCausalLM` and speaks ChatML,
    so matching on the type alone wrapped it in Llama 3 headers and it replied by
    restating the question with an `<|eot_id|>` on the end. The type is the fallback for
    the artifacts that carry no template.
    """
    if declared:
        for marker, family in _FORMAT_MARKERS:
            if marker in declared:
                return dict(_CHAT_FORMATS)[family]

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


def _declared_template(path: Path) -> str | None:
    """The chat template an artifact ships, wherever the exporter happened to put it.

    Recent transformers writes it to `chat_template.jinja`; older ones leave it inside
    `tokenizer_config.json`. Only the markers in it are read, so an unparsed Jinja
    string is all this needs to return.
    """
    jinja = path / "chat_template.jinja"
    try:
        if jinja.is_file():
            return jinja.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("could not read chat_template.jinja in %s: %s", path, exc)

    try:
        config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    template = config.get("chat_template")
    if isinstance(template, list):
        # Some tokenizers ship several named templates; the default is the chat one.
        template = next(
            (t.get("template") for t in template if isinstance(t, dict)),
            None,
        )
    return template if isinstance(template, str) else None


def _model_type(path: Path) -> str | None:
    """Read `model.type` out of an artifact's genai_config.json."""
    try:
        config = json.loads((path / "genai_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("could not read genai_config.json in %s: %s", path, exc)
        return None
    model = config.get("model")
    return model.get("type") if isinstance(model, dict) else None


def _input_length(inputs: Any, prompt: str, image_count: int) -> int:
    """How many tokens the processor actually produced for a prompt and its images.

    A picture expands into hundreds of tokens that the prompt text gives no sign of, so
    a budget estimated from the words alone is far smaller than the input it has to
    cover and onnxruntime-genai rejects the prompt outright. The processor knows the
    real figure; the estimate is kept only for a build that does not report a shape.
    """
    try:
        return int(inputs["input_ids"].shape[-1])
    except Exception:  # noqa: BLE001 - any processor that does not report a shape
        return estimate_tokens(prompt) + image_prefill_tokens(image_count)


# What each image format starts with, since the loader dispatches on the file name and
# a caller may send any of them under any content type.
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF8", ".gif"),
    (b"BM", ".bmp"),
    (b"II*\x00", ".tiff"),
    (b"MM\x00*", ".tiff"),
)


def _image_suffix(data: bytes) -> str:
    """Name a staged image after what it actually is, not what it was labelled."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    for magic, suffix in _IMAGE_MAGIC:
        if data.startswith(magic):
            return suffix
    return ".png"


@contextlib.contextmanager
def _staged_images(og: Any, request: GenerationRequest) -> Any:
    """Stage the request's images on disk for as long as the processor needs them.

    onnxruntime-genai opens images by path, so the bytes have to be written somewhere
    first. A directory of its own per request keeps two things right that a shared name
    did not: `id()` is reused as soon as an object is freed, so naming files after the
    request eventually handed one request's path to another, and the files were left
    behind on every call. The directory goes away once the processor has read it.
    """
    if not request.images:
        yield None
        return
    folder = Path(tempfile.mkdtemp(prefix="hipep-img-"))
    try:
        paths = []
        for i, data in enumerate(request.images):
            tmp = folder / f"{i}{_image_suffix(data)}"
            tmp.write_bytes(data)
            paths.append(str(tmp))
        yield og.Images.open(*paths)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
