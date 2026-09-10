# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Streaming behaviour of the OGA engine that does not need a GPU to check.

The engine decodes one token at a time and must decide, after each one, how much text
it may hand to the client. Release too much and a stop marker split across two tokens
reaches the user as visible markup; hold back too much and every chunk is delayed by
several tokens, which reads as the model pausing before it answers.
"""

from __future__ import annotations

import queue
import sys
import threading
import types

import pytest
from hipep_server.engines.oga import OgaEngine, _stop_prefix_len

LLAMA_STOPS = ["<|eot_id|>", "<|end_of_text|>"]


@pytest.mark.parametrize(
    ("text", "held"),
    [
        ("Hello there", 0),
        ("", 0),
        ("Hello <", 1),
        ("Hello <|eot", 5),
        ("Hello <|end_of_tex", 12),
        ("trailing angle >", 0),
    ],
)
def test_only_a_possible_marker_start_is_withheld(text: str, held: int) -> None:
    assert _stop_prefix_len(text, LLAMA_STOPS) == held


def test_ordinary_text_streams_immediately() -> None:
    """Nothing that cannot begin a marker may be delayed.

    The regression this guards: reserving the longest marker's width on every chunk
    delayed the first visible token by several decodes.
    """
    for text in ("The answer is 42.", "one two three", "a" * 200):
        assert _stop_prefix_len(text, LLAMA_STOPS) == 0


def test_a_marker_split_across_tokens_never_leaks() -> None:
    """Replays the engine's release loop one token at a time.

    A stop marker is frequently decoded as several tokens, so the boundary case is the
    normal case rather than an edge one.
    """
    pieces = ["Done", "<|", "eot", "_id|>", " trailing"]
    released = ""
    buffered = ""
    for piece in pieces:
        buffered += piece
        hit = next((s for s in LLAMA_STOPS if s in buffered), None)
        if hit:
            released += buffered.split(hit, 1)[0]
            break
        keep = _stop_prefix_len(buffered, LLAMA_STOPS)
        if len(buffered) > keep:
            released += buffered[: len(buffered) - keep]
            buffered = buffered[len(buffered) - keep :]

    assert released == "Done", f"leaked markup or lost text: {released!r}"
    for stop in LLAMA_STOPS:
        assert stop not in released


def test_text_that_merely_resembles_a_marker_is_still_delivered() -> None:
    """A partial marker that turns out to be prose must be released, not swallowed."""
    released = ""
    buffered = ""
    for piece in ("use <", "b> for bold"):
        buffered += piece
        keep = _stop_prefix_len(buffered, LLAMA_STOPS)
        released += buffered[: len(buffered) - keep]
        buffered = buffered[len(buffered) - keep :]
    assert released + buffered == "use <b> for bold"


def test_custom_stops_from_the_request_are_honoured() -> None:
    assert _stop_prefix_len("answer: ST", ["STOP"]) == 2
    assert _stop_prefix_len("answer: ST", []) == 0


class _FakeGenerator:
    """Records what it was asked to read, so a test can tell prefill from continuation."""

    built: list[_FakeGenerator] = []

    def __init__(self, model, params) -> None:
        self.produced = 0
        self.sequence: list[int] = []
        self.appended: list[list[int]] = []
        self.rewound: list[int] = []
        _FakeGenerator.built.append(self)

    def append_tokens(self, tokens) -> None:
        tokens = list(tokens)
        self.appended.append(tokens)
        self.sequence.extend(tokens)
        # Feeding a generator opens a fresh turn on it, as onnxruntime-genai does.
        self.produced = 0

    def is_done(self) -> bool:
        return self.produced >= 4

    def generate_next_token(self) -> None:
        self.produced += 1
        self.sequence.append(900 + self.produced)

    def get_next_tokens(self) -> list[int]:
        return [self.produced]

    def get_sequence(self, index: int) -> list[int]:
        return list(self.sequence)

    def rewind_to(self, length: int) -> None:
        self.rewound.append(length)
        del self.sequence[length:]


def _fake_genai() -> types.ModuleType:
    module = types.ModuleType("onnxruntime_genai")

    class Params:
        def set_search_options(self, **kwargs) -> None:
            pass

    module.GeneratorParams = lambda model: Params()
    module.Generator = _FakeGenerator
    return module


class _FakeLoaded:
    def __init__(self, prompt: str = "prompt") -> None:
        self.og_model = object()
        self.processor = None
        self.lock = threading.Lock()
        self.session = None
        self.model = types.SimpleNamespace(id="m")
        self.chat = types.SimpleNamespace(render=lambda messages, tools: prompt, stops=[])
        # Stands in for a tokenizer: each word becomes a token, so a prompt that extends
        # another shares its leading tokens the way a continued conversation does.
        self.tokenizer = types.SimpleNamespace(
            encode=lambda text: [len(w) * 10 + i for i, w in enumerate(text.split())],
            create_stream=lambda: types.SimpleNamespace(decode=lambda token: f"t{token}"),
        )


def test_the_generation_loop_stands_aside_once_per_token(monkeypatch, tmp_path) -> None:
    """onnxruntime-genai holds the GIL for the whole of a decode, and this thread would
    otherwise reacquire it before the event loop is ever scheduled: the frames pile up
    and the answer arrives in one burst at the end instead of streaming. Losing the
    yield is invisible in output and shows only as that timing, so the interleaving is
    asserted rather than measured, which would be flaky on a loaded machine."""
    from hipep_server.catalog import Model
    from hipep_server.engines import oga as oga_module
    from hipep_server.engines.base import ChatMessage, GenerationRequest

    monkeypatch.setitem(sys.modules, "onnxruntime_genai", _fake_genai())

    events: list[str] = []
    real_sleep = oga_module.time.sleep

    def record(seconds: float) -> None:
        events.append(f"yield {seconds}")
        real_sleep(0)

    monkeypatch.setattr(oga_module.time, "sleep", record)
    original = _FakeGenerator.generate_next_token

    def traced(self) -> None:
        events.append("decode")
        original(self)

    monkeypatch.setattr(_FakeGenerator, "generate_next_token", traced)

    out: queue.Queue = queue.Queue()
    request = GenerationRequest(
        model=Model(id="m", name="m", tier="llm", type="llm"),
        messages=[ChatMessage(role="user", content="hi")],
    )
    engine = OgaEngine(models_dir=tmp_path)
    engine._generate_blocking(_FakeLoaded(), request, out, threading.Event())

    decodes = [e for e in events if e == "decode"]
    assert decodes, "the fake generator never ran"
    # Every decode must be preceded by a yield, or the loop can hold the GIL for the
    # whole generation.
    assert events[: 2 * len(decodes)] == [f"yield {oga_module._YIELD_S}", "decode"] * len(
        decodes
    )
    assert oga_module._YIELD_S > 0


def _run_turn(engine, loaded, prompt: str, tmp_path) -> None:
    from hipep_server.catalog import Model
    from hipep_server.engines.base import ChatMessage, GenerationRequest

    loaded.chat = types.SimpleNamespace(render=lambda messages, tools: prompt, stops=[])
    request = GenerationRequest(
        model=Model(id="m", name="m", tier="llm", type="llm"),
        messages=[ChatMessage(role="user", content="hi")],
    )
    engine._generate_blocking(loaded, request, queue.Queue(), threading.Event())


def test_a_follow_up_turn_reads_only_what_was_added(monkeypatch, tmp_path) -> None:
    """A conversation is re-sent in full every turn, so without a kept cache the model
    re-reads the entire transcript before answering and time-to-first-token climbs with
    the length of the chat. Continuing the previous generator leaves only the new words
    to read. The saving is a timing that would be flaky to measure, so what is asserted
    is that the second turn was handed just the added tokens."""
    from hipep_server.engines import oga as oga_module

    monkeypatch.setitem(sys.modules, "onnxruntime_genai", _fake_genai())
    _FakeGenerator.built.clear()

    engine = OgaEngine(models_dir=tmp_path)
    loaded = _FakeLoaded()

    _run_turn(engine, loaded, "one two three", tmp_path)
    first = _FakeGenerator.built[0]
    assert len(first.appended[0]) == 3, "the opening turn reads the whole prompt"

    # The follow-up repeats the conversation and adds to it, as a chat client does.
    _run_turn(engine, loaded, "one two three four five", tmp_path)

    assert len(_FakeGenerator.built) == 1, "the follow-up built a second generator"
    assert first.appended[1] == [43, 44], "the follow-up re-read more than the new words"
    # The kept cache also holds the previous answer, which the new prompt does not
    # repeat, so the generator is wound back to where the two still agree.
    assert first.rewound == [3]

    # The cache now records the prompt and the answer generated from it, so a later
    # turn compares against what the model actually saw.
    assert oga_module._KV_REUSE
    assert loaded.session is not None
    assert loaded.session.fed == first.sequence


def test_a_different_conversation_does_not_join_the_kept_cache(monkeypatch, tmp_path) -> None:
    """The cache is keyed on the prompt agreeing with what built it. A conversation that
    shares nothing with the kept one must be prefilled on its own generator, or it would
    answer with another chat's context behind it."""
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", _fake_genai())
    _FakeGenerator.built.clear()

    engine = OgaEngine(models_dir=tmp_path)
    loaded = _FakeLoaded()

    _run_turn(engine, loaded, "one two three", tmp_path)
    _run_turn(engine, loaded, "wholly different opening", tmp_path)

    assert len(_FakeGenerator.built) == 2, "an unrelated chat reused the kept cache"
    assert _FakeGenerator.built[1].appended[0] == [60, 91, 72]


def test_reuse_can_be_turned_off(monkeypatch, tmp_path) -> None:
    """Continuing a cache makes an answer depend on what the server holds rather than on
    the conversation alone, so it has to be possible to switch off."""
    from hipep_server.engines import oga as oga_module

    monkeypatch.setitem(sys.modules, "onnxruntime_genai", _fake_genai())
    monkeypatch.setattr(oga_module, "_KV_REUSE", False)
    _FakeGenerator.built.clear()

    engine = OgaEngine(models_dir=tmp_path)
    loaded = _FakeLoaded()

    _run_turn(engine, loaded, "one two three", tmp_path)
    _run_turn(engine, loaded, "one two three four five", tmp_path)

    assert len(_FakeGenerator.built) == 2, "reuse happened with it switched off"
    assert loaded.session is None


def test_an_image_is_refused_rather_than_answered_without_being_seen(
    monkeypatch, tmp_path
) -> None:
    """A model whose multimodal processor failed to load still answers a question about
    a picture, and what it invents reads exactly like a description of the one that was
    sent. Nothing downstream can tell that apart from a real answer, so the request has
    to fail instead of being served from the text alone."""
    from hipep_server.catalog import Model
    from hipep_server.engines.base import ChatMessage, GenerationRequest

    monkeypatch.setitem(sys.modules, "onnxruntime_genai", _fake_genai())
    _FakeGenerator.built.clear()

    loaded = _FakeLoaded()  # a text-only tokenizer, as a failed processor leaves it
    assert loaded.processor is None
    out: queue.Queue = queue.Queue()
    request = GenerationRequest(
        model=Model(id="m", name="m", tier="vlm", type="image-text-to-text"),
        messages=[ChatMessage(role="user", content="what is this?", images=[b"png"])],
    )
    OgaEngine(models_dir=tmp_path)._generate_blocking(loaded, request, out, threading.Event())

    raised = [item for item in out.queue if isinstance(item, Exception)]
    assert raised, "the request was answered without the picture being read"
    assert "cannot read images" in str(raised[0])
    assert not _FakeGenerator.built, "a generator ran for a picture nobody could see"


def test_the_token_budget_counts_what_an_image_expands_into() -> None:
    """A picture becomes hundreds of tokens that the prompt text gives no sign of. A
    budget estimated from the words alone is smaller than the input it has to cover, and
    onnxruntime-genai rejects that outright rather than clamping it."""
    from hipep_server.engines.oga import _input_length

    class _Tensor:
        shape = [1, 217]

    class _Inputs:
        def __getitem__(self, name):
            if name != "input_ids":
                raise RuntimeError(f"Tensor with name: {name} not found.")
            return _Tensor()

    short_prompt = "what is this?"
    assert _input_length(_Inputs(), short_prompt, 1) == 217

    # A processor that reports no shape must still leave room for the picture.
    class _Silent:
        def __getitem__(self, name):
            raise RuntimeError("no shape here")

    estimated = _input_length(_Silent(), short_prompt, 1)
    assert estimated > 217, f"an image was not accounted for at all ({estimated})"
