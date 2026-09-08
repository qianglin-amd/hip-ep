# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Streaming behaviour of the OGA engine that does not need a GPU to check.

The engine decodes one token at a time and must decide, after each one, how much text
it may hand to the client. Release too much and a stop marker split across two tokens
reaches the user as visible markup; hold back too much and every chunk is delayed by
several tokens, which reads as the model pausing before it answers.
"""

from __future__ import annotations

import pytest
from hipep_server.engines.oga import _stop_prefix_len

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
