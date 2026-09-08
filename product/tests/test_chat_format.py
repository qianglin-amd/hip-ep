# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Chat markup per model family.

onnxruntime-genai applies no chat template, so the server supplies one. Getting it
wrong does not raise: the markers arrive as ordinary text, the model continues the
transcript it appears to have been handed, and it answers on the user's behalf until it
hits max_tokens. Nothing in a throughput benchmark catches that, because
model_benchmark.exe never prints what the model said. These tests are the safety net
instead.
"""

from __future__ import annotations

import pytest
from hipep_server.engines import ChatMessage
from hipep_server.engines.oga import chat_format


def rendered(model_type: str, text: str = "Hi") -> str:
    return chat_format(model_type).render([ChatMessage(role="user", content=text)])


@pytest.mark.parametrize(
    ("model_type", "marker"),
    [
        ("qwen3_5_moe", "<|im_start|>"),
        ("qwen2", "<|im_start|>"),
        ("llama", "<|start_header_id|>"),
        ("phi3", "<|user|>"),
        ("phi4", "<|im_sep|>"),
        ("gemma", "<start_of_turn>"),
        ("mistral", "[INST]"),
    ],
)
def test_each_family_gets_its_own_markup(model_type: str, marker: str) -> None:
    assert marker in rendered(model_type), f"{model_type} was not given its own markup"


def test_families_do_not_borrow_each_others_markers() -> None:
    """The bug this guards: every model was handed Phi's markup."""
    qwen = rendered("qwen3_5_moe")
    assert "<|end|>" not in qwen, "Qwen is being sent Phi's end marker"
    assert "<|start_header_id|>" not in qwen, "Qwen is being sent Llama's header marker"

    llama = rendered("llama")
    assert "<|im_start|>" not in llama, "Llama is being sent ChatML markup"
    assert "<|end|>" not in llama, "Llama is being sent Phi's end marker"


def test_every_family_can_be_stopped() -> None:
    """A format with no stop marker generates until max_tokens.

    Some converted artifacts under-declare their EOS ids -- Llama 3.1 lists
    `<|end_of_text|>` but emits `<|eot_id|>` -- so the format has to carry the marker.
    """
    for model_type in ("qwen3_5_moe", "llama", "phi3", "phi4", "gemma", "mistral"):
        fmt = chat_format(model_type)
        assert fmt.stops, f"{model_type} can never stop"
        for stop in fmt.stops:
            assert stop, f"{model_type} has an empty stop marker"

    assert "<|eot_id|>" in chat_format("llama").stops, (
        "Llama must stop on the marker it actually emits, not just the declared EOS"
    )


def test_the_prompt_ends_where_the_model_should_speak() -> None:
    """Every format has to hand the turn over, or the model completes the user."""
    for model_type, opening in (
        ("qwen3_5_moe", "<|im_start|>assistant\n"),
        ("llama", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("phi3", "<|assistant|>\n"),
        ("gemma", "<start_of_turn>model\n"),
    ):
        assert rendered(model_type).endswith(opening), (
            f"{model_type} does not end by opening the assistant turn"
        )


def test_an_unknown_family_still_gets_usable_markup() -> None:
    """New artifacts appear before this table learns about them."""
    fmt = chat_format("some-model-released-tomorrow")
    assert fmt.stops, "an unrecognised model would never stop"
    assert "<|im_start|>" in rendered("some-model-released-tomorrow"), (
        "the fallback should be ChatML, which most current artifacts use"
    )
    assert chat_format(None).stops, "a missing model type would never stop"


def test_multi_turn_conversations_keep_the_roles_apart() -> None:
    messages = [
        ChatMessage(role="system", content="Be brief."),
        ChatMessage(role="user", content="First"),
        ChatMessage(role="assistant", content="Second"),
        ChatMessage(role="user", content="Third"),
    ]
    prompt = chat_format("qwen3_5_moe").render(messages)
    assert prompt.count("<|im_start|>") == 5, "one opener per turn plus the reply"
    assert prompt.index("First") < prompt.index("Second") < prompt.index("Third"), (
        "turns are out of order"
    )
    assert "system" in prompt, "the system message was dropped"
