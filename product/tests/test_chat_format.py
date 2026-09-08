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


class TestDeclaredTemplateWins:
    """An artifact's own chat template beats a guess from its architecture.

    SmolLM2 is `LlamaForCausalLM` and speaks ChatML. Matching on the type alone wrapped
    it in Llama 3 headers, and it answered by restating the question with a stray
    `<|eot_id|>` on the end -- markup that had arrived as ordinary text.
    """

    # Trimmed from the file SmolLM2-360M-Instruct ships.
    SMOLLM2 = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
        "{% endfor %}"
    )

    def test_a_chatml_template_on_a_llama_architecture_gets_chatml(self) -> None:
        fmt = chat_format("llama", self.SMOLLM2)
        prompt = fmt.render([ChatMessage(role="user", content="Hi")])

        assert "<|im_start|>" in prompt
        assert "<|start_header_id|>" not in prompt, "still being sent Llama markup"
        assert "<|eot_id|>" not in fmt.stops

    def test_a_llama_template_still_gets_llama_markup(self) -> None:
        template = "{{'<|start_header_id|>' + role + '<|end_header_id|>'}}"
        assert "<|start_header_id|>" in rendered_with("llama", template)

    def test_phi4_is_not_mistaken_for_chatml(self) -> None:
        # Phi-4 uses ChatML's turn markers with its own separator, so the more
        # specific marker has to be tested first.
        template = "{{'<|im_start|>' + role + '<|im_sep|>' + content + '<|im_end|>'}}"
        assert "<|im_sep|>" in rendered_with("llama", template)

    def test_a_template_naming_no_known_markup_falls_back_to_the_type(self) -> None:
        fmt = chat_format("gemma", "{{ messages | join('\n') }}")
        assert "<start_of_turn>" in fmt.render([ChatMessage(role="user", content="Hi")])

    def test_no_template_behaves_as_before(self) -> None:
        assert chat_format("llama", None).stops == chat_format("llama").stops


def rendered_with(model_type: str, template: str) -> str:
    return chat_format(model_type, template).render([ChatMessage(role="user", content="Hi")])


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
