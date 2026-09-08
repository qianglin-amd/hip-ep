# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Tool calling.

The catalog advertises a `tools` capability and the console shows a badge for it, so
the API has to honour it: a harness that sends tool definitions and receives prose has
no way to tell whether the model declined or the server ignored the request.

The parsing tests matter more than they look. Families disagree about the wrapper they
were trained to emit, and a wrapper we fail to recognise does not raise -- it silently
turns a tool call into a paragraph of JSON shown to the user.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from hipep_server.engines.tools import describe_tools, extract_tool_calls

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


# --- parsing ------------------------------------------------------------------


def test_chatml_wrapper_is_recognised() -> None:
    text, calls = extract_tool_calls(
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    )
    assert text == ""
    assert [(c.name, json.loads(c.arguments)) for c in calls] == [
        ("get_weather", {"city": "Paris"})
    ]


def test_llama_python_tag_is_recognised() -> None:
    """Llama 3.1 marks a call with `<|python_tag|>` and names the arguments
    `parameters`."""
    _, calls = extract_tool_calls(
        '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}'
    )
    assert len(calls) == 1
    assert json.loads(calls[0].arguments) == {"city": "Paris"}


def test_a_fenced_call_is_recognised() -> None:
    _, calls = extract_tool_calls(
        'Sure.\n```json\n{"name": "get_weather", "arguments": {"city": "Oslo"}}\n```'
    )
    assert len(calls) == 1
    assert calls[0].name == "get_weather"


def test_a_bare_object_is_recognised() -> None:
    _, calls = extract_tool_calls('{"name": "get_weather", "arguments": {"city": "Rome"}}')
    assert len(calls) == 1


def test_narration_around_a_call_is_kept_and_the_markup_is_not() -> None:
    """A model that explains itself first must not have its markup shown."""
    text, calls = extract_tool_calls(
        'Let me look that up. <tool_call>{"name": "get_weather", '
        '"arguments": {"city": "Paris"}}</tool_call>'
    )
    assert calls and "tool_call" not in text
    assert text == "Let me look that up."


def test_ordinary_json_in_an_answer_is_not_a_tool_call() -> None:
    """The bare-object path is the loosest one, so it has to stay narrow."""
    text, calls = extract_tool_calls(
        'Here is an example config:\n{"name": "server", "port": 8080}'
    )
    assert calls == []
    assert text.endswith("}")


def test_debris_inside_the_wrapper_does_not_lose_the_call() -> None:
    """Exactly what Llama 3.1 emits on the GPU, every time, at temperature 0.

    The wrapper is right and the object inside it is right, but the model repeats the
    tool name in between, so the body as a whole is not JSON. Rejecting it turned a
    correct tool call into an empty reply.
    """
    text, calls = extract_tool_calls(
        '<tool_call>get_weather>{"name": "get_weather", '
        '"arguments": {"city": "Paris"}}</tool_call> '
    )
    assert len(calls) == 1, "the call the model plainly made was dropped"
    assert calls[0].name == "get_weather"
    assert json.loads(calls[0].arguments) == {"city": "Paris"}
    assert text == ""


def test_unparseable_markup_never_reaches_the_user() -> None:
    """What Llama 3.1 actually did on the GPU: a real call marked one way, and a
    malformed fragment marked another. Stripping only the wrapper that parsed left
    `<tool_call>get_weather></tool_call>` sitting in the reply."""
    text, calls = extract_tool_calls(
        "<tool_call>get_weather></tool_call>"
        '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}'
    )
    assert len(calls) == 1
    assert "tool_call" not in text and text == ""


def test_an_unterminated_marker_does_not_become_the_answer() -> None:
    """What Qwen2.5-0.5B did on the GPU: it opened `<tool_call>`, wrote the object and
    never closed the tag. The closed-tag pattern missed it, the bare-object path took
    the JSON, and the lone opener was returned as the assistant's content -- which
    every OpenAI client renders."""
    text, calls = extract_tool_calls(
        '<tool_call>\n{"name": "get_current_time", "arguments": {}}'
    )
    assert len(calls) == 1
    assert calls[0].name == "get_current_time"
    assert text == "", f"marker leaked into the reply: {text!r}"


def test_an_orphan_closing_tag_is_not_shown() -> None:
    text, calls = extract_tool_calls(
        '<tool_call>{"name": "get_time", "arguments": {}}</tool_call></tool_call>'
    )
    assert len(calls) == 1
    assert "tool_call" not in text


def test_narration_before_an_unterminated_marker_survives() -> None:
    """A model that explains itself first should keep the explanation; only the markup
    is the part the user must not see."""
    text, calls = extract_tool_calls(
        'Let me check the clock.\n<tool_call>{"name": "get_time", "arguments": {}}'
    )
    assert len(calls) == 1
    assert text == "Let me check the clock."


def test_prose_is_left_alone() -> None:
    text, calls = extract_tool_calls("The weather in Paris is mild today.")
    assert calls == []
    assert text == "The weather in Paris is mild today."


def test_two_calls_in_one_reply_are_both_returned() -> None:
    _, calls = extract_tool_calls(
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
    )
    assert [c.name for c in calls] == ["a", "b"]


def test_every_call_gets_its_own_id() -> None:
    _, calls = extract_tool_calls(
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
    )
    assert calls[0].id != calls[1].id


def test_arguments_are_always_json_text() -> None:
    """A caller does `json.loads` on this field, so it can never be a bare string."""
    _, calls = extract_tool_calls('<tool_call>{"name": "a", "arguments": "Paris"}</tool_call>')
    assert json.loads(calls[0].arguments) == {"input": "Paris"}


def test_the_schema_reaches_the_model_intact() -> None:
    described = describe_tools([WEATHER_TOOL])
    assert "get_weather" in described
    assert '"required": ["city"]' in described.replace("'", '"')
    assert "<tool_call>" in described


def test_a_tool_without_a_name_is_skipped() -> None:
    assert describe_tools([{"type": "function", "function": {"description": "x"}}]) == ""


# --- the OpenAI surface -------------------------------------------------------


def test_chat_returns_tool_calls_and_the_matching_finish_reason(
    client: TestClient, chat_model: str
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [WEATHER_TOOL],
        },
    )
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]

    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    json.loads(call["function"]["arguments"])  # must parse
    assert choice["message"]["content"] is None


def test_a_request_without_tools_is_unchanged(client: TestClient, chat_model: str) -> None:
    """The tool path must not alter ordinary chat."""
    choice = client.post(
        "/v1/chat/completions",
        json={"model": chat_model, "messages": [{"role": "user", "content": "Hi"}]},
    ).json()["choices"][0]

    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]
    assert isinstance(choice["message"]["content"], str)


def test_a_tool_result_can_be_handed_back(client: TestClient, chat_model: str) -> None:
    """The whole point of the feature: the second leg of an agent loop."""
    first = client.post(
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [WEATHER_TOOL],
        },
    ).json()["choices"][0]["message"]

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                first,
                {
                    "role": "tool",
                    "tool_call_id": first["tool_calls"][0]["id"],
                    "name": "get_weather",
                    "content": "18C, clear",
                },
            ],
            "tools": [WEATHER_TOOL],
        },
    )
    assert second.status_code == 200, second.text
    choice = second.json()["choices"][0]
    # Having been given the answer, the loop has to be able to end.
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"]


def test_tool_choice_none_suppresses_the_tools(client: TestClient, chat_model: str) -> None:
    choice = client.post(
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": "none",
        },
    ).json()["choices"][0]

    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]


def test_an_unsupported_tool_choice_is_refused(client: TestClient, chat_model: str) -> None:
    """Silently ignoring it would leave a caller believing a guarantee was applied."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": "sometimes",
        },
    )
    assert response.status_code == 400
    assert "tool_choice" in response.json()["detail"]


def test_a_malformed_tool_is_refused(client: TestClient, chat_model: str) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"description": "no name"}}],
        },
    )
    assert response.status_code == 400


def test_tool_calls_arrive_on_the_stream(client: TestClient, chat_model: str) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [WEATHER_TOOL],
            "stream": True,
        },
    ) as response:
        frames = [
            json.loads(line[6:])
            for line in response.iter_lines()
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]

    calls = [
        call for frame in frames for call in frame["choices"][0]["delta"].get("tool_calls", [])
    ]
    assert calls, "no tool call was streamed"
    assert calls[0]["function"]["name"] == "get_weather"
    assert calls[0]["index"] == 0
    assert frames[-1]["choices"][0]["finish_reason"] == "tool_calls"


# --- the Ollama surface -------------------------------------------------------


def test_ollama_chat_returns_tool_calls(client: TestClient, chat_model: str) -> None:
    """Ollama nests the same call under `message.tool_calls` with parsed arguments."""
    body = client.post(
        "/api/chat",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [WEATHER_TOOL],
            "stream": False,
        },
    ).json()

    call = body["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    # Ollama gives arguments as an object, not as JSON text.
    assert isinstance(call["function"]["arguments"], dict)
    assert body["done_reason"] == "tool_calls"
