# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""OpenAI compatibility.

The promise made in the docs is that an application can point an OpenAI SDK at this
server and change nothing else, so these tests assert the shapes that SDK depends on
rather than the shapes we happen to emit.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient


def sse_events(response) -> list[dict | str]:
    """Decode an SSE body into its payloads, dropping the terminator."""
    events: list[dict | str] = []
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


def test_models_list_is_openai_shaped(client: TestClient) -> None:
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert body["data"], "catalog should not be empty"
    for model in body["data"]:
        assert model["object"] == "model"
        assert model["id"] and model["owned_by"]


def test_models_list_filters_by_capability(client: TestClient) -> None:
    vision = client.get("/v1/models", params={"capability": "vision"}).json()["data"]
    every = client.get("/v1/models").json()["data"]
    assert 0 < len(vision) < len(every)


def test_retrieve_unknown_model_is_404(client: TestClient) -> None:
    assert client.get("/v1/models/nope:0b").status_code == 404


def test_a_model_that_is_not_downloaded_is_404_not_500(
    client: TestClient, chat_model: str, monkeypatch
) -> None:
    """A missing download is the caller's to fix, and the status has to say so.

    Every engine failure mapped to 500, so an OpenAI client met a retryable server
    fault for a request that could not succeed until the model was pulled. The message
    was already right; only the code disagreed with it.
    """
    from hipep_server.engines import ModelNotAvailable

    async def refuse(model):
        raise ModelNotAvailable(f"model '{model.id}' is not downloaded. Pull it first.")

    monkeypatch.setattr(client.app.state.scheduler.engine, "load", refuse)

    response = client.post(
        "/v1/chat/completions",
        json={"model": chat_model, "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 404, response.text
    assert "not downloaded" in response.json()["detail"]


def test_a_genuine_engine_failure_is_still_500(
    client: TestClient, chat_model: str, monkeypatch
) -> None:
    """The narrower status must not swallow real faults."""
    from hipep_server.engines import EngineError

    async def explode(model):
        raise EngineError("the GPU fell over")

    monkeypatch.setattr(client.app.state.scheduler.engine, "load", explode)

    response = client.post(
        "/v1/chat/completions",
        json={"model": chat_model, "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 500, response.text


def test_chat_completion_returns_content_and_timing(
    client: TestClient, chat_model: str
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": chat_model, "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["object"] == "chat.completion"
    assert body["model"] == chat_model
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"].strip()
    assert choice["finish_reason"] == "stop"

    usage = body["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    # The extension the console and the CLI both rely on.
    assert {"ttft_s", "decode_s", "tps", "load_s"} <= usage.keys()


def test_chat_completion_honours_max_tokens(client: TestClient, chat_model: str) -> None:
    body = client.post(
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "Write at length"}],
            "max_tokens": 5,
        },
    ).json()
    assert body["usage"]["completion_tokens"] <= 5


def test_chat_completion_rejects_empty_messages(client: TestClient, chat_model: str) -> None:
    response = client.post("/v1/chat/completions", json={"model": chat_model, "messages": []})
    assert response.status_code == 400


def test_streaming_frames_reassemble_to_the_same_answer(
    client: TestClient, chat_model: str
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-hipep-session"]

    events = sse_events(response)
    assert all(e["object"] == "chat.completion.chunk" for e in events)
    # First frame opens the message, last frame carries the finish reason and usage.
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["usage"]["tps"] >= 0

    streamed = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
    assert streamed.strip()


def test_image_is_accepted_by_a_vision_model(
    client: TestClient, vision_model: str, png_data_url: str
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": png_data_url}},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200
    # Image prefill dominates the prompt, so the count must reflect it.
    assert response.json()["usage"]["prompt_tokens"] > 100


def test_image_is_refused_by_a_text_only_model(
    client: TestClient, chat_model: str, png_data_url: str
) -> None:
    """A text model silently ignoring an image is the worst outcome: the answer looks
    plausible and is unrelated to the picture."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": png_data_url}},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 400
    assert "vision" in response.json()["detail"].lower()


def test_legacy_completions_endpoint(client: TestClient, chat_model: str) -> None:
    body = client.post(
        "/v1/completions", json={"model": chat_model, "prompt": "Once upon a time"}
    ).json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"].strip()


def test_embeddings_refuses_rather_than_returning_zeros(client: TestClient) -> None:
    response = client.post("/v1/embeddings", json={"model": "any", "input": "text"})
    assert response.status_code == 501
