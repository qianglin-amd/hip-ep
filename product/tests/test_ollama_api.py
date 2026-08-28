# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Ollama compatibility.

The server listens on Ollama's default port so that existing Ollama clients work with
no configuration at all. That only holds if the response bodies match what those
clients parse, including the nanosecond duration fields they use to show timings.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

NS = 1_000_000_000


def ndjson_frames(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_version_reports_a_modern_baseline(client: TestClient) -> None:
    body = client.get("/api/version").json()
    assert body["version"]
    assert body["hipep_version"]


def test_tags_lists_generative_models_with_details(client: TestClient) -> None:
    models = client.get("/api/tags").json()["models"]
    assert models
    for entry in models:
        assert entry["name"] == entry["model"]
        assert entry["details"]["format"] == "onnx"
        assert entry["details"]["quantization_level"]
        # Our own annotation, so a client can tell listed from downloaded.
        assert "installed" in entry["hipep"]


def test_tags_excludes_non_generative_tiers(client: TestClient) -> None:
    """`ollama list` means "what can I chat with", so a speech model does not belong."""
    tags = {m["name"] for m in client.get("/api/tags").json()["models"]}
    speech = {
        m["id"] for m in client.get("/v1/catalog", params={"tier": "speech"}).json()["models"]
    }
    assert speech, "expected the catalog to carry speech models"
    assert not (tags & speech)


def test_show_returns_modelfile_and_capabilities(
    client: TestClient, vision_model: str
) -> None:
    body = client.post("/api/show", json={"model": vision_model}).json()
    assert body["modelfile"].startswith("# ")
    assert body["model_info"]["context.length"] > 0
    assert "vision" in body["capabilities"]


def test_non_streaming_chat_carries_durations(client: TestClient, chat_model: str) -> None:
    body = client.post(
        "/api/chat",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    ).json()

    assert body["done"] is True
    assert body["done_reason"] == "stop"
    assert body["message"]["role"] == "assistant"
    assert body["message"]["content"].strip()

    # Ollama clients divide these by 1e9 and by eval_count to show tok/s.
    assert body["eval_count"] > 0
    assert body["eval_duration"] > 0
    assert body["prompt_eval_count"] > 0
    assert body["total_duration"] >= body["eval_duration"] + body["prompt_eval_duration"]


def test_load_duration_is_reported_on_a_cold_model(
    paced_client: TestClient, chat_model: str
) -> None:
    """A first request pays for the model load. Reporting zero there, as a placeholder
    implementation would, makes the load look like slow prefill."""

    def ask(text: str) -> dict:
        return paced_client.post(
            "/api/chat",
            json={
                "model": chat_model,
                "messages": [{"role": "user", "content": text}],
                "stream": False,
            },
        ).json()

    cold = ask("Hello")
    assert cold["load_duration"] > 0
    assert cold["total_duration"] > cold["load_duration"]

    assert ask("Hello again")["load_duration"] == 0


def test_streaming_chat_emits_ndjson_ending_in_done(
    client: TestClient, chat_model: str
) -> None:
    response = client.post(
        "/api/chat",
        json={"model": chat_model, "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.headers["content-type"].startswith("application/x-ndjson")

    frames = ndjson_frames(response)
    assert all(f["model"] == chat_model for f in frames)
    assert not any(f["done"] for f in frames[:-1])
    assert frames[-1]["done"] is True
    assert frames[-1]["eval_count"] > 0

    text = "".join(f.get("message", {}).get("content", "") for f in frames)
    assert text.strip()


def test_generate_supports_a_system_prompt(client: TestClient, chat_model: str) -> None:
    body = client.post(
        "/api/generate",
        json={
            "model": chat_model,
            "prompt": "Hello",
            "system": "You are terse.",
            "stream": False,
        },
    ).json()
    assert body["response"].strip()
    assert body["done"] is True


def test_generate_accepts_ollama_style_images(
    client: TestClient, vision_model: str, png_data_url: str
) -> None:
    # Ollama passes bare base64, not a data URL.
    body = client.post(
        "/api/generate",
        json={
            "model": vision_model,
            "prompt": "Describe this",
            "images": [png_data_url.split(",", 1)[1]],
            "stream": False,
        },
    ).json()
    assert body["prompt_eval_count"] > 100


def test_unknown_model_is_404(client: TestClient) -> None:
    response = client.post(
        "/api/chat",
        json={"model": "nope:0b", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 404


def test_pull_then_delete_round_trip(client: TestClient, chat_model: str, finished) -> None:
    started = client.post("/api/pull", json={"model": chat_model, "stream": False}).json()
    assert started["status"] == "started"

    snapshot = finished(client, started["job"])
    assert snapshot["state"] == "succeeded", snapshot.get("error")

    tags = {m["name"]: m for m in client.get("/api/tags").json()["models"]}
    assert tags[chat_model]["hipep"]["installed"] is True

    def delete():
        return client.request("DELETE", "/api/delete", json={"model": chat_model})

    assert delete().status_code == 200
    assert delete().status_code == 404
