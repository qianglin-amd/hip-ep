# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Status, health and hardware reporting.

The status payload is what a user reads when the server misbehaves, and its most
important job is to be honest about the engine. A replay-engine server that presented
itself as a GPU server would let someone publish placeholder timings as measurements.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient


def test_health_is_cheap_and_names_the_engine(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["engine"] == "replay"


def test_status_admits_it_is_replaying(client: TestClient) -> None:
    body = client.get("/v1/status").json()
    assert body["engine"]["name"] == "replay"
    assert body["engine"]["real"] is False
    # And explains the choice, so nobody has to guess which precondition failed.
    assert body["engine_reason"]


def test_status_reports_limits_storage_and_catalog(client: TestClient) -> None:
    body = client.get("/v1/status").json()

    assert body["limits"]["max_loaded_models"] > 0
    assert body["limits"]["max_queue_depth"] > 0
    assert body["catalog"]["models"] > 30
    assert body["catalog"]["release"]
    assert body["storage"]["models"] == 0
    assert body["jobs"] == {"running": 0, "total": 0}
    assert body["uptime_s"] >= 0


def test_status_tracks_sessions_across_requests(client: TestClient) -> None:
    """The multi-user view: several clients on one GPU should each be visible."""
    first = client.post(
        "/v1/chat/completions",
        json={"model": "llama3.1:8b", "messages": [{"role": "user", "content": "Hi"}]},
    )
    session = first.headers["x-hipep-session"]

    client.post(
        "/v1/chat/completions",
        headers={"X-HIPEP-Session": session},
        json={"model": "llama3.1:8b", "messages": [{"role": "user", "content": "Again"}]},
    )

    body = client.get("/v1/status").json()
    assert body["totals"]["requests"] == 2
    assert body["totals"]["errors"] == 0
    mine = next(s for s in body["sessions_detail"] if s["id"] == session)
    assert mine["requests"] == 2, "a reused session id should not open a second session"
    assert mine["completion_tokens"] > 0


def test_recent_history_records_completed_work(client: TestClient, chat_model: str) -> None:
    client.post(
        "/v1/chat/completions",
        json={"model": chat_model, "messages": [{"role": "user", "content": "Hi"}]},
    )
    recent = client.get("/v1/status").json()["recent"]
    assert recent[0]["model"] == chat_model
    assert recent[0]["api"] == "openai"
    assert recent[0]["completion_tokens"] > 0


def test_the_whole_status_payload_survives_json(client: TestClient) -> None:
    """The console streams this frame by frame for as long as the page is open, so a
    single unserializable value - an unquoted YAML date, say - takes the page down
    mid-session rather than failing at startup."""
    from hipep_server.api.common import ndjson, sse

    payload = client.get("/v1/status").json()
    assert json.loads(sse(payload).removeprefix("data: ").strip()) == payload
    assert json.loads(ndjson(payload)) == payload


def test_the_stream_formatters_do_not_choke_on_yaml_dates() -> None:
    """`yaml.safe_load` turns an unquoted date into a `datetime.date`, which json has
    no encoder for."""
    from datetime import date

    from hipep_server.api.common import ndjson, sse

    assert "2026-08-27" in sse({"run_date": date(2026, 8, 27)})
    assert "2026-08-27" in ndjson({"run_date": date(2026, 8, 27)})


def test_hardware_is_redetected_on_request(client: TestClient) -> None:
    body = client.get("/v1/hardware").json()
    assert body["hostname"]
    assert "gpus" in body
    assert isinstance(body["has_amd_gpu"], bool)


def test_console_is_served_at_the_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
