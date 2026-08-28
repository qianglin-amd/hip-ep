# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Job pipeline: pull and port.

Porting a model is the longest operation the product performs, so the contract that
matters is not just "it finished" but that every stage reported, the quantization plan
the user asked for was the one carried through, and the result is immediately usable
without a restart.

These run against a replay-engine server, where the export toolchain is absent and the
stages are simulated. Simulation being *labelled* is itself part of the contract.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

PORT_STAGES = [
    "fetch",
    "inspect",
    "export",
    "quantize",
    "convert",
    "validate",
    "benchmark",
    "publish",
]


def test_pull_runs_every_stage_and_installs(
    client: TestClient, chat_model: str, finished
) -> None:
    job = client.post("/v1/pull", json={"model": chat_model}).json()["job"]
    assert job["kind"] == "pull"

    snapshot = finished(client, job["id"])
    assert snapshot["state"] == "succeeded", snapshot.get("error")
    assert [s["name"] for s in snapshot["stages"]] == [
        "resolve",
        "download",
        "verify",
        "register",
    ]
    assert all(s["state"] == "succeeded" for s in snapshot["stages"])

    installed = {m["id"] for m in client.get("/v1/local").json()["models"]}
    assert chat_model in installed


def test_pull_reports_transfer_progress(client: TestClient, chat_model: str, finished) -> None:
    job = client.post("/v1/pull", json={"model": chat_model}).json()["job"]
    result = finished(client, job["id"])["result"]
    assert result["total_bytes"] > 0
    assert result["completed_bytes"] == result["total_bytes"]


def test_a_simulated_pull_says_so(client: TestClient, chat_model: str, finished) -> None:
    """Silent simulation would let someone believe they have weights they do not
    have."""
    job = client.post("/v1/pull", json={"model": chat_model}).json()["job"]
    snapshot = finished(client, job["id"])
    assert snapshot["result"]["simulated"] is True
    assert any("SIMULATED" in line["text"] for line in snapshot["logs"])
    assert any(line["level"] == "warning" for line in snapshot["logs"])


def test_pull_of_an_unknown_model_is_404(client: TestClient) -> None:
    assert client.post("/v1/pull", json={"model": "nope:1b"}).status_code == 404


def test_port_requires_a_source(client: TestClient) -> None:
    response = client.post("/v1/port", json={})
    assert response.status_code == 400
    assert "source is required" in response.json()["detail"]


def test_port_runs_the_full_stage_sequence(client: TestClient, finished) -> None:
    job = client.post(
        "/v1/port", json={"source": "Qwen/Qwen2.5-7B-Instruct", "name": "qwen2.5:7b-mine"}
    ).json()["job"]

    snapshot = finished(client, job["id"], timeout=60.0)
    assert snapshot["state"] == "succeeded", snapshot.get("error")
    assert [s["name"] for s in snapshot["stages"]] == PORT_STAGES
    assert all(s["state"] == "succeeded" for s in snapshot["stages"])


def test_port_applies_the_quark_defaults(client: TestClient, finished) -> None:
    """chunk_size 512 and logit pruning are the defaults the quantization guidance
    calls for, and a caller who omits them should still get them."""
    job = client.post("/v1/port", json={"source": "org/model", "name": "d:1b"}).json()["job"]
    snapshot = finished(client, job["id"], timeout=60.0)

    params = snapshot["params"]
    assert params["chunk_size"] == 512
    assert params["prune_logit"] is True
    assert params["quantization"] == "int4-rtn-g128"

    plan = next(line["text"] for line in snapshot["logs"] if line["text"].startswith("plan:"))
    assert "chunk_size=512" in plan
    assert "prune_logit=True" in plan


def test_port_honours_an_explicit_quantization_plan(client: TestClient, finished) -> None:
    job = client.post(
        "/v1/port",
        json={
            "source": "org/model",
            "name": "e:1b",
            "quantization": "int4-awq-g128",
            "chunk_size": 1024,
            "prune_logit": False,
            "calibration_samples": 64,
        },
    ).json()["job"]
    params = finished(client, job["id"], timeout=60.0)["params"]

    assert params["quantization"] == "int4-awq-g128"
    assert params["chunk_size"] == 1024
    assert params["prune_logit"] is False
    assert params["calibration_samples"] == 64


def test_a_ported_model_is_usable_without_a_restart(client: TestClient, finished) -> None:
    """The point of registering in-catalog at publish time: no bounce required."""
    name = "mymodel:7b"
    job = client.post("/v1/port", json={"source": "org/mymodel", "name": name}).json()["job"]
    assert finished(client, job["id"], timeout=60.0)["state"] == "succeeded"

    assert name in {m["id"] for m in client.get("/v1/models").json()["data"]}

    answer = client.post(
        "/v1/chat/completions",
        json={"model": name, "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert answer.status_code == 200
    assert answer.json()["choices"][0]["message"]["content"].strip()


def test_a_ported_model_keeps_its_quantization_metadata(client: TestClient, finished) -> None:
    job = client.post(
        "/v1/port",
        json={"source": "org/f", "name": "f:1b", "quantization": "int4-awq-g128"},
    ).json()["job"]
    finished(client, job["id"], timeout=60.0)

    entry = client.get("/v1/catalog/f:1b").json()
    assert entry["status"] == "imported"
    assert entry["installed"] is True
    assert "AWQ" in entry["quant_label"].upper() or "awq" in str(entry["quantization"])


def test_cancelling_a_running_job_stops_it(client: TestClient, finished) -> None:
    job = client.post("/v1/port", json={"source": "org/g", "name": "g:1b"}).json()["job"]
    assert client.post(f"/v1/jobs/{job['id']}/cancel").json()["cancelled"] == job["id"]

    snapshot = finished(client, job["id"], timeout=60.0)
    assert snapshot["state"] == "cancelled"
    assert "g:1b" not in {m["id"] for m in client.get("/v1/local").json()["models"]}


def test_cancelling_a_finished_job_is_a_conflict(
    client: TestClient, chat_model: str, finished
) -> None:
    job = client.post("/v1/pull", json={"model": chat_model}).json()["job"]
    finished(client, job["id"])
    assert client.post(f"/v1/jobs/{job['id']}/cancel").status_code == 409


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/v1/jobs/j-nope").status_code == 404
    assert client.post("/v1/jobs/j-nope/cancel").status_code == 404


def test_jobs_list_is_filterable_by_kind(
    client: TestClient, chat_model: str, finished
) -> None:
    pull = client.post("/v1/pull", json={"model": chat_model}).json()["job"]
    finished(client, pull["id"])
    port = client.post("/v1/port", json={"source": "org/h", "name": "h:1b"}).json()["job"]
    finished(client, port["id"], timeout=60.0)

    kinds = {j["kind"] for j in client.get("/v1/jobs").json()["jobs"]}
    assert kinds == {"pull", "port"}
    only_ports = client.get("/v1/jobs", params={"kind": "port"}).json()["jobs"]
    assert [j["id"] for j in only_ports] == [port["id"]]
