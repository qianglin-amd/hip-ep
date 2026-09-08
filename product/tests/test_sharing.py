# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Sharing a ported model between machines.

Porting a model costs half an hour of quantization, so the second machine that wants it
should copy the artifact rather than repeat the work. That path crosses a real network
boundary, so the peer here is a real uvicorn server on a real port: the import job
builds its own HTTP client and would not go through an in-process transport.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def peer(live_server) -> tuple[str, httpx.Client]:
    """A second server holding a model to share."""
    return live_server("peer")


@pytest.fixture
def shared(peer, finished) -> tuple[str, str]:
    """A model ported on the peer and ready to share. Returns (url, model id)."""
    url, peer_client = peer
    job = peer_client.post(
        "/v1/port",
        json={"source": "org/shareme", "name": "shareme:7b", "quantization": "int4-rtn-g32"},
    ).json()["job"]
    assert finished(peer_client, job["id"], timeout=60.0)["state"] == "succeeded"
    return url, "shareme:7b"


def test_manifest_describes_the_artifact(peer, shared) -> None:
    url, model_id = shared
    _, peer_client = peer

    manifest = peer_client.get(f"/v1/artifacts/{model_id}").json()
    assert manifest["model"] == model_id
    assert manifest["origin"] == "imported"
    assert manifest["files"], "an artifact with no files cannot be shared"
    assert manifest["total_bytes"] == sum(f["size_bytes"] for f in manifest["files"])
    # The catalog entry travels with the files so the importer can reproduce the card.
    assert manifest["catalog"]["quantization"]
    assert manifest["server"]["base_url"].endswith(url.rsplit(":", 1)[1])


def test_manifest_omits_the_install_sidecar(peer, shared) -> None:
    """The sidecar records how *this* machine installed the model; the importer writes
    its own, and copying it would misattribute the origin."""
    _, model_id = shared
    _, peer_client = peer
    paths = {f["path"] for f in peer_client.get(f"/v1/artifacts/{model_id}").json()["files"]}
    assert not any(p.endswith(".hipep.json") for p in paths)


def test_files_are_served_individually(peer, shared) -> None:
    _, model_id = shared
    _, peer_client = peer
    manifest = peer_client.get(f"/v1/artifacts/{model_id}").json()
    first = manifest["files"][0]

    response = peer_client.get(f"/v1/artifacts/{model_id}/files/{first['path']}")
    assert response.status_code == 200
    assert len(response.content) == first["size_bytes"]


@pytest.mark.parametrize(
    "attempt",
    ["../../../../Windows/win.ini", "../../etc/passwd", "..%2f..%2fsecret"],
)
def test_a_path_outside_the_model_directory_is_refused(peer, shared, attempt: str) -> None:
    """`relative` is client-controlled, so it must not be able to escape the model
    directory."""
    _, model_id = shared
    _, peer_client = peer
    response = peer_client.get(f"/v1/artifacts/{model_id}/files/{attempt}")
    assert response.status_code == 404


def test_sharing_a_model_that_is_not_installed_is_404(peer) -> None:
    _, peer_client = peer
    response = peer_client.get("/v1/artifacts/llama3.1:8b")
    assert response.status_code == 404
    assert "not installed" in response.json()["detail"]


def test_import_requires_a_url_and_a_model(client: TestClient) -> None:
    assert client.post("/v1/import", json={"model": "x:1b"}).status_code == 400
    assert client.post("/v1/import", json={"url": "http://x"}).status_code == 400


def test_import_installs_the_model_from_the_peer(
    client: TestClient, peer, shared, finished
) -> None:
    url, model_id = shared

    job = client.post("/v1/import", json={"url": url, "model": model_id}).json()["job"]
    snapshot = finished(client, job["id"], timeout=60.0)
    assert snapshot["state"] == "succeeded", snapshot.get("error")
    assert [s["name"] for s in snapshot["stages"]] == [
        "manifest",
        "transfer",
        "verify",
        "register",
    ]

    installed = {m["id"] for m in client.get("/v1/local").json()["models"]}
    assert model_id in installed


def test_an_imported_model_keeps_the_peers_metadata(
    client: TestClient, peer, shared, finished
) -> None:
    url, model_id = shared
    job = client.post("/v1/import", json={"url": url, "model": model_id}).json()["job"]
    finished(client, job["id"], timeout=60.0)

    entry = client.get(f"/v1/catalog/{model_id}").json()
    assert entry["status"] == "imported"
    assert entry["quantization"], "quantization schema did not survive the import"
    assert "g32" in f"{entry['quantization']} {entry['quant_label']}".lower()


def test_an_imported_model_can_be_chatted_with(
    client: TestClient, peer, shared, finished
) -> None:
    url, model_id = shared
    job = client.post("/v1/import", json={"url": url, "model": model_id}).json()["job"]
    finished(client, job["id"], timeout=60.0)

    response = client.post(
        "/v1/chat/completions",
        json={"model": model_id, "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"].strip()


def test_import_can_rename_on_arrival(client: TestClient, peer, shared, finished) -> None:
    """Two peers may both share a `mymodel:7b`; the importer needs to disambiguate."""
    url, model_id = shared
    job = client.post(
        "/v1/import", json={"url": url, "model": model_id, "name": "shareme:7b-from-peer"}
    ).json()["job"]
    assert finished(client, job["id"], timeout=60.0)["state"] == "succeeded"

    installed = {m["id"] for m in client.get("/v1/local").json()["models"]}
    assert installed == {"shareme:7b-from-peer"}


def test_importing_from_a_peer_that_lacks_the_model_fails_clearly(
    client: TestClient, peer, finished
) -> None:
    url, _ = peer
    job = client.post("/v1/import", json={"url": url, "model": "absent:1b"}).json()["job"]
    snapshot = finished(client, job["id"], timeout=60.0)

    assert snapshot["state"] == "failed"
    assert "not sharing" in snapshot["error"]
    # The message should tell the user how to find the right id.
    assert "/v1/local" in snapshot["error"]


def test_a_second_import_reuses_the_files_already_present(
    client: TestClient, peer, shared, finished
) -> None:
    """An interrupted transfer should resume at the file it stopped on, so a re-import
    of an intact artifact must not re-fetch anything."""
    url, model_id = shared
    first = client.post("/v1/import", json={"url": url, "model": model_id}).json()["job"]
    finished(client, first["id"], timeout=60.0)

    second = client.post("/v1/import", json={"url": url, "model": model_id}).json()["job"]
    snapshot = finished(client, second["id"], timeout=60.0)
    assert snapshot["state"] == "succeeded"
    assert any("already present, skipping" in line["text"] for line in snapshot["logs"])
