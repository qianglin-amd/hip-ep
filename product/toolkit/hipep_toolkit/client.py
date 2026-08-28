# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""HTTP client for oga-server, including the two streaming formats it speaks.

Server-sent events and newline-delimited JSON are both line protocols, so both are
consumed by the same iterator with a different frame parser. Everything here is
synchronous: a CLI has one request in flight at a time, and an event loop would buy
nothing but a harder traceback.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover - the message is the point
    raise SystemExit(
        "httpx is required by the hipep CLI.\nRun: pip install -e product  (or: pip install httpx)"
    ) from None

DEFAULT_URL = "http://127.0.0.1:11434"

# Generation has no useful timeout: a long prompt on a large model on a laptop GPU can
# legitimately take minutes, and a client that gives up mid-generation looks like a
# server fault. Connection setup is bounded, the read is not.
STREAM_TIMEOUT = httpx.Timeout(10.0, read=None)


class ServerUnreachable(RuntimeError):
    """The server is not answering on the address the CLI was pointed at."""


class ServerError(RuntimeError):
    """The server answered with an error. Carries the server's own message."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def base_url(explicit: str | None = None) -> str:
    url = explicit or os.environ.get("HIPEP_URL") or DEFAULT_URL
    if "://" not in url:
        url = "http://" + url
    return url.rstrip("/")


def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
        if payload.get("detail"):
            return str(payload["detail"])
    return f"HTTP {response.status_code}"


class Client:
    """Thin wrapper: URL joining, error translation and line streaming."""

    def __init__(self, url: str | None = None, *, session: str | None = None):
        self.url = base_url(url)
        headers = {"User-Agent": "hipep-cli"}
        if session:
            headers["X-HIPEP-Session"] = session
        self._http = httpx.Client(base_url=self.url, headers=headers, timeout=30.0)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # --- plain requests -------------------------------------------------------

    def get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params={k: v for k, v in params.items() if v})

    def post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        return self._request("POST", path, json=payload or {})

    def delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise ServerUnreachable(
                f"no server at {self.url}. Start one with 'hipep serve', "
                "or pass --url if it is listening elsewhere."
            ) from exc
        except httpx.HTTPError as exc:
            raise ServerUnreachable(f"{self.url}: {exc}") from exc
        if response.status_code >= 400:
            raise ServerError(response.status_code, _detail(response))
        if not response.content:
            return None
        return response.json()

    # --- streaming ------------------------------------------------------------

    def stream_lines(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Iterator[str]:
        try:
            with self._http.stream(
                method, path, json=payload, timeout=STREAM_TIMEOUT
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise ServerError(response.status_code, _detail(response))
                yield from response.iter_lines()
        except httpx.ConnectError as exc:
            raise ServerUnreachable(
                f"no server at {self.url}. Start one with 'hipep serve'."
            ) from exc

    def stream_events(
        self, path: str, payload: dict[str, Any] | None = None, *, method: str = "GET"
    ) -> Iterator[Any]:
        """Server-sent events, yielding each decoded ``data:`` frame.

        A frame of ``[DONE]`` ends the stream. Undecodable frames are skipped rather
        than raised: losing one status snapshot is not worth aborting a job the user
        is watching.
        """
        for line in self.stream_lines(method, path, payload):
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body or body == '"[DONE]"' or body == "[DONE]":
                return
            try:
                yield json.loads(body)
            except json.JSONDecodeError:
                continue

    def stream_ndjson(self, path: str, payload: dict[str, Any] | None = None) -> Iterator[Any]:
        for line in self.stream_lines("POST", path, payload):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    # --- convenience ----------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self.get("/health")

    def status(self) -> dict[str, Any]:
        return self.get("/v1/status")

    def catalog(self, *, tier: str | None = None, q: str | None = None) -> dict[str, Any]:
        return self.get("/v1/catalog", tier=tier, q=q)

    def resolve_model(self, model: str | None) -> dict[str, Any]:
        """Resolve a name against the catalog, or pick a sensible default.

        With no name, prefer something already installed: a first-time user running
        `hipep run` should get the model they just pulled, not a 40 GB download.
        """
        data = self.catalog()
        models = data.get("models") or []
        if not models:
            raise ServerError(404, "the catalog is empty")
        if model:
            for entry in models:
                if entry["id"] == model:
                    return entry
            lowered = model.lower()
            partial = [e for e in models if lowered in e["id"].lower()]
            if len(partial) == 1:
                return partial[0]
            if partial:
                names = ", ".join(e["id"] for e in partial[:6])
                raise ServerError(400, f"'{model}' matches several models: {names}")
            raise ServerError(404, f"no model matching '{model}'. Try 'hipep models'.")
        generative = [e for e in models if e["tier"] in {"llm", "vlm"}]
        installed = [e for e in generative if e.get("installed")]
        pool = installed or [e for e in generative if e.get("fits")] or generative
        return pool[0]
