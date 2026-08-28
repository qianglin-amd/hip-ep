# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The local model store.

Tracks what is actually on this machine, which is a different question from what is in
the certified catalog. A model can be certified but not downloaded, downloaded but not
certified (a user's own import), or downloaded and stale.

Layout under $HIPEP_HOME/models/:

    <model-id with ':' replaced by '--'>/
        genai_config.json        the OGA entry point
        *.onnx, *.onnx.data      weights
        hipep.json               our metadata sidecar
"""

from __future__ import annotations

import contextlib
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SIDECAR = "hipep.json"


def dir_name_for(model_id: str) -> str:
    """Filesystem-safe directory name for a model id.

    Ids contain a colon, which is legal on Linux but is an alternate-data-stream
    separator on Windows, so it is always replaced.
    """
    return model_id.replace(":", "--").replace("/", "__")


def _dir_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


@dataclass
class LocalModel:
    id: str
    path: str
    size_bytes: int
    installed_at: float
    origin: str = "certified"  # certified | imported | converted
    source: str | None = None
    recipe: str | None = None
    ready: bool = True
    note: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / (1024**3), 2)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["size_gb"] = self.size_gb
        return data


class ModelStore:
    def __init__(self, models_dir: Path):
        self.root = models_dir
        self.root.mkdir(parents=True, exist_ok=True)

    # --- paths ---------------------------------------------------------------

    def path_for(self, model_id: str) -> Path:
        return self.root / dir_name_for(model_id)

    def is_installed(self, model_id: str) -> bool:
        path = self.path_for(model_id)
        return (path / "genai_config.json").is_file() or (path / SIDECAR).is_file()

    # --- metadata ------------------------------------------------------------

    def read(self, model_id: str) -> LocalModel | None:
        path = self.path_for(model_id)
        if not path.is_dir():
            return None
        meta: dict[str, Any] = {}
        sidecar = path / SIDECAR
        if sidecar.is_file():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
        known = set(LocalModel.__dataclass_fields__)
        return LocalModel(
            id=meta.get("id", model_id),
            path=str(path),
            size_bytes=meta.get("size_bytes") or _dir_size(path),
            installed_at=meta.get("installed_at", path.stat().st_mtime),
            origin=meta.get("origin", "certified"),
            source=meta.get("source"),
            recipe=meta.get("recipe"),
            ready=(path / "genai_config.json").is_file(),
            note=meta.get("note"),
            extra={k: v for k, v in meta.items() if k not in known},
        )

    def write(self, model: LocalModel) -> LocalModel:
        path = Path(model.path)
        path.mkdir(parents=True, exist_ok=True)
        model.size_bytes = model.size_bytes or _dir_size(path)
        payload = model.to_dict()
        payload.pop("size_gb", None)
        (path / SIDECAR).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return model

    def register(
        self,
        model_id: str,
        *,
        origin: str = "certified",
        source: str | None = None,
        recipe: str | None = None,
        note: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LocalModel:
        path = self.path_for(model_id)
        path.mkdir(parents=True, exist_ok=True)
        model = LocalModel(
            id=model_id,
            path=str(path),
            size_bytes=_dir_size(path),
            installed_at=time.time(),
            origin=origin,
            source=source,
            recipe=recipe,
            ready=(path / "genai_config.json").is_file(),
            note=note,
            extra=extra or {},
        )
        return self.write(model)

    # --- collection ----------------------------------------------------------

    def list(self) -> list[LocalModel]:
        models: list[LocalModel] = []
        for child in sorted(self.root.iterdir()) if self.root.is_dir() else []:
            if not child.is_dir():
                continue
            sidecar = child / SIDECAR
            model_id = child.name.replace("--", ":").replace("__", "/")
            if sidecar.is_file():
                with contextlib.suppress(OSError, json.JSONDecodeError):
                    model_id = json.loads(sidecar.read_text(encoding="utf-8")).get(
                        "id", model_id
                    )
            local = self.read(model_id)
            if local:
                models.append(local)
        models.sort(key=lambda m: m.id)
        return models

    def remove(self, model_id: str) -> bool:
        path = self.path_for(model_id)
        if not path.is_dir():
            return False
        shutil.rmtree(path, ignore_errors=True)
        return True

    def total_bytes(self) -> int:
        return sum(m.size_bytes for m in self.list())
