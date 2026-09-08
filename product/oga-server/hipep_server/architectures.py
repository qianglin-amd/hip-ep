# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""What is known about porting each model architecture.

The knowledge base answers the question someone asks before they spend an hour and
thirty gigabytes: will this model port, which recipe should it use, and is there
anything about it that is known to go wrong. A port that fails after the download is a
much worse answer than a refusal before it.

The entries hold only what cannot be read from somewhere else. Which recipe suits an
architecture is written in the recipe's own `applies_to` block and is joined in here, so
that adding a recipe for a new architecture needs no second edit -- and so the two
cannot disagree.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .recipes import list_recipes, recipe_architectures, recipe_plan

# Where a model goes when nothing more specific can be built for it.
GENERIC_RECIPE = "int4-rtn-g128"

KB_PATH = (
    Path(__file__).resolve().parents[2] / "toolkit" / "hipep_toolkit" / "architectures.yaml"
)

# Block sizes the ONNX model builder accepts, largest first.
BLOCK_SIZES = (256, 128, 64, 32, 16)


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, Any]:
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def rules(kb_path: Path | None = None) -> list[dict[str, Any]]:
    """The architecture-independent findings, each of which cost a broken artifact."""
    return list(_load(str(kb_path or KB_PATH)).get("rules") or [])


def architectures(
    recipes_dir: Path | None = None, kb_path: Path | None = None
) -> list[dict[str, Any]]:
    """Every known architecture, with the recipes that name it joined in."""
    entries = [dict(e) for e in _load(str(kb_path or KB_PATH)).get("architectures") or []]
    by_id = {e["id"]: e for e in entries}

    for recipe in list_recipes(recipes_dir) if recipes_dir else []:
        # Recipes name their audience in prose as often as by class -- "MoE causal
        # decoders" alongside `Qwen2ForCausalLM` -- and only a class name belongs in a
        # list of architectures.
        for name in recipe_architectures(recipe):
            entry = by_id.get(name)
            if entry is None:
                # A recipe naming an architecture the knowledge base has not heard of is
                # still the best answer available for it, so it earns an entry.
                entry = by_id[name] = {"id": name, "kind": "unknown"}
                entries.append(entry)
            entry.setdefault("recipes", []).append(recipe["id"])
            if recipe.get("buildable") and not entry.get("recipe"):
                entry["recipe"] = recipe["id"]

    for entry in entries:
        if entry.get("recipe"):
            continue
        # Every recipe written for this architecture needs a calibration pass the
        # pipeline cannot run. The generic one is what a port would actually use, and
        # saying so is more use than leaving the field empty.
        entry["recipe"] = GENERIC_RECIPE
        if entry.get("recipes"):
            entry["recipe_note"] = (
                f"the certified artifacts for this architecture used "
                f"{', '.join(entry['recipes'])}, which need a calibration pass this "
                f"pipeline does not run; a port here uses {GENERIC_RECIPE}"
            )

    return sorted(entries, key=lambda e: e["id"])


def find(name: str, recipes_dir: Path | None = None, kb_path: Path | None = None):
    """The entry for one architecture, or None."""
    return next(
        (e for e in architectures(recipes_dir, kb_path) if e["id"] == name),
        None,
    )


def fit_block_size(block: int, hidden: int | None) -> int:
    """The largest supported block at most `block` that divides `hidden`.

    Kept in step with the export stage, which applies the same rule. A block that does
    not divide the row it splits leaves a short final group whose weights come back
    wrong, and the model still loads and runs at full speed.
    """
    if not hidden:
        return block
    return next(
        (size for size in BLOCK_SIZES if size <= block and hidden % size == 0),
        min(BLOCK_SIZES),
    )


def assess(
    config: dict[str, Any],
    recipes_dir: Path | None = None,
    kb_path: Path | None = None,
) -> dict[str, Any]:
    """What porting this model would do, read from its Hugging Face config alone.

    `config` is the model's `config.json`. Nothing else is needed, which is the point:
    it is a few kilobytes, so this answers before any weights are fetched.
    """
    architecture = (config.get("architectures") or ["unknown"])[0]
    hidden = config.get("hidden_size")
    entry = find(architecture, recipes_dir, kb_path)

    verdict: dict[str, Any] = {
        "architecture": architecture,
        "hidden_size": hidden,
        "num_hidden_layers": config.get("num_hidden_layers"),
        "vocab_size": config.get("vocab_size"),
        "context": config.get("max_position_embeddings"),
        "known": entry is not None,
        "kind": (entry or {}).get("kind", "unknown"),
        "recipe": (entry or {}).get("recipe") or GENERIC_RECIPE,
        "notes": [],
        "warnings": [],
    }
    if entry and entry.get("verified"):
        verdict["verified"] = entry["verified"]
    for key in ("note", "recipe_note"):
        if entry and entry.get(key):
            verdict["notes"].append(entry[key])

    if not entry:
        verdict["warnings"].append(
            f"{architecture} is not in the knowledge base. The generic INT4 "
            "round-to-nearest recipe is the starting point, and the port will report "
            "what it finds."
        )

    requested = _block_of(verdict["recipe"], recipes_dir)
    effective = fit_block_size(requested, hidden)
    verdict["block_size"] = effective
    if effective != requested:
        verdict["warnings"].append(
            f"a hidden size of {hidden} is not a multiple of {requested}, so the port "
            f"will quantize in blocks of {effective} instead. A block that does not "
            "divide the row it splits corrupts the weights."
        )
    return verdict


def _block_of(recipe_id: str, recipes_dir: Path | None) -> int:
    """The block size a recipe asks for, defaulting to the generic 128."""
    if recipes_dir:
        for recipe in list_recipes(recipes_dir):
            if recipe["id"] == recipe_id and recipe.get("buildable"):
                return int(recipe_plan(recipe).get("block_size") or 128)
    return 128
