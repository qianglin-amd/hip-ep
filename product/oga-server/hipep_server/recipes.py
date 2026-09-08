# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Recipe discovery.

Recipes are the reproducible half of the model zoo: each certified model names the
recipe that produced it, and a user porting their own model picks one as a starting
point. They are plain YAML in the hipep_toolkit package so a user can copy one, edit
two fields and run it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_recipe(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("id", path.stem)
    data["path"] = str(path)

    # Whether this recipe can be carried out here, so a caller can say so before someone
    # picks it. Every recipe stays listed either way: they document how the certified
    # artifacts were produced, which is worth reading even where it cannot be rerun.
    try:
        recipe_plan(data)
    except RecipeError as exc:
        data["buildable"] = False
        data["unavailable_reason"] = str(exc)
    else:
        data["buildable"] = True
    return data


def list_recipes(recipes_dir: Path) -> list[dict[str, Any]]:
    if not recipes_dir.is_dir():
        return []
    recipes = []
    for path in sorted(recipes_dir.glob("*.yaml")):
        try:
            recipes.append(load_recipe(path))
        except (OSError, yaml.YAMLError):
            continue
    return recipes


# Quantization schemes the ONNX model builder can carry out on its own. AWQ and qMoE
# need a Quark calibration pass the port pipeline does not run, and quietly substituting
# round-to-nearest would hand back a model that is not the one the recipe describes.
BUILDABLE_SCHEMES = {"rtn", "k_quant", "kquant", "default", "none"}

# Recipes name the model they suit in prose as often as by class -- "MoE causal decoders"
# alongside `Qwen2ForCausalLM` -- and only a class name can be checked against a config.
_ARCHITECTURE_HINT = "ForCausalLM", "ForConditionalGeneration"


class RecipeError(ValueError):
    """A recipe cannot be carried out as written."""


def recipe_plan(recipe: dict[str, Any]) -> dict[str, Any]:
    """Turn a recipe's quantization block into port parameters.

    Without this a recipe is decorative: the port endpoint took the chosen id as a
    free-form scheme string and parsed what it could out of the name, so
    `phi-rtn-int4-block32` -- whose whole point is the block size -- was built at 128
    like everything else.
    """
    quant = recipe.get("quantization") or {}
    scheme = str(quant.get("scheme") or "rtn").lower()
    if scheme not in BUILDABLE_SCHEMES:
        raise RecipeError(
            f"{recipe.get('id', 'this recipe')} asks for {scheme.upper()}, which needs a "
            "calibration pass the port pipeline does not run yet. Use a round-to-nearest "
            "recipe, or pull the certified artifact this recipe produced."
        )

    weight = str(quant.get("weight_dtype") or "int4").lower()
    symmetry = quant.get("symmetry")
    plan: dict[str, Any] = {
        # uint4 is how a recipe spells asymmetric int4, so it decides the symmetry when
        # the recipe does not say outright.
        "precision": "int4" if weight == "uint4" else weight,
        "algorithm": "k_quant" if scheme in {"k_quant", "kquant"} else scheme,
        "symmetric": (
            str(symmetry).lower() != "asymmetric" if symmetry else weight != "uint4"
        ),
        "prune_logit": bool(quant.get("prune_logit", True)),
    }

    # Recipes spell the same thing both ways depending on which tool they came from.
    group = quant.get("group_size") or quant.get("block_size")
    if group:
        plan["block_size"] = int(group)
    if quant.get("chunk_size"):
        plan["chunk_size"] = int(quant["chunk_size"])
    return plan


def recipe_architectures(recipe: dict[str, Any]) -> list[str]:
    """The model classes a recipe names, ignoring the ones it describes in prose."""
    listed = (recipe.get("applies_to") or {}).get("architectures") or []
    return [a for a in listed if any(hint in a for hint in _ARCHITECTURE_HINT)]


def find_recipe(recipes_dir: Path, recipe_id: str) -> dict[str, Any] | None:
    candidate = recipes_dir / f"{recipe_id}"
    if not candidate.suffix:
        candidate = candidate.with_suffix(".yaml")
    if candidate.is_file():
        return load_recipe(candidate)
    for recipe in list_recipes(recipes_dir):
        if recipe.get("id") == recipe_id:
            return recipe
    return None
