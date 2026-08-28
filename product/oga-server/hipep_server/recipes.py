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
