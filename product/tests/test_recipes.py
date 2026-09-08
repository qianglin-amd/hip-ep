# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Recipes, and whether choosing one changes what gets built.

A recipe is the reproducible half of the model zoo: each certified model names the one
that produced it, and someone porting their own model starts from the closest match.
That only means something if the settings in it reach the builder. They did not -- the
port endpoint took the chosen recipe's id as a free-form scheme string and parsed what
it could out of the name, so `phi-rtn-int4-block32`, whose entire purpose is the block
size, was built at 128 like everything else.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from hipep_server.recipes import (
    RecipeError,
    list_recipes,
    recipe_architectures,
    recipe_plan,
)

RECIPES_DIR = Path(__file__).resolve().parents[1] / "toolkit" / "hipep_toolkit" / "recipes"


def recipe(name: str) -> dict:
    return yaml.safe_load((RECIPES_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


class TestShippedRecipes:
    """Read against the files that ship, not fixtures, so drift shows up here."""

    def test_the_collection_is_present(self):
        assert len(list_recipes(RECIPES_DIR)) >= 15

    def test_every_recipe_declares_a_scheme_and_dtype(self):
        for entry in list_recipes(RECIPES_DIR):
            quant = entry.get("quantization") or {}
            assert quant.get("scheme"), f"{entry['id']} names no scheme"
            assert quant.get("weight_dtype"), f"{entry['id']} names no weight dtype"

    def test_every_buildable_recipe_can_be_turned_into_a_plan(self):
        for entry in list_recipes(RECIPES_DIR):
            scheme = (entry.get("quantization") or {}).get("scheme", "")
            if scheme in {"awq", "qmoe"}:
                continue
            plan = recipe_plan(entry)
            assert plan["precision"], f"{entry['id']} produced no precision"

    def test_the_block_size_a_recipe_exists_for_survives(self):
        # This recipe drops the block size from 128 to 32 and says so in its name; if
        # that does not reach the builder the recipe does nothing at all.
        assert recipe_plan(recipe("phi-rtn-int4-block32"))["block_size"] == 32
        assert recipe_plan(recipe("qwen-rtn-int4"))["block_size"] == 128

    def test_k_quant_is_spelled_the_way_the_builder_wants(self):
        assert recipe_plan(recipe("qwen-vlm-kquant-int4"))["algorithm"] == "k_quant"

    def test_a_recipe_needing_calibration_is_refused_by_name(self):
        with pytest.raises(RecipeError, match="AWQ"):
            recipe_plan(recipe("llama-awq-int4-sym"))

    def test_the_refusal_says_what_to_do_instead(self):
        with pytest.raises(RecipeError) as caught:
            recipe_plan(recipe("mistral-awq-int4"))
        assert "round-to-nearest" in str(caught.value)


class TestUnbuildableRecipesStayVisible:
    """A recipe that cannot be run here still documents a certified artifact.

    Dropping them from the list would hide how half the zoo was produced; offering them
    as ordinary choices would give the user a dropdown entry that can only ever fail.
    """

    def test_they_are_still_listed(self, client: TestClient):
        listed = {r["id"] for r in client.get("/v1/recipes").json()["recipes"]}
        assert "llama-awq-int4-sym" in listed

    def test_they_are_marked_as_not_buildable(self, client: TestClient):
        entry = next(
            r
            for r in client.get("/v1/recipes").json()["recipes"]
            if r["id"] == "llama-awq-int4-sym"
        )
        assert entry["buildable"] is False
        assert "calibration" in entry["unavailable_reason"]

    def test_the_ones_that_work_are_marked_too(self, client: TestClient):
        entry = next(
            r
            for r in client.get("/v1/recipes").json()["recipes"]
            if r["id"] == "qwen-rtn-int4"
        )
        assert entry["buildable"] is True
        assert not entry.get("unavailable_reason")

    def test_most_of_the_collection_can_actually_be_run(self, client: TestClient):
        entries = client.get("/v1/recipes").json()["recipes"]
        buildable = [r for r in entries if r["buildable"]]
        assert len(buildable) >= len(entries) // 2


class TestSymmetry:
    """uint4 is how these recipes spell asymmetric."""

    def test_an_asymmetric_recipe_is_read_as_asymmetric(self):
        plan = recipe_plan(
            {"id": "x", "quantization": {"scheme": "rtn", "symmetry": "asymmetric"}}
        )
        assert plan["symmetric"] is False

    def test_uint4_implies_asymmetric_when_nothing_says_otherwise(self):
        plan = recipe_plan(
            {"id": "x", "quantization": {"scheme": "rtn", "weight_dtype": "uint4"}}
        )
        assert plan["symmetric"] is False
        assert plan["precision"] == "int4", "uint4 is int4 storage, not a fifth dtype"

    def test_symmetric_is_the_default(self):
        assert recipe_plan({"id": "x", "quantization": {"scheme": "rtn"}})["symmetric"]


class TestArchitectures:
    """Recipes name their audience in prose as often as by class."""

    def test_class_names_are_returned(self):
        assert recipe_architectures(recipe("qwen-rtn-int4")) == [
            "Qwen2ForCausalLM",
            "Qwen3ForCausalLM",
        ]

    def test_prose_is_not_mistaken_for_a_class_name(self):
        # "any causal decoder exportable to ONNX" cannot be compared to a config.
        assert recipe_architectures(recipe("int4-rtn-g128")) == []

    def test_conditional_generation_classes_count(self):
        assert "Gemma3ForConditionalGeneration" in recipe_architectures(
            recipe("gemma-vlm-rtn-int4")
        )


class TestPortEndpointAppliesTheRecipe:
    """The choice has to reach the job, or the dropdown is decoration."""

    def test_choosing_a_recipe_carries_its_block_size_into_the_job(self, client: TestClient):
        job = client.post(
            "/v1/port",
            json={"source": "org/model", "name": "x:1b", "recipe": "phi-rtn-int4-block32"},
        ).json()["job"]

        assert job["params"]["plan"]["block_size"] == 32
        assert job["params"]["recipe"] == "phi-rtn-int4-block32"

    def test_the_console_sending_the_id_as_the_scheme_still_works(self, client: TestClient):
        # The port view sends `quantization: <recipe id>`, so a recipe is identified by
        # whichever field names one.
        job = client.post(
            "/v1/port",
            json={
                "source": "org/model",
                "name": "y:1b",
                "quantization": "phi-rtn-int4-block32",
            },
        ).json()["job"]

        assert job["params"]["plan"]["block_size"] == 32

    def test_the_expected_architectures_travel_with_the_job(self, client: TestClient):
        job = client.post(
            "/v1/port",
            json={"source": "org/model", "name": "z:1b", "recipe": "qwen-rtn-int4"},
        ).json()["job"]

        assert "Qwen2ForCausalLM" in job["params"]["expects_architectures"]

    def test_a_recipe_that_cannot_be_built_is_refused_before_the_job_starts(
        self, client: TestClient
    ):
        reply = client.post(
            "/v1/port",
            json={"source": "org/model", "name": "a:1b", "recipe": "llama-awq-int4-sym"},
        )
        assert reply.status_code == 400
        assert "AWQ" in reply.json()["detail"]

    def test_an_unknown_recipe_is_a_404(self, client: TestClient):
        reply = client.post(
            "/v1/port",
            json={"source": "org/model", "name": "b:1b", "recipe": "no-such-recipe"},
        )
        assert reply.status_code == 404

    def test_a_bare_scheme_string_is_still_accepted(self, client: TestClient):
        # Not every caller picks from the collection.
        job = client.post(
            "/v1/port",
            json={"source": "org/model", "name": "c:1b", "quantization": "int4-rtn-g32"},
        ).json()["job"]

        assert job["params"]["quantization"] == "int4-rtn-g32"

    def test_an_explicit_override_beats_the_recipe(self, client: TestClient):
        job = client.post(
            "/v1/port",
            json={
                "source": "org/model",
                "name": "d:1b",
                "recipe": "qwen-rtn-int4",
                "prune_logit": False,
            },
        ).json()["job"]

        assert job["params"]["prune_logit"] is False
