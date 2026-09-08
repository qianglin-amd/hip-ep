# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The model architecture knowledge base, and the preflight built on it.

The question it answers is the one asked before an hour and thirty gigabytes are spent:
will this model port, which recipe suits it, and is there anything about it known to go
wrong. Getting that answer after the download is a much worse answer.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from hipep_server import architectures as kb

RECIPES_DIR = Path(__file__).resolve().parents[1] / "toolkit" / "hipep_toolkit" / "recipes"


def entries() -> list[dict]:
    return kb.architectures(RECIPES_DIR)


class TestTheKnowledgeBase:
    def test_the_certified_architectures_are_all_covered(self):
        known = {e["id"] for e in entries()}
        for architecture in (
            "Qwen2ForCausalLM",
            "LlamaForCausalLM",
            "MistralForCausalLM",
            "Phi3ForCausalLM",
            "Gemma3ForConditionalGeneration",
            "WhisperForConditionalGeneration",
        ):
            assert architecture in known, f"{architecture} is not described anywhere"

    def test_prose_is_not_listed_as_an_architecture(self):
        # Recipes name their audience in prose as often as by class -- "MoE causal
        # decoders", "BEV perception" -- and those are not architectures.
        for entry in entries():
            assert "ForCausalLM" in entry["id"] or "ForConditionalGeneration" in entry["id"]

    def test_every_architecture_has_a_recipe_that_can_be_built(self):
        # An entry whose only recipes need calibration used to come back with none at
        # all, which is the least useful possible answer.
        for entry in entries():
            assert entry.get("recipe"), f"{entry['id']} offers no recipe"

    def test_an_architecture_served_only_by_awq_explains_itself(self):
        llama = next(e for e in entries() if e["id"] == "LlamaForCausalLM")
        assert llama["recipe"] == kb.GENERIC_RECIPE
        assert "calibration" in llama["recipe_note"]

    def test_the_recipe_link_comes_from_the_recipe_not_a_copy(self):
        # Which recipe suits an architecture is written in the recipe's own applies_to
        # block. Storing it here as well would let the two disagree.
        qwen = next(e for e in entries() if e["id"] == "Qwen2ForCausalLM")
        assert qwen["recipe"] == "qwen-rtn-int4"
        assert "qwen-rtn-int4" in qwen["recipes"]

    def test_the_rules_that_cost_a_broken_artifact_are_recorded(self):
        ids = {r["id"] for r in kb.rules()}
        assert "block-size-divides-hidden-size" in ids
        assert "chat-markup-is-not-implied-by-architecture" in ids
        assert "builder-target-decides-numeric-shape" in ids

    def test_every_rule_says_how_it_was_established(self):
        for rule in kb.rules():
            assert rule.get("verified"), f"{rule['id']} claims no evidence"
            assert rule.get("detail"), f"{rule['id']} explains nothing"


class TestFitBlockSize:
    """Kept in step with the export stage, which applies the same rule."""

    def test_a_block_that_divides_is_kept(self):
        assert kb.fit_block_size(128, 896) == 128

    def test_a_block_that_does_not_divide_is_reduced(self):
        assert kb.fit_block_size(128, 960) == 64

    def test_it_agrees_with_the_export_stage(self):
        from hipep_toolkit import stages

        for hidden in (576, 896, 960, 2048, 3072, 4096, 5120):
            assert kb.fit_block_size(128, hidden) == stages._fit_block_size(128, hidden)


class TestAssess:
    def test_a_known_architecture_is_recognised(self):
        verdict = kb.assess(
            {"architectures": ["Qwen2ForCausalLM"], "hidden_size": 896}, RECIPES_DIR
        )
        assert verdict["known"] is True
        assert verdict["recipe"] == "qwen-rtn-int4"
        assert not verdict["warnings"]

    def test_a_hidden_size_that_forces_a_smaller_block_is_called_out(self):
        verdict = kb.assess(
            {"architectures": ["LlamaForCausalLM"], "hidden_size": 960}, RECIPES_DIR
        )
        assert verdict["block_size"] == 64
        assert any("960" in w for w in verdict["warnings"])

    def test_an_unknown_architecture_still_gets_a_starting_point(self):
        verdict = kb.assess(
            {"architectures": ["SomethingNewForCausalLM"], "hidden_size": 2048},
            RECIPES_DIR,
        )
        assert verdict["known"] is False
        assert verdict["recipe"] == kb.GENERIC_RECIPE
        assert verdict["warnings"], "an unknown architecture should say so"

    def test_what_was_actually_ported_here_is_reported_as_verified(self):
        verdict = kb.assess(
            {"architectures": ["Qwen2ForCausalLM"], "hidden_size": 896}, RECIPES_DIR
        )
        assert verdict["verified"]["gpu"] == "gfx1151"

    def test_a_config_with_no_architectures_does_not_raise(self):
        assert kb.assess({}, RECIPES_DIR)["architecture"] == "unknown"


class TestEndpoints:
    def test_the_knowledge_base_is_served(self, client: TestClient):
        body = client.get("/v1/architectures").json()
        assert len(body["architectures"]) >= 10
        assert len(body["rules"]) >= 3

    def test_preflight_reads_a_local_model_without_downloading_anything(
        self, client: TestClient, tmp_path
    ):
        (tmp_path / "config.json").write_text(
            '{"architectures": ["LlamaForCausalLM"], "hidden_size": 960}'
        )
        verdict = client.post("/v1/port/preflight", json={"source": str(tmp_path)}).json()

        assert verdict["architecture"] == "LlamaForCausalLM"
        assert verdict["block_size"] == 64
        assert any("960" in w for w in verdict["warnings"])

    def test_preflight_needs_a_source(self, client: TestClient):
        assert client.post("/v1/port/preflight", json={}).status_code == 400

    def test_a_source_that_cannot_be_read_says_gated_as_well_as_missing(
        self, client: TestClient
    ):
        # Hugging Face answers 404 both for a repository that is absent and for one
        # behind an unaccepted licence, so the message must not claim it is absent.
        reply = client.post(
            "/v1/port/preflight", json={"source": "org/definitely-not-real-xyz"}
        )
        assert reply.status_code == 404
        assert "gated" in reply.json()["detail"]
