# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Tests for the real porting stages.

These cover the parts that do not need a toolchain or a GPU: the reading of a recipe
into builder arguments, and the check that decides whether a ported artifact actually
produces language.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from hipep_toolkit import stages

# What the ONNX model builder's output looks like: hundreds of redraws of one line.
PROGRESS_BAR = (
    "import sys\n"
    "for i in range(4000):\n"
    "    sys.stdout.write(f'\\rSaving weight {i} ' + 'x' * 40)\n"
    "sys.stdout.write('\\ndone\\n')\n"
)


class RecordingContext:
    """The parts of a job context the command runner touches."""

    def __init__(self):
        self.lines: list[str] = []

    def log(self, message, level="info"):
        self.lines.append(message)

    def check_cancelled(self):
        return None


class TestCommandOutput:
    """Streaming a toolchain command's output into the job log."""

    @pytest.mark.asyncio
    async def test_progress_bar_redraws_do_not_overrun_the_buffer(self):
        # The exporter draws progress with carriage returns and no newline. Reading that
        # by line makes the whole export one unterminated line, which past 64 KB failed
        # the job with 'Separator is not found, and chunk exceed the limit'.
        ctx = RecordingContext()
        await stages._run(ctx, [sys.executable, "-c", PROGRESS_BAR], "export")

        assert any(line.endswith("done") for line in ctx.lines)

    @pytest.mark.asyncio
    async def test_progress_bar_redraws_do_not_flood_the_log(self):
        # Treating each redraw as its own line does not crash, it buries: a job keeps a
        # bounded log, and a few hundred progress frames pushed out the warning saying
        # the requested block size had been changed.
        ctx = RecordingContext()
        await stages._run(ctx, [sys.executable, "-c", PROGRESS_BAR], "export")

        assert len(ctx.lines) < 20, f"{len(ctx.lines)} lines for one progress bar"

    @pytest.mark.asyncio
    async def test_a_line_is_logged_as_it_finally_read(self):
        # A carriage return rewrites the line, so a half-drawn frame is not what the
        # exporter said.
        script = "import sys\nsys.stdout.write('\\rfirst\\rsecond\\rfinal state\\n')\n"
        ctx = RecordingContext()
        await stages._run(ctx, [sys.executable, "-c", script], "export")

        assert ctx.lines == ["export: final state"]

    @pytest.mark.asyncio
    async def test_output_with_no_newline_at_all_is_still_logged(self):
        ctx = RecordingContext()
        await stages._run(
            ctx, [sys.executable, "-c", "import sys; sys.stdout.write('no newline')"], "x"
        )
        assert ctx.lines == ["x: no newline"]

    @pytest.mark.asyncio
    async def test_a_failing_command_is_reported_with_its_status(self):
        ctx = RecordingContext()
        with pytest.raises(RuntimeError, match="export exited with status 3"):
            await stages._run(ctx, [sys.executable, "-c", "raise SystemExit(3)"], "export")


class TestDegenerateOutput:
    """The gate that separates a working port from a numerically broken one."""

    def test_ordinary_prose_passes(self):
        text = " Paris, which has been the seat of French government for centuries."
        assert stages._degenerate(text) is None

    def test_a_short_answer_is_not_called_repetitive(self):
        assert stages._degenerate(" Paris.") is None

    def test_punctuation_only_is_rejected(self):
        # The first broken port emitted exactly this.
        complaint = stages._degenerate(" ......................")
        assert complaint and "no words" in complaint

    def test_an_empty_continuation_is_rejected(self):
        assert stages._degenerate("") is not None

    def test_one_word_repeated_to_the_limit_is_rejected(self):
        complaint = stages._degenerate(" is" * 20)
        assert complaint and "repeated" in complaint
        assert "'is'" in complaint

    def test_a_repetitive_but_real_sentence_survives(self):
        # Naming the same thing a few times is not the failure being looked for.
        text = " Paris. Paris is the capital, and Paris has been so since 508."
        assert stages._degenerate(text) is None

    def test_the_complaint_names_the_scale_of_the_repetition(self):
        complaint = stages._degenerate(" the" * 12)
        assert "1 distinct words across 12" in complaint

    def test_word_salad_is_rejected(self):
        # Recorded from a real port quantized in blocks that did not divide its rows.
        salad = (
            " PariswnsMetaInfoerianMetaInfo Ig__':estialrowaveearchers tearKeyNottysmill"
            "anoriouslyosi):**ipyicumountainspliance TType))[TSD scrat:')araderie"
        )
        complaint = stages._degenerate(salad)
        assert complaint and "word salad" in complaint

    def test_undecodable_bytes_are_rejected(self):
        complaint = stages._degenerate(" Paris is the \ufffd capital of France today")
        assert complaint and "undecodable" in complaint

    @pytest.mark.parametrize(
        "sample",
        [
            # Every one of these came out of a port that works.
            " Paris, which is located in the northern part of the country. The capital"
            " is known for its historical landmarks and cultural significance.",
            " Paris. The capital of the United States is Washington D. C. The capital"
            " of the United Kingdom is London.",
            " The sky is blue because of the scattering of sunlight by the atmosphere,"
            " with blue light being the predominant color of the visible spectrum.",
        ],
    )
    def test_real_output_from_working_ports_passes(self, sample):
        assert stages._degenerate(sample) is None


class TestBlockSizeFitting:
    """A quantization block that does not divide a row produces a broken model."""

    def test_a_block_that_divides_is_left_alone(self):
        # Qwen2.5-0.5B: 896 is a multiple of 128, and it ported correctly at g128.
        assert stages._fit_block_size(128, 896) == 128

    def test_a_block_that_does_not_divide_is_reduced(self):
        # SmolLM2-360M: 960 is not a multiple of 128, and g128 produced word salad.
        assert stages._fit_block_size(128, 960) == 64

    def test_the_largest_workable_block_is_chosen(self):
        # Reducing further than needed costs accuracy for nothing.
        assert stages._fit_block_size(256, 512) == 256
        assert stages._fit_block_size(256, 960) == 64

    def test_it_never_increases_the_requested_block(self):
        assert stages._fit_block_size(32, 4096) == 32

    def test_an_unknown_hidden_size_changes_nothing(self):
        # `inspect` could not read a config; guessing would be worse than proceeding.
        assert stages._fit_block_size(128, None) == 128


class TestSchemeLabelling:
    """A corrected build has to be labelled with what was built, not what was asked."""

    def test_the_group_size_in_the_id_is_rewritten(self):
        assert stages._scheme_with_block("int4-rtn-g128", 64) == "int4-rtn-g64"

    def test_an_id_with_no_group_size_gains_one(self):
        assert stages._scheme_with_block("int4-rtn", 64) == "int4-rtn-g64"

    def test_the_rest_of_the_id_is_preserved(self):
        assert stages._scheme_with_block("int4-rtn-g128-asym", 32) == "int4-rtn-g32-asym"


class TestQuantizationPlan:
    """Reading a recipe id into arguments the builder understands."""

    def test_the_default_recipe_is_read(self):
        plan = stages._quantization_plan({"quantization": "int4-rtn-g128"})
        assert plan == {
            "precision": "int4",
            "algo_config": "rtn",
            "block_size": 128,
            "is_symmetric": True,
        }

    def test_the_group_size_is_taken_from_the_id(self):
        plan = stages._quantization_plan({"quantization": "int4-rtn-g32"})
        assert plan["block_size"] == 32

    def test_a_group_size_the_builder_cannot_take_falls_back(self):
        plan = stages._quantization_plan({"quantization": "int4-rtn-g100"})
        assert plan["block_size"] == 128

    def test_asymmetric_quantization_is_recognised(self):
        plan = stages._quantization_plan({"quantization": "int4-rtn-g128-asym"})
        assert plan["is_symmetric"] is False

    def test_k_quant_is_spelled_the_way_the_builder_wants(self):
        plan = stages._quantization_plan({"quantization": "int4-kquant-g32"})
        assert plan["algo_config"] == "k_quant"

    def test_an_unsupported_algorithm_is_refused_by_name(self):
        # AWQ needs a calibration pass this stage does not run, and silently doing RTN
        # instead would mislabel the artifact.
        with pytest.raises(ValueError, match="(?i)awq"):
            stages._quantization_plan({"quantization": "int4-awq-g128"})


class TestToolchainDiscovery:
    """Finding the interpreter that can export a model."""

    def test_the_managed_environment_is_found_by_convention(self, tmp_path, monkeypatch):
        # Installing the toolchain has to be enough. Requiring an environment variable
        # as well means the install appears to do nothing until someone is told about a
        # variable the installer's output is the only place that mentions.
        monkeypatch.setenv("HIPEP_HOME", str(tmp_path))
        scripts = tmp_path / "toolchain" / "Scripts"
        scripts.mkdir(parents=True)
        (scripts / "python.exe").write_text("")

        assert stages.managed_toolchain() == scripts / "python.exe"

    def test_a_posix_layout_is_found_too(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HIPEP_HOME", str(tmp_path))
        binaries = tmp_path / "toolchain" / "bin"
        binaries.mkdir(parents=True)
        (binaries / "python3").write_text("")

        assert stages.managed_toolchain() == binaries / "python3"

    def test_no_managed_environment_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HIPEP_HOME", str(tmp_path))
        assert stages.managed_toolchain() is None

    def test_an_explicit_interpreter_wins(self, tmp_path, monkeypatch):
        chosen = tmp_path / "chosen-python"
        chosen.write_text("")
        monkeypatch.setenv("HIPEP_PORT_PYTHON", str(chosen))
        stages._probe.cache_clear()

        assert stages.toolchain_python() == str(chosen)


class TestGraphInspection:
    """Counting the quantized matmuls in the graph the builder produced."""

    def test_the_graph_is_read_on_the_toolchain_interpreter(self, tmp_path, monkeypatch):
        # `onnx` belongs to the porting toolchain, not the server: a clean server
        # install has no protobuf library, so importing it in-process turned every real
        # port into "No module named 'onnx'" at the quantize stage, after the source
        # weights had already been downloaded and the graph built.
        interpreter = tmp_path / "toolchain-python"
        interpreter.write_text("")
        monkeypatch.setattr(stages, "toolchain_python", lambda: str(interpreter))
        seen = {}

        def fake_run(command, **kwargs):
            seen["command"] = command
            return subprocess.CompletedProcess(command, 0, '{"MatMulNBits": 168}', "")

        monkeypatch.setattr(stages.subprocess, "run", fake_run)

        counts = stages._count_quantized_ops(tmp_path / "model.onnx")

        assert counts == {"MatMulNBits": 168}
        assert seen["command"][0] == str(interpreter)

    def test_the_weights_are_not_loaded_to_read_the_node_list(self):
        # On a large model the difference between reading the nodes and reading the
        # weights is megabytes against tens of gigabytes.
        assert "load_external_data=False" in stages._COUNT_SCRIPT

    def test_a_failure_reading_the_graph_is_reported(self, tmp_path, monkeypatch):
        interpreter = tmp_path / "toolchain-python"
        interpreter.write_text("")
        monkeypatch.setattr(stages, "toolchain_python", lambda: str(interpreter))
        monkeypatch.setattr(
            stages.subprocess,
            "run",
            lambda command, **kw: subprocess.CompletedProcess(command, 1, "", "boom"),
        )

        with pytest.raises(RuntimeError, match="boom"):
            stages._count_quantized_ops(tmp_path / "model.onnx")

    def test_no_toolchain_is_named_as_such(self, tmp_path, monkeypatch):
        monkeypatch.setattr(stages, "toolchain_python", lambda: None)

        with pytest.raises(stages.ToolchainMissing):
            stages._count_quantized_ops(tmp_path / "model.onnx")


class TestBuildTarget:
    """The builder target decides the numeric shape of the graph."""

    def test_the_target_is_not_cpu(self):
        # `int4` plus `cpu` is the one combination that yields an FP32 interface and
        # int8 compute, which the AMD GPU turns into punctuation.
        assert stages.BUILD_TARGET != "cpu"
