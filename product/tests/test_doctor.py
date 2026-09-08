# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""`hipep doctor`, the preflight a user runs before asking for help.

Its whole value is that it works on a broken install, so the tests care less about the
verdicts than about two properties: it never raises, and it never fails a machine that
can still serve. A preflight that exits non-zero on a GPU-free CI runner would make the
signal useless, and one that raises while diagnosing has nothing to report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
from hipep_toolkit import cli, commands
from hipep_toolkit import doctor as doc

PRODUCT_DIR = Path(__file__).resolve().parent.parent

STATES = {doc.OK, doc.WARN, doc.FAIL}


def _args(**overrides) -> argparse.Namespace:
    defaults = {"url": None, "json": False, "no_server": True}
    return argparse.Namespace(**{**defaults, **overrides})


# --- shape --------------------------------------------------------------------


def test_every_check_reports_a_known_state() -> None:
    report = doc.run_checks(include_server=False)
    assert report.checks
    for check in report.checks:
        assert check.state in STATES, check
        assert check.name and check.detail, check


def test_a_failing_check_carries_a_remedy() -> None:
    """A verdict a user cannot act on is a verdict that generates a support ticket."""
    report = doc.run_checks(include_server=False)
    for check in report.failures:
        assert check.remedy, f"{check.name} failed without telling the user what to do"


def test_python_and_catalog_pass_in_a_working_checkout() -> None:
    report = doc.run_checks(include_server=False)
    by_name = {c.name: c for c in report.checks}
    assert by_name["python"].state == doc.OK
    assert by_name["catalog"].state == doc.OK
    assert "certified models" in by_name["catalog"].detail


def test_report_is_json_serializable() -> None:
    import json

    payload = doc.run_checks(include_server=False).to_dict()
    assert json.loads(json.dumps(payload))["checks"]


def test_ok_ignores_warnings_and_respects_failures() -> None:
    report = doc.Report()
    report.add("a", doc.OK, "fine")
    report.add("b", doc.WARN, "no GPU")
    assert report.ok is True
    assert len(report.warnings) == 1
    report.add("c", doc.FAIL, "broken", "fix it")
    assert report.ok is False


# --- individual checks --------------------------------------------------------


def test_python_version_check_reads_the_running_interpreter() -> None:
    report = doc.Report()
    doc._check_python(report)
    assert report.checks[0].state == doc.OK
    assert sys.executable in report.checks[0].detail


def test_a_too_old_interpreter_fails(monkeypatch) -> None:
    monkeypatch.setattr(doc, "MIN_PYTHON", (99, 0))
    report = doc.Report()
    doc._check_python(report)
    check = report.checks[0]
    assert check.state == doc.FAIL
    assert "99.0" in check.remedy


def test_missing_dependencies_fail_with_the_pip_command(monkeypatch) -> None:
    monkeypatch.setattr(doc, "_installed", lambda module: False)
    report = doc.Report()
    doc._check_packages(report)
    deps = report.checks[0]
    assert deps.state == doc.FAIL
    assert "fastapi" in deps.detail
    assert "pip install" in deps.remedy


@pytest.fixture
def exporter(monkeypatch):
    """Control what the porting check finds, without needing a real toolchain."""

    def found(interpreter: str | None):
        from hipep_toolkit import stages

        monkeypatch.setattr(stages, "toolchain_python", lambda: interpreter)

    return found


def test_a_missing_porting_toolchain_warns_rather_than_fails(exporter) -> None:
    """Serving certified models needs no exporter, so this must not block anything."""
    exporter(None)
    report = doc.Report()
    doc._check_quantization(report)
    assert report.checks[0].state == doc.WARN
    assert "simulation" in report.checks[0].detail


def test_an_exporter_in_another_environment_still_counts(exporter) -> None:
    """The check has to follow the pipeline, not this interpreter.

    The exporter deliberately lives in its own environment, so asking whether *this*
    one can import Quark reports simulation on a machine that ports perfectly well --
    telling the reader to go and fix something that is already working.
    """
    exporter(r"C:\Users\a\.hipep\toolchain\Scripts\python.exe")
    report = doc.Report()
    doc._check_quantization(report)
    check = report.checks[0]

    assert check.state == doc.OK
    assert "toolchain" in check.detail
    assert not check.remedy


def test_an_interpreter_too_new_for_quark_says_so(exporter, monkeypatch) -> None:
    """Otherwise the remedy sends the reader round a loop that cannot terminate.

    No AMD Quark release supports 3.14, and `pip install amd-quark` there resolves to
    an unrelated 0.1.0 placeholder that ships no modules: the install reports success,
    this check still finds nothing, and the advice was to run the install again.
    """
    exporter(None)
    monkeypatch.setattr(doc.sys, "version_info", (3, 14, 7))
    monkeypatch.setattr(doc.platform, "python_version", lambda: "3.14.7")

    report = doc.Report()
    doc._check_quantization(report)
    check = report.checks[0]

    assert check.state == doc.WARN, "serving does not need Quark, so this cannot fail"
    assert "3.14.7" in check.detail
    assert "pip install" not in check.remedy, "the install being suggested is the no-op"
    assert "install-port-toolchain" in check.remedy


def test_the_remedy_names_a_script_that_exists(exporter, monkeypatch) -> None:
    """A remedy pointing at a missing file is worse than no remedy."""
    exporter(None)
    monkeypatch.setattr(doc.sys, "version_info", (3, 14, 7))

    report = doc.Report()
    doc._check_quantization(report)
    named = report.checks[0].remedy.split()[1]

    assert (PRODUCT_DIR.parent / named).exists(), f"{named} is not in the repository"


def test_a_supported_interpreter_still_suggests_the_install(exporter, monkeypatch) -> None:
    exporter(None)
    monkeypatch.setattr(doc.sys, "version_info", (3, 12, 10))

    report = doc.Report()
    doc._check_quantization(report)
    assert "pip install" in report.checks[0].remedy


def test_the_quant_extra_cannot_resolve_to_the_placeholder() -> None:
    """`amd-quark` 0.1.0 is a different project that installs nothing at all."""
    import tomllib

    text = (PRODUCT_DIR / "pyproject.toml").read_bytes()
    quant = tomllib.loads(text.decode())["project"]["optional-dependencies"]["quant"]
    pin = next(spec for spec in quant if spec.startswith("amd-quark"))
    assert pin != "amd-quark", "an unpinned amd-quark can resolve to the empty 0.1.0"
    assert ">=" in pin


def test_store_check_follows_hipep_home(monkeypatch, tmp_path) -> None:
    home = tmp_path / "store"
    monkeypatch.setenv("HIPEP_HOME", str(home))
    report = doc.Report()
    doc._check_store(report)
    check = report.checks[0]
    assert check.state in {doc.OK, doc.WARN}
    assert "GB free" in check.detail
    assert home.exists(), "the check should create the store it was pointed at"


def test_an_unwritable_store_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HIPEP_HOME", str(tmp_path / "store"))

    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("pathlib.Path.mkdir", refuse)
    report = doc.Report()
    doc._check_store(report)
    check = report.checks[0]
    assert check.state == doc.FAIL
    assert "HIPEP_HOME" in check.remedy


def test_requesting_the_oga_engine_without_it_installed_is_a_failure(monkeypatch) -> None:
    """`HIPEP_ENGINE=oga` is an explicit demand for GPU inference.

    Silently replaying published numbers instead is the one outcome that could be
    mistaken for a successful run, so this is the only engine verdict that fails.
    """
    monkeypatch.setenv("HIPEP_ENGINE", "oga")
    monkeypatch.setattr(
        "hipep_server.engines.oga.oga_available", lambda: (False, "not installed")
    )
    report = doc.Report()
    doc._check_engine(report)
    assert report.checks[0].state == doc.FAIL

    monkeypatch.delenv("HIPEP_ENGINE")
    report = doc.Report()
    doc._check_engine(report)
    assert report.checks[0].state == doc.WARN


def test_gpu_detection_failure_never_raises(monkeypatch) -> None:
    monkeypatch.setattr("hipep_server.hardware.detect_gpus", lambda: [])
    report = doc.Report()
    doc._check_gpu(report)
    assert report.checks[0].state == doc.WARN
    assert "replay" in report.checks[0].remedy


# --- the server check ---------------------------------------------------------


def test_no_server_running_is_not_a_failure() -> None:
    """Running the preflight before starting anything is the normal case."""
    report = doc.Report()
    doc._check_server(report, "http://127.0.0.1:1")
    check = report.checks[0]
    assert check.state in {doc.OK, doc.WARN}
    assert check.state != doc.FAIL


def test_a_running_server_is_reported_with_its_engine(live_server) -> None:
    url, _client = live_server("doctor")
    report = doc.Report()
    doc._check_server(report, url)
    check = report.checks[0]
    assert check.state == doc.OK
    assert "replay" in check.detail


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:11434", ("127.0.0.1", 11434)),
        ("127.0.0.1:8080", ("127.0.0.1", 8080)),
        ("https://gpu-host", ("gpu-host", 443)),
        ("http://gpu-host", ("gpu-host", 80)),
    ],
)
def test_host_and_port_are_split_for_the_port_probe(url: str, expected) -> None:
    assert doc._split_hostport(url) == expected


# --- the command --------------------------------------------------------------


def test_command_exits_zero_when_only_warnings(monkeypatch, capsys) -> None:
    report = doc.Report()
    report.add("gpu", doc.WARN, "no AMD GPU detected", "install the driver")
    monkeypatch.setattr(doc, "run_checks", lambda *a, **k: report)
    assert commands.doctor(_args()) == 0
    out = capsys.readouterr().out
    assert "warning" in out
    assert "install the driver" in out


def test_command_exits_non_zero_on_a_failure(monkeypatch, capsys) -> None:
    report = doc.Report()
    report.add("server package", doc.FAIL, "not importable", "pip install -e product")
    monkeypatch.setattr(doc, "run_checks", lambda *a, **k: report)
    assert commands.doctor(_args()) == 1
    assert "failed" in capsys.readouterr().out


def test_json_output_is_machine_readable(monkeypatch, capsys) -> None:
    import json

    report = doc.Report()
    report.add("python", doc.OK, "3.12.0")
    monkeypatch.setattr(doc, "run_checks", lambda *a, **k: report)
    commands.doctor(_args(json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": True,
        "checks": [{"name": "python", "state": "ok", "detail": "3.12.0", "remedy": ""}],
    }


def test_no_server_flag_skips_the_server_check(monkeypatch) -> None:
    seen = {}

    def record(url=None, *, include_server=True):
        seen["include_server"] = include_server
        return doc.Report()

    monkeypatch.setattr(doc, "run_checks", record)
    commands.doctor(_args(no_server=True))
    assert seen["include_server"] is False
    commands.doctor(_args(no_server=False))
    assert seen["include_server"] is True


def test_doctor_is_reachable_from_the_parser() -> None:
    args = cli.build_parser().parse_args(["doctor", "--json", "--no-server"])
    assert args.handler is commands.doctor
    assert args.json and args.no_server


def test_platform_summary_names_the_operating_system() -> None:
    summary = doc.platform_summary()
    assert summary
    assert any(token in summary for token in ("Windows", "Linux", "Darwin", "Ubuntu", "/"))
