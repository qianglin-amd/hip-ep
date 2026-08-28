# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""The `hipep` command line.

Two things here are easy to get subtly wrong and hard to notice: column alignment once
cells carry colour, and resolving a partial model name. A table that pads on `len()`
looks fine until the first coloured cell, and a name resolver that guesses wrongly
sends a user a 40 GB download they did not ask for.
"""

from __future__ import annotations

import re

import pytest
from hipep_toolkit import cli, term
from hipep_toolkit.client import Client, ServerError, ServerUnreachable, base_url

# --- formatting ---------------------------------------------------------------


def test_visible_width_ignores_colour() -> None:
    assert term.visible(term.paint("ok", "green")) == 2
    assert term.visible("ok") == 2


def test_table_columns_align_even_when_cells_are_coloured(monkeypatch) -> None:
    """Padding on len() rather than screen columns indents every coloured column by
    the width of its escape sequence."""
    monkeypatch.setattr(term, "COLOUR", True)
    rendered = term.table(
        ["MODEL", "STATE"],
        [["llama3.1:8b", term.paint("ok", "green")], ["qwen2.5:14b", "pending"]],
    )
    # Where the second column starts, measured in screen columns on each line.
    starts = {
        re.match(r"\S+\s+", term._ANSI.sub("", line)).end() for line in rendered.splitlines()
    }
    assert len(starts) == 1, f"second column starts at differing offsets: {starts}"


def test_table_does_not_truncate_ids() -> None:
    """Users copy model ids out of this output."""
    long_id = "some-org/a-very-long-model-name-that-would-be-tempting-to-shorten:70b"
    assert long_id in term.table(["MODEL"], [[long_id]])


def test_an_empty_table_renders_nothing() -> None:
    assert term.table(["MODEL"], []) == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "-"), (0, "0 B"), (512, "512 B"), (1536, "1.5 KB"), (5 * 1024**3, "5.0 GB")],
)
def test_byte_formatting(value, expected: str) -> None:
    assert term.bytes_human(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "-"), (1.5, "1.5s"), (75, "1m15s"), (3600 + 30 * 60, "1h30m")],
)
def test_duration_formatting(value, expected: str) -> None:
    assert term.duration(value) == expected


def test_progress_bar_is_clamped() -> None:
    assert term.bar(0.0, 10) == "[..........]"
    assert term.bar(1.0, 10) == "[##########]"
    assert term.bar(0.5, 10) == "[#####.....]"
    # A stage that reports 130% should not draw a longer bar than the terminal has.
    assert term.visible(term.bar(1.3, 10)) == 12
    assert term.visible(term.bar(-1.0, 10)) == 12


def test_output_stays_within_ascii() -> None:
    """A Windows console defaults to a code page that cannot encode a box-drawing
    character, and a UnicodeEncodeError while reporting progress is worse than a
    plain-looking table."""
    rendered = "\n".join(
        [
            term.table(["A", "B"], [["1", "2"]]),
            term.bar(0.5),
            term.state_label("succeeded"),
            term.keyvalues([("key", "value")]),
        ]
    )
    rendered.encode("cp1252")


# --- client -------------------------------------------------------------------


def test_base_url_accepts_a_bare_host_and_port(monkeypatch) -> None:
    monkeypatch.delenv("HIPEP_URL", raising=False)
    assert base_url("127.0.0.1:11500") == "http://127.0.0.1:11500"
    assert base_url("http://host:1/") == "http://host:1"


def test_base_url_falls_back_to_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("HIPEP_URL", "http://elsewhere:9999")
    assert base_url(None) == "http://elsewhere:9999"


def test_an_unreachable_server_says_how_to_start_one(monkeypatch) -> None:
    monkeypatch.delenv("HIPEP_URL", raising=False)
    with Client("http://127.0.0.1:1") as client, pytest.raises(ServerUnreachable) as exc:
        client.health()
    assert "hipep serve" in str(exc.value)


# --- model resolution ---------------------------------------------------------


@pytest.fixture
def cli_client(live_server) -> Client:
    url, _ = live_server("cli")
    with Client(url) as client:
        yield client


def test_health_and_status_round_trip(cli_client: Client) -> None:
    assert cli_client.health()["status"] == "ok"
    assert cli_client.status()["engine"]["name"] == "replay"


def test_an_exact_id_resolves(cli_client: Client) -> None:
    assert cli_client.resolve_model("llama3.1:8b")["id"] == "llama3.1:8b"


def test_an_unambiguous_prefix_resolves(cli_client: Client) -> None:
    assert cli_client.resolve_model("deepseek")["id"].startswith("deepseek")


def test_an_ambiguous_name_lists_the_candidates(cli_client: Client) -> None:
    """Guessing here would start a download the user did not ask for."""
    with pytest.raises(ServerError) as exc:
        cli_client.resolve_model("qwen")
    assert "matches several models" in str(exc.value)
    assert "qwen" in str(exc.value)


def test_an_unknown_name_points_at_the_model_list(cli_client: Client) -> None:
    with pytest.raises(ServerError) as exc:
        cli_client.resolve_model("not-a-model")
    assert "hipep models" in str(exc.value)


def test_no_name_picks_a_generative_model_that_fits(cli_client: Client) -> None:
    chosen = cli_client.resolve_model(None)
    assert chosen["tier"] in {"llm", "vlm"}


def test_no_name_prefers_something_already_installed(cli_client: Client) -> None:
    """A first-time user running `hipep run` should get the model they just pulled,
    not the first entry in the catalog."""
    job = cli_client.post("/v1/pull", {"model": "mistral:7b"})["job"]
    for _ in range(2000):
        if cli_client.get(f"/v1/jobs/{job['id']}")["state"] != "running":
            break
    assert cli_client.resolve_model(None)["id"] == "mistral:7b"


# --- argument parsing ---------------------------------------------------------


def test_every_documented_subcommand_parses() -> None:
    parser = cli.build_parser()
    for argv in (
        ["serve"],
        ["models", "--tier", "llm"],
        ["show", "llama3.1:8b"],
        ["ls"],
        ["rm", "llama3.1:8b", "-y"],
        ["recipes"],
        ["run", "llama3.1:8b", "hello"],
        ["pull", "llama3.1:8b"],
        ["port", "org/model", "--recipe", "int4-awq-g128"],
        ["bench", "llama3.1:8b", "--contexts", "128"],
        ["status"],
        ["ps"],
        ["jobs"],
        ["cancel", "j-1"],
    ):
        assert parser.parse_args(argv) is not None, argv


def test_quark_defaults_are_the_parser_defaults() -> None:
    args = cli.build_parser().parse_args(["port", "org/model"])
    assert args.chunk_size == 512
    assert args.prune_logit is True


def test_run_accepts_repeated_images() -> None:
    args = cli.build_parser().parse_args(
        ["run", "qwen3.6-vl:27b", "describe", "--image", "a.png", "--image", "b.png"]
    )
    assert args.image == ["a.png", "b.png"]
