"""Fixed-width tables must not fuse columns when a name reaches the width (issue #819).

The run ledger and the summary tables used literals such as ``{name:<24}`` with
no separator, so a 24-character model name ran straight into the ``kind``
column and a longer one pushed the rest of its row out of line with the header.
Astrolabe has models of 26, 28 and 32 characters, so this is the normal case.
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from stel.cli import cli
from stel.progress import OutputLevel, _TerminalReporter
from stel.synth import generate_invoices
from stel.table_layout import column_width

# Straddle the old ledger width (24) and the old summary width (22): below,
# exactly at, one past, and far past each.
NAME_LENGTHS = [21, 22, 23, 24, 25, 40]


def _name(length: int) -> str:
    return "m" * length


def _ledger(names: list[str], *, kind: str = "extraction") -> list[str]:
    stream = io.StringIO()
    reporter = _TerminalReporter(stream, level=OutputLevel.NORMAL, bars=False)
    reporter.run_started(
        len(names), target="dev", warehouse="duckdb", project_total=len(names), model_names=names
    )
    for name in names[:-1]:
        reporter.model_finished(name, kind, 3, 0.04, None, failed=False)
    reporter.model_skipped(names[-1], "upstream failed")
    return stream.getvalue().splitlines()[1:]


def test_column_width_is_never_narrower_than_the_historical_width() -> None:
    assert column_width(["a"], minimum=24) == 24
    assert column_width([], minimum=24) == 24
    # Longest value plus one separating space once it outgrows the minimum.
    assert column_width(["x" * 23], minimum=24) == 24
    assert column_width(["x" * 24], minimum=24) == 25
    assert column_width(["x" * 40, "y"], minimum=24) == 41


@pytest.mark.parametrize("length", NAME_LENGTHS)
def test_live_ledger_separates_name_from_kind_and_aligns_rows(length: int) -> None:
    long_name = _name(length)
    lines = _ledger(["short", long_name, "other"])
    # `[n/3] ` is six characters; the name is followed by padding or a
    # separator, never by the kind.
    row = next(ln for ln in lines if ln[6:].startswith(long_name))
    assert row[6 + len(long_name)] == " "
    kind_columns = {line.index("extraction") for line in lines if "extraction" in line}
    assert len(kind_columns) == 1
    # The skipped row's dash sits in the same column as the kind of the others.
    skipped = lines[-1]
    assert skipped[kind_columns.pop()] == "-"


def test_live_ledger_for_ordinary_names_is_unchanged() -> None:
    """Names that fit keep the exact historical layout: a 24-wide name column."""
    lines = _ledger(["raw_invoices", "invoice_summary", "totals"])
    assert lines[0] == "[1/3] raw_invoices            extraction          3 rows     0.0s  OK"
    assert lines[1] == "[2/3] invoice_summary         extraction          3 rows     0.0s  OK"
    assert lines[2] == (
        "[3/3] totals".ljust(30) + "-".ljust(12) + "-".rjust(9) + " " * 5 + "-".rjust(8)
        + "  SKIPPED (upstream failed)"
    )


@pytest.fixture
def long_named_project(tmp_path: Path, example_project_dir: Path) -> Path:
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    generate_invoices(2, dst / "data" / "invoices", seed=1)
    return dst


def _rename_summary(project: Path, new_name: str) -> None:
    path = project / "models" / "invoice_summary.yml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("name: invoice_summary", f"name: {new_name}"),
        encoding="utf-8",
    )


def _assert_table_aligned(lines: list[str], long_name: str, header_word: str, cell: str) -> None:
    """The row for `long_name` has a separator after it, and `cell` starts in
    the column the header's `header_word` does."""
    header = next(ln for ln in lines if ln.startswith("model"))
    row = next(ln for ln in lines if ln.startswith(long_name))
    assert row[len(long_name)] == " "
    assert row.index(cell, len(long_name)) == header.index(header_word)


@pytest.mark.parametrize("length", [24, 40])
def test_run_summary_and_ledger_do_not_fuse_long_model_names(
    long_named_project: Path, monkeypatch: pytest.MonkeyPatch, length: int
) -> None:
    monkeypatch.delenv("STEL_VERBOSE", raising=False)
    long_name = "s" * length
    _rename_summary(long_named_project, long_name)

    result = CliRunner().invoke(cli, ["--project-dir", str(long_named_project), "run"])
    assert result.exit_code == 0, (result.stdout, result.stderr)

    out = result.stdout.splitlines()
    _assert_table_aligned(out, long_name, "kind", "transform")
    _assert_table_aligned(out, long_name, "mater.", "full")
    assert next(ln for ln in out if ln.startswith("model")).index("processed") > 0

    ledger = [ln for ln in result.stderr.splitlines() if long_name in ln and ln.startswith("[")]
    assert len(ledger) == 1
    kind_start = ledger[0].index("transform")
    assert ledger[0][kind_start - 1] == " "
    raw = next(ln for ln in result.stderr.splitlines() if "raw_invoices" in ln and "[" in ln[:1])
    assert raw.index("extraction") == kind_start


def test_build_summary_and_test_table_do_not_fuse_long_model_names(
    long_named_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STEL_VERBOSE", raising=False)
    long_name = "s" * 40
    _rename_summary(long_named_project, long_name)

    result = CliRunner().invoke(cli, ["--project-dir", str(long_named_project), "build"])
    assert result.exit_code == 0, (result.stdout, result.stderr)

    out = result.stdout.splitlines()
    model_table = out[: next(i for i, ln in enumerate(out) if not ln.strip())]
    _assert_table_aligned(model_table, long_name, "kind", "transform")
    test_table = out[len(model_table) + 1 :]
    # Test rows: model, then test name, then column, each separated.
    _assert_table_aligned(test_table, long_name, "test", "not_null")

    test_only = CliRunner().invoke(cli, ["--project-dir", str(long_named_project), "test"])
    assert test_only.exit_code == 0, (test_only.stdout, test_only.stderr)
    _assert_table_aligned(test_only.stdout.splitlines(), long_name, "test", "not_null")
