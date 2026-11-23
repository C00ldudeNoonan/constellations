"""`orphans`, `pending` and the `watch` re-run lines must not fuse columns (issue #822).

Same defect as the run ledger (issue #819): a literal ``{name:<32}`` with no
separator runs a 32-character name straight into the next column, and a longer
one pushes the rest of its row out of line with the header.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stel.adapters.base import StateScope, StateScopeSummary
from stel.cli_services.watch import format_rerun_lines
from stel.execution import ModelRunResult
from stel.orphans import OrphanReport, OrphanStateScope, OrphanTable, format_orphans
from stel.pending import ModelPending, PendingReport, format_pending_table

# Straddle the old widths (orphans 32, pending 32 + a space, watch 22 + a
# space): below, exactly at, one past, and far past.
ORPHAN_LENGTHS = [30, 31, 32, 33, 45]
PENDING_LENGTHS = [31, 32, 33, 45]
WATCH_LENGTHS = [20, 21, 22, 23, 40]


def _name(length: int) -> str:
    return "m" * length


def _orphans(names: list[str]) -> list[str]:
    scope = StateScopeSummary(
        scope=StateScope(names[-1]), rows=7, code_versions=1, last_run_at=None
    )
    report = OrphanReport(
        tables=tuple(OrphanTable(name=name, rows=3) for name in names[:-1]),
        state_scopes=(OrphanStateScope(scope),),
        schema="main",
    )
    return format_orphans(report)


def _pending(names: list[str]) -> list[str]:
    report = PendingReport(
        models=tuple(
            ModelPending(
                name=name,
                kind="extraction",
                status="pending",
                reason="",
                new_rows=5,
                deleted_rows=0,
                partition_min="2026-01-01",
                partition_max="2026-02-01",
            )
            for name in names
        ),
        target={},
        generated_at="now",
        dry_run=False,
        artifact_dir=Path("."),
    )
    return format_pending_table(report)


def _rerun(names: list[str]) -> list[str]:
    results = [
        ModelRunResult(
            model_name=name,
            materialization="table",
            kind="extraction",
            documents_processed=2,
            documents_skipped=1,
            rows_written=9,
        )
        for name in names
    ]
    return format_rerun_lines(results)


@pytest.mark.parametrize("length", ORPHAN_LENGTHS)
def test_orphans_table_separates_name_from_kind_and_aligns_rows(length: int) -> None:
    long_name = _name(length)
    lines = _orphans(["short", long_name, "other"])
    header, rule = lines[0], lines[1]
    row = next(ln for ln in lines if ln.startswith(long_name))
    assert row[length] == " "
    kind_column = header.index("kind")
    assert row[kind_column:].startswith("table")
    assert lines[2].index("table") == kind_column
    assert lines[-3].index("state_scope") == kind_column
    assert len(rule) == len(header)


@pytest.mark.parametrize("length", PENDING_LENGTHS)
def test_pending_table_separates_name_from_status_and_aligns_rows(length: int) -> None:
    long_name = _name(length)
    lines = _pending(["short", long_name, "other"])
    header = lines[0]
    status_column = header.index("status")
    for row in lines[1:4]:
        assert row[status_column:].startswith("pending")
        assert row[status_column - 1] == " "
    assert lines[2].startswith(long_name + " ")


@pytest.mark.parametrize("length", WATCH_LENGTHS)
def test_watch_rerun_lines_separate_name_from_kind_and_align_rows(length: int) -> None:
    long_name = _name(length)
    lines = _rerun(["short", long_name, "other"])
    kind_columns = {line.index("extraction") for line in lines}
    assert len(kind_columns) == 1
    kind_column = kind_columns.pop()
    for line in lines:
        assert line[kind_column - 1] == " "
    assert lines[1].startswith("  " + long_name + " ")


def test_orphans_table_for_ordinary_names_is_unchanged() -> None:
    lines = _orphans(["stray_table", "old_scope"])
    assert lines[0] == "name".ljust(32) + "kind".ljust(13) + "rows".rjust(10) + "  detail"
    assert lines[1] == "-" * len(lines[0])
    assert lines[2] == "stray_table".ljust(32) + "table".ljust(13) + "3".rjust(10)
    assert lines[3] == (
        "old_scope".ljust(32)
        + "state_scope".ljust(13)
        + "7".rjust(10)
        + "  stage=materialization code_versions=1 last_run_at=-"
    )


def test_pending_table_for_ordinary_names_is_unchanged() -> None:
    lines = _pending(["raw_invoices"])
    assert lines[0] == (
        "model".ljust(32) + " " + "status".ljust(17) + " " + "new".rjust(10) + " "
        + "deleted".rjust(9) + "  span"
    )
    assert lines[1] == (
        "raw_invoices".ljust(32) + " " + "pending".ljust(17) + " " + "5".rjust(10) + " "
        + "0".rjust(9) + "  2026-01-01 .. 2026-02-01"
    )


def test_watch_rerun_lines_for_ordinary_names_are_unchanged() -> None:
    lines = _rerun(["raw_invoices"])
    assert lines == [
        "  " + "raw_invoices".ljust(22) + " " + "extraction".ljust(12)
        + " processed=2     skipped=1     rows=9"
    ]
