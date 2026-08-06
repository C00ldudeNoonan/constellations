from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

import duckdb
import pytest

from dbt_ml.config.model import ModelConfig
from dbt_ml.runner import RunError, _prepare_subset_run, run_project
from dbt_ml.synth import generate_invoices


def _extraction_model(
    name: str, materialization: Literal["full", "incremental"] = "incremental"
) -> ModelConfig:
    return ModelConfig(
        name=name,
        source="ref('src')",
        extraction={"backend": "json"},
        materialization=materialization,
    )


# ── the --source-filter guardrails (unit) ────────────────────────────────────


def test_no_filter_is_not_a_subset_run() -> None:
    assert (
        _prepare_subset_run(
            (), full_refresh=False, selected=["m"], models=[_extraction_model("m")]
        )
        is False
    )


def test_filter_returns_subset_run_for_incremental() -> None:
    assert (
        _prepare_subset_run(
            ("AAPL/*",),
            full_refresh=False,
            selected=["m"],
            models=[_extraction_model("m")],
        )
        is True
    )


def test_filter_rejects_full_refresh() -> None:
    with pytest.raises(RunError, match="cannot be combined with --full-refresh"):
        _prepare_subset_run(
            ("AAPL/*",),
            full_refresh=True,
            selected=["m"],
            models=[_extraction_model("m")],
        )


def test_filter_rejects_non_incremental_extraction_model() -> None:
    with pytest.raises(RunError, match="additive extraction models"):
        _prepare_subset_run(
            ("AAPL/*",),
            full_refresh=False,
            selected=["m"],
            models=[_extraction_model("m", materialization="full")],
        )


def test_filter_rejects_insert_overwrite_strategy() -> None:
    # insert_overwrite replaces whole partitions, so a filtered (partial) batch
    # would clobber sibling documents — not additive (#266 review).
    model = ModelConfig(
        name="m",
        source="ref('src')",
        extraction={"backend": "json"},
        materialization="incremental",
        warehouse_options={"incremental_strategy": "insert_overwrite"},
    )
    with pytest.raises(RunError, match="insert_overwrite"):
        _prepare_subset_run(
            ("AAPL/*",), full_refresh=False, selected=["m"], models=[model]
        )


# ── end-to-end: a filtered run is additive and never clobbers siblings ───────


def _source_uris(project: Path) -> list[str]:
    con = duckdb.connect(str(project / "target" / "dbt_ml.duckdb"), read_only=True)
    try:
        return [
            r[0]
            for r in con.execute(
                'SELECT source_uri FROM "dbt_ml".dbt_ml.raw_invoices'
            ).fetchall()
        ]
    finally:
        con.close()


def test_source_filter_does_not_clobber_other_partitions(
    tmp_path: Path, example_project_dir: Path
) -> None:
    project = tmp_path / "proj"
    shutil.copytree(
        example_project_dir,
        project,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    invoices = project / "data" / "invoices"
    # Partition the corpus by top-level directory -> `relative_path` = AAPL/… .
    generate_invoices(2, invoices / "AAPL", seed=1)
    generate_invoices(1, invoices / "MSFT", seed=50)

    # Full run materializes both partitions.
    run_project(project)
    initial = _source_uris(project)
    assert sum("/AAPL/" in u for u in initial) == 2
    assert sum("/MSFT/" in u for u in initial) == 1

    # A filtered run discovers only AAPL. Before #266 this made every MSFT
    # document look "removed" and delete_rows_and_state'd it — the crux. It must
    # instead leave MSFT's rows (and state) untouched.
    run_project(project, source_filter=["AAPL/*"])
    after = _source_uris(project)
    assert sum("/MSFT/" in u for u in after) == 1  # MSFT NOT clobbered
    assert sum("/AAPL/" in u for u in after) == 2


def test_source_filter_selects_only_the_matching_partition(
    tmp_path: Path, example_project_dir: Path
) -> None:
    project = tmp_path / "proj"
    shutil.copytree(
        example_project_dir,
        project,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    invoices = project / "data" / "invoices"
    generate_invoices(2, invoices / "AAPL", seed=1)
    generate_invoices(3, invoices / "MSFT", seed=50)

    # First run only MSFT; AAPL must not be materialized yet.
    run_project(project, source_filter=["MSFT/*"])
    uris = _source_uris(project)
    assert sum("/MSFT/" in u for u in uris) == 3
    assert not any("/AAPL/" in u for u in uris)
