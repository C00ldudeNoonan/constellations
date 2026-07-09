"""Streaming/chunked materialization for extraction models (issue #77).

Runs examples/invoice_pipeline (json backend) with a tiny flush_every so a
handful of documents exercises multiple flushes: per-flush incremental
upserts + state (crash recovery), the staged full-load swap, and the batch
path's chunked materialization.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import pytest

from dbt_ml.adapters.base import AdapterError
from dbt_ml.adapters.duckdb import DuckDBAdapter
from dbt_ml.backends import llm_backend
from dbt_ml.runner import RunError, run_project
from dbt_ml.synth import generate_invoice_texts, generate_invoices


@pytest.fixture
def flushing_project(tmp_path: Path, example_project_dir: Path) -> Path:
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    raw = dst / "models" / "raw_invoices.yml"
    raw.write_text(
        raw.read_text().replace(
            "    extraction:", "    extraction:\n      flush_every: 2", 1
        )
    )
    generate_invoices(5, dst / "data" / "invoices", seed=1)
    return dst


def _table_count(project: Path, table: str) -> int:
    con = duckdb.connect(str(project / "target" / "dbt_ml.duckdb"), read_only=True)
    try:
        row = con.execute(
            f'SELECT COUNT(*) FROM "dbt_ml"."dbt_ml"."{table}"'
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        con.close()


def _state_count(project: Path, model: str) -> int:
    con = duckdb.connect(str(project / "target" / "dbt_ml.duckdb"), read_only=True)
    try:
        row = con.execute(
            'SELECT COUNT(*) FROM "dbt_ml"."dbt_ml"."dbt_ml_state" '
            "WHERE model_name = ?",
            [model],
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        con.close()


def test_incremental_flushes_in_chunks(
    monkeypatch: pytest.MonkeyPatch, flushing_project: Path
) -> None:
    calls = {"n": 0}
    orig = DuckDBAdapter.materialize_incremental

    def spy(self: DuckDBAdapter, *args: Any, **kwargs: Any) -> int:
        calls["n"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(DuckDBAdapter, "materialize_incremental", spy)

    results = run_project(flushing_project, select="raw_invoices")
    r = results[0]
    assert calls["n"] == 3, "5 docs at flush_every=2 → 3 flushes"
    assert r.rows_written == 5
    assert _table_count(flushing_project, "raw_invoices") == 5
    assert _state_count(flushing_project, "raw_invoices") == 5

    # Everything already flushed+stated: second run skips all docs.
    results2 = run_project(flushing_project, select="raw_invoices")
    assert results2[0].documents_skipped == 5


def test_crash_mid_run_keeps_completed_flushes(
    monkeypatch: pytest.MonkeyPatch, flushing_project: Path
) -> None:
    calls = {"n": 0}
    orig = DuckDBAdapter.materialize_incremental

    def failing(self: DuckDBAdapter, *args: Any, **kwargs: Any) -> int:
        calls["n"] += 1
        if calls["n"] == 2:
            raise AdapterError("simulated crash on second flush")
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(DuckDBAdapter, "materialize_incremental", failing)
    with pytest.raises(RunError, match="simulated crash"):
        run_project(flushing_project, select="raw_invoices")

    # Chunk 1 (2 docs) survived — rows AND state.
    assert _table_count(flushing_project, "raw_invoices") == 2
    assert _state_count(flushing_project, "raw_invoices") == 2

    # Recovery run processes only the remaining 3 documents.
    monkeypatch.undo()
    results = run_project(flushing_project, select="raw_invoices")
    r = results[0]
    assert r.documents_skipped == 2
    assert r.documents_processed == 3
    assert _table_count(flushing_project, "raw_invoices") == 5
    assert _state_count(flushing_project, "raw_invoices") == 5


def test_full_materialization_streams_through_staging(
    monkeypatch: pytest.MonkeyPatch, flushing_project: Path
) -> None:
    raw = flushing_project / "models" / "raw_invoices.yml"
    raw.write_text(
        raw.read_text().replace("materialization: incremental", "materialization: full")
    )

    chunk_counts: list[int] = []
    orig = DuckDBAdapter.materialize_full_chunks

    def spy(self: DuckDBAdapter, table: str, chunks: Any) -> int:
        def counting() -> Any:
            for df in chunks:
                chunk_counts.append(df.height)
                yield df

        return orig(self, table, counting())

    monkeypatch.setattr(DuckDBAdapter, "materialize_full_chunks", spy)

    results = run_project(flushing_project, select="raw_invoices")
    assert results[0].rows_written == 5
    assert chunk_counts == [2, 2, 1]
    assert _table_count(flushing_project, "raw_invoices") == 5

    # Re-run replaces, never appends.
    run_project(flushing_project, select="raw_invoices")
    assert _table_count(flushing_project, "raw_invoices") == 5


def test_full_refresh_uses_staged_path(flushing_project: Path) -> None:
    run_project(flushing_project, select="raw_invoices")
    results = run_project(flushing_project, select="raw_invoices", full_refresh=True)
    r = results[0]
    assert r.documents_processed == 5
    assert r.rows_written == 5
    assert _table_count(flushing_project, "raw_invoices") == 5
    assert _state_count(flushing_project, "raw_invoices") == 5


def test_batch_mode_single_submission_chunked_flushes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = Path(__file__).resolve().parents[1]
    dst = tmp_path / "llm_proj"
    shutil.copytree(
        repo / "examples" / "llm_invoice_pipeline",
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    model = dst / "models" / "raw_invoices_llm.yml"
    text = model.read_text()
    text = text.replace("    extraction:", "    extraction:\n      flush_every: 2", 1)
    text = text.replace("      options:", "      options:\n        batch: true", 1)
    model.write_text(text)
    generate_invoice_texts(5, dst / "data" / "invoices_text", 1)

    batch_calls = {"n": 0}

    def fake_batch(
        requests: list[dict[str, Any]], *, poll_seconds: float
    ) -> dict[str, Any]:
        batch_calls["n"] += 1
        out = {}
        for i, req in enumerate(requests):
            msg = SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="extract",
                        input={
                            "vendor": "V",
                            "invoice_id": f"I-{i}",
                            "issue_date": "2026-01-01",
                            "currency": "USD",
                            "total": 1.0,
                        },
                    )
                ],
                usage=SimpleNamespace(
                    input_tokens=100,
                    output_tokens=10,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                ),
            )
            out[req["custom_id"]] = SimpleNamespace(
                result=SimpleNamespace(type="succeeded", message=msg)
            )
        return out

    monkeypatch.setattr(llm_backend, "_run_message_batch", fake_batch)

    flushes = {"n": 0}
    orig = DuckDBAdapter.materialize_incremental

    def spy(self: DuckDBAdapter, *args: Any, **kwargs: Any) -> int:
        flushes["n"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(DuckDBAdapter, "materialize_incremental", spy)

    results = run_project(dst)
    r = next(x for x in results if x.model_name == "raw_invoices_llm")
    assert batch_calls["n"] == 1, "batch mode still submits once for the whole model"
    assert flushes["n"] == 3, "…while materialization flushes in chunks"
    assert r.rows_written == 5
    assert r.metrics["batch"] is True
