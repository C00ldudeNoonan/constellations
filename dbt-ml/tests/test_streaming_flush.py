"""Streaming/chunked materialization for extraction models (issue #77).

Runs examples/invoice_pipeline (json backend) with a tiny flush_every so a
handful of documents exercises multiple flushes: per-flush incremental
upserts + state (crash recovery), the staged full-load swap, and the batch
path's chunked materialization.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import duckdb
import pytest

from dbt_ml.adapters.base import AdapterError
from dbt_ml.adapters.duckdb import DuckDBAdapter
from dbt_ml.backends import llm_backend
from dbt_ml.providers import (
    BatchInferenceItem,
    BatchInferenceRequest,
    BatchInferenceResult,
    InferenceResult,
    ProviderUsage,
)
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


@pytest.fixture
def typed_empty_project(tmp_path: Path, example_project_dir: Path) -> Path:
    dst = tmp_path / "typed_project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    (dst / "models" / "raw_invoices.yml").write_text(
        """version: 2
models:
  - name: raw_invoices
    source: ref('vendor_invoices')
    extraction:
      backend: json
      options:
        fields: [invoice_id, total, quantity, paid, note, payload, event_date, event_at]
    materialization: incremental
    fields:
      - {name: invoice_id, data_type: string}
      - {name: total, data_type: float}
      - {name: quantity, data_type: integer}
      - {name: paid, data_type: boolean}
      - {name: note, data_type: string}
      - {name: payload, data_type: json}
      - {name: event_date, data_type: date}
      - {name: event_at, data_type: timestamp}
"""
    )
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


def _table_snapshot(project: Path, table: str) -> list[tuple[Any, ...]]:
    con = duckdb.connect(str(project / "target" / "dbt_ml.duckdb"), read_only=True)
    try:
        return con.execute(
            f'SELECT * FROM "dbt_ml"."dbt_ml"."{table}" ORDER BY document_id'
        ).fetchall()
    finally:
        con.close()


def _table_types(project: Path, table: str) -> dict[str, str]:
    con = duckdb.connect(str(project / "target" / "dbt_ml.duckdb"), read_only=True)
    try:
        return dict(
            con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'dbt_ml' AND table_name = ?",
                [table],
            ).fetchall()
        )
    finally:
        con.close()


def test_full_zero_match_streams_typed_empty_relation(
    typed_empty_project: Path,
) -> None:
    model = typed_empty_project / "models" / "raw_invoices.yml"
    model.write_text(
        model.read_text().replace("materialization: incremental", "materialization: full")
    )

    result = run_project(typed_empty_project, select="raw_invoices")[0]

    assert result.rows_written == 0
    assert _table_count(typed_empty_project, "raw_invoices") == 0
    types = _table_types(typed_empty_project, "raw_invoices")
    assert types["invoice_id"] == "VARCHAR"
    assert types["total"] == "DOUBLE"
    assert types["quantity"] == "BIGINT"
    assert types["paid"] == "BOOLEAN"
    assert types["event_date"] == "DATE"
    assert types["event_at"] == "TIMESTAMP WITH TIME ZONE"


def test_incremental_typed_empty_then_nonempty_preserves_types(
    typed_empty_project: Path,
) -> None:
    first = run_project(typed_empty_project, select="raw_invoices")[0]
    assert first.rows_written == 0
    assert first.warnings
    assert _table_count(typed_empty_project, "raw_invoices") == 0

    data_dir = typed_empty_project / "data" / "invoices"
    data_dir.mkdir(parents=True)
    (data_dir / "one.json").write_text(
        '{"invoice_id":"INV-1","total":19.5,"quantity":2,"paid":true,'
        '"note":"ready","payload":{"kind":"sample"},'
        '"event_date":"2026-07-11","event_at":"2026-07-11T12:30:00Z"}'
    )

    second = run_project(typed_empty_project, select="raw_invoices")[0]

    assert second.rows_written == 1
    types = _table_types(typed_empty_project, "raw_invoices")
    assert types["invoice_id"] == "VARCHAR"
    assert types["total"] == "DOUBLE"
    assert types["quantity"] == "BIGINT"
    assert types["paid"] == "BOOLEAN"
    assert types["event_date"] == "DATE"
    assert types["event_at"] == "TIMESTAMP WITH TIME ZONE"
    con = duckdb.connect(
        str(typed_empty_project / "target" / "dbt_ml.duckdb"), read_only=True
    )
    try:
        row = con.execute(
            'SELECT invoice_id, total, quantity, paid, note, payload '
            'FROM "dbt_ml"."dbt_ml"."raw_invoices"'
        ).fetchone()
    finally:
        con.close()
    assert row == ("INV-1", 19.5, 2, True, "ready", '{"kind": "sample"}')


def test_declared_extraction_fields_are_an_exact_projection(
    typed_empty_project: Path,
) -> None:
    model = typed_empty_project / "models" / "raw_invoices.yml"
    model.write_text(
        model.read_text().replace(
            "      options:\n"
            "        fields: [invoice_id, total, quantity, paid, note, payload, "
            "event_date, event_at]\n",
            "      options: {}\n",
        )
    )
    data_dir = typed_empty_project / "data" / "invoices"
    data_dir.mkdir(parents=True)
    (data_dir / "one.json").write_text(
        '{"invoice_id":"INV-1","total":19.5,"quantity":2,"paid":true,'
        '"note":"ready","payload":{},"event_date":"2026-07-11",'
        '"event_at":"2026-07-11T12:30:00Z","undeclared":"drop me"}'
    )

    run_project(typed_empty_project, select="raw_invoices")

    assert "undeclared" not in _table_types(typed_empty_project, "raw_invoices")


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


def test_publish_every_coalesces_flushes(
    monkeypatch: pytest.MonkeyPatch, flushing_project: Path
) -> None:
    """publish_every>1 shares one MERGE across that many flushes (issue #293):
    5 docs at flush_every=2 still stream in 3 flushes but publish in 2 upserts
    (flushes 1-2 coalesced, flush 3 trailing), and state advances per publish."""
    raw = flushing_project / "models" / "raw_invoices.yml"
    raw.write_text(
        raw.read_text().replace(
            "      flush_every: 2", "      flush_every: 2\n      publish_every: 2", 1
        )
    )

    calls = {"n": 0}
    orig = DuckDBAdapter.materialize_incremental

    def spy(self: DuckDBAdapter, table: str, df: Any, **kwargs: Any) -> int:
        calls["n"] += 1
        return orig(self, table, df, **kwargs)

    monkeypatch.setattr(DuckDBAdapter, "materialize_incremental", spy)

    r = run_project(flushing_project, select="raw_invoices")[0]
    assert calls["n"] == 2, "3 flushes at publish_every=2 → 2 coalesced upserts"
    assert r.rows_written == 5
    assert _table_count(flushing_project, "raw_invoices") == 5
    assert _state_count(flushing_project, "raw_invoices") == 5

    # State advanced only for published docs: the second run skips all five.
    assert run_project(flushing_project, select="raw_invoices")[0].documents_skipped == 5


def test_publish_every_crash_keeps_published_batch(
    monkeypatch: pytest.MonkeyPatch, flushing_project: Path
) -> None:
    """A crash with a partial buffer discards only the unpublished flush; the
    already-published (coalesced) batch and its state survive and the run stays
    retryable — coarser than per-flush recovery but state-safe (issue #293)."""
    raw = flushing_project / "models" / "raw_invoices.yml"
    raw.write_text(
        raw.read_text().replace(
            "      flush_every: 2", "      flush_every: 2\n      publish_every: 2", 1
        )
    )

    calls = {"n": 0}
    orig = DuckDBAdapter.materialize_incremental

    def failing(self: DuckDBAdapter, *args: Any, **kwargs: Any) -> int:
        calls["n"] += 1
        if calls["n"] == 2:
            raise AdapterError("simulated crash on second publication")
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(DuckDBAdapter, "materialize_incremental", failing)
    with pytest.raises(RunError, match="simulated crash"):
        run_project(flushing_project, select="raw_invoices")

    # Publication 1 coalesced flushes 1-2 (4 docs) — rows AND state survived;
    # the buffered trailing flush was never published.
    assert _table_count(flushing_project, "raw_invoices") == 4
    assert _state_count(flushing_project, "raw_invoices") == 4

    monkeypatch.undo()
    r = run_project(flushing_project, select="raw_invoices")[0]
    assert r.documents_skipped == 4
    assert r.documents_processed == 1
    assert _table_count(flushing_project, "raw_invoices") == 5
    assert _state_count(flushing_project, "raw_invoices") == 5


def test_publish_every_preserves_later_flush_schema_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, example_project_dir: Path
) -> None:
    """Coalescing must not fold a later flush's new column into the first
    publication's `on_schema_change` policy (issue #293 review). A schema-on-read
    model with `on_schema_change: ignore` that gains a column in a *later* flush
    must still publish that column via `append_new_columns` — exactly as the
    per-flush path did — instead of dropping it and advancing state (data loss).
    The buffer is published at the schema boundary so each publication is uniform.
    """
    project = tmp_path / "drift_project"
    shutil.copytree(
        example_project_dir,
        project,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    (project / "models" / "raw_invoices.yml").write_text(
        "version: 2\n"
        "models:\n"
        "  - name: raw_invoices\n"
        "    source: ref('vendor_invoices')\n"
        "    extraction:\n"
        "      backend: json\n"
        "      flush_every: 2\n"
        "      publish_every: 10\n"
        "    materialization: incremental\n"
        "    on_schema_change: ignore\n"
    )
    docs = project / "data" / "invoices"
    docs.mkdir(parents=True)
    # document_id order follows the sorted relative path, so numeric prefixes fix
    # flush grouping. Baseline run creates the target with schema {val}.
    (docs / "01a.json").write_text('{"val": "a"}')
    (docs / "02b.json").write_text('{"val": "b"}')
    run_project(project, select="raw_invoices")

    # Second run adds four new docs across two flushes: flush 1 = {03c, 04d}
    # (schema {val}, matches the target), flush 2 = {05e, 06f} where 05e
    # introduces `extra`. Coalescing them naively would subject `extra` to the
    # first publication's `ignore` policy and drop it.
    (docs / "03c.json").write_text('{"val": "c"}')
    (docs / "04d.json").write_text('{"val": "d"}')
    (docs / "05e.json").write_text('{"val": "e", "extra": "kept"}')
    (docs / "06f.json").write_text('{"val": "f"}')

    calls: list[tuple[list[str], str]] = []
    orig = DuckDBAdapter.materialize_incremental

    def spy(self: DuckDBAdapter, table: str, df: Any, **kwargs: Any) -> int:
        calls.append((list(df.columns), str(kwargs.get("on_schema_change"))))
        return orig(self, table, df, **kwargs)

    monkeypatch.setattr(DuckDBAdapter, "materialize_incremental", spy)
    run_project(project, select="raw_invoices")

    # Two publications split at the schema boundary: flush 1 under the model's
    # `ignore` policy (no `extra`), then the drifting flush 2 under
    # `append_new_columns` (carrying `extra`).
    assert len(calls) == 2
    assert "extra" not in calls[0][0] and calls[0][1] == "ignore"
    assert "extra" in calls[1][0] and calls[1][1] == "append_new_columns"

    # The later flush's column and value survive rather than being dropped.
    assert "extra" in _table_types(project, "raw_invoices")
    con = duckdb.connect(
        str(project / "target" / "dbt_ml.duckdb"), read_only=True
    )
    try:
        kept = con.execute(
            'SELECT extra FROM "dbt_ml"."dbt_ml"."raw_invoices" WHERE val = ?',
            ["e"],
        ).fetchone()
    finally:
        con.close()
    assert kept == ("kept",)


def test_full_materialization_streams_through_staging(
    monkeypatch: pytest.MonkeyPatch, flushing_project: Path
) -> None:
    raw = flushing_project / "models" / "raw_invoices.yml"
    raw.write_text(
        raw.read_text().replace("materialization: incremental", "materialization: full")
    )

    chunk_counts: list[int] = []
    orig = DuckDBAdapter.materialize_full_chunks

    def spy(self: DuckDBAdapter, table: str, chunks: Any, **kwargs: Any) -> int:
        def counting() -> Any:
            for df in chunks:
                chunk_counts.append(df.height)
                yield df

        return orig(self, table, counting(), **kwargs)

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


def test_full_document_failure_preserves_target_and_state(
    flushing_project: Path,
) -> None:
    raw = flushing_project / "models" / "raw_invoices.yml"
    raw.write_text(
        raw.read_text().replace("materialization: incremental", "materialization: full")
    )
    run_project(flushing_project, select="raw_invoices")
    before = _table_snapshot(flushing_project, "raw_invoices")
    assert _state_count(flushing_project, "raw_invoices") == 5

    (flushing_project / "data" / "invoices" / "corrupt.json").write_text("{broken")
    result = run_project(flushing_project, select="raw_invoices")[0]

    assert result.rows_written == 0
    assert any("corrupt.json" in error for error in result.errors)
    assert _table_snapshot(flushing_project, "raw_invoices") == before
    assert _state_count(flushing_project, "raw_invoices") == 5


def test_full_materialization_replaces_state_snapshot(flushing_project: Path) -> None:
    raw = flushing_project / "models" / "raw_invoices.yml"
    raw.write_text(
        raw.read_text().replace("materialization: incremental", "materialization: full")
    )
    run_project(flushing_project, select="raw_invoices")
    removed = next((flushing_project / "data" / "invoices").glob("*.json"))
    removed.unlink()

    result = run_project(flushing_project, select="raw_invoices")[0]

    assert result.rows_written == 4
    assert _table_count(flushing_project, "raw_invoices") == 4
    assert _state_count(flushing_project, "raw_invoices") == 4


def test_dynamic_backend_fields_cannot_overwrite_lineage(
    tmp_path: Path, example_project_dir: Path
) -> None:
    project = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        project,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    model = project / "models" / "raw_invoices.yml"
    model.write_text(
        model.read_text().replace(
            "      options:\n"
            "        fields: [invoice_id, vendor, issue_date, line_items, total, currency]\n",
            "      options: {}\n",
        )
    )
    data = project / "data" / "invoices"
    data.mkdir(parents=True)
    (data / "hostile.json").write_text('{"document_id": "attacker-controlled"}')

    with pytest.raises(RunError, match="reserved dbt-ml lineage columns"):
        run_project(project, select="raw_invoices")


def test_batch_mode_single_submission_chunked_flushes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
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
        requests: list[BatchInferenceRequest],
        *,
        provider: str,
        poll_seconds: float,
        api_key_env: str,
        max_retries: int,
        **_kwargs: object,
    ) -> tuple[BatchInferenceResult, bool]:
        batch_calls["n"] += 1
        out: list[BatchInferenceItem] = []
        for i, req in enumerate(requests):
            out.append(
                BatchInferenceItem(
                    req.request_id,
                    result=InferenceResult(
                        {
                            "vendor": "V",
                            "invoice_id": f"I-{i}",
                            "issue_date": "2026-01-01",
                            "currency": "USD",
                            "total": 1.0,
                        },
                        usage=ProviderUsage(
                    input_tokens=100,
                    output_tokens=10,
                        ),
                    ),
                ),
            )
        return BatchInferenceResult(tuple(out), batch_submissions=1), False

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
