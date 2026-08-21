"""Append-only warehouse logs (issues #306, #329 phase 1).

Two histories, one mechanism. The tests below pin the mechanism (append not
replace, create on first write), the contract that makes it safe to turn on
(a log failure never fails the work being logged), and the privacy rule that
makes the query log usable at all (fingerprint always, text only on a second
explicit opt-in).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pytest

from stel.adapters import create_adapter, parse_warehouse_config
from stel.append_log import (
    QUERY_LOG_SCHEMA,
    RUN_LOG_SCHEMA,
    query_fingerprint,
    run_log_rows,
    write_rows,
)
from stel.config.profile import QueryLogConfig, RunLogConfig
from stel.execution.contracts import ModelRunResult

# ─── the shared mechanism ───────────────────────────────────────────────────


def _adapter(tmp_path: Path):
    return create_adapter(
        parse_warehouse_config(
            {"type": "duckdb", "path": str(tmp_path / "log.duckdb"), "schema": "main"}
        )
    )


def test_append_creates_the_relation_on_first_write(tmp_path: Path) -> None:
    # Opt-in with no provisioning step: turning the log on is the whole setup.
    with _adapter(tmp_path) as adapter:
        written = adapter.append_rows("a_log", pl.DataFrame({"n": [1, 2]}))

        assert written == 2
        assert adapter.row_count("a_log") == 2


def test_append_adds_rather_than_replaces(tmp_path: Path) -> None:
    """The history is the artifact — a second write must not overwrite."""
    with _adapter(tmp_path) as adapter:
        adapter.append_rows("a_log", pl.DataFrame({"n": [1]}))
        adapter.append_rows("a_log", pl.DataFrame({"n": [2]}))

        assert adapter.row_count("a_log") == 2


def test_append_tolerates_reordered_columns(tmp_path: Path) -> None:
    with _adapter(tmp_path) as adapter:
        adapter.append_rows("a_log", pl.DataFrame({"a": [1], "b": ["x"]}))
        adapter.append_rows("a_log", pl.DataFrame({"b": ["y"], "a": [2]}))

        rows = adapter.read_table("a_log").sort("a").to_dicts()
        assert rows == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


def test_an_empty_frame_writes_nothing(tmp_path: Path) -> None:
    with _adapter(tmp_path) as adapter:
        assert adapter.append_rows("a_log", pl.DataFrame()) == 0


# ─── best-effort by contract ────────────────────────────────────────────────


class _BrokenAdapter:
    def append_rows(self, table: str, df: pl.DataFrame) -> int:
        raise RuntimeError("warehouse said no")


def test_a_failed_log_write_does_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    """A log is observability, not the work.

    A warehouse that rejects the write must not turn a successful run into a
    failed one, or a served MCP answer into an error.
    """
    config = RunLogConfig(enabled=True, relation="stel_run_log")

    written = write_rows(
        _BrokenAdapter(),
        config,
        [{"n": 1}],
        schema={"n": pl.Int64},
        what="the run log",
    )

    assert written == 0
    assert "Could not write the run log" in caplog.text
    # The exception class locates the failure; its text is not echoed.
    assert "warehouse said no" not in caplog.text


def test_a_disabled_log_writes_nothing() -> None:
    disabled = RunLogConfig(enabled=False, relation="stel_run_log")

    # The broken adapter proves it is never called.
    schema = {"n": pl.Int64}
    assert (
        write_rows(_BrokenAdapter(), disabled, [{"n": 1}], schema=schema, what="x")
        == 0
    )
    assert (
        write_rows(_BrokenAdapter(), None, [{"n": 1}], schema=schema, what="x") == 0
    )


def test_logs_are_off_by_default() -> None:
    from stel.config.profile import TargetConfig

    target = TargetConfig.model_validate({"warehouse": {"type": "duckdb"}})

    assert target.run_log is None
    assert target.mcp_query_log is None


# ─── the query fingerprint ──────────────────────────────────────────────────


def test_the_fingerprint_groups_the_same_question() -> None:
    # The point is grouping repeats and spotting zero-result questions, not
    # distinguishing typography.
    assert query_fingerprint("What is CPI?") == query_fingerprint(
        "  what is   CPI?  "
    )
    assert query_fingerprint("What is CPI?") != query_fingerprint("What is PPI?")


def test_the_fingerprint_does_not_contain_the_query() -> None:
    assert "cpi" not in query_fingerprint("What is CPI?").lower()


# ─── run log rows (issue #306) ──────────────────────────────────────────────


def _result(**overrides: Any) -> ModelRunResult:
    values: dict[str, Any] = {
        "model_name": "notes",
        "materialization": "incremental",
        "kind": "extraction",
    }
    values.update(overrides)
    return ModelRunResult(**values)


def test_run_log_rows_carry_identity_and_aggregates() -> None:
    rows = run_log_rows(
        [
            _result(
                provider="vertex",
                provider_model="gemini-2.5-flash-lite",
                documents_processed=10,
                documents_skipped=2,
                metrics={"api_calls": 4, "input_tokens": 900, "output_tokens": 50},
            )
        ],
        invocation_id="abc",
        started_at="2026-08-21T00:00:00+00:00",
        completed_at="2026-08-21T00:00:05+00:00",
        profile_target="dev",
    )

    assert rows[0]["invocation_id"] == "abc"
    assert rows[0]["provider_model"] == "gemini-2.5-flash-lite"
    assert rows[0]["rows_processed"] == 10
    assert rows[0]["input_tokens"] == 900
    assert rows[0]["status"] == "success"


def test_a_budget_exceeded_run_is_visible_after_the_fact() -> None:
    # The terminal output of the run that tripped is otherwise the only record.
    rows = run_log_rows(
        [_result(status="budget_exceeded")],
        invocation_id="abc",
        started_at="t0",
        completed_at="t1",
        profile_target="dev",
    )

    assert rows[0]["status"] == "budget_exceeded"


def test_run_log_rows_carry_no_text_or_credentials() -> None:
    """Same rules as artifacts: resolved identity and aggregates only."""
    rows = run_log_rows(
        [
            _result(
                metrics={"api_calls": 1},
                errors=["a very specific failure message"],
            )
        ],
        invocation_id="abc",
        started_at="t0",
        completed_at="t1",
        profile_target="dev",
    )

    serialized = json.dumps(rows[0])
    assert "a very specific failure message" not in serialized
    assert not any(
        "prompt" in key or "text" in key or "key" in key for key in rows[0]
    )


# ─── end to end ─────────────────────────────────────────────────────────────


def _log_project(tmp_path: Path, *, run_log: bool) -> Path:
    project = tmp_path / "proj"
    (project / "models").mkdir(parents=True)
    (project / "sources").mkdir()
    docs = project / "data" / "docs"
    docs.mkdir(parents=True)

    (project / "stel_project.yml").write_text(
        "name: logs\nversion: '0.1.0'\nprofile: logs\n"
    )
    log_block = "      run_log:\n        enabled: true\n" if run_log else ""
    (project / "profiles.yml").write_text(
        "logs:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n"
        "        schema: main\n" + log_block
    )
    (project / "sources" / "src.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data/docs\n"
        "    file_pattern: '*.json'\n"
    )
    (project / "models" / "m.yml").write_text(
        "version: 2\nmodels:\n  - name: notes\n    source: ref('docs')\n"
        "    extraction:\n      backend: json\n      options:\n"
        "        fields: [note_id, body]\n    materialization: incremental\n"
    )
    for index in range(3):
        (docs / f"d{index}.json").write_text(
            json.dumps({"note_id": f"n{index}", "body": f"text {index}"})
        )
    return project


def _log_rows(project: Path) -> list[tuple[Any, ...]]:
    con = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        return con.execute(
            "SELECT invocation_id, model_name, status, rows_processed, "
            "rows_skipped FROM main.stel_run_log ORDER BY started_at"
        ).fetchall()
    finally:
        con.close()


def test_each_invocation_appends_a_row(tmp_path: Path) -> None:
    """The cross-run history #306 asked for: two runs, two rows, one table."""
    from stel.runner import run_project

    project = _log_project(tmp_path, run_log=True)
    run_project(project)
    run_project(project)

    rows = _log_rows(project)
    assert len(rows) == 2
    # Distinct invocations...
    assert rows[0][0] != rows[1][0]
    # ...and the second run's skips are visible as history.
    assert rows[0][3:] == (3, 0)
    assert rows[1][3:] == (0, 3)


def test_no_log_relation_exists_when_disabled(tmp_path: Path) -> None:
    from stel.runner import run_project

    project = _log_project(tmp_path, run_log=False)
    run_project(project)

    con = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    finally:
        con.close()
    assert "stel_run_log" not in tables


# ─── MCP query log (issue #329 phase 1) ─────────────────────────────────────


def test_query_log_config_defaults_to_fingerprint_only() -> None:
    """`capture_query_text` is a second opt-in, off even when the log is on."""
    config = QueryLogConfig(enabled=True)

    assert config.relation == "stel_mcp_query_log"
    assert config.capture_query_text is False


# ─── review follow-ups (PR #333) ────────────────────────────────────────────


def test_a_first_all_null_batch_does_not_poison_the_schema(tmp_path: Path) -> None:
    """The first invocation must not decide the persisted column types.

    A first run with no LLM model leaves `provider` null. Inferred, that
    becomes an integer column in DuckDB, and every later row carrying an
    actual provider name fails to convert — silently, because the write is
    best-effort, leaving the history stuck at its first row forever.
    """
    config = RunLogConfig(enabled=True, relation="stel_run_log")
    first = run_log_rows(
        [_result()],  # no provider, no metrics: every optional column null
        invocation_id="one",
        started_at="t0",
        completed_at="t1",
        profile_target="dev",
    )
    second = run_log_rows(
        [_result(provider="vertex", metrics={"api_calls": 3})],
        invocation_id="two",
        started_at="t2",
        completed_at="t3",
        profile_target="dev",
    )

    with _adapter(tmp_path) as adapter:
        assert write_rows(
            adapter, config, first, schema=RUN_LOG_SCHEMA, what="the run log"
        ) == 1
        assert write_rows(
            adapter, config, second, schema=RUN_LOG_SCHEMA, what="the run log"
        ) == 1

        rows = adapter.read_table("stel_run_log").sort("invocation_id")
        assert rows.height == 2
        assert rows["provider"].to_list() == [None, "vertex"]


def test_a_zero_result_query_row_does_not_poison_the_schema(
    tmp_path: Path,
) -> None:
    """Same hazard on the query log: the first query returning nothing.

    `returned_chunk_ids: []` and a null `top_score` would infer an empty list
    and an integer, so the first successful search afterwards could never be
    logged.
    """
    config = QueryLogConfig(enabled=True)
    empty = {
        "logged_at": "t0",
        "principal_id": "p",
        "tenant_id": None,
        "model_name": "m",
        "mode": "text",
        "query_fingerprint": "f",
        "query_text": None,
        "requested_limit": 5,
        "result_count": 0,
        "zero_results": True,
        "returned_chunk_ids": [],
        "top_score": None,
        "elapsed_ms": 1.0,
    }
    populated = {
        **empty,
        "logged_at": "t1",
        "result_count": 1,
        "zero_results": False,
        "returned_chunk_ids": ["chunk-a"],
        "top_score": 0.75,
    }

    with _adapter(tmp_path) as adapter:
        write_rows(
            adapter, config, [empty], schema=QUERY_LOG_SCHEMA, what="q"
        )
        written = write_rows(
            adapter, config, [populated], schema=QUERY_LOG_SCHEMA, what="q"
        )

        assert written == 1
        rows = adapter.read_table("stel_mcp_query_log").sort("logged_at")
        assert rows.height == 2
        assert rows["returned_chunk_ids"].to_list() == [[], ["chunk-a"]]
        assert rows["top_score"].to_list() == [None, 0.75]


def test_the_estimated_cost_metric_is_the_one_extraction_publishes() -> None:
    """`estimated_cost_usd` is what a priced run actually stores.

    Reading only `reported_cost_usd` left the column null on the normal
    estimator path — the exact query this log exists to answer.
    """
    rows = run_log_rows(
        [_result(metrics={"estimated_cost_usd": 0.0421, "api_calls": 2})],
        invocation_id="abc",
        started_at="t0",
        completed_at="t1",
        profile_target="dev",
    )

    assert rows[0]["estimated_cost_usd"] == pytest.approx(0.0421)


def test_provider_reported_cost_stands_in_when_no_estimate_exists() -> None:
    rows = run_log_rows(
        [_result(metrics={"reported_cost_usd": 0.5})],
        invocation_id="abc",
        started_at="t0",
        completed_at="t1",
        profile_target="dev",
    )

    assert rows[0]["estimated_cost_usd"] == pytest.approx(0.5)
