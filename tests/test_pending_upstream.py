"""The adapter read behind `stel plan --pending` (issue #751)."""
from __future__ import annotations

from pathlib import Path

from stel.adapters import StateRecord, StateScope, create_adapter, parse_warehouse_config
from stel.adapters.base import PendingUpstreamRequest


def _adapter(tmp_path: Path):
    config = parse_warehouse_config({"type": "duckdb", "path": str(tmp_path / "w.duckdb")})
    return create_adapter(config, project_dir=tmp_path)


def _seed_upstream(adapter, rows: list[tuple[int, str, str]]) -> None:
    table = adapter.table_ref("chunks")
    adapter.execute(
        f"CREATE TABLE {table} (context_id BIGINT, filing_date DATE, symbol VARCHAR)"
    )
    for key, day, symbol in rows:
        adapter.execute(f"INSERT INTO {table} VALUES ({key}, DATE '{day}', '{symbol}')")


def _request(**overrides: object) -> PendingUpstreamRequest:
    fields: dict[str, object] = {
        "scope": StateScope("embeddings"),
        "upstream": "chunks",
        "key_column": "context_id",
        "partition_column": "filing_date",
        "cluster_column": "symbol",
    }
    fields.update(overrides)
    return PendingUpstreamRequest(**fields)  # type: ignore[arg-type]


def test_new_rows_are_upstream_keys_with_no_state_row(tmp_path: Path) -> None:
    with _adapter(tmp_path) as adapter:
        _seed_upstream(
            adapter,
            [
                (1, "2026-01-05", "AAA"),
                (2, "2026-02-10", "BBB"),
                (3, "2026-03-15", "BBB"),
                (4, "2026-03-20", "CCC"),
            ],
        )
        # A numeric key is stored as its string, which is the cast under test.
        adapter.upsert_state(
            StateScope("embeddings"), [StateRecord("1", "fp", "v1"), StateRecord("2", "fp", "v1")]
        )

        summary = adapter.pending_upstream(_request(), dry_run=False)

    assert summary.new_rows == 2
    assert str(summary.partition_min) == "2026-03-15"
    assert str(summary.partition_max) == "2026-03-20"
    assert summary.cluster_distinct == 2
    assert summary.deleted_rows == 0


def test_deleted_rows_are_state_keys_with_no_upstream_row(tmp_path: Path) -> None:
    with _adapter(tmp_path) as adapter:
        _seed_upstream(adapter, [(1, "2026-01-05", "AAA")])
        adapter.upsert_state(
            StateScope("embeddings"),
            [StateRecord("1", "fp", "v1"), StateRecord("gone", "fp", "v1")],
        )
        # Another model's state must not count as this one's.
        adapter.upsert_state(StateScope("other"), [StateRecord("zzz", "fp", "v1")])

        summary = adapter.pending_upstream(_request(), dry_run=False)

    assert (summary.new_rows, summary.deleted_rows) == (0, 1)
    assert summary.partition_min is None


def test_never_run_model_has_every_upstream_row_pending(tmp_path: Path) -> None:
    with _adapter(tmp_path) as adapter:
        _seed_upstream(adapter, [(1, "2026-01-05", "AAA"), (2, "2026-01-06", "AAA")])

        summary = adapter.pending_upstream(
            _request(partition_column=None, cluster_column=None), dry_run=False
        )

    assert summary.new_rows == 2
    assert summary.partition_min is None
    assert summary.cluster_distinct is None


def test_dry_run_reads_nothing(tmp_path: Path) -> None:
    with _adapter(tmp_path) as adapter:
        _seed_upstream(adapter, [(1, "2026-01-05", "AAA")])

        summary = adapter.pending_upstream(_request(), dry_run=True)

    assert summary.new_rows == 0
    assert summary.estimated_bytes is None  # DuckDB does not bill by bytes


def test_null_keys_are_skipped_and_repeated_keys_count_once(tmp_path: Path) -> None:
    """A NULL key never matches a state row, and a repeated key is one record
    to the runners, so neither may inflate `new_rows` (issue #751)."""
    with _adapter(tmp_path) as adapter:
        _seed_upstream(adapter, [(1, "2026-01-05", "AAA"), (1, "2026-01-06", "AAA")])
        adapter.execute(
            f"INSERT INTO {adapter.table_ref('chunks')} VALUES (NULL, DATE '2026-01-07', 'AAA')"
        )
        adapter.upsert_state(StateScope("embeddings"), [StateRecord("9", "fp", "v1")])

        summary = adapter.pending_upstream(_request(), dry_run=False)
        fresh = adapter.pending_upstream(_request(scope=StateScope("never_run")), dry_run=False)

    assert summary.new_rows == 1
    assert fresh.new_rows == 1


def test_a_key_type_whose_text_differs_by_warehouse_is_unsupported(tmp_path: Path) -> None:
    """State keys are Python `str(value)`; `CAST(true AS VARCHAR)` is 'true', so
    comparing them in SQL would report every row as new and deleted."""
    from types import SimpleNamespace

    from stel.pending import PendingSource, _pending_for_model

    with _adapter(tmp_path) as adapter:
        adapter.execute(f"CREATE TABLE {adapter.table_ref('flags')} (flag BOOLEAN)")
        model = SimpleNamespace(name="m", kind_label=lambda: "embed")

        result = _pending_for_model(
            model,  # ty: ignore[invalid-argument-type]
            PendingSource("flags", "flag"),
            adapter=adapter,
            dry_run=False,
            include_changed=False,
            predicates=(),
        )

    assert result.status == "unsupported"
    assert "string and integer" in result.reason
