"""The sync-watermark primitive (issue #611): the parent generation a model
last successfully caught up with.

`state_max_last_run_at` is the cheap generation signal (`stel_state`'s own
`last_run_at`, aggregated -- no new column on the parent). `read_sync_watermark`/
`write_sync_watermark` are the new, tiny table recording what a child last
synced to. Both are read-only/best-effort by design: an absent watermark or an
absent state scope resolve to "never", never to an error.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from stel.adapters import StateRecord, StateScope, create_adapter, parse_warehouse_config


def _duckdb_config(tmp_path: Path) -> dict[str, str]:
    return {"type": "duckdb", "path": str(tmp_path / "w.duckdb")}


def test_max_last_run_at_is_none_for_a_scope_with_no_published_state(
    tmp_path: Path,
) -> None:
    config = parse_warehouse_config(_duckdb_config(tmp_path))
    with create_adapter(config, project_dir=tmp_path) as adapter:
        assert adapter.state_max_last_run_at(StateScope("docs")) is None


def test_max_last_run_at_advances_only_when_a_row_is_actually_written(
    tmp_path: Path,
) -> None:
    """The generation a watermark compares against: untouched by a scope that
    publishes nothing, exactly like the `last_run_at` column it reads."""
    config = parse_warehouse_config(_duckdb_config(tmp_path))
    scope = StateScope("docs")
    with create_adapter(config, project_dir=tmp_path) as adapter:
        adapter.upsert_state(scope, [StateRecord("a", "fp-a", "v1")])
        first = adapter.state_max_last_run_at(scope)
        assert first is not None

        # A second upsert of the SAME row still counts as a real write --
        # this adapter-level primitive does not itself classify new vs
        # unchanged, that is the caller's job. What it must not do is move on
        # a scope nothing touched at all: a second, unrelated scope's write
        # leaves `docs`'s own generation exactly where it was.
        adapter.upsert_state(StateScope("other"), [StateRecord("z", "fp-z", "v9")])
        assert adapter.state_max_last_run_at(scope) == first


def test_read_sync_watermark_is_none_before_any_write(tmp_path: Path) -> None:
    config = parse_warehouse_config(_duckdb_config(tmp_path))
    with create_adapter(config, project_dir=tmp_path) as adapter:
        assert (
            adapter.read_sync_watermark(StateScope("child"), StateScope("parent"))
            is None
        )


def _naive(value: datetime) -> datetime:
    # DuckDB's TIMESTAMP round-trips naive, same as `stel_state.last_run_at`
    # elsewhere in this adapter -- the production comparison this feeds never
    # crosses that boundary (both sides of an equality check are read back
    # from the same adapter), so the loss is a test-literal detail, not a bug.
    return value.replace(tzinfo=None)


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    config = parse_warehouse_config(_duckdb_config(tmp_path))
    child, parent = StateScope("child"), StateScope("parent")
    generation = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    with create_adapter(config, project_dir=tmp_path) as adapter:
        adapter.write_sync_watermark(child, parent, generation)
        assert adapter.read_sync_watermark(child, parent) == _naive(generation)


def test_a_second_write_overwrites_rather_than_duplicates(tmp_path: Path) -> None:
    config = parse_warehouse_config(_duckdb_config(tmp_path))
    child, parent = StateScope("child"), StateScope("parent")
    first = datetime(2026, 9, 1, tzinfo=UTC)
    second = datetime(2026, 9, 2, tzinfo=UTC)
    with create_adapter(config, project_dir=tmp_path) as adapter:
        adapter.write_sync_watermark(child, parent, first)
        adapter.write_sync_watermark(child, parent, second)
        assert adapter.read_sync_watermark(child, parent) == _naive(second)
        # Overwritten, not appended: exactly one row for this (child, parent).
        rows = adapter.rows(
            f"SELECT COUNT(*) FROM {adapter.table_ref('stel_sync_watermark')} "
            "WHERE model_name = 'child' AND parent_model_name = 'parent'"
        )
        assert rows == [(1,)]


def test_watermarks_for_different_parents_do_not_collide(tmp_path: Path) -> None:
    """One child can have synced to several parents (a multi-dependency
    transform); each pair gets its own row."""
    config = parse_warehouse_config(_duckdb_config(tmp_path))
    child = StateScope("child")
    parent_a, parent_b = StateScope("parent_a"), StateScope("parent_b")
    gen_a = datetime(2026, 9, 1, tzinfo=UTC)
    gen_b = datetime(2026, 9, 2, tzinfo=UTC)
    with create_adapter(config, project_dir=tmp_path) as adapter:
        adapter.write_sync_watermark(child, parent_a, gen_a)
        adapter.write_sync_watermark(child, parent_b, gen_b)
        assert adapter.read_sync_watermark(child, parent_a) == _naive(gen_a)
        assert adapter.read_sync_watermark(child, parent_b) == _naive(gen_b)


def test_sync_watermark_table_is_hidden_from_list_tables(tmp_path: Path) -> None:
    """Purely internal bookkeeping, like `stel_state` -- never a user-visible
    model in `stel ls`."""
    config = parse_warehouse_config(_duckdb_config(tmp_path))
    with create_adapter(config, project_dir=tmp_path) as adapter:
        adapter.write_sync_watermark(
            StateScope("child"), StateScope("parent"), datetime.now(UTC)
        )
        assert "stel_sync_watermark" not in adapter.list_tables()
