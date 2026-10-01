"""The sync-watermark primitive (issue #611): what a child last synced to.

Two signals, both read fresh every time, never trusted from anything other
than the warehouse itself:

- `state_generation` is cheap (one aggregate query over `stel_state`, the same
  shape as `state_code_version_counts`) and catches every ordinary case,
  including a real deletion -- but it is blind to a write that bypassed stel
  entirely.
- `table_content_fingerprint` is the authoritative, real-cost confirmation: an
  aggregate hash over the parent's actual current rows, which changes under a
  direct `UPDATE`/`DELETE`/`ALTER TABLE` too, because it is re-derived from
  the table every time rather than trusted from anything persisted.

`read_sync_watermark`/`write_sync_watermark` are the new, tiny table
recording both signals together for one (child, parent) pair. All of these
are read-only/best-effort by design: an absent watermark or an absent state
scope resolve to "never", never to an error.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import duckdb

from stel.adapters import (
    StateGeneration,
    StateRecord,
    StateScope,
    SyncWatermark,
    TableContentFingerprint,
    create_adapter,
    parse_warehouse_config,
)
from stel.adapters.duckdb import DuckDBAdapter


def _duckdb_config(tmp_path: Path) -> dict[str, str]:
    return {"type": "duckdb", "path": str(tmp_path / "w.duckdb")}


# ─── state_generation ────────────────────────────────────────────────────────


def test_state_generation_is_none_for_a_scope_with_no_published_state(
    tmp_path: Path,
) -> None:
    config = parse_warehouse_config(_duckdb_config(tmp_path))
    with create_adapter(config, project_dir=tmp_path) as adapter:
        assert adapter.state_generation(StateScope("docs")) is None


def test_state_generation_advances_only_when_this_scope_is_actually_written(
    tmp_path: Path,
) -> None:
    """The generation a watermark compares against: untouched by a scope that
    publishes nothing, exactly like the `last_run_at` column it reads."""
    config = parse_warehouse_config(_duckdb_config(tmp_path))
    scope = StateScope("docs")
    with create_adapter(config, project_dir=tmp_path) as adapter:
        adapter.upsert_state(scope, [StateRecord("a", "fp-a", "v1")])
        first = adapter.state_generation(scope)
        assert first is not None and first.rows == 1

        # A second, unrelated scope's write leaves `docs`'s own generation
        # exactly where it was.
        adapter.upsert_state(StateScope("other"), [StateRecord("z", "fp-z", "v9")])
        assert adapter.state_generation(scope) == first


def test_state_generation_catches_a_pure_deletion_the_timestamp_alone_would_miss(
    tmp_path: Path,
) -> None:
    """Issue #611's own regression: removing a row touches no surviving row's
    `last_run_at`, so a generation signal built from the timestamp alone
    cannot tell a deletion happened. `rows` is what catches it."""
    config = parse_warehouse_config(_duckdb_config(tmp_path))
    scope = StateScope("docs")
    with create_adapter(config, project_dir=tmp_path) as adapter:
        adapter.upsert_state(
            scope, [StateRecord("a", "fp-a", "v1"), StateRecord("b", "fp-b", "v1")]
        )
        before = adapter.state_generation(scope)
        assert before is not None and before.rows == 2

        adapter.delete_state(scope, ["b"])
        after = adapter.state_generation(scope)
        assert after is not None
        assert after.rows == 1
        # The surviving row's own last_run_at is untouched by the deletion --
        # exactly the blind spot a timestamp-only signal would have.
        assert after.last_run_at == before.last_run_at
        assert after != before


# ─── table_content_fingerprint ───────────────────────────────────────────────


def test_content_fingerprint_is_none_for_a_table_that_does_not_exist(
    tmp_path: Path,
) -> None:
    config = parse_warehouse_config(_duckdb_config(tmp_path))
    with create_adapter(config, project_dir=tmp_path) as adapter:
        assert adapter.table_content_fingerprint("nope") is None


def test_content_fingerprint_changes_on_a_direct_update_state_cannot_see(
    tmp_path: Path,
) -> None:
    """The gap `state_generation` alone cannot close: a model's table edited
    directly, with no stel write and so no `stel_state` row touched at all
    (issue #611). This repo's own test suite edits model tables this way
    routinely to simulate scenarios cheaply, so this is not a hypothetical."""
    config = parse_warehouse_config(_duckdb_config(tmp_path))
    with create_adapter(config, project_dir=tmp_path) as base_adapter:
        adapter = cast(DuckDBAdapter, base_adapter)
        docs = adapter.table_ref("docs")
        adapter.connection.execute(f"CREATE TABLE {docs} (a VARCHAR, b INTEGER)")
        adapter.connection.execute(f"INSERT INTO {docs} VALUES ('x', 1), ('y', 2)")
        before = adapter.table_content_fingerprint("docs")
        assert before is not None and before.rows == 2

        adapter.connection.execute(f"UPDATE {docs} SET b = 99 WHERE a = 'x'")
        after_update = adapter.table_content_fingerprint("docs")
        assert after_update is not None
        assert after_update.rows == before.rows
        assert after_update.fingerprint != before.fingerprint

        adapter.connection.execute(f"DELETE FROM {docs} WHERE a = 'y'")
        after_delete = adapter.table_content_fingerprint("docs")
        assert after_delete is not None and after_delete.rows == 1

        adapter.connection.execute(f"ALTER TABLE {docs} ADD COLUMN c VARCHAR")
        after_alter = adapter.table_content_fingerprint("docs")
        assert after_alter is not None
        assert after_alter.fingerprint != after_delete.fingerprint


# ─── the watermark table itself ──────────────────────────────────────────────


def _watermark(rows: int, last_run_at: str, content_rows: int, fingerprint: str) -> SyncWatermark:
    return SyncWatermark(
        state=StateGeneration(rows=rows, last_run_at=last_run_at),
        content=TableContentFingerprint(rows=content_rows, fingerprint=fingerprint),
    )


def test_read_sync_watermark_is_none_before_any_write(tmp_path: Path) -> None:
    config = parse_warehouse_config(_duckdb_config(tmp_path))
    with create_adapter(config, project_dir=tmp_path) as adapter:
        assert (
            adapter.read_sync_watermark(StateScope("child"), StateScope("parent"))
            is None
        )


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    config = parse_warehouse_config(_duckdb_config(tmp_path))
    child, parent = StateScope("child"), StateScope("parent")
    watermark = _watermark(3, "2026-09-01T00:00:00", 3, "fp-1")
    with create_adapter(config, project_dir=tmp_path) as adapter:
        adapter.write_sync_watermark(child, parent, watermark)
        assert adapter.read_sync_watermark(child, parent) == watermark


def test_a_second_write_overwrites_rather_than_duplicates(tmp_path: Path) -> None:
    config = parse_warehouse_config(_duckdb_config(tmp_path))
    child, parent = StateScope("child"), StateScope("parent")
    first = _watermark(1, "2026-09-01T00:00:00", 1, "fp-1")
    second = _watermark(2, "2026-09-02T00:00:00", 2, "fp-2")
    with create_adapter(config, project_dir=tmp_path) as adapter:
        adapter.write_sync_watermark(child, parent, first)
        adapter.write_sync_watermark(child, parent, second)
        assert adapter.read_sync_watermark(child, parent) == second
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
    gen_a = _watermark(1, "2026-09-01T00:00:00", 1, "fp-a")
    gen_b = _watermark(2, "2026-09-02T00:00:00", 2, "fp-b")
    with create_adapter(config, project_dir=tmp_path) as adapter:
        adapter.write_sync_watermark(child, parent_a, gen_a)
        adapter.write_sync_watermark(child, parent_b, gen_b)
        assert adapter.read_sync_watermark(child, parent_a) == gen_a
        assert adapter.read_sync_watermark(child, parent_b) == gen_b


def test_sync_watermark_table_is_hidden_from_list_tables(tmp_path: Path) -> None:
    """Purely internal bookkeeping, like `stel_state` -- never a user-visible
    model in `stel ls`."""
    config = parse_warehouse_config(_duckdb_config(tmp_path))
    with create_adapter(config, project_dir=tmp_path) as adapter:
        adapter.write_sync_watermark(
            StateScope("child"),
            StateScope("parent"),
            _watermark(1, datetime.now(UTC).isoformat(), 1, "fp"),
        )
        assert "stel_sync_watermark" not in adapter.list_tables()


# ─── invalidated by a state-shape migration ──────────────────────────────────


def test_a_v1_state_migration_clears_every_sync_watermark(tmp_path: Path) -> None:
    """A v1 state row's fingerprint was keyed on a document, not necessarily
    the grain v2's `record_key` expects (a chunk model needs one per chunk,
    not one per source document) -- carried over unchanged by the migration,
    verified only by row count (issue #611). A child trusting a stale
    watermark across that shape change could skip the one real scan that
    would notice its migrated fingerprints are wrong, so the migration must
    invalidate every watermark, not just the model it happens to run for."""
    db_path = tmp_path / "w.duckdb"
    config = parse_warehouse_config({"type": "duckdb", "path": str(db_path)})

    # Write a watermark the ordinary way, then downgrade `stel_state` to the
    # v1 shape underneath it -- the same transformation `_migrate_v1_state`
    # itself reverses, so the next connect detects v1 and migrates again.
    with create_adapter(config, project_dir=tmp_path) as adapter:
        child, parent = StateScope("document_chunks"), StateScope("document_registry")
        adapter.upsert_state(parent, [StateRecord("a", "fp-a", "v1")])
        state = adapter.state_generation(parent)
        assert state is not None
        adapter.write_sync_watermark(
            child, parent, _watermark(state.rows, state.last_run_at, 1, "fp")
        )
        assert adapter.read_sync_watermark(child, parent) is not None

    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE stel.stel_state_v1 (
                model_name VARCHAR NOT NULL,
                document_id VARCHAR NOT NULL,
                content_hash VARCHAR NOT NULL,
                code_version VARCHAR NOT NULL,
                last_run_at TIMESTAMP NOT NULL,
                PRIMARY KEY (model_name, document_id)
            )
            """
        )
        con.execute(
            "INSERT INTO stel.stel_state_v1 "
            "SELECT model_name, record_key, input_fingerprint, code_version, last_run_at "
            "FROM stel.stel_state"
        )
        con.execute("DROP TABLE stel.stel_state")
        con.execute("ALTER TABLE stel.stel_state_v1 RENAME TO stel_state")
    finally:
        con.close()

    # Reconnecting detects the v1 shape and migrates it back to v2
    # automatically -- the same path a real `stel run` would take.
    with create_adapter(config, project_dir=tmp_path) as adapter:
        assert adapter.read_sync_watermark(child, parent) is None
