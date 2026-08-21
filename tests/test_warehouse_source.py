"""Warehouse-table document source (issue #322).

Rows of a relation as documents: identity from `key_column`, change detection
from a row fingerprint, `--source-filter` composition through `path_columns`.
The design's promise is that the existing incremental machinery works
unchanged — the end-to-end tests here assert exactly that (skip, re-extract,
prune) over a table instead of a directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import polars as pl
import pytest

from stel.adapters import create_adapter, parse_warehouse_config
from stel.config.profile import WarehouseConfig
from stel.config.source import SourceConfig, validate_relation_name
from stel.sources import DocumentRef, SourceError, get_document_source
from stel.sources.warehouse import WarehouseDocumentSource

# ─── config surface ─────────────────────────────────────────────────────────


def _source(**overrides: object) -> SourceConfig:
    values: dict[str, object] = {
        "name": "reddit_rows",
        "path": "warehouse://rawdata.reddit_posts",
        "key_column": "post_id",
    }
    values.update(overrides)
    return SourceConfig.model_validate(values)


def test_warehouse_source_requires_a_key_column() -> None:
    with pytest.raises(ValueError, match="must declare `key_column:`"):
        _source(key_column=None)


def test_key_column_outside_a_warehouse_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="apply only to warehouse://"):
        _source(path="data/docs")


def test_object_source_knobs_are_rejected_when_set() -> None:
    # Silently ignoring `file_pattern:` on a relation would let a filtering
    # intent pass validation while filtering nothing.
    with pytest.raises(ValueError, match="does not apply"):
        _source(file_pattern="*.json")


def test_path_columns_must_not_repeat_the_key() -> None:
    with pytest.raises(ValueError, match="must not repeat the key column"):
        _source(path_columns=["post_id"])


def test_relation_names_are_validated_per_part() -> None:
    validate_relation_name("table")
    validate_relation_name("schema.table")
    # The leading part may be a BigQuery project (dashes are legal there).
    validate_relation_name("econ-data-project-478800.dataset.table")
    for bad in ("bad-table", "schema.bad-table", "a.b.c.d", "", "sch ema.t"):
        with pytest.raises(ValueError):
            validate_relation_name(bad)


def test_dispatch_without_a_resolved_warehouse_is_a_source_error() -> None:
    with pytest.raises(SourceError, match="resolved warehouse config"):
        get_document_source("warehouse://a.b")


# ─── discovery, identity, fetch ─────────────────────────────────────────────


@pytest.fixture
def warehouse(tmp_path: Path) -> WarehouseConfig:
    db = tmp_path / "wh.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE SCHEMA rawdata")
    con.execute(
        "CREATE TABLE rawdata.reddit_posts AS SELECT * FROM (VALUES "
        "('p1','wallstreetbets','first post text'),"
        "('p2','economics','second post text'),"
        "('p3','wallstreetbets','third post text')"
        ") t(post_id, subreddit, selftext)"
    )
    con.close()
    return parse_warehouse_config(
        {"type": "duckdb", "path": str(db), "schema": "main"}
    )


def _discover(
    warehouse: WarehouseConfig, tmp_path: Path, source: SourceConfig
) -> tuple[WarehouseDocumentSource, list[DocumentRef]]:
    backend = WarehouseDocumentSource(warehouse, tmp_path)
    return backend, backend.discover(source, tmp_path)


def test_rows_become_documents_with_stable_identity(
    warehouse: WarehouseConfig, tmp_path: Path
) -> None:
    source = _source(path_columns=["subreddit"])

    _, first = _discover(warehouse, tmp_path, source)
    _, second = _discover(warehouse, tmp_path, source)

    assert [r.relative_path for r in first] == [
        "wallstreetbets/p1",
        "economics/p2",
        "wallstreetbets/p3",
    ]
    # Identity is deterministic across discoveries — the incremental contract.
    assert [(r.document_id, r.content_hash) for r in first] == [
        (r.document_id, r.content_hash) for r in second
    ]
    assert first[0].source_uri == (
        "warehouse://rawdata.reddit_posts#post_id=p1"
    )


def test_a_changed_row_changes_only_its_own_hash(
    warehouse: WarehouseConfig, tmp_path: Path
) -> None:
    source = _source()
    _, before = _discover(warehouse, tmp_path, source)

    con = duckdb.connect(str(tmp_path / "wh.duckdb"))
    con.execute(
        "UPDATE rawdata.reddit_posts SET selftext = 'edited' "
        "WHERE post_id = 'p2'"
    )
    con.close()
    _, after = _discover(warehouse, tmp_path, source)

    by_id_before = {r.document_id: r.content_hash for r in before}
    by_id_after = {r.document_id: r.content_hash for r in after}
    changed = [
        document_id
        for document_id, content_hash in by_id_after.items()
        if by_id_before[document_id] != content_hash
    ]
    assert len(changed) == 1


def test_fetch_serves_the_discovered_snapshot_as_plain_json(
    warehouse: WarehouseConfig, tmp_path: Path
) -> None:
    backend, refs = _discover(warehouse, tmp_path, _source())

    work_dir = tmp_path / "scratch"
    local = backend.fetch(refs[0], work_dir)

    payload = json.loads(local.read_text(encoding="utf-8"))
    assert payload == {
        "post_id": "p1",
        "subreddit": "wallstreetbets",
        "selftext": "first post text",
    }


def test_fetch_rejects_a_ref_from_another_discovery(
    warehouse: WarehouseConfig, tmp_path: Path
) -> None:
    # The snapshot rule: extraction consumes what discovery saw, never a
    # re-query of a table that may have moved since.
    _, refs = _discover(warehouse, tmp_path, _source())
    fresh_backend = WarehouseDocumentSource(warehouse, tmp_path)

    with pytest.raises(SourceError, match="not part of this run's discovery"):
        fresh_backend.fetch(refs[0], tmp_path / "scratch")


def test_typed_values_render_as_plain_json(tmp_path: Path) -> None:
    db = tmp_path / "typed.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE events AS SELECT 'e1' AS event_id, "
        "TIMESTAMP '2026-03-14 09:30:00' AS at, "
        "DECIMAL '12.30' AS amount"
    )
    con.close()
    warehouse = parse_warehouse_config(
        {"type": "duckdb", "path": str(db), "schema": "main"}
    )
    source = _source(path="warehouse://events", key_column="event_id")

    backend, refs = _discover(warehouse, tmp_path, source)
    payload = json.loads(
        backend.fetch(refs[0], tmp_path / "scratch").read_text(encoding="utf-8")
    )

    assert payload["at"] == "2026-03-14T09:30:00"
    # Decimals become strings at the warehouse's declared scale (DuckDB's
    # bare DECIMAL is (18,3)): a JSON float would silently round them.
    assert payload["amount"] == "12.300"


# ─── malformed relations are loud ───────────────────────────────────────────


def test_null_keys_are_a_hard_error(
    warehouse: WarehouseConfig, tmp_path: Path
) -> None:
    con = duckdb.connect(str(tmp_path / "wh.duckdb"))
    con.execute(
        "INSERT INTO rawdata.reddit_posts VALUES (NULL, 'economics', 'x')"
    )
    con.close()

    with pytest.raises(SourceError, match="null `post_id`"):
        _discover(warehouse, tmp_path, _source())


def test_duplicate_keys_are_a_hard_error(
    warehouse: WarehouseConfig, tmp_path: Path
) -> None:
    con = duckdb.connect(str(tmp_path / "wh.duckdb"))
    con.execute(
        "INSERT INTO rawdata.reddit_posts VALUES ('p1', 'economics', 'dupe')"
    )
    con.close()

    with pytest.raises(SourceError, match="duplicate document path"):
        _discover(warehouse, tmp_path, _source())


def test_missing_columns_fail_before_any_row_is_processed(
    warehouse: WarehouseConfig, tmp_path: Path
) -> None:
    with pytest.raises(SourceError, match="no column"):
        _discover(warehouse, tmp_path, _source(key_column="nonexistent"))


def test_the_row_cap_refuses_rather_than_truncates(
    warehouse: WarehouseConfig, tmp_path: Path
) -> None:
    # Truncation would silently drop documents; the object sources refuse for
    # the same reason.
    with pytest.raises(SourceError, match="more than 2 rows"):
        _discover(warehouse, tmp_path, _source(max_objects=2))


def test_freshness_scan_reports_row_count_and_no_mtime(
    warehouse: WarehouseConfig, tmp_path: Path
) -> None:
    backend = WarehouseDocumentSource(warehouse, tmp_path)

    scan = backend.scan(_source(), tmp_path)

    assert scan.exists
    assert scan.file_count == 3
    assert scan.newest_epoch is None


def test_read_relation_rejects_hostile_names(tmp_path: Path) -> None:
    cfg = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "q.duckdb"), "schema": "main"}
    )
    with create_adapter(cfg) as adapter:
        with pytest.raises(Exception, match="invalid"):
            adapter.read_relation("rawdata.posts; DROP TABLE x")


# ─── end to end: table → extraction → chunk ─────────────────────────────────


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    (project / "models").mkdir(parents=True)
    (project / "sources").mkdir()
    (project / "target").mkdir()

    (project / "stel_project.yml").write_text(
        "name: whsrc\nversion: '0.1.0'\nprofile: whsrc\n"
    )
    (project / "profiles.yml").write_text(
        "whsrc:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n"
        "        schema: main\n"
    )
    (project / "sources" / "src.yml").write_text(
        "version: 2\nsources:\n  - name: reddit_rows\n"
        "    path: warehouse://rawdata.reddit_posts\n"
        "    key_column: post_id\n    path_columns: [subreddit]\n"
    )
    (project / "models" / "models.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: post_registry\n"
        "    source: ref('reddit_rows')\n"
        "    extraction:\n      backend: json\n      options:\n"
        "        fields: [post_id, subreddit, selftext]\n"
        "    materialization: incremental\n"
    )
    con = duckdb.connect(str(project / "target" / "db.duckdb"))
    con.execute("CREATE SCHEMA rawdata")
    con.execute(
        "CREATE TABLE rawdata.reddit_posts AS SELECT * FROM (VALUES "
        "('p1','wallstreetbets','first post text'),"
        "('p2','economics','second post text'),"
        "('p3','wallstreetbets','third post text')"
        ") t(post_id, subreddit, selftext)"
    )
    con.close()
    return project


def _registry(project: Path) -> pl.DataFrame:
    con = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        return pl.DataFrame(
            con.execute(
                "SELECT post_id, subreddit, selftext, source_uri "
                'FROM "db".main.post_registry ORDER BY post_id'
            ).pl()
        )
    finally:
        con.close()


def test_table_rows_flow_through_extraction(tmp_path: Path) -> None:
    from stel.runner import run_project

    project = _project(tmp_path)

    results = {r.model_name: r for r in run_project(project)}

    assert results["post_registry"].documents_processed == 3
    registry = _registry(project)
    assert registry.height == 3
    assert registry["source_uri"].to_list()[0].startswith(
        "warehouse://rawdata.reddit_posts#post_id="
    )


def test_incremental_contract_over_rows(tmp_path: Path) -> None:
    """The design's promise: skip, re-extract, prune — machinery unchanged."""
    from stel.runner import run_project

    project = _project(tmp_path)
    run_project(project)

    # Unchanged table: everything skips.
    rerun = {r.model_name: r for r in run_project(project)}
    assert rerun["post_registry"].documents_skipped == 3
    assert rerun["post_registry"].documents_processed == 0

    # One row edited, one deleted: one re-extracts, one prunes.
    con = duckdb.connect(str(project / "target" / "db.duckdb"))
    con.execute(
        "UPDATE rawdata.reddit_posts SET selftext = 'edited' "
        "WHERE post_id = 'p2'"
    )
    con.execute("DELETE FROM rawdata.reddit_posts WHERE post_id = 'p3'")
    con.close()

    third = {r.model_name: r for r in run_project(project)}
    assert third["post_registry"].documents_processed == 1
    assert third["post_registry"].documents_skipped == 1
    assert third["post_registry"].documents_deleted == 1
    assert _registry(project).height == 2


def test_source_filter_scopes_rows_by_path_columns(tmp_path: Path) -> None:
    """`--source-filter 'wallstreetbets/*'` addresses a row subset the same
    way it addresses an object prefix — the partition seam for orchestrators
    (astrolabe #167 shape) with no new concept."""
    from stel.runner import run_project

    project = _project(tmp_path)

    results = {
        r.model_name: r
        for r in run_project(project, source_filter=["wallstreetbets/*"])
    }

    assert results["post_registry"].documents_processed == 2
    assert set(_registry(project)["subreddit"].to_list()) == {"wallstreetbets"}
