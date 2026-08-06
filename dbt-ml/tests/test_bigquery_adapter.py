"""BigQuery adapter (issue #83): config, SQL dialect, load planning, and
client interactions against a fake; real round-trips run only when
DBT_ML_BQ_TEST_PROJECT is set."""
from __future__ import annotations

import io
import os
import pickle
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from dbt_ml.adapters import (
    AdapterError,
    StaleStateFenceError,
    StateAbsenceProbe,
    StateRecord,
    StateScope,
    StateScopeFence,
    StateValue,
    WarehouseCapability,
    adapter_capabilities,
    create_adapter,
    list_adapter_types,
    parse_warehouse_config,
)
from dbt_ml.adapters.bigquery import (
    BigQueryAdapter,
    BigQueryWarehouseConfig,
    BigQueryWarehouseOptions,
    to_query_parameters,
)
from dbt_ml.credentials import ProtectedCredential

# ─── config ─────────────────────────────────────────────────────────────────


def test_bigquery_registered() -> None:
    assert "bigquery" in list_adapter_types()


def test_bigquery_declares_only_implemented_guarantees() -> None:
    capabilities = adapter_capabilities("bigquery")

    assert WarehouseCapability.TABULAR_READS in capabilities
    assert WarehouseCapability.SQL_SCHEMA_TESTS in capabilities
    assert WarehouseCapability.STREAMING_TABULAR_READS in capabilities
    assert WarehouseCapability.TABULAR_PREDICATE_PUSHDOWN in capabilities
    assert WarehouseCapability.ATOMIC_FULL_REPLACE in capabilities
    assert WarehouseCapability.ATOMIC_KEYED_UPSERT in capabilities
    assert WarehouseCapability.TRANSACTIONS not in capabilities


def test_config_dataset_alias() -> None:
    cfg = parse_warehouse_config(
        {"type": "bigquery", "project": "my-proj", "dataset": "docs"}
    )
    assert isinstance(cfg, BigQueryWarehouseConfig)
    assert cfg.schema_name == "docs"


def test_config_schema_alias_and_defaults() -> None:
    cfg = parse_warehouse_config({"type": "bigquery", "project": "my-proj"})
    assert isinstance(cfg, BigQueryWarehouseConfig)
    assert cfg.schema_name == "dbt_ml"
    assert cfg.location is None
    assert cfg.keyfile is None
    assert cfg.catalog_name() == "my-proj"
    assert cfg.storage_location() == "my-proj.dbt_ml"


def test_config_requires_project() -> None:
    with pytest.raises(AdapterError, match="bigquery"):
        parse_warehouse_config({"type": "bigquery", "dataset": "docs"})


def test_config_rejects_unknown_field() -> None:
    with pytest.raises(AdapterError, match="datset_typo"):
        parse_warehouse_config(
            {"type": "bigquery", "project": "p", "datset_typo": "oops"}
        )


def test_keyfile_absolutized_relative_to_project(tmp_path: Path) -> None:
    cfg = parse_warehouse_config(
        {"type": "bigquery", "project": "p", "keyfile": "./secrets/sa.json"}
    )
    assert isinstance(cfg, BigQueryWarehouseConfig)
    resolved = cfg.absolutize(tmp_path)
    assert resolved.keyfile == (tmp_path / "secrets" / "sa.json").resolve()


# ─── SQL dialect ────────────────────────────────────────────────────────────


def _adapter(client: Any = None, **cfg_extra: Any) -> BigQueryAdapter:
    cfg = parse_warehouse_config(
        {"type": "bigquery", "project": "proj", "dataset": "ds", **cfg_extra}
    )
    adapter = create_adapter(cfg)
    assert isinstance(adapter, BigQueryAdapter)
    if client is not None:
        adapter._client = client
    return adapter


def test_quote_ident_backticks() -> None:
    adapter = _adapter()
    assert adapter.quote_ident("order") == "`order`"
    assert adapter.quote_ident("has`tick") == "`has\\`tick`"


def test_table_ref_fully_qualified() -> None:
    # catalog comes from config, so no connection is needed
    assert _adapter().table_ref("chunks") == "`proj`.`ds`.`chunks`"


def test_query_parameters_type_inference() -> None:
    params = to_query_parameters(["s", 3, 2.5, True, datetime.now(UTC), ["a", "b"], []])
    types = [getattr(p, "type_", None) or p.array_type for p in params]
    assert types == ["STRING", "INT64", "FLOAT64", "BOOL", "TIMESTAMP", "STRING", "STRING"]
    assert params[5].values == ["a", "b"]


# The schema-change planner is warehouse-independent; its contract now lives in
# tests/test_adapter_invariants.py (issue #190, Workstream C).


# ─── client interactions against a fake ─────────────────────────────────────


class _FakeRow(tuple[Any, ...]):
    def values(self) -> tuple[Any, ...]:
        return tuple(self)


class _FakeJob:
    def __init__(self, rows: list[tuple[Any, ...]] | None = None, affected: int | None = None):
        self._rows = [_FakeRow(r) for r in (rows or [])]
        self.num_dml_affected_rows = affected
        self.result_timeout: Any = "unset"

    def result(self, timeout: Any = None) -> list[_FakeRow]:
        self.result_timeout = timeout
        return list(self._rows)


class _FailingJob(_FakeJob):
    def result(self, timeout: Any = None) -> list[_FakeRow]:
        raise RuntimeError("simulated merge failure")


_STATE_V1_SCHEMA = [
    SimpleNamespace(name=name, field_type=field_type, mode="REQUIRED")
    for name, field_type in (
        ("model_name", "STRING"),
        ("document_id", "STRING"),
        ("content_hash", "STRING"),
        ("code_version", "STRING"),
        ("last_run_at", "TIMESTAMP"),
    )
]
_STATE_V2_SCHEMA = [
    SimpleNamespace(name=name, field_type=field_type, mode="REQUIRED")
    for name, field_type in (
        ("model_name", "STRING"),
        ("state_scope", "STRING"),
        ("target_identity", "STRING"),
        ("record_key", "STRING"),
        ("input_fingerprint", "STRING"),
        ("code_version", "STRING"),
        ("last_run_at", "TIMESTAMP"),
    )
]


class _FakeClient:
    def __init__(self) -> None:
        self.queries: list[tuple[str, Any]] = []
        self.query_kwargs: list[dict[str, Any]] = []
        self.loads: list[tuple[bytes, str, Any]] = []
        self.tables: dict[str, list[Any]] = {}
        self.table_meta: dict[str, dict[str, Any]] = {}
        self.listing: list[str] = []
        self.dropped: list[str] = []
        self.query_results: list[_FakeJob] = []

    def query(self, sql: str, job_config: Any = None, **kwargs: Any) -> _FakeJob:
        self.queries.append((sql, job_config))
        self.query_kwargs.append(kwargs)
        return self.query_results.pop(0) if self.query_results else _FakeJob()

    def load_table_from_file(
        self, fobj: io.BytesIO, table_id: str, job_config: Any = None
    ) -> _FakeJob:
        self.loads.append((fobj.read(), table_id, job_config))
        return _FakeJob()

    def get_table(self, table_id: str) -> Any:
        from google.api_core.exceptions import NotFound

        if table_id not in self.tables:
            raise NotFound(table_id)
        schema = [
            field
            if not isinstance(field, str)
            else SimpleNamespace(name=field, field_type="STRING", mode="NULLABLE")
            for field in self.tables[table_id]
        ]
        return SimpleNamespace(schema=schema, **self.table_meta.get(table_id, {}))

    def list_tables(self, dataset_id: str) -> list[Any]:
        return [SimpleNamespace(table_id=n) for n in self.listing]

    def delete_table(self, table_id: str, not_found_ok: bool = False) -> None:
        self.dropped.append(table_id)

    def close(self) -> None:
        pass


def test_materialize_full_truncating_parquet_load() -> None:
    from google.cloud import bigquery

    client = _FakeClient()
    adapter = _adapter(client)
    df = pl.DataFrame({"document_id": ["a", "b"], "x": [1, 2]})
    assert adapter.materialize_full("docs", df) == 2

    payload, table_id, job_config = client.loads[0]
    assert table_id == "proj.ds.docs"
    assert job_config.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE
    assert pl.read_parquet(io.BytesIO(payload)).rows() == [("a", 1), ("b", 2)]


def test_parquet_load_enables_list_inference() -> None:
    # Without list inference BigQuery loads a Parquet LIST as a nested RECORD
    # instead of ARRAY<T>, which breaks the embed→search vector contract
    # (issue #226). Every parquet load must opt in.
    client = _FakeClient()
    adapter = _adapter(client)
    df = pl.DataFrame(
        {"document_id": ["a"], "embedding": [[0.1, 0.2, 0.3]]},
        schema={"document_id": pl.Utf8, "embedding": pl.List(pl.Float64)},
    )
    assert adapter.materialize_full("chunks", df) == 1

    _, _, job_config = client.loads[0]
    assert job_config.parquet_options is not None
    assert job_config.parquet_options.enable_list_inference is True


def test_incremental_first_load_creates_table() -> None:
    client = _FakeClient()  # get_table -> NotFound
    adapter = _adapter(client)
    df = pl.DataFrame({"document_id": ["a"], "x": [1]})
    assert adapter.materialize_incremental("docs", df, key_col="document_id") == 1

    assert client.queries == []  # no DELETE against a table that doesn't exist
    _, _, job_config = client.loads[0]
    assert job_config.write_disposition == "WRITE_APPEND"


def test_incremental_upsert_uses_staging_merge() -> None:
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id", "x"]
    adapter = _adapter(client)
    df = pl.DataFrame({"document_id": ["a", "b"], "x": [1, 2]})
    adapter.materialize_incremental("docs", df, key_col="document_id")

    payload, staging_id, job_config = client.loads[0]
    assert staging_id.startswith("proj.ds.dbt_ml_staging__docs__")
    assert job_config.write_disposition == "WRITE_TRUNCATE"
    assert pl.read_parquet(io.BytesIO(payload)).rows() == [("a", 1), ("b", 2)]

    sql, query_config = client.queries[0]
    assert sql.startswith("MERGE `proj`.`ds`.`docs` AS target")
    assert "WHEN MATCHED THEN UPDATE SET" in sql
    assert "WHEN NOT MATCHED THEN INSERT" in sql
    assert "DELETE FROM" not in sql
    assert query_config is None
    assert len(client.loads) == 1
    assert client.dropped == [staging_id]


def test_incremental_append_new_columns_sets_schema_update() -> None:
    from google.cloud import bigquery

    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id"]
    adapter = _adapter(client)
    df = pl.DataFrame({"document_id": ["a"], "extra": ["new"]})
    adapter.materialize_incremental(
        "docs", df, key_col="document_id", on_schema_change="append_new_columns"
    )
    schema_payload, table_id, schema_config = client.loads[0]
    assert table_id == "proj.ds.docs"
    assert pl.read_parquet(io.BytesIO(schema_payload)).height == 0
    assert schema_config.schema_update_options == [
        bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
    ]
    data_payload, staging_id, staging_config = client.loads[1]
    assert staging_id.startswith("proj.ds.dbt_ml_staging__docs__")
    assert staging_config.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE
    assert pl.read_parquet(io.BytesIO(data_payload)).rows() == [("a", "new")]


@pytest.mark.parametrize(
    ("df", "message"),
    [
        (pl.DataFrame({"x": [1]}), "missing required key"),
        (pl.DataFrame({"document_id": [None], "x": [1]}), "contains 1 NULL"),
        (
            pl.DataFrame({"document_id": ["a", "a"], "x": [1, 2]}),
            "contains 1 duplicate",
        ),
    ],
)
def test_incremental_rejects_invalid_keys(df: pl.DataFrame, message: str) -> None:
    client = _FakeClient()
    adapter = _adapter(client)
    with pytest.raises(AdapterError, match=message):
        adapter.materialize_incremental("docs", df, key_col="document_id")
    assert client.loads == []
    assert client.queries == []


def test_incremental_rejects_target_without_key() -> None:
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["x"]
    adapter = _adapter(client)
    with pytest.raises(AdapterError, match=r"target.*missing key"):
        adapter.materialize_incremental(
            "docs",
            pl.DataFrame({"document_id": ["a"], "x": [1]}),
            key_col="document_id",
        )
    assert client.loads == []
    assert client.queries == []


def test_incremental_merge_failure_keeps_target_and_cleans_staging() -> None:
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id", "x"]
    client.query_results = [_FailingJob()]
    adapter = _adapter(client)

    with pytest.raises(RuntimeError, match="simulated merge failure"):
        adapter.materialize_incremental(
            "docs",
            pl.DataFrame({"document_id": ["a"], "x": [99]}),
            key_col="document_id",
        )

    sql, _ = client.queries[0]
    assert sql.startswith("MERGE")
    assert "DELETE FROM" not in sql
    assert len(client.dropped) == 1
    assert client.dropped[0].startswith("proj.ds.dbt_ml_staging__docs__")


def test_incremental_fail_policy_raises_before_writing() -> None:
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id"]
    adapter = _adapter(client)
    df = pl.DataFrame({"document_id": ["a"], "extra": ["new"]})
    with pytest.raises(AdapterError, match="full-refresh"):
        adapter.materialize_incremental("docs", df, key_col="document_id")
    assert client.loads == []
    assert client.queries == []


def test_delete_rows_reports_affected() -> None:
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id"]
    client.query_results = [_FakeJob(affected=2)]
    adapter = _adapter(client)
    assert adapter.delete_rows("docs", key_col="document_id", keys=["a", "b"]) == 2


def test_delete_rows_missing_table_is_noop() -> None:
    client = _FakeClient()
    adapter = _adapter(client)
    assert adapter.delete_rows("gone", key_col="document_id", keys=["a"]) == 0
    assert client.queries == []


def test_state_table_create_uses_v2_schema() -> None:
    client = _FakeClient()
    adapter = _adapter(client)

    adapter._ensure_state_table()

    assert len(client.queries) == 1
    sql, _ = client.queries[0]
    assert "CREATE TABLE IF NOT EXISTS `proj`.`ds`.`dbt_ml_state`" in sql
    assert "state_scope STRING NOT NULL" in sql
    assert "target_identity STRING NOT NULL" in sql
    assert "record_key STRING NOT NULL" in sql
    assert "input_fingerprint STRING NOT NULL" in sql
    assert "document_id" not in sql
    assert "content_hash" not in sql


def test_state_table_v2_schema_is_an_exact_noop() -> None:
    client = _FakeClient()
    client.tables["proj.ds.dbt_ml_state"] = list(_STATE_V2_SCHEMA)
    adapter = _adapter(client)

    adapter._ensure_state_table()

    assert client.queries == []
    assert client.dropped == []


def test_state_table_rejects_unrecognized_schema_without_mutation() -> None:
    client = _FakeClient()
    client.tables["proj.ds.dbt_ml_state"] = [
        *_STATE_V2_SCHEMA,
        SimpleNamespace(name="unexpected", field_type="STRING", mode="NULLABLE"),
    ]
    adapter = _adapter(client)

    with pytest.raises(AdapterError, match="Unsupported dbt_ml_state schema"):
        adapter._ensure_state_table()

    assert client.queries == []
    assert client.dropped == []


def test_state_table_migrates_legacy_rows_through_verified_copy() -> None:
    client = _FakeClient()
    client.tables["proj.ds.dbt_ml_state"] = list(_STATE_V1_SCHEMA)
    protected_table_ids = {
        "proj.ds.dbt_ml_staging__state_migration_v2",
        "proj.ds.dbt_ml_staging__state_migration_v2__user_owned",
    }
    for table_id in protected_table_ids:
        client.tables[table_id] = ["user_data"]
    client.query_results = [
        _FakeJob(rows=[(0,)]),
        _FakeJob(),
        _FakeJob(rows=[(2, 2)]),
        _FakeJob(),
    ]
    adapter = _adapter(client)

    adapter._ensure_state_table()

    assert len(client.queries) == 4
    duplicate_sql = client.queries[0][0]
    assert "GROUP BY model_name, document_id HAVING COUNT(*) > 1" in duplicate_sql
    stage_sql = client.queries[1][0]
    assert len(client.dropped) == 1
    migration_id = client.dropped[0]
    migration_table = migration_id.removeprefix("proj.ds.")
    assert migration_table.startswith("dbt_ml_staging__state_migration_v2__")
    suffix = migration_table.removeprefix("dbt_ml_staging__state_migration_v2__")
    assert len(suffix) == 32
    assert set(suffix) <= set("0123456789abcdef")
    migration_ref = f"`proj`.`ds`.`{migration_table}`"
    assert f"CREATE TABLE {migration_ref}" in stage_sql
    assert "CREATE OR REPLACE" not in stage_sql
    assert "IF NOT EXISTS" not in stage_sql
    assert "model_name AS model_name" in stage_sql
    assert "'materialization' AS state_scope" in stage_sql
    assert "'warehouse-v1' AS target_identity" in stage_sql
    assert "document_id AS record_key" in stage_sql
    assert "content_hash AS input_fingerprint" in stage_sql
    assert "code_version AS code_version" in stage_sql
    assert "last_run_at AS last_run_at" in stage_sql
    count_sql = client.queries[2][0]
    assert count_sql.count("SELECT COUNT(*)") == 2
    assert migration_ref in count_sql
    copy_sql = client.queries[3][0]
    assert copy_sql.startswith("CREATE OR REPLACE TABLE `proj`.`ds`.`dbt_ml_state` COPY")
    assert copy_sql.endswith(f"COPY {migration_ref}")
    assert protected_table_ids.isdisjoint(client.dropped)

    client.tables["proj.ds.dbt_ml_state"] = list(_STATE_V2_SCHEMA)
    adapter._ensure_state_table()
    assert len(client.queries) == 4


def test_state_table_migration_count_mismatch_keeps_legacy_table() -> None:
    client = _FakeClient()
    client.tables["proj.ds.dbt_ml_state"] = list(_STATE_V1_SCHEMA)
    client.query_results = [
        _FakeJob(rows=[(0,)]),
        _FakeJob(),
        _FakeJob(rows=[(2, 1)]),
    ]
    adapter = _adapter(client)

    with pytest.raises(AdapterError, match="row-count verification failed"):
        adapter._ensure_state_table()

    assert len(client.queries) == 3
    assert all(" COPY " not in sql for sql, _ in client.queries)
    assert len(client.dropped) == 1
    assert client.dropped[0].startswith(
        "proj.ds.dbt_ml_staging__state_migration_v2__"
    )


def test_state_table_migration_rejects_duplicate_legacy_keys_before_writes() -> None:
    client = _FakeClient()
    client.tables["proj.ds.dbt_ml_state"] = list(_STATE_V1_SCHEMA)
    client.query_results = [_FakeJob(rows=[(1,)])]
    adapter = _adapter(client)

    with pytest.raises(AdapterError, match="keys contain duplicates"):
        adapter._ensure_state_table()

    assert len(client.queries) == 1
    assert client.queries[0][0].lstrip().startswith("SELECT COUNT(*)")
    assert client.dropped == []


def test_state_table_migration_collision_never_deletes_existing_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collision_hex = "a" * 32
    collision_id = (
        "proj.ds.dbt_ml_staging__state_migration_v2__" + collision_hex
    )
    client = _FakeClient()
    client.tables["proj.ds.dbt_ml_state"] = list(_STATE_V1_SCHEMA)
    client.tables[collision_id] = ["user_data"]
    client.query_results = [_FakeJob(rows=[(0,)]), _FailingJob()]
    adapter = _adapter(client)
    monkeypatch.setattr(
        "dbt_ml.adapters.bigquery.uuid4",
        lambda: SimpleNamespace(hex=collision_hex),
    )

    with pytest.raises(RuntimeError, match="simulated merge failure"):
        adapter._ensure_state_table()

    assert collision_id in client.tables
    assert client.dropped == []
    create_sql = client.queries[1][0]
    assert f"CREATE TABLE `proj`.`ds`.`{collision_id.removeprefix('proj.ds.')}`" in create_sql
    assert "CREATE OR REPLACE" not in create_sql
    assert "IF NOT EXISTS" not in create_sql


def test_state_upsert_is_single_scoped_merge() -> None:
    client = _FakeClient()
    adapter = _adapter(client)
    scope = StateScope("m1", stage="publication", target_identity="lancedb:one")
    adapter.upsert_state(
        scope,
        [
            StateRecord("chunk-1", "h1", "v1"),
            StateRecord("chunk-2", "h2", "v2"),
        ],
    )

    sql, job_config = client.queries[0]
    assert sql.strip().startswith("MERGE")
    assert "OFFSET" in sql
    assert "target.state_scope = source.state_scope" in sql
    assert "target.target_identity = source.target_identity" in sql
    assert "target.record_key = source.record_key" in sql
    assert "WHEN NOT MATCHED BY SOURCE" not in sql
    params = job_config.query_parameters
    assert params[0].value == "m1"
    assert params[1].value == "publication"
    assert params[2].value == "lancedb:one"
    assert params[3].values == ["chunk-1", "chunk-2"]
    assert params[4].values == ["h1", "h2"]


def test_replace_state_is_one_scoped_merge_for_empty_snapshot() -> None:
    client = _FakeClient()
    adapter = _adapter(client)
    scope = StateScope("m1", stage="publication", target_identity="target-b")

    adapter.replace_state(scope, [])

    assert len(client.queries) == 1
    sql, job_config = client.queries[0]
    assert "WHEN NOT MATCHED BY SOURCE" in sql
    assert "target.model_name = ?" in sql
    assert "target.state_scope = ?" in sql
    assert "target.target_identity = ?" in sql
    params = job_config.query_parameters
    assert params[3].values == []
    assert params[4].values == []
    assert params[5].values == []
    assert [param.value for param in params[6:]] == [
        "m1",
        "publication",
        "target-b",
    ]


def test_state_upsert_stages_and_merges_at_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Above the inline threshold the record set must NOT be passed as array query
    # params (issue #256): it is staged via bounded Parquet loads and merged from
    # the staging table in one statement.
    monkeypatch.setattr("dbt_ml.adapters.bigquery._STATE_MERGE_INLINE_MAX", 3)
    monkeypatch.setattr("dbt_ml.adapters.bigquery._STATE_MERGE_LOAD_BATCH", 2)
    client = _FakeClient()
    adapter = _adapter(client)
    scope = StateScope("m1", stage="publication", target_identity="t")

    records = [StateRecord(f"chunk-{i}", f"h{i}", "v1") for i in range(5)]
    adapter.upsert_state(scope, records)

    # 5 records at load batch 2 -> 3 loads, all into one staging table.
    assert len(client.loads) == 3
    staging_ids = {table_id for _, table_id, _ in client.loads}
    assert len(staging_ids) == 1
    assert "dbt_ml_staging__state_merge__" in next(iter(staging_ids))

    merge_sqls = [sql for sql, _ in client.queries if sql.strip().startswith("MERGE")]
    assert len(merge_sqls) == 1
    merge_sql, merge_config = next(
        q for q in client.queries if q[0].strip().startswith("MERGE")
    )
    assert "dbt_ml_staging__state_merge__" in merge_sql  # reads the staging table
    assert "OFFSET" not in merge_sql                     # not the inline array form
    assert "WHEN NOT MATCHED BY SOURCE" not in merge_sql  # upsert: no delete
    # No array parameters — only the three scope scalars ride inline.
    from google.cloud import bigquery

    assert not any(
        isinstance(p, bigquery.ArrayQueryParameter)
        for p in merge_config.query_parameters
    )
    # Staging table is cleaned up.
    assert any("dbt_ml_staging__state_merge__" in t for t in client.dropped)


def test_state_replace_stages_with_delete_at_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dbt_ml.adapters.bigquery._STATE_MERGE_INLINE_MAX", 3)
    client = _FakeClient()
    adapter = _adapter(client)
    scope = StateScope("m1", stage="publication", target_identity="t")

    adapter.replace_state(scope, [StateRecord(f"c{i}", f"h{i}", "v1") for i in range(5)])

    merge_sql = next(
        sql for sql, _ in client.queries if sql.strip().startswith("MERGE")
    )
    assert "dbt_ml_staging__state_merge__" in merge_sql
    assert "OFFSET" not in merge_sql
    # Replace still deletes rows absent from the staged snapshot, atomically.
    assert "WHEN NOT MATCHED BY SOURCE" in merge_sql
    assert any("dbt_ml_staging__state_merge__" in t for t in client.dropped)


def test_fetch_state_round_trip_shape() -> None:
    client = _FakeClient()
    client.query_results = [_FakeJob(rows=[("chunk-1", "h1", "v1"), ("chunk-2", "h2", "v2")])]
    adapter = _adapter(client)
    scope = StateScope("m1", stage="publication", target_identity="target-a")

    assert adapter.fetch_state(scope) == {
        "chunk-1": StateValue("h1", "v1"),
        "chunk-2": StateValue("h2", "v2"),
    }
    sql, job_config = client.queries[0]
    assert "record_key, input_fingerprint, code_version" in sql
    assert "state_scope = ? AND target_identity = ?" in sql
    assert [param.value for param in job_config.query_parameters] == [
        "m1",
        "publication",
        "target-a",
    ]


def test_fetch_state_rejects_duplicate_keys_without_disclosing_them() -> None:
    secret_key = "sensitive-record-key"
    client = _FakeClient()
    client.query_results = [
        _FakeJob(
            rows=[
                (secret_key, "h1", "v1"),
                (secret_key, "h2", "v2"),
            ]
        )
    ]
    adapter = _adapter(client)

    with pytest.raises(AdapterError, match="duplicate record keys") as error:
        adapter.fetch_state(StateScope("m1"))

    assert secret_key not in str(error.value)
    assert "h1" not in str(error.value)
    assert "h2" not in str(error.value)


def test_state_delete_and_clear_are_exactly_scoped() -> None:
    client = _FakeClient()
    adapter = _adapter(client)
    scope = StateScope("m1", stage="publication", target_identity="target-a")

    adapter.delete_state(scope, ["chunk-1", "chunk-2"])
    adapter.clear_state(scope)

    delete_sql, delete_config = client.queries[0]
    clear_sql, clear_config = client.queries[1]
    for sql in (delete_sql, clear_sql):
        assert "model_name = ? AND state_scope = ? AND target_identity = ?" in sql
    delete_params = delete_config.query_parameters
    assert [param.value for param in delete_params[:3]] == [
        "m1",
        "publication",
        "target-a",
    ]
    assert delete_params[3].values == ["chunk-1", "chunk-2"]
    assert [param.value for param in clear_config.query_parameters] == [
        "m1",
        "publication",
        "target-a",
    ]


def test_delete_rows_and_state_uses_one_scoped_transaction() -> None:
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id"]
    client.query_results = [_FakeJob(rows=[(2,)])]
    adapter = _adapter(client)
    scope = StateScope("chunks", stage="publication", target_identity="target-a")

    deleted = adapter.delete_rows_and_state(
        "docs",
        key_col="document_id",
        keys=["doc-1"],
        state_scope=scope,
        state_record_keys=["chunk-1", "chunk-2"],
    )

    assert deleted == 2
    assert len(client.queries) == 1
    sql, job_config = client.queries[0]
    assert "BEGIN TRANSACTION;" in sql
    assert "COMMIT TRANSACTION;" in sql
    assert "DELETE FROM `proj`.`ds`.`docs`" in sql
    assert "DELETE FROM `proj`.`ds`.`dbt_ml_state`" in sql
    assert "model_name = ? AND state_scope = ? AND target_identity = ?" in sql
    params = job_config.query_parameters
    assert params[0].values == ["doc-1"]
    assert [param.value for param in params[1:4]] == [
        "chunks",
        "publication",
        "target-a",
    ]
    assert params[4].values == ["chunk-1", "chunk-2"]


def test_list_tables_filters_internal() -> None:
    client = _FakeClient()
    client.listing = ["docs", "dbt_ml_state", "dbt_ml_test_failures__docs__not_null"]
    adapter = _adapter(client)
    assert adapter.list_tables() == ["docs"]


# ─── model-level warehouse_options (issue #91) ──────────────────────────────


def _parse_options(payload: dict[str, Any]) -> Any:
    return _adapter().parse_warehouse_options(payload, model_name="filings")


def test_warehouse_options_partition_and_cluster_parse() -> None:
    opts = _parse_options(
        {
            "partition_by": {"field": "filing_date", "granularity": "day"},
            "cluster_by": ["cik", "form_type"],
        }
    )
    assert opts.partition_by.field == "filing_date"
    assert opts.partition_by.data_type == "date"
    assert opts.cluster_by == ["cik", "form_type"]


def test_warehouse_options_single_cluster_column_string() -> None:
    assert _parse_options({"cluster_by": "cik"}).cluster_by == ["cik"]


def test_warehouse_options_reject_unknown_key() -> None:
    with pytest.raises(AdapterError, match="partiton_by"):
        _parse_options({"partiton_by": {"field": "d"}})


def test_warehouse_options_int64_requires_range() -> None:
    with pytest.raises(AdapterError, match="range"):
        _parse_options({"partition_by": {"field": "bucket", "data_type": "int64"}})


def test_warehouse_options_range_only_for_int64() -> None:
    with pytest.raises(AdapterError, match="int64"):
        _parse_options(
            {
                "partition_by": {
                    "field": "filing_date",
                    "range": {"start": 0, "end": 10, "interval": 1},
                }
            }
        )


def test_warehouse_options_date_hour_rejected() -> None:
    with pytest.raises(AdapterError, match="hour"):
        _parse_options({"partition_by": {"field": "d", "granularity": "hour"}})


def test_warehouse_options_cluster_limit_four() -> None:
    with pytest.raises(AdapterError, match="cluster_by"):
        _parse_options({"cluster_by": ["a", "b", "c", "d", "e"]})


def test_partition_expression_ddl_forms() -> None:
    adapter = _adapter()

    def expr(**payload: Any) -> str:
        from dbt_ml.adapters.bigquery import BigQueryPartitionBy

        return adapter._partition_expression(BigQueryPartitionBy(**payload))

    assert expr(field="d") == "`d`"
    assert expr(field="d", granularity="month") == "DATE_TRUNC(`d`, MONTH)"
    assert (
        expr(field="ts", data_type="timestamp", granularity="hour")
        == "TIMESTAMP_TRUNC(`ts`, HOUR)"
    )
    assert (
        expr(field="dt", data_type="datetime", granularity="year")
        == "DATETIME_TRUNC(`dt`, YEAR)"
    )
    assert (
        expr(field="n", data_type="int64", range={"start": 0, "end": 100, "interval": 10})
        == "RANGE_BUCKET(`n`, GENERATE_ARRAY(0, 100, 10))"
    )
    assert expr() == "_PARTITIONDATE"
    assert expr(granularity="month", data_type="timestamp") == (
        "TIMESTAMP_TRUNC(_PARTITIONTIME, MONTH)"
    )


def test_materialize_full_applies_layout_and_replaces_atomically() -> None:
    from google.cloud import bigquery

    client = _FakeClient()
    adapter = _adapter(client)
    opts = _parse_options(
        {
            "partition_by": {"field": "filing_date"},
            "cluster_by": ["cik"],
        }
    )
    df = pl.DataFrame({"document_id": ["a"], "filing_date": ["2026-01-01"]})
    adapter.materialize_full("docs", df, options=opts)

    # Replacement is staged with the layout (validating it), then one
    # CREATE OR REPLACE swaps it in atomically — the target never disappears.
    _, staging_id, job_config = client.loads[0]
    assert staging_id.startswith("proj.ds.dbt_ml_staging__docs__")
    assert job_config.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE
    assert job_config.time_partitioning.type_ == "DAY"
    assert job_config.time_partitioning.field == "filing_date"
    assert job_config.clustering_fields == ["cik"]
    swap_sql = client.queries[-1][0]
    staging_table = staging_id.removeprefix("proj.ds.")
    assert swap_sql == (
        "CREATE OR REPLACE TABLE `proj`.`ds`.`docs` "
        "PARTITION BY `filing_date` CLUSTER BY `cik` "
        f"AS SELECT * FROM `proj`.`ds`.`{staging_table}`"
    )
    assert client.dropped == [staging_id]  # only staging cleans up


def test_materialize_full_matching_partition_spec_replaces_atomically() -> None:
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id", "filing_date"]
    client.table_meta["proj.ds.docs"] = {
        "time_partitioning": SimpleNamespace(type_="DAY", field="filing_date")
    }
    adapter = _adapter(client)
    opts = _parse_options({"partition_by": {"field": "filing_date"}})
    df = pl.DataFrame({"document_id": ["a"], "filing_date": ["2026-01-01"]})
    adapter.materialize_full("docs", df, options=opts)

    swap_sql = client.queries[-1][0]
    assert swap_sql.startswith("CREATE OR REPLACE TABLE `proj`.`ds`.`docs` ")
    assert "proj.ds.docs" not in client.dropped


def test_materialize_full_partition_migration_falls_back_to_staged_swap() -> None:
    client = _FakeClient()
    # existing target is unpartitioned; the declared layout partitions it
    client.tables["proj.ds.docs"] = ["document_id", "filing_date"]
    adapter = _adapter(client)
    opts = _parse_options({"partition_by": {"field": "filing_date"}})
    df = pl.DataFrame({"document_id": ["a"], "filing_date": ["2026-01-01"]})
    adapter.materialize_full("docs", df, options=opts)

    # BigQuery cannot replace a table under a different partitioning spec:
    # the configured replacement renames in after the target drops.
    assert client.dropped == ["proj.ds.docs"]
    rename_sql = client.queries[-1][0]
    _, staging_id, _ = client.loads[0]
    staging_table = staging_id.removeprefix("proj.ds.")
    assert rename_sql == (
        f"ALTER TABLE `proj`.`ds`.`{staging_table}` RENAME TO `docs`"
    )


def test_materialize_full_bad_layout_keeps_target() -> None:
    class _FailingLoadClient(_FakeClient):
        def load_table_from_file(self, fobj: Any, table_id: str, job_config: Any = None) -> Any:
            raise RuntimeError("partition column type mismatch")

    client = _FailingLoadClient()
    adapter = _adapter(client)
    opts = _parse_options({"partition_by": {"field": "no_such_column"}})
    with pytest.raises(RuntimeError, match="mismatch"):
        adapter.materialize_full(
            "docs", pl.DataFrame({"document_id": ["a"]}), options=opts
        )
    assert "proj.ds.docs" not in client.dropped  # last good table survives


def test_materialize_full_without_options_keeps_plain_truncate() -> None:
    client = _FakeClient()
    adapter = _adapter(client)
    adapter.materialize_full("docs", pl.DataFrame({"document_id": ["a"]}))
    assert client.dropped == []
    _, _, job_config = client.loads[0]
    assert job_config.time_partitioning is None
    assert job_config.clustering_fields is None


def test_incremental_first_load_creates_partitioned_table() -> None:
    client = _FakeClient()  # get_table -> NotFound
    adapter = _adapter(client)
    opts = _parse_options(
        {
            "partition_by": {
                "field": "bucket",
                "data_type": "int64",
                "range": {"start": 0, "end": 100, "interval": 10},
            }
        }
    )
    df = pl.DataFrame({"document_id": ["a"], "bucket": [7]})
    adapter.materialize_incremental(
        "docs", df, key_col="document_id", options=opts
    )
    _, _, job_config = client.loads[0]
    assert job_config.range_partitioning.field == "bucket"
    assert job_config.range_partitioning.range_.start == 0
    assert job_config.range_partitioning.range_.end == 100
    assert job_config.range_partitioning.range_.interval == 10


def test_incremental_existing_table_keeps_layout() -> None:
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id", "filing_date"]
    adapter = _adapter(client)
    opts = _parse_options({"partition_by": {"field": "filing_date"}})
    df = pl.DataFrame({"document_id": ["a"], "filing_date": ["2026-01-01"]})
    adapter.materialize_incremental("docs", df, key_col="document_id", options=opts)

    # the staging load and MERGE never carry a partitioning spec
    for _, _, job_config in client.loads:
        assert job_config.time_partitioning is None


def test_full_chunks_ctas_carries_partition_and_cluster_clauses() -> None:
    client = _FakeClient()
    adapter = _adapter(client)
    opts = _parse_options(
        {
            "partition_by": {"field": "filing_date", "granularity": "month"},
            "cluster_by": ["cik", "form_type"],
        }
    )
    total = adapter.materialize_full_chunks(
        "docs",
        iter([pl.DataFrame({"document_id": ["a"], "filing_date": ["2026-01-01"]})]),
        options=opts,
    )
    assert total == 1
    create_sql = client.queries[0][0]
    # the layout validates inside the atomic CREATE OR REPLACE — no staged swap
    assert create_sql.startswith("CREATE OR REPLACE TABLE `proj`.`ds`.`docs` ")
    assert (
        "PARTITION BY DATE_TRUNC(`filing_date`, MONTH) "
        "CLUSTER BY `cik`, `form_type` AS SELECT" in create_sql
    )
    assert len(client.queries) == 1
    assert all(d.startswith("proj.ds.dbt_ml_staging__docs__") for d in client.dropped)


def test_full_chunks_partition_migration_falls_back_to_staged_swap() -> None:
    client = _FakeClient()
    # existing target is unpartitioned; the declared layout partitions it
    client.tables["proj.ds.docs"] = ["document_id", "filing_date"]
    adapter = _adapter(client)
    opts = _parse_options({"partition_by": {"field": "filing_date"}})
    adapter.materialize_full_chunks(
        "docs",
        iter([pl.DataFrame({"document_id": ["a"], "filing_date": ["2026-01-01"]})]),
        options=opts,
    )
    create_sql = client.queries[0][0]
    # the replacement is created (validating the layout) before any drop
    assert create_sql.startswith("CREATE TABLE `proj`.`ds`.`dbt_ml_staging__docs__")
    assert client.dropped[0] == "proj.ds.docs"
    assert "RENAME TO `docs`" in client.queries[1][0]


def test_full_chunks_bad_layout_keeps_target() -> None:
    client = _FakeClient()
    client.query_results = [_FailingJob()]  # replacement CTAS rejected
    adapter = _adapter(client)
    opts = _parse_options({"cluster_by": ["no_such_column"]})
    with pytest.raises(RuntimeError, match="simulated"):
        adapter.materialize_full_chunks(
            "docs", iter([pl.DataFrame({"document_id": ["a"]})]), options=opts
        )
    assert "proj.ds.docs" not in client.dropped  # last good table survives
    # the staging table is cleaned up; the failed CREATE OR REPLACE never
    # touched the target
    assert all(d.startswith("proj.ds.dbt_ml_staging__docs__") for d in client.dropped)
    assert len(client.dropped) == 1


def test_table_options_require_partitioning_where_relevant() -> None:
    with pytest.raises(AdapterError, match="require_partition_filter"):
        _parse_options({"require_partition_filter": True})
    with pytest.raises(AdapterError, match="partition_expiration_days"):
        _parse_options({"partition_expiration_days": 30})


def test_labels_validated_client_side() -> None:
    with pytest.raises(AdapterError, match="label key"):
        _parse_options({"labels": {"Team": "econ"}})
    with pytest.raises(AdapterError, match="label value"):
        _parse_options({"labels": {"team": "Econ!"}})
    assert _parse_options({"labels": {"team": "econ"}}).labels == {"team": "econ"}


def test_materialize_full_applies_table_options_and_kms() -> None:
    client = _FakeClient()
    adapter = _adapter(client)
    opts = _parse_options(
        {
            "partition_by": {"field": "filing_date"},
            "require_partition_filter": True,
            "partition_expiration_days": 90,
            "hours_to_expiration": 48,
            "labels": {"team": "econ", "env": "prod"},
            "kms_key_name": "projects/p/locations/us/keyRings/r/cryptoKeys/k",
        }
    )
    df = pl.DataFrame({"document_id": ["a"], "filing_date": ["2026-01-01"]})
    adapter.materialize_full("docs", df, options=opts)

    _, _, job_config = client.loads[0]
    assert job_config.destination_encryption_configuration.kms_key_name == (
        "projects/p/locations/us/keyRings/r/cryptoKeys/k"
    )
    assert job_config.labels == {"team": "econ", "env": "prod"}

    swap_sql, swap_config = client.queries[0]
    # every table option — kms included — rides the atomic CREATE OR REPLACE
    assert swap_sql.startswith("CREATE OR REPLACE TABLE `proj`.`ds`.`docs` ")
    assert " OPTIONS (" in swap_sql
    assert "require_partition_filter = TRUE" in swap_sql
    assert "partition_expiration_days = 90" in swap_sql
    assert (
        "expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), "
        "INTERVAL 48 HOUR)" in swap_sql
    )
    assert "labels = [('env', 'prod'), ('team', 'econ')]" in swap_sql
    assert (
        "kms_key_name = 'projects/p/locations/us/keyRings/r/cryptoKeys/k'"
        in swap_sql
    )
    assert swap_config.labels == {"team": "econ", "env": "prod"}


def test_partition_migration_configures_staging_before_swap() -> None:
    client = _FakeClient()
    # existing target is unpartitioned; the declared layout partitions it
    client.tables["proj.ds.docs"] = ["document_id", "filing_date"]
    adapter = _adapter(client)
    opts = _parse_options(
        {
            "partition_by": {"field": "filing_date"},
            "require_partition_filter": True,
            "kms_key_name": "projects/p/locations/us/keyRings/r/cryptoKeys/k",
        }
    )
    df = pl.DataFrame({"document_id": ["a"], "filing_date": ["2026-01-01"]})
    adapter.materialize_full("docs", df, options=opts)

    alter_sql, _ = client.queries[0]
    # options are configured on the staging replacement before the swap
    assert alter_sql.startswith("ALTER TABLE `proj`.`ds`.`dbt_ml_staging__docs__")
    assert " SET OPTIONS (" in alter_sql
    assert "require_partition_filter = TRUE" in alter_sql
    assert "kms_key_name" not in alter_sql  # set at create, never re-keyed
    assert client.dropped == ["proj.ds.docs"]


def test_incremental_create_applies_post_create_options() -> None:
    client = _FakeClient()  # get_table -> NotFound
    adapter = _adapter(client)
    opts = _parse_options(
        {"partition_by": {"field": "filing_date"}, "require_partition_filter": True}
    )
    df = pl.DataFrame({"document_id": ["a"], "filing_date": ["2026-01-01"]})
    adapter.materialize_incremental("docs", df, key_col="document_id", options=opts)
    alter_sql, _ = client.queries[0]
    assert alter_sql.startswith("ALTER TABLE")
    assert "require_partition_filter = TRUE" in alter_sql


def test_full_chunks_ctas_carries_options_clause() -> None:
    client = _FakeClient()
    adapter = _adapter(client)
    opts = _parse_options(
        {
            "partition_by": {"field": "filing_date"},
            "labels": {"team": "econ"},
            "kms_key_name": "projects/p/locations/us/keyRings/r/cryptoKeys/k",
        }
    )
    adapter.materialize_full_chunks(
        "docs",
        iter([pl.DataFrame({"document_id": ["a"], "filing_date": ["2026-01-01"]})]),
        options=opts,
    )
    swap_sql = client.queries[0][0]
    assert "PARTITION BY `filing_date` OPTIONS (" in swap_sql
    assert "labels = [('team', 'econ')]" in swap_sql
    assert (
        "kms_key_name = 'projects/p/locations/us/keyRings/r/cryptoKeys/k'"
        in swap_sql
    )
    # labels also ride the load + swap jobs
    assert client.loads[0][2].labels == {"team": "econ"}
    assert client.queries[0][1].labels == {"team": "econ"}


# ─── incremental_strategy: insert_overwrite ──────────────────────────────────


def test_insert_overwrite_requires_time_partitioning() -> None:
    with pytest.raises(AdapterError, match="requires"):
        _parse_options({"incremental_strategy": "insert_overwrite"})
    with pytest.raises(AdapterError, match="field"):
        _parse_options(
            {
                "incremental_strategy": "insert_overwrite",
                "partition_by": {"data_type": "timestamp"},  # ingestion-time
            }
        )
    with pytest.raises(AdapterError, match="time"):
        _parse_options(
            {
                "incremental_strategy": "insert_overwrite",
                "partition_by": {
                    "field": "n",
                    "data_type": "int64",
                    "range": {"start": 0, "end": 10, "interval": 1},
                },
            }
        )


def test_insert_overwrite_replaces_partitions_via_script() -> None:
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id", "filing_date"]
    adapter = _adapter(client)
    opts = _parse_options(
        {
            "incremental_strategy": "insert_overwrite",
            "partition_by": {"field": "filing_date"},
        }
    )
    df = pl.DataFrame({"document_id": ["a"], "filing_date": ["2026-01-01"]})
    adapter.materialize_incremental("docs", df, key_col="document_id", options=opts)

    script, _ = client.queries[0]
    assert "DECLARE dbt_ml_partitions ARRAY<DATE>;" in script
    assert (
        "SET dbt_ml_partitions = ARRAY(SELECT DISTINCT `filing_date` FROM"
        in script
    )
    assert "WHERE `filing_date` IS NOT NULL" in script
    assert "USING" in script and "ON FALSE" in script
    assert (
        "WHEN NOT MATCHED BY SOURCE AND target.`filing_date` "
        "IN UNNEST(dbt_ml_partitions) THEN DELETE" in script
    )
    assert "WHEN NOT MATCHED THEN INSERT (`document_id`, `filing_date`)" in script
    assert "WHEN MATCHED THEN UPDATE" not in script
    assert len(client.dropped) == 1  # staging cleanup


def test_insert_overwrite_monthly_truncates_partition_identity() -> None:
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id", "ts"]
    adapter = _adapter(client)
    opts = _parse_options(
        {
            "incremental_strategy": "insert_overwrite",
            "partition_by": {
                "field": "ts",
                "data_type": "timestamp",
                "granularity": "month",
            },
        }
    )
    df = pl.DataFrame({"document_id": ["a"], "ts": ["2026-01-01T00:00:00Z"]})
    adapter.materialize_incremental("docs", df, key_col="document_id", options=opts)
    script, _ = client.queries[0]
    assert "ARRAY<TIMESTAMP>" in script
    assert "SELECT DISTINCT TIMESTAMP_TRUNC(`ts`, MONTH)" in script
    assert "TIMESTAMP_TRUNC(target.`ts`, MONTH) IN UNNEST" in script


def test_insert_overwrite_needs_partition_column_in_batch() -> None:
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id", "x"]
    adapter = _adapter(client)
    opts = _parse_options(
        {
            "incremental_strategy": "insert_overwrite",
            "partition_by": {"field": "filing_date"},
        }
    )
    df = pl.DataFrame({"document_id": ["a"], "x": [1]})
    with pytest.raises(AdapterError, match="filing_date"):
        adapter.materialize_incremental(
            "docs", df, key_col="document_id", options=opts
        )
    assert client.loads == []  # rejected before staging anything


def test_merge_stays_default_with_table_options() -> None:
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id", "filing_date"]
    adapter = _adapter(client)
    opts = _parse_options({"labels": {"team": "econ"}})
    df = pl.DataFrame({"document_id": ["a"], "filing_date": ["2026-01-01"]})
    adapter.materialize_incremental("docs", df, key_col="document_id", options=opts)
    sql, query_config = client.queries[0]
    assert sql.startswith("MERGE")
    assert "WHEN MATCHED THEN UPDATE SET" in sql
    assert query_config.labels == {"team": "econ"}


def test_duckdb_ignores_warehouse_options(tmp_path: Path) -> None:
    from dbt_ml.adapters.duckdb import DuckDBAdapter

    cfg = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "wh.duckdb")}
    )
    adapter = create_adapter(cfg)
    assert isinstance(adapter, DuckDBAdapter)
    assert adapter.warehouse_options_model() is None
    parsed = adapter.parse_warehouse_options(
        {"partition_by": {"field": "filing_date"}}, model_name="filings"
    )
    assert parsed is None
    with adapter:
        rows = adapter.materialize_full(
            "docs", pl.DataFrame({"document_id": ["a"]}), options=parsed
        )
    assert rows == 1


# ─── dbt sources export ─────────────────────────────────────────────────────


def test_emit_dbt_sources_for_bigquery(tmp_path: Path) -> None:
    from dbt_ml.dbt_export import build_dbt_sources

    (tmp_path / "dbt_ml_project.yml").write_text(
        "name: econ\nversion: '0.1.0'\nprofile: econ\n"
    )
    (tmp_path / "profiles.yml").write_text(
        "\n".join(
            [
                "econ:",
                "  target: prod",
                "  outputs:",
                "    prod:",
                "      warehouse:",
                "        type: bigquery",
                "        project: econ-lakehouse",
                "        dataset: documents",
            ]
        )
        + "\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m.yml").write_text(
        "version: 2\nmodels:\n  - name: filings\n    transform:\n"
        "      type: python\n      module: transforms.x\n"
    )

    payload = build_dbt_sources(tmp_path)
    source = payload["sources"][0]
    assert source["database"] == "econ-lakehouse"
    assert source["schema"] == "documents"
    assert source["tables"][0]["name"] == "filings"


# ─── Iceberg / BigLake managed tables (issue #163) ──────────────────────────

_ICEBERG = {
    "table_format": "iceberg",
    "storage_uri": "gs://bucket/tbl",
    "connection": "proj.us.conn",
}


def test_iceberg_options_require_storage_uri_and_connection() -> None:
    with pytest.raises(ValidationError, match="requires"):
        BigQueryWarehouseOptions.model_validate({"table_format": "iceberg"})
    with pytest.raises(ValidationError, match="requires"):
        BigQueryWarehouseOptions.model_validate(
            {"table_format": "iceberg", "storage_uri": "gs://b/t"}
        )


def test_iceberg_keys_require_table_format() -> None:
    with pytest.raises(ValidationError, match="require `table_format: iceberg`"):
        BigQueryWarehouseOptions.model_validate({"storage_uri": "gs://b/t"})


def test_iceberg_storage_uri_must_be_gs() -> None:
    with pytest.raises(ValidationError, match="gs:// Cloud Storage URI"):
        BigQueryWarehouseOptions.model_validate({**_ICEBERG, "storage_uri": "s3://b/t"})


def test_iceberg_rejects_kms_and_int64_partitioning() -> None:
    with pytest.raises(ValidationError, match=r"kms_key_name.*iceberg"):
        BigQueryWarehouseOptions.model_validate(
            {**_ICEBERG, "kms_key_name": "projects/p/locations/l/keyRings/r/cryptoKeys/k"}
        )
    with pytest.raises(ValidationError, match="time partitioning only"):
        BigQueryWarehouseOptions.model_validate(
            {
                **_ICEBERG,
                "partition_by": {
                    "field": "n",
                    "data_type": "int64",
                    "range": {"start": 0, "end": 10, "interval": 1},
                },
            }
        )


def test_iceberg_column_ddl_maps_polars_dtypes() -> None:
    adapter = _adapter()
    df = pl.DataFrame(
        schema={
            "s": pl.String,
            "i": pl.Int32,
            "f": pl.Float64,
            "b": pl.Boolean,
            "d": pl.Date,
            "ts": pl.Datetime("us", "UTC"),
            "dt": pl.Datetime("us"),
            "vec": pl.List(pl.Float64),
            "obj": pl.Struct({"a": pl.Int64, "c": pl.String}),
            "n": pl.Decimal(10, 2),
        }
    )
    assert adapter._iceberg_column_ddl(df) == (
        "`s` STRING, `i` INT64, `f` FLOAT64, `b` BOOL, `d` DATE, "
        "`ts` TIMESTAMP, `dt` DATETIME, `vec` ARRAY<FLOAT64>, "
        "`obj` STRUCT<`a` INT64, `c` STRING>, `n` NUMERIC"
    )


def test_iceberg_column_ddl_rejects_unsupported_types() -> None:
    adapter = _adapter()
    duration = pl.DataFrame(schema={"d": pl.Duration})
    with pytest.raises(AdapterError, match="column 'd'"):
        adapter._iceberg_column_ddl(duration)
    big = pl.DataFrame(schema={"n": pl.Decimal(20, 10)})
    with pytest.raises(AdapterError, match="BIGNUMERIC"):
        adapter._iceberg_column_ddl(big)


def test_materialize_full_iceberg_drops_creates_and_appends() -> None:
    from google.cloud import bigquery

    client = _FakeClient()
    adapter = _adapter(client)
    df = pl.DataFrame({"document_id": ["a", "b"], "x": [1, 2]})
    assert adapter.materialize_full("docs", df, options=_parse_options(_ICEBERG)) == 2

    assert client.dropped == ["proj.ds.docs"]
    assert client.queries[0][0] == (
        "CREATE TABLE `proj`.`ds`.`docs` (`document_id` STRING, `x` INT64) "
        "WITH CONNECTION `proj.us.conn` OPTIONS (file_format = 'PARQUET', "
        "table_format = 'ICEBERG', storage_uri = 'gs://bucket/tbl')"
    )
    payload, table_id, job_config = client.loads[0]
    assert table_id == "proj.ds.docs"
    assert job_config.write_disposition == bigquery.WriteDisposition.WRITE_APPEND
    assert pl.read_parquet(io.BytesIO(payload)).rows() == [("a", 1), ("b", 2)]


def test_materialize_full_iceberg_default_connection_and_partition() -> None:
    client = _FakeClient()
    adapter = _adapter(client)
    opts = _parse_options(
        {
            **_ICEBERG,
            "connection": "DEFAULT",
            "partition_by": {"field": "filing_date"},
            "cluster_by": ["cik"],
        }
    )
    df = pl.DataFrame({"cik": ["1"], "filing_date": ["2026-01-01"]})
    adapter.materialize_full("docs", df, options=opts)

    assert client.queries[0][0] == (
        "CREATE TABLE `proj`.`ds`.`docs` (`cik` STRING, `filing_date` STRING) "
        "PARTITION BY `filing_date` CLUSTER BY `cik` "
        "WITH CONNECTION DEFAULT OPTIONS (file_format = 'PARQUET', "
        "table_format = 'ICEBERG', storage_uri = 'gs://bucket/tbl')"
    )


def test_incremental_iceberg_first_run_creates_then_appends() -> None:
    from google.cloud import bigquery

    client = _FakeClient()  # get_table -> NotFound
    adapter = _adapter(client)
    df = pl.DataFrame({"document_id": ["a"], "x": [1]})
    adapter.materialize_incremental(
        "docs", df, key_col="document_id", options=_parse_options(_ICEBERG)
    )

    assert client.queries[0][0].startswith("CREATE TABLE `proj`.`ds`.`docs` (")
    assert "table_format = 'ICEBERG'" in client.queries[0][0]
    _, table_id, job_config = client.loads[0]
    assert table_id == "proj.ds.docs"
    assert job_config.write_disposition == bigquery.WriteDisposition.WRITE_APPEND


def test_incremental_iceberg_existing_table_merges() -> None:
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id", "x"]
    adapter = _adapter(client)
    df = pl.DataFrame({"document_id": ["a"], "x": [9]})
    adapter.materialize_incremental(
        "docs", df, key_col="document_id", options=_parse_options(_ICEBERG)
    )

    merge_sql = client.queries[-1][0]
    assert merge_sql.startswith("MERGE `proj`.`ds`.`docs` AS target ")
    assert "WHEN MATCHED THEN UPDATE SET" in merge_sql


def test_incremental_iceberg_adds_columns_with_ddl() -> None:
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id", "x"]
    adapter = _adapter(client)
    df = pl.DataFrame({"document_id": ["a"], "x": [9], "y": ["new"]})
    adapter.materialize_incremental(
        "docs",
        df,
        key_col="document_id",
        on_schema_change="append_new_columns",
        options=_parse_options(_ICEBERG),
    )

    alter = next(q[0] for q in client.queries if q[0].startswith("ALTER TABLE"))
    assert alter == "ALTER TABLE `proj`.`ds`.`docs` ADD COLUMN `y` STRING"


def test_materialize_full_iceberg_validates_schema_before_dropping() -> None:
    # An unsupported dtype must fail before the existing target is dropped, so a
    # bad schema never destroys the last good table.
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id"]
    adapter = _adapter(client)
    df = pl.DataFrame(schema={"document_id": pl.String, "d": pl.Duration})
    with pytest.raises(AdapterError, match="column 'd'"):
        adapter.materialize_full("docs", df, options=_parse_options(_ICEBERG))
    assert client.dropped == []
    assert client.queries == []


def test_iceberg_connection_rejects_injection() -> None:
    with pytest.raises(ValidationError, match="Cloud Resource connection identifier"):
        BigQueryWarehouseOptions.model_validate(
            {**_ICEBERG, "connection": "proj.us.c` OPTIONS(x); DROP TABLE t"}
        )


def test_iceberg_connection_accepts_resource_path() -> None:
    opts = BigQueryWarehouseOptions.model_validate(
        {**_ICEBERG, "connection": "projects/p/locations/us/connections/c"}
    )
    assert opts.connection == "projects/p/locations/us/connections/c"


def test_full_chunks_iceberg_inserts_from_staging() -> None:
    client = _FakeClient()
    schema_df = pl.DataFrame(schema={"document_id": pl.String, "x": pl.Int64})

    class _ArrowJob(_FakeJob):
        def to_arrow(self) -> Any:
            return schema_df.to_arrow()

    client.query_results.append(_ArrowJob())  # answers the LIMIT 0 schema probe
    adapter = _adapter(client)
    chunks = [pl.DataFrame({"document_id": ["a"], "x": [1]})]
    adapter.materialize_full_chunks(
        "docs", iter(chunks), options=_parse_options(_ICEBERG)
    )

    create = next(q[0] for q in client.queries if q[0].startswith("CREATE TABLE"))
    assert "table_format = 'ICEBERG'" in create
    insert = next(q[0] for q in client.queries if q[0].startswith("INSERT INTO"))
    assert insert.startswith(
        "INSERT INTO `proj`.`ds`.`docs` SELECT * FROM `proj`.`ds`.`dbt_ml_staging__docs__"
    )
    assert "proj.ds.docs" in client.dropped


# ─── optional integration (needs real GCP credentials) ─────────────────────

_BQ_PROJECT = os.environ.get("DBT_ML_BQ_TEST_PROJECT")


@pytest.mark.skipif(
    not _BQ_PROJECT, reason="set DBT_ML_BQ_TEST_PROJECT to run BigQuery integration"
)
def test_integration_full_round_trip() -> None:
    cfg = parse_warehouse_config(
        {
            "type": "bigquery",
            "project": _BQ_PROJECT,
            "dataset": "dbt_ml_it_" + os.urandom(3).hex(),
        }
    )
    adapter = create_adapter(cfg)
    try:
        with adapter:
            adapter.materialize_full(
                "docs", pl.DataFrame({"document_id": ["a", "b"], "x": [1, 2]})
            )
            adapter.materialize_incremental(
                "docs",
                pl.DataFrame({"document_id": ["a", "c"], "x": [99, 3]}),
                key_col="document_id",
            )
            rows = adapter.rows(
                f"SELECT document_id, x FROM {adapter.table_ref('docs')} "
                "ORDER BY document_id"
            )
            assert rows == [("a", 99), ("b", 2), ("c", 3)]

            scope = StateScope("m")
            adapter.upsert_state(scope, [StateRecord("a", "h", "v")])
            adapter.upsert_state(scope, [StateRecord("a", "h2", "v2")])
            assert adapter.fetch_state(scope) == {"a": StateValue("h2", "v2")}
            assert adapter.list_tables() == ["docs"]
    finally:
        assert isinstance(adapter, BigQueryAdapter)
        adapter._reset_storage_for_test()


@pytest.mark.skipif(
    not _BQ_PROJECT, reason="set DBT_ML_BQ_TEST_PROJECT to run BigQuery integration"
)
def test_integration_warehouse_options_round_trip() -> None:
    from datetime import date as date_type

    cfg = parse_warehouse_config(
        {
            "type": "bigquery",
            "project": _BQ_PROJECT,
            "dataset": "dbt_ml_it_" + os.urandom(3).hex(),
        }
    )
    adapter = create_adapter(cfg)
    assert isinstance(adapter, BigQueryAdapter)
    opts = adapter.parse_warehouse_options(
        {
            "partition_by": {"field": "filing_date"},
            "cluster_by": ["vendor"],
            "labels": {"managed-by": "dbt-ml-tests"},
            "incremental_strategy": "insert_overwrite",
        },
        model_name="docs",
    )
    try:
        with adapter:
            adapter.materialize_full(
                "docs",
                pl.DataFrame(
                    {
                        "document_id": ["a", "b"],
                        "vendor": ["acme", "zenith"],
                        "filing_date": [date_type(2026, 1, 1), date_type(2026, 1, 2)],
                    }
                ),
                options=opts,
            )
            table = adapter.client.get_table(adapter._table_id("docs"))
            assert table.time_partitioning.field == "filing_date"
            assert table.clustering_fields == ["vendor"]
            assert table.labels == {"managed-by": "dbt-ml-tests"}

            # insert_overwrite replaces the 2026-01-01 partition only
            adapter.materialize_incremental(
                "docs",
                pl.DataFrame(
                    {
                        "document_id": ["a2"],
                        "vendor": ["acme"],
                        "filing_date": [date_type(2026, 1, 1)],
                    }
                ),
                key_col="document_id",
                options=opts,
            )
            rows = adapter.rows(
                f"SELECT document_id FROM {adapter.table_ref('docs')} "
                "ORDER BY document_id"
            )
            assert rows == [("a2",), ("b",)]
    finally:
        adapter._reset_storage_for_test()


_BQ_ICEBERG_CONNECTION = os.environ.get("DBT_ML_BQ_TEST_CONNECTION")
_BQ_ICEBERG_STORAGE_URI = os.environ.get("DBT_ML_BQ_TEST_STORAGE_URI")


@pytest.mark.skipif(
    not (_BQ_PROJECT and _BQ_ICEBERG_CONNECTION and _BQ_ICEBERG_STORAGE_URI),
    reason=(
        "set DBT_ML_BQ_TEST_PROJECT, DBT_ML_BQ_TEST_CONNECTION (a BigLake Cloud "
        "Resource connection), and DBT_ML_BQ_TEST_STORAGE_URI (a gs:// prefix) to "
        "run the Iceberg round-trip"
    ),
)
def test_integration_iceberg_round_trip() -> None:
    dataset = "dbt_ml_it_" + os.urandom(3).hex()
    cfg = parse_warehouse_config(
        {"type": "bigquery", "project": _BQ_PROJECT, "dataset": dataset}
    )
    adapter = create_adapter(cfg)
    assert isinstance(adapter, BigQueryAdapter)
    opts = adapter.parse_warehouse_options(
        {
            "table_format": "iceberg",
            "connection": _BQ_ICEBERG_CONNECTION,
            "storage_uri": f"{(_BQ_ICEBERG_STORAGE_URI or '').rstrip('/')}/{dataset}",
        },
        model_name="docs",
    )
    try:
        with adapter:
            adapter.materialize_full(
                "docs",
                pl.DataFrame(
                    {"document_id": ["a", "b"], "vec": [[0.1, 0.2], [0.3, 0.4]]}
                ),
                options=opts,
            )
            adapter.materialize_incremental(
                "docs",
                pl.DataFrame({"document_id": ["a", "c"], "vec": [[9.0, 9.0], [0.5, 0.6]]}),
                key_col="document_id",
                options=opts,
            )
            rows = adapter.rows(
                f"SELECT document_id FROM {adapter.table_ref('docs')} "
                "ORDER BY document_id"
            )
            assert rows == [("a",), ("b",), ("c",)]
    finally:
        adapter._reset_storage_for_test()


# ─── auth parity with dbt-bigquery ──────────────────────────────────────────


def test_auth_method_inference() -> None:
    assert _bq_cfg().method == "oauth"
    assert _bq_cfg(keyfile="./sa.json").method == "service-account"
    assert _bq_cfg(
        keyfile_json="{{ env_var('BQ_SERVICE_ACCOUNT_JSON') }}"
    ).method == (
        "service-account-json"
    )
    assert _bq_cfg(token="{{ env_var('BQ_ACCESS_TOKEN') }}").method == (
        "oauth-secrets"
    )


def _bq_cfg(**extra: Any) -> BigQueryWarehouseConfig:
    cfg = parse_warehouse_config({"type": "bigquery", "project": "p", **extra})
    assert isinstance(cfg, BigQueryWarehouseConfig)
    return cfg


def test_auth_method_mismatch_rejected() -> None:
    with pytest.raises(AdapterError, match="conflicts"):
        _bq_cfg(method="oauth", keyfile="./sa.json")


def test_service_account_requires_keyfile() -> None:
    with pytest.raises(AdapterError, match="keyfile"):
        _bq_cfg(method="service-account")


def test_keyfile_and_keyfile_json_conflict() -> None:
    with pytest.raises(AdapterError, match="not both"):
        _bq_cfg(
            keyfile="./sa.json",
            keyfile_json="{{ env_var('BQ_SERVICE_ACCOUNT_JSON') }}",
        )


def test_oauth_secrets_requires_token_or_full_refresh_set() -> None:
    with pytest.raises(AdapterError, match="oauth-secrets"):
        _bq_cfg(
            method="oauth-secrets",
            refresh_token="{{ env_var('BQ_REFRESH_TOKEN') }}",
            client_id="c",
        )
    cfg = _bq_cfg(
        refresh_token="{{ env_var('BQ_REFRESH_TOKEN') }}",
        client_id="c",
        client_secret="{{ env_var('BQ_CLIENT_SECRET') }}",
        token_uri="https://oauth2.googleapis.com/token",
    )
    assert cfg.method == "oauth-secrets"


@pytest.mark.parametrize(
    "field_name",
    ["keyfile_json", "token", "refresh_token", "client_secret"],
)
def test_literal_secret_fields_are_rejected_before_pydantic(
    field_name: str,
) -> None:
    sentinel = "distinctive-literal-secret-sentinel"

    with pytest.raises(AdapterError) as exc_info:
        _bq_cfg(**{field_name: sentinel})

    message = str(exc_info.value)
    assert field_name in message
    assert sentinel not in message


def test_direct_config_validation_protects_cross_field_error_inputs() -> None:
    first_name = "DISTINCTIVE_DIRECT_KEYFILE_REFERENCE"
    second_name = "DISTINCTIVE_DIRECT_TOKEN_REFERENCE"
    raw = {
        "type": "bigquery",
        "project": "p",
        "keyfile_json": f"{{{{ env_var('{first_name}') }}}}",
        "token": f"{{{{ env_var('{second_name}') }}}}",
    }

    with pytest.raises(ValidationError) as exc_info:
        BigQueryWarehouseConfig.model_validate(raw)

    error = exc_info.value
    rendered = "\n".join(
        (str(error), repr(error), repr(error.errors()), error.json())
    )
    assert first_name not in rendered
    assert second_name not in rendered
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(error)


def test_direct_config_validation_clears_rejected_literal_input() -> None:
    sentinel = "distinctive-direct-literal-secret"

    with pytest.raises(ValidationError) as exc_info:
        BigQueryWarehouseConfig.model_validate(
            {"type": "bigquery", "project": "p", "token": sentinel}
        )

    error = exc_info.value
    rendered = "\n".join(
        (str(error), repr(error), repr(error.errors()), error.json())
    )
    assert sentinel not in rendered
    assert sentinel.encode() not in pickle.dumps(error)


def test_bigquery_schema_describes_exact_environment_references() -> None:
    properties = BigQueryWarehouseConfig.model_json_schema()["properties"]

    for field_name in (
        "keyfile_json",
        "token",
        "refresh_token",
        "client_secret",
    ):
        credential_schema = next(
            candidate
            for candidate in properties[field_name]["anyOf"]
            if candidate.get("format") == "password"
        )
        assert "env_var" in credential_schema["pattern"]
        assert "Exact {{ env_var('NAME') }}" in credential_schema["description"]


@pytest.mark.parametrize(
    "token_uri",
    [
        "https://user@oauth2.googleapis.com/token",
        "https://user:distinctive-secret@oauth2.googleapis.com/token",
        "https://%75ser:%73ecret@oauth2.googleapis.com/token",
    ],
)
def test_token_uri_rejects_url_user_information(token_uri: str) -> None:
    with pytest.raises(AdapterError, match="must not contain URL user information") as exc_info:
        _bq_cfg(
            refresh_token="{{ env_var('BQ_REFRESH_TOKEN') }}",
            client_id="client",
            client_secret="{{ env_var('BQ_CLIENT_SECRET') }}",
            token_uri=token_uri,
        )

    assert "distinctive-secret" not in str(exc_info.value)


def test_auth_methods_reject_credentials_for_other_methods() -> None:
    with pytest.raises(AdapterError, match="cannot be combined"):
        _bq_cfg(
            keyfile="./sa.json",
            token="{{ env_var('BQ_ACCESS_TOKEN') }}",
        )
    with pytest.raises(AdapterError, match="conflicts"):
        _bq_cfg(
            method="oauth",
            token="{{ env_var('BQ_ACCESS_TOKEN') }}",
        )


def test_token_credentials_reject_partial_refresh_metadata() -> None:
    with pytest.raises(AdapterError, match="all of"):
        _bq_cfg(
            token="{{ env_var('BQ_ACCESS_TOKEN') }}",
            client_id="client-without-refresh-set",
        )


def test_default_scopes_match_dbt_bigquery() -> None:
    from dbt_ml.adapters.bigquery import DEFAULT_SCOPES

    assert _bq_cfg().scopes == list(DEFAULT_SCOPES)
    assert len(DEFAULT_SCOPES) == 3


def test_parse_keyfile_json_forms() -> None:
    import base64
    import json

    from dbt_ml.adapters.bigquery import parse_keyfile_json

    info = {"type": "service_account", "project_id": "p"}
    assert parse_keyfile_json(info) == info
    assert parse_keyfile_json(json.dumps(info)) == info
    encoded = base64.b64encode(json.dumps(info).encode()).decode()
    assert parse_keyfile_json(encoded) == info
    with pytest.raises(AdapterError, match="keyfile_json"):
        parse_keyfile_json("not json at all !!")
    with pytest.raises(AdapterError, match="JSON object"):
        parse_keyfile_json('["a", "list"]')


def test_parse_keyfile_json_error_scrubs_resolved_input() -> None:
    from dbt_ml.adapters.bigquery import parse_keyfile_json

    sentinel = "distinctive-invalid-service-account-secret"

    with pytest.raises(AdapterError) as exc_info:
        parse_keyfile_json(sentinel)

    error = exc_info.value
    assert sentinel not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback = error.__traceback__
    while traceback is not None:
        if "/src/dbt_ml/" in traceback.tb_frame.f_code.co_filename:
            assert sentinel not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


def test_credentials_service_account_json_scopes_and_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.oauth2 import service_account

    captured: dict[str, Any] = {}

    class _FakeCreds:
        def with_quota_project(self, qp: str) -> _FakeCreds:
            captured["quota_project"] = qp
            return self

    def fake_from_info(info: dict[str, Any], scopes: Any = None) -> _FakeCreds:
        captured["info"] = info
        captured["scopes"] = scopes
        return _FakeCreds()

    monkeypatch.setattr(
        service_account.Credentials,
        "from_service_account_info",
        staticmethod(fake_from_info),
    )
    monkeypatch.setenv(
        "BQ_SERVICE_ACCOUNT_JSON", '{"type":"service_account"}'
    )
    adapter = _adapter(
        keyfile_json="{{ env_var('BQ_SERVICE_ACCOUNT_JSON') }}",
        quota_project="bill-here",
    )
    creds = adapter._credentials()
    assert isinstance(creds, _FakeCreds)
    assert captured["info"] == {"type": "service_account"}
    assert list(captured["scopes"]) == _bq_cfg().scopes
    assert captured["quota_project"] == "bill-here"


def test_credentials_oauth_reveals_each_value_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.oauth2 import credentials as oauth_credentials

    captured: dict[str, Any] = {}
    reveal_count = 0
    original_reveal = ProtectedCredential.reveal

    class _FakeCredentials:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    def counted_reveal(credential: ProtectedCredential) -> str:
        nonlocal reveal_count
        reveal_count += 1
        return original_reveal(credential)

    monkeypatch.setenv("BQ_ACCESS_TOKEN", "access-secret")
    monkeypatch.setenv("BQ_REFRESH_TOKEN", "refresh-secret")
    monkeypatch.setenv("BQ_CLIENT_SECRET", "client-secret")
    monkeypatch.setattr(ProtectedCredential, "reveal", counted_reveal)
    monkeypatch.setattr(oauth_credentials, "Credentials", _FakeCredentials)

    adapter = _adapter(
        token="{{ env_var('BQ_ACCESS_TOKEN') }}",
        refresh_token="{{ env_var('BQ_REFRESH_TOKEN') }}",
        client_id="public-client-id",
        client_secret="{{ env_var('BQ_CLIENT_SECRET') }}",
        token_uri="https://oauth2.googleapis.com/token",
    )
    credentials = adapter._credentials()

    assert isinstance(credentials, _FakeCredentials)
    assert captured["token"] == "access-secret"
    assert captured["refresh_token"] == "refresh-secret"
    assert captured["client_secret"] == "client-secret"
    assert captured["client_id"] == "public-client-id"
    assert reveal_count == 3


def test_missing_oauth_reference_fails_before_sdk_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.oauth2 import credentials as oauth_credentials

    called = False
    env_name = "DISTINCTIVE_MISSING_BQ_TOKEN"

    def unexpected_credentials(**kwargs: Any) -> None:
        del kwargs
        nonlocal called
        called = True

    monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(
        oauth_credentials,
        "Credentials",
        unexpected_credentials,
    )
    adapter = _adapter(token=f"{{{{ env_var('{env_name}') }}}}")

    with pytest.raises(AdapterError) as exc_info:
        adapter._credentials()

    assert called is False
    assert env_name not in str(exc_info.value)


def test_environment_token_uri_rejects_user_information_before_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.oauth2 import credentials as oauth_credentials

    called = False

    def unexpected_credentials(**kwargs: Any) -> None:
        del kwargs
        nonlocal called
        called = True

    monkeypatch.setenv("BQ_REFRESH_TOKEN", "refresh-secret")
    monkeypatch.setenv("BQ_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv(
        "BQ_TOKEN_URI",
        "https://user:distinctive-url-secret@oauth2.googleapis.com/token",
    )
    monkeypatch.setattr(
        oauth_credentials,
        "Credentials",
        unexpected_credentials,
    )
    adapter = _adapter(
        refresh_token="{{ env_var('BQ_REFRESH_TOKEN') }}",
        client_id="public-client-id",
        client_secret="{{ env_var('BQ_CLIENT_SECRET') }}",
        token_uri="{{ env_var('BQ_TOKEN_URI') }}",
    )

    with pytest.raises(AdapterError) as exc_info:
        adapter._credentials()

    assert called is False
    assert "distinctive-url-secret" not in str(exc_info.value)
    assert "BQ_TOKEN_URI" not in str(exc_info.value)


def test_native_credential_error_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.oauth2 import credentials as oauth_credentials

    sentinel = "distinctive-native-sdk-secret"
    env_name = "DISTINCTIVE_NATIVE_ERROR_TOKEN"
    monkeypatch.setenv(env_name, sentinel)

    def fail_with_secret(**kwargs: Any) -> None:
        del kwargs
        raise RuntimeError(sentinel)

    monkeypatch.setattr(oauth_credentials, "Credentials", fail_with_secret)
    adapter = _adapter(token=f"{{{{ env_var('{env_name}') }}}}")

    with pytest.raises(AdapterError) as exc_info:
        adapter._credentials()

    cause = exc_info.value.__cause__
    assert isinstance(cause, AdapterError)
    assert str(cause) == "Native adapter error type: RuntimeError"
    assert cause.__traceback__ is None
    assert cause.__context__ is None
    assert exc_info.value.__context__ is None
    rendered = "".join(
        (str(exc_info.value), repr(exc_info.value), str(cause), repr(cause))
    )
    assert "credential construction failed" in rendered
    assert sentinel not in rendered
    assert env_name not in rendered


def test_credentials_impersonation_wraps_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.auth import impersonated_credentials
    from google.oauth2 import service_account

    captured: dict[str, Any] = {}

    class _FakeSource:
        pass

    class _FakeImpersonated:
        def __init__(
            self, source_credentials: Any, target_principal: str, target_scopes: list[Any]
        ) -> None:
            captured["source"] = source_credentials
            captured["principal"] = target_principal

    monkeypatch.setattr(
        service_account.Credentials,
        "from_service_account_file",
        staticmethod(lambda path, scopes=None: _FakeSource()),
    )
    monkeypatch.setattr(impersonated_credentials, "Credentials", _FakeImpersonated)

    adapter = _adapter(
        keyfile="./sa.json",
        impersonate_service_account="runner@proj.iam.gserviceaccount.com",
    )
    creds = adapter._credentials()
    assert isinstance(creds, _FakeImpersonated)
    assert isinstance(captured["source"], _FakeSource)
    assert captured["principal"] == "runner@proj.iam.gserviceaccount.com"


def test_environment_keyfile_resolves_relative_to_project_at_sdk_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.oauth2 import service_account

    captured: dict[str, Any] = {}

    class _FakeCredentials:
        pass

    def fake_from_file(path: str, scopes: Any = None) -> _FakeCredentials:
        captured["path"] = path
        captured["scopes"] = scopes
        return _FakeCredentials()

    monkeypatch.setenv("BQ_KEYFILE_PATH", "./secrets/service-account.json")
    monkeypatch.setattr(
        service_account.Credentials,
        "from_service_account_file",
        staticmethod(fake_from_file),
    )
    config = parse_warehouse_config(
        {
            "type": "bigquery",
            "project": "proj",
            "keyfile": "{{ env_var('BQ_KEYFILE_PATH') }}",
        }
    )
    adapter = create_adapter(config, project_dir=tmp_path)
    assert isinstance(adapter, BigQueryAdapter)

    credentials = adapter._credentials()

    assert isinstance(credentials, _FakeCredentials)
    assert captured["path"] == str(
        (tmp_path / "secrets" / "service-account.json").resolve()
    )


# ─── execution / billing options ────────────────────────────────────────────


def test_default_job_config_priority_and_cost_cap() -> None:
    adapter = _adapter(priority="batch", maximum_bytes_billed=10**9)
    job_config = adapter._default_job_config()
    assert job_config is not None
    assert job_config.priority == "BATCH"
    assert job_config.maximum_bytes_billed == 10**9

    assert _adapter()._default_job_config() is None


def test_execution_project_bills_elsewhere_data_stays_qualified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.cloud import bigquery
    from google.oauth2 import service_account

    captured: dict[str, Any] = {}

    class _FakeCreds:
        pass

    monkeypatch.setattr(
        service_account.Credentials,
        "from_service_account_info",
        staticmethod(lambda info, scopes=None: _FakeCreds()),
    )

    def fake_client(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "client"

    monkeypatch.setattr(bigquery, "Client", fake_client)
    monkeypatch.setenv(
        "BQ_SERVICE_ACCOUNT_JSON", '{"type":"service_account"}'
    )

    adapter = _adapter(
        keyfile_json="{{ env_var('BQ_SERVICE_ACCOUNT_JSON') }}",
        execution_project="billing-proj",
    )
    assert adapter._make_client() == "client"
    assert captured["project"] == "billing-proj"
    # table refs still point at the data project
    assert adapter.table_ref("docs") == "`proj`.`ds`.`docs`"


def test_job_retry_and_timeout_wiring() -> None:
    client = _FakeClient()
    adapter = _adapter(
        client,
        job_retries=0,
        job_creation_timeout_seconds=7.0,
        job_execution_timeout_seconds=99.0,
    )
    job = adapter._run_query("SELECT 1")
    assert client.query_kwargs[0]["timeout"] == 7.0
    assert client.query_kwargs[0]["job_retry"] is None
    assert job.result_timeout == 99.0


def test_job_retry_deadline_applied() -> None:
    client = _FakeClient()
    adapter = _adapter(client, job_retry_deadline_seconds=120.0)
    adapter._run_query("SELECT 1")
    job_retry = client.query_kwargs[0]["job_retry"]
    assert job_retry is not None
    assert job_retry._deadline == 120.0


# ─── materialize_full_chunks (issue #77) ─────────────────────────────────────


def test_full_chunks_stages_then_swaps() -> None:
    from google.cloud import bigquery

    client = _FakeClient()
    adapter = _adapter(client)
    total = adapter.materialize_full_chunks(
        "docs",
        iter(
            [
                pl.DataFrame({"document_id": ["a"], "x": [1]}),
                pl.DataFrame({"document_id": ["b"], "extra": ["?"]}),
            ]
        ),
    )
    assert total == 2

    # Chunk 1 truncates the staging table; chunk 2 appends with field addition.
    _, staging_id, cfg1 = client.loads[0]
    assert staging_id.startswith("proj.ds.dbt_ml_staging__docs__")
    assert cfg1.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE
    _, _, cfg2 = client.loads[1]
    assert cfg2.write_disposition == bigquery.WriteDisposition.WRITE_APPEND
    assert cfg2.schema_update_options == [
        bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
    ]

    # Swap into the target, then drop staging.
    swap_sql = client.queries[0][0]
    assert "CREATE OR REPLACE TABLE `proj`.`ds`.`docs`" in swap_sql
    staging_table = staging_id.removeprefix("proj.ds.")
    assert f"`proj`.`ds`.`{staging_table}`" in swap_sql
    assert client.dropped == [staging_id]


def test_full_chunks_empty_iterator_drops_target() -> None:
    client = _FakeClient()
    adapter = _adapter(client)
    assert adapter.materialize_full_chunks("docs", iter([])) == 0
    assert client.loads == []
    # materialize_full(empty) drops the target; staging cleanup is a no-op drop.
    assert "proj.ds.docs" in client.dropped


def test_full_chunks_typed_empty_frame_loads_and_swaps() -> None:
    client = _FakeClient()
    adapter = _adapter(client)
    frame = pl.DataFrame(
        schema={
            "document_id": pl.String,
            "count": pl.Int64,
            "score": pl.Float64,
            "active": pl.Boolean,
        }
    )

    assert adapter.materialize_full_chunks("docs", iter([frame])) == 0

    payload, staging_id, _ = client.loads[0]
    loaded = pl.read_parquet(io.BytesIO(payload))
    assert loaded.schema == frame.schema
    assert loaded.height == 0
    assert "CREATE OR REPLACE TABLE `proj`.`ds`.`docs`" in client.queries[0][0]
    assert client.dropped == [staging_id]


def test_list_tables_excludes_staging() -> None:
    client = _FakeClient()
    client.listing = ["docs", "dbt_ml_staging__docs", "dbt_ml_state"]
    adapter = _adapter(client)
    assert adapter.list_tables() == ["docs"]
# ─── paged state reconciliation (issue #153) ────────────────────────────────


class _MessageFailingJob(_FakeJob):
    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def result(self, timeout: Any = None) -> list[_FakeRow]:
        raise RuntimeError(self._message)


def _param_values(job_config: Any) -> list[Any]:
    return [
        parameter.values
        if hasattr(parameter, "values")
        else parameter.value
        for parameter in job_config.query_parameters
    ]


def test_fetch_state_subset_is_a_bounded_array_lookup() -> None:
    client = _FakeClient()
    client.query_results = [_FakeJob(rows=[("a", "fp-a", "v1")])]
    adapter = _adapter(client)
    subset = adapter.fetch_state_subset(StateScope("m"), ["a", "b"])
    assert subset == {"a": StateValue("fp-a", "v1")}
    sql, job_config = client.queries[0]
    assert "record_key IN UNNEST(?)" in sql
    assert _param_values(job_config)[-1] == ["a", "b"]


def test_state_page_reader_pins_one_system_time_snapshot() -> None:
    ts = datetime.now(UTC)
    client = _FakeClient()
    client.query_results = [
        _FakeJob(rows=[(ts,)]),
        _FakeJob(rows=[("a", "fp-a", "v1", ts), ("b", "fp-b", "v1", ts)]),
        _FakeJob(rows=[("c", "fp-c", "v1", ts)]),
    ]
    adapter = _adapter(client)
    with adapter.state_page_reader(StateScope("m"), page_size=2) as reader:
        first = reader.fetch_page(None)
        assert [record.record_key for record in first.records] == ["a", "b"]
        assert first.next_cursor is not None
        second = reader.fetch_page(first.next_cursor)
        assert [record.record_key for record in second.records] == ["c"]
        assert second.next_cursor is None

    assert client.queries[0][0] == "SELECT CURRENT_TIMESTAMP()"
    page_sql, page_config = client.queries[1]
    assert "FOR SYSTEM_TIME AS OF ?" in page_sql
    # BigQuery requires the alias before the time-travel clause; the reverse
    # order is a syntax error that blocks every state page read (#249).
    assert "AS state FOR SYSTEM_TIME AS OF ?" in page_sql
    assert "FOR SYSTEM_TIME AS OF ? AS state" not in page_sql
    assert "ORDER BY state.record_key" in page_sql
    assert "LIMIT ?" in page_sql
    assert _param_values(page_config)[0] == ts
    keyset_sql, keyset_config = client.queries[2]
    assert "state.record_key > ?" in keyset_sql
    # Same pinned timestamp plus the keyset position from page one.
    values = _param_values(keyset_config)
    assert values[0] == ts
    assert "b" in values


def test_state_page_reader_rejects_foreign_cursors() -> None:
    ts = datetime.now(UTC)
    client = _FakeClient()
    client.query_results = [_FakeJob(rows=[(ts,)])]
    adapter = _adapter(client)
    with adapter.state_page_reader(StateScope("m"), page_size=2) as reader:
        with pytest.raises(AdapterError, match="does not belong"):
            reader.fetch_page("deadbeef:aGVsbG8=")


def test_state_page_reader_absence_probe_time_travels_both_relations() -> None:
    ts = datetime.now(UTC)
    client = _FakeClient()
    client.tables["proj.ds.chunks"] = ["chunk_id", "text"]
    client.query_results = [_FakeJob(rows=[(ts,)]), _FakeJob(rows=[])]
    adapter = _adapter(client)
    probe = StateAbsenceProbe(table="chunks", key_column="chunk_id")
    with adapter.state_page_reader(
        StateScope("m"), page_size=5, absent_from=probe
    ) as reader:
        page = reader.fetch_page(None)
    assert page.records == ()
    assert page.next_cursor is None
    sql, _ = client.queries[1]
    assert sql.count("FOR SYSTEM_TIME AS OF ?") == 2
    # Both relations must alias before the time-travel clause (#249).
    assert "AS state FOR SYSTEM_TIME AS OF ?" in sql
    assert "AS probe FOR SYSTEM_TIME AS OF ?" in sql
    assert "FOR SYSTEM_TIME AS OF ? AS" not in sql
    assert "NOT EXISTS" in sql
    assert "`chunk_id` = state.record_key" in sql


def test_state_page_reader_missing_probe_relation_fails() -> None:
    ts = datetime.now(UTC)
    client = _FakeClient()
    client.query_results = [_FakeJob(rows=[(ts,)])]
    adapter = _adapter(client)
    probe = StateAbsenceProbe(table="missing", key_column="chunk_id")
    with pytest.raises(AdapterError, match="absence probe"):
        with adapter.state_page_reader(
            StateScope("m"), page_size=5, absent_from=probe
        ):
            pass

    client.query_results = [_FakeJob(rows=[(ts,)])]
    client.tables["proj.ds.chunks"] = ["other_column"]
    probe = StateAbsenceProbe(table="chunks", key_column="chunk_id")
    with pytest.raises(AdapterError, match="absence probe"):
        with adapter.state_page_reader(
            StateScope("m"), page_size=5, absent_from=probe
        ):
            pass


def test_state_page_read_failure_preserves_cause_without_leaking_text() -> None:
    # The read used to swallow the warehouse error with `from None`, hiding
    # syntax faults like #249. The exception chain must be preserved via
    # `from exc`, but the artifact-visible message must stay generic so raw
    # warehouse text (SQL/response details) never reaches run_results.json
    # or the CLI (AGENTS.md).
    ts = datetime.now(UTC)
    client = _FakeClient()
    client.query_results = [_FakeJob(rows=[(ts,)]), _FailingJob()]
    adapter = _adapter(client)
    with adapter.state_page_reader(StateScope("m"), page_size=2) as reader:
        with pytest.raises(AdapterError) as excinfo:
            reader.fetch_page(None)
    assert str(excinfo.value) == "BigQuery state page read failed"
    assert "simulated merge failure" not in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_replace_state_scope_stages_batches_and_merges_once() -> None:
    client = _FakeClient()
    adapter = _adapter(client)
    written = adapter.replace_state_scope(
        StateScope("m"),
        iter(
            [
                [StateRecord("a", "fp-a", "v1")],
                [StateRecord("b", "fp-b", "v1")],
            ]
        ),
    )
    assert written == 2
    assert len(client.loads) == 2
    staging_id = client.loads[0][1]
    assert staging_id.startswith("proj.ds.dbt_ml_staging__state_replace__")
    assert client.loads[1][1] == staging_id
    assert all(
        load[2].write_disposition == "WRITE_APPEND" for load in client.loads
    )
    sql, _ = client.queries[0]
    assert sql.strip().startswith("MERGE `proj`.`ds`.`dbt_ml_state` AS target")
    assert "UNION ALL" in sql and "is_sentinel" in sql
    assert "WHEN NOT MATCHED BY SOURCE" in sql
    assert "dbt-ml-state-replace-invalid" in sql
    # Unfenced replacement must not consult the serving ledger.
    assert "dbt_ml_serving_ledger" not in sql
    assert client.dropped == [staging_id]


def test_replace_state_scope_empty_input_still_replaces_atomically() -> None:
    client = _FakeClient()
    adapter = _adapter(client)
    assert adapter.replace_state_scope(StateScope("m"), iter([])) == 0
    assert client.loads == []
    create_sql, _ = client.queries[0]
    assert create_sql.startswith("CREATE TABLE `proj`.`ds`.`dbt_ml_staging__")
    merge_sql, _ = client.queries[1]
    assert "WHEN NOT MATCHED BY SOURCE" in merge_sql
    assert len(client.dropped) == 1


def test_fenced_replace_conditions_on_the_serving_ledger() -> None:
    client = _FakeClient()
    adapter = _adapter(client)
    fence = StateScopeFence("pub123", 7)
    adapter.replace_state_scope(
        StateScope("m"),
        iter([[StateRecord("a", "fp-a", "v1")]]),
        fence=fence,
    )
    sql, job_config = client.queries[0]
    assert "`proj`.`ds`.`dbt_ml_serving_ledger`" in sql
    assert "publication_id = ? AND fencing_token = ?" in sql
    assert "dbt-ml-stale-state-fence" in sql
    values = _param_values(job_config)
    assert "pub123" in values
    assert 7 in values


def test_fenced_replace_maps_gate_errors_without_leaking() -> None:
    scope = StateScope("m")
    fence = StateScopeFence("pub123", 7)

    client = _FakeClient()
    client.query_results = [
        _MessageFailingJob("400 dbt-ml-stale-state-fence: publication reassigned")
    ]
    adapter = _adapter(client)
    with pytest.raises(StaleStateFenceError, match="reassigned"):
        adapter.replace_state_scope(
            scope, iter([[StateRecord("a", "fp", "v1")]]), fence=fence
        )
    assert len(client.dropped) == 1

    client = _FakeClient()
    client.query_results = [
        _MessageFailingJob("400 dbt-ml-state-replace-invalid: duplicate keys")
    ]
    adapter = _adapter(client)
    with pytest.raises(AdapterError, match="duplicate record keys"):
        adapter.replace_state_scope(
            scope, iter([[StateRecord("a", "fp", "v1")]]), fence=fence
        )
    assert len(client.dropped) == 1

    client = _FakeClient()
    client.query_results = [
        _MessageFailingJob("404 Not found: Table proj:ds.dbt_ml_serving_ledger")
    ]
    adapter = _adapter(client)
    with pytest.raises(StaleStateFenceError, match="serving ledger"):
        adapter.replace_state_scope(
            scope, iter([[StateRecord("a", "fp", "v1")]]), fence=fence
        )


# ─── SQL incremental materialization (issue #142) ──────────────────────────

# materialize_sql_incremental stages select_sql into a real table (named with a
# uuid4 suffix) before validating/merging it, so the checked and merged rowsets
# are always identical. Fixing uuid4 makes the staging table id predictable.
_STAGING_SUFFIX = "0" * 12
_STAGING_TABLE = f"dbt_ml_staging__tgt__{_STAGING_SUFFIX}"
_STAGING_ID = f"proj.ds.{_STAGING_TABLE}"


@pytest.fixture
def _fixed_staging_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    import uuid as uuid_module

    monkeypatch.setattr(
        "dbt_ml.adapters.bigquery.uuid4", lambda: uuid_module.UUID(int=0)
    )


def _stage_schema(client: _FakeClient, columns: list[tuple[str, str]]) -> None:
    # materialize_sql_incremental reads the staged table's schema via
    # get_table(), not a dry-run query, so register it the same way as any
    # other existing table.
    client.tables[_STAGING_ID] = [
        SimpleNamespace(name=name, field_type=field_type) for name, field_type in columns
    ]


def test_bigquery_relation_exists() -> None:
    client = _FakeClient()
    adapter = _adapter(client)
    assert adapter.relation_exists("missing") is False
    client.tables["proj.ds.present"] = ["id"]
    assert adapter.relation_exists("present") is True


def test_bigquery_materialize_sql_incremental_builds_merge(_fixed_staging_uuid) -> None:
    client = _FakeClient()
    client.tables["proj.ds.tgt"] = ["id", "v"]
    _stage_schema(client, [("id", "INTEGER"), ("v", "STRING")])
    client.query_results = [
        _FakeJob(),  # CREATE TABLE staging AS select_sql
        _FakeJob(rows=[(0, 0)]),  # unique-key check: 0 null, 0 duplicate
        _FakeJob(affected=3),  # the MERGE itself
    ]
    adapter = _adapter(client)
    result = adapter.materialize_sql_incremental(
        "tgt", "SELECT id, v FROM src", unique_key="id"
    )
    assert result.rows_written == 3

    create_sql, _ = client.queries[0]
    assert create_sql == f"CREATE TABLE `proj`.`ds`.`{_STAGING_TABLE}` AS SELECT id, v FROM src"
    check_sql, _ = client.queries[1]
    # The key check reads the staged table, never select_sql a second time.
    assert "SELECT id, v FROM src" not in check_sql
    assert f"`proj`.`ds`.`{_STAGING_TABLE}`" in check_sql
    merge_sql, _ = client.queries[-1]
    assert merge_sql.startswith(
        f"MERGE `proj`.`ds`.`tgt` AS T USING `proj`.`ds`.`{_STAGING_TABLE}` AS S"
    )
    assert "SELECT id, v FROM src" not in merge_sql  # merged from staging, not re-run
    assert "ON T.`id` = S.`id`" in merge_sql
    assert "WHEN MATCHED THEN UPDATE SET T.`v` = S.`v`" in merge_sql
    assert "WHEN NOT MATCHED THEN INSERT (`id`, `v`) VALUES (S.`id`, S.`v`)" in merge_sql
    # The staging table is always dropped, success or failure.
    assert client.dropped == [_STAGING_ID]


def test_bigquery_materialize_sql_incremental_rejects_bad_key(_fixed_staging_uuid) -> None:
    client = _FakeClient()
    client.tables["proj.ds.tgt"] = ["id", "v"]
    client.query_results = [
        _FakeJob(),  # CREATE TABLE staging
        _FakeJob(rows=[(1, 2)]),  # 1 null, 2 duplicates
    ]
    adapter = _adapter(client)
    with pytest.raises(AdapterError, match="1 null and 2 duplicate"):
        adapter.materialize_sql_incremental(
            "tgt", "SELECT id, v FROM src", unique_key="id"
        )
    assert client.dropped == [_STAGING_ID]


def test_bigquery_materialize_sql_incremental_rejects_key_missing_from_target(
    _fixed_staging_uuid,
) -> None:
    # The target's unique_key changed (or never had this column); appending it
    # as a schema-drift "new column" would leave existing rows keyless.
    client = _FakeClient()
    client.tables["proj.ds.tgt"] = ["old_id", "v"]
    _stage_schema(client, [("id", "INTEGER"), ("v", "STRING")])
    client.query_results = [
        _FakeJob(),  # CREATE TABLE staging
        _FakeJob(rows=[(0, 0)]),  # key check passes
    ]
    adapter = _adapter(client)
    with pytest.raises(AdapterError, match="does not exist in the current target"):
        adapter.materialize_sql_incremental(
            "tgt",
            "SELECT id, v FROM src",
            unique_key="id",
            on_schema_change="append_new_columns",
        )
    assert client.dropped == [_STAGING_ID]


def test_bigquery_materialize_sql_incremental_schema_change_fail(
    _fixed_staging_uuid,
) -> None:
    client = _FakeClient()
    client.tables["proj.ds.tgt"] = ["id", "v"]
    _stage_schema(client, [("id", "INTEGER"), ("v", "STRING"), ("w", "STRING")])
    client.query_results = [
        _FakeJob(),  # CREATE TABLE staging
        _FakeJob(rows=[(0, 0)]),
    ]
    adapter = _adapter(client)
    with pytest.raises(AdapterError, match="Schema change"):
        adapter.materialize_sql_incremental(
            "tgt", "SELECT id, v, w FROM src", unique_key="id"
        )
    assert client.dropped == [_STAGING_ID]


def test_bigquery_materialize_sql_incremental_appends_new_columns(
    _fixed_staging_uuid,
) -> None:
    client = _FakeClient()
    client.tables["proj.ds.tgt"] = ["id", "v"]
    _stage_schema(client, [("id", "INTEGER"), ("v", "STRING"), ("w", "STRING")])
    client.query_results = [
        _FakeJob(),  # CREATE TABLE staging
        _FakeJob(rows=[(0, 0)]),  # key check
        _FakeJob(),  # ALTER TABLE ADD COLUMN
        _FakeJob(affected=1),  # the MERGE
    ]
    adapter = _adapter(client)
    adapter.materialize_sql_incremental(
        "tgt",
        "SELECT id, v, w FROM src",
        unique_key="id",
        on_schema_change="append_new_columns",
    )
    alter_sql, _ = client.queries[-2]
    assert alter_sql == "ALTER TABLE `proj`.`ds`.`tgt` ADD COLUMN `w` STRING"
    merge_sql, _ = client.queries[-1]
    assert "`w`" in merge_sql
    assert client.dropped == [_STAGING_ID]


# ── #260: batch the unbounded array-param delete/fetch methods ───────────────


def _array_param(job_config: Any) -> Any:
    from google.cloud import bigquery

    arrays = [
        p
        for p in job_config.query_parameters
        if isinstance(p, bigquery.ArrayQueryParameter)
    ]
    assert len(arrays) == 1
    return arrays[0]


def test_delete_rows_batches_large_key_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dbt_ml.adapters.bigquery._KEY_REQUEST_BATCH", 2)
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id"]
    client.query_results = [_FakeJob(affected=2), _FakeJob(affected=2), _FakeJob(affected=1)]
    adapter = _adapter(client)

    total = adapter.delete_rows(
        "docs", key_col="document_id", keys=["a", "b", "c", "d", "e"]
    )
    assert total == 5  # affected counts summed across batches
    deletes = [cfg for sql, cfg in client.queries if sql.strip().startswith("DELETE")]
    assert len(deletes) == 3  # 5 keys / batch 2 -> 3 bounded requests
    for cfg in deletes:
        assert len(_array_param(cfg).values) <= 2
    batched = [k for cfg in deletes for k in _array_param(cfg).values]
    assert sorted(batched) == ["a", "b", "c", "d", "e"]


def test_delete_state_batches_large_key_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dbt_ml.adapters.bigquery._KEY_REQUEST_BATCH", 2)
    client = _FakeClient()
    adapter = _adapter(client)

    adapter.delete_state(StateScope("m"), [f"k{i}" for i in range(5)])
    deletes = [cfg for sql, cfg in client.queries if sql.strip().startswith("DELETE")]
    assert len(deletes) == 3
    for cfg in deletes:
        assert len(_array_param(cfg).values) <= 2
    batched = [k for cfg in deletes for k in _array_param(cfg).values]
    assert sorted(batched) == [f"k{i}" for i in range(5)]


def test_fetch_state_subset_batches_and_merges(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dbt_ml.adapters.bigquery._KEY_REQUEST_BATCH", 2)
    client = _FakeClient()
    client.query_results = [
        _FakeJob(rows=[("k0", "h0", "v0"), ("k1", "h1", "v1")]),
        _FakeJob(rows=[("k2", "h2", "v2"), ("k3", "h3", "v3")]),
        _FakeJob(rows=[("k4", "h4", "v4")]),
    ]
    adapter = _adapter(client)

    result = adapter._fetch_state_subset(StateScope("m"), [f"k{i}" for i in range(5)])
    assert result == {f"k{i}": StateValue(f"h{i}", f"v{i}") for i in range(5)}
    selects = [sql for sql, _ in client.queries if sql.strip().startswith("SELECT")]
    assert len(selects) == 3


# ── #260: stage the atomic array-param methods at scale ──────────────────────


def test_delete_rows_and_state_stages_at_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dbt_ml.adapters.bigquery._STATE_MERGE_INLINE_MAX", 2)
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id"]
    client.query_results = [_FakeJob(rows=[(3,)])]  # SELECT deleted_count
    adapter = _adapter(client)

    deleted = adapter.delete_rows_and_state(
        "docs",
        key_col="document_id",
        keys=[f"k{i}" for i in range(3)],
        state_scope=StateScope("m"),
    )
    assert deleted == 3
    # Keys were staged via a Parquet load, not shipped as an array param.
    staged = {tbl for _, tbl, _ in client.loads}
    assert any("dbt_ml_staging__delete_target__" in t for t in staged)
    from google.cloud import bigquery

    script, cfg = client.queries[-1]
    assert "BEGIN TRANSACTION" in script  # still one atomic transaction
    assert "IN (SELECT k FROM" in script
    assert "IN UNNEST" not in script
    assert not any(
        isinstance(p, bigquery.ArrayQueryParameter) for p in (cfg.query_parameters or [])
    )
    # scoped_keys defaulted to target_keys, so the state delete reuses one staging
    # table (no second load).
    assert len(staged) == 1
    assert any("dbt_ml_staging__delete_target__" in t for t in client.dropped)


def test_replace_children_stages_parents_and_state_at_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dbt_ml.adapters.bigquery._STATE_MERGE_INLINE_MAX", 2)
    client = _FakeClient()
    client.tables["proj.ds.chunks"] = ["parent_id", "child_id"]  # existing target
    client.tables["proj.ds.dbt_ml_state"] = list(_STATE_V2_SCHEMA)
    adapter = _adapter(client)

    empty = pl.DataFrame(schema={"parent_id": pl.String, "child_id": pl.String})
    adapter.replace_children(
        "chunks",
        parent_key="parent_id",
        parent_ids=[f"p{i}" for i in range(3)],
        child_key="child_id",
        new_rows=empty,
        state_scope=StateScope("m"),
        state_records=[StateRecord(f"p{i}", f"h{i}", "v1") for i in range(3)],
    )
    staged = {tbl for _, tbl, _ in client.loads}
    assert any("dbt_ml_staging__replace_parents__" in t for t in staged)
    assert any("dbt_ml_staging__replace_state__" in t for t in staged)

    script = next(sql for sql, _ in client.queries if "BEGIN TRANSACTION" in sql)
    assert "IN (SELECT k FROM" in script  # parent delete reads the staging table
    assert "dbt_ml_staging__replace_state__" in script  # state MERGE reads staging
    assert "UNNEST(?)" not in script and "GENERATE_ARRAY" not in script
    # Both staging tables are cleaned up.
    assert any("dbt_ml_staging__replace_parents__" in t for t in client.dropped)
    assert any("dbt_ml_staging__replace_state__" in t for t in client.dropped)


def test_delete_rows_and_state_staging_preserves_native_key_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A non-string key column (e.g. INT64) must not be coerced to STRING in the
    # staging table, or the staged subquery would mismatch the native column and
    # BigQuery would reject the delete (#260 review).
    import io

    monkeypatch.setattr("dbt_ml.adapters.bigquery._STATE_MERGE_INLINE_MAX", 2)
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["id"]
    client.query_results = [_FakeJob(rows=[(0,)])]
    adapter = _adapter(client)

    adapter.delete_rows_and_state(
        "docs",
        key_col="id",
        keys=[1, 2, 3],  # INT64 target keys
        state_scope=StateScope("m"),
        state_record_keys=["a", "b", "c"],  # STRING state keys
    )
    loads = {tbl: payload for payload, tbl, _ in client.loads}
    target_tbl = next(t for t in loads if "delete_target" in t)
    state_tbl = next(t for t in loads if "delete_state" in t)
    target_df = pl.read_parquet(io.BytesIO(loads[target_tbl]))
    state_df = pl.read_parquet(io.BytesIO(loads[state_tbl]))
    assert target_df["k"].dtype == pl.Int64   # native INT64 preserved
    assert state_df["k"].dtype == pl.String


def test_bigquery_sql_error_omits_raw_warehouse_text() -> None:
    # #262: BigQuery SQL-model errors must not interpolate raw warehouse text
    # (which can carry SQL fragments or row values) into the artifact-visible
    # AdapterError message. Only the exception class is surfaced; the raw detail
    # stays on the chained cause.
    from google.api_core.exceptions import BadRequest

    class _RaisingClient(_FakeClient):
        def query(self, sql: str, job_config: Any = None, **kwargs: Any) -> _FakeJob:
            raise BadRequest("Syntax error: unexpected SECRET_VALUE at [1:5]")

    adapter = _adapter(_RaisingClient())

    with pytest.raises(AdapterError) as excinfo:
        adapter.dry_run_sql("SELECT SECRET_VALUE")

    message = str(excinfo.value)
    assert message == "SQL dry-run failed [BadRequest]"
    assert "SECRET_VALUE" not in message
    cause = excinfo.value.__cause__
    assert cause is not None
    assert "SECRET_VALUE" in str(cause)
