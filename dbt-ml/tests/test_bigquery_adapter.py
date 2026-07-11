"""BigQuery adapter (issue #83): config, SQL dialect, load planning, and
client interactions against a fake; real round-trips run only when
DBT_ML_BQ_TEST_PROJECT is set."""
from __future__ import annotations

import io
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from dbt_ml.adapters import (
    AdapterError,
    create_adapter,
    list_adapter_types,
    parse_warehouse_config,
)
from dbt_ml.adapters.bigquery import (
    BigQueryAdapter,
    BigQueryWarehouseConfig,
    plan_schema_change,
    to_query_parameters,
)

# ─── config ─────────────────────────────────────────────────────────────────


def test_bigquery_registered() -> None:
    assert "bigquery" in list_adapter_types()


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


# ─── schema change planning ─────────────────────────────────────────────────


def test_plan_no_drift() -> None:
    plan = plan_schema_change(["a", "b"], ["a", "b"], "fail", "t")
    assert plan.columns_to_load == ["a", "b"]
    assert not plan.allow_field_addition


def test_plan_fail_on_new_and_removed() -> None:
    with pytest.raises(AdapterError, match="full-refresh"):
        plan_schema_change(["a"], ["a", "b"], "fail", "t")
    with pytest.raises(AdapterError, match="removed columns"):
        plan_schema_change(["a", "b"], ["a"], "fail", "t")


def test_plan_append_new_columns() -> None:
    plan = plan_schema_change(["a"], ["a", "b"], "append_new_columns", "t")
    assert plan.columns_to_load == ["a", "b"]
    assert plan.allow_field_addition
    # removed-only drift needs no field addition
    plan = plan_schema_change(["a", "b"], ["a"], "append_new_columns", "t")
    assert not plan.allow_field_addition


def test_plan_ignore_drops_new_columns() -> None:
    plan = plan_schema_change(["a"], ["a", "b"], "ignore", "t")
    assert plan.columns_to_load == ["a"]


def test_plan_unknown_policy() -> None:
    with pytest.raises(AdapterError, match="Unknown on_schema_change"):
        plan_schema_change(["a"], ["a", "b"], "sync_all", "t")


# ─── client interactions against a fake ─────────────────────────────────────


class _FakeRow(tuple):
    def values(self) -> tuple[Any, ...]:
        return tuple(self)


class _FakeJob:
    def __init__(self, rows: list[tuple] | None = None, affected: int | None = None):
        self._rows = [_FakeRow(r) for r in (rows or [])]
        self.num_dml_affected_rows = affected
        self.result_timeout: Any = "unset"

    def result(self, timeout: Any = None) -> list[_FakeRow]:
        self.result_timeout = timeout
        return list(self._rows)


class _FailingJob(_FakeJob):
    def result(self, timeout: Any = None) -> list[_FakeRow]:
        raise RuntimeError("simulated merge failure")


class _FakeClient:
    def __init__(self) -> None:
        self.queries: list[tuple[str, Any]] = []
        self.query_kwargs: list[dict[str, Any]] = []
        self.loads: list[tuple[bytes, str, Any]] = []
        self.tables: dict[str, list[str]] = {}
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
        return SimpleNamespace(
            schema=[SimpleNamespace(name=c) for c in self.tables[table_id]]
        )

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


def test_state_upsert_is_single_merge() -> None:
    client = _FakeClient()
    adapter = _adapter(client)
    adapter.upsert_state("m1", [("d1", "h1", "v1"), ("d2", "h2", "v2")])

    sql, job_config = client.queries[0]
    assert sql.strip().startswith("MERGE")
    assert "OFFSET" in sql
    params = job_config.query_parameters
    assert params[0].values == ["d1", "d2"]
    assert params[3].value == "m1"


def test_fetch_state_round_trip_shape() -> None:
    client = _FakeClient()
    client.query_results = [_FakeJob(rows=[("d1", "h1", "v1"), ("d2", "h2", "v2")])]
    adapter = _adapter(client)
    assert adapter.fetch_state("m1") == {"d1": ("h1", "v1"), "d2": ("h2", "v2")}


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


def test_materialize_full_applies_layout_and_recreates() -> None:
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

    # dropped first: a load job cannot change an existing table's layout
    assert client.dropped == ["proj.ds.docs"]
    _, _, job_config = client.loads[0]
    assert job_config.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE
    assert job_config.time_partitioning.type_ == "DAY"
    assert job_config.time_partitioning.field == "filing_date"
    assert job_config.clustering_fields == ["cik"]


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
    swap_sql = client.queries[0][0]
    assert (
        "CREATE OR REPLACE TABLE `proj`.`ds`.`docs` "
        "PARTITION BY DATE_TRUNC(`filing_date`, MONTH) "
        "CLUSTER BY `cik`, `form_type` AS SELECT" in swap_sql
    )
    # target dropped before the swap so the new layout can apply
    assert client.dropped[0] == "proj.ds.docs"


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

            adapter.upsert_state("m", [("a", "h", "v")])
            adapter.upsert_state("m", [("a", "h2", "v2")])
            assert adapter.fetch_state("m") == {"a": ("h2", "v2")}
            assert adapter.list_tables() == ["docs"]
    finally:
        assert isinstance(adapter, BigQueryAdapter)
        adapter._reset_storage_for_test()


# ─── auth parity with dbt-bigquery ──────────────────────────────────────────


def test_auth_method_inference() -> None:
    assert _bq_cfg().method == "oauth"
    assert _bq_cfg(keyfile="./sa.json").method == "service-account"
    assert _bq_cfg(keyfile_json={"type": "service_account"}).method == (
        "service-account-json"
    )
    assert _bq_cfg(token="tok").method == "oauth-secrets"


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
        _bq_cfg(keyfile="./sa.json", keyfile_json={"a": 1})


def test_oauth_secrets_requires_token_or_full_refresh_set() -> None:
    with pytest.raises(AdapterError, match="oauth-secrets"):
        _bq_cfg(method="oauth-secrets", refresh_token="r", client_id="c")
    cfg = _bq_cfg(
        refresh_token="r", client_id="c", client_secret="s",
        token_uri="https://oauth2.googleapis.com/token",
    )
    assert cfg.method == "oauth-secrets"


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


def test_credentials_service_account_json_scopes_and_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.oauth2 import service_account

    captured: dict[str, Any] = {}

    class _FakeCreds:
        def with_quota_project(self, qp: str) -> _FakeCreds:
            captured["quota_project"] = qp
            return self

    def fake_from_info(info: dict, scopes: Any = None) -> _FakeCreds:
        captured["info"] = info
        captured["scopes"] = scopes
        return _FakeCreds()

    monkeypatch.setattr(
        service_account.Credentials,
        "from_service_account_info",
        staticmethod(fake_from_info),
    )
    adapter = _adapter(
        keyfile_json={"type": "service_account"}, quota_project="bill-here"
    )
    creds = adapter._credentials()
    assert isinstance(creds, _FakeCreds)
    assert captured["info"] == {"type": "service_account"}
    assert list(captured["scopes"]) == _bq_cfg().scopes
    assert captured["quota_project"] == "bill-here"


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
            self, source_credentials: Any, target_principal: str, target_scopes: list
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


# ─── execution / billing options ────────────────────────────────────────────


def test_default_job_config_priority_and_cost_cap() -> None:
    adapter = _adapter(priority="batch", maximum_bytes_billed=10**9)
    job_config = adapter._default_job_config()
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

    adapter = _adapter(
        keyfile_json={"type": "service_account"}, execution_project="billing-proj"
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
