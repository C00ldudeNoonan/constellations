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


def test_incremental_upsert_deletes_then_appends() -> None:
    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id", "x"]
    adapter = _adapter(client)
    df = pl.DataFrame({"document_id": ["a", "b"], "x": [1, 2]})
    adapter.materialize_incremental("docs", df, key_col="document_id")

    sql, job_config = client.queries[0]
    assert "DELETE FROM `proj`.`ds`.`docs`" in sql
    assert "IN UNNEST(?)" in sql
    assert job_config.query_parameters[0].values == ["a", "b"]
    assert len(client.loads) == 1


def test_incremental_append_new_columns_sets_schema_update() -> None:
    from google.cloud import bigquery

    client = _FakeClient()
    client.tables["proj.ds.docs"] = ["document_id"]
    adapter = _adapter(client)
    df = pl.DataFrame({"document_id": ["a"], "extra": ["new"]})
    adapter.materialize_incremental(
        "docs", df, key_col="document_id", on_schema_change="append_new_columns"
    )
    _, _, job_config = client.loads[0]
    assert job_config.schema_update_options == [
        bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
    ]


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
        adapter.clean()


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
