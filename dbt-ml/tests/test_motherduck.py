"""MotherDuck as a DuckDB deployment mode (issue #186).

MotherDuck reuses the DuckDB adapter and its capability contract; only the
connection differs (`md:` URI + a protected token). These tests pin the config
surface, the credential protection (no token in repr/serialization/manifest),
and that the token is revealed only into the native connection. A
credential-gated integration test exercises a real round-trip when
`MOTHERDUCK_TOKEN` is present.
"""

from __future__ import annotations

import os

import pytest

from dbt_ml.adapters import AdapterError, create_adapter, parse_warehouse_config
from dbt_ml.adapters.duckdb import DuckDBAdapter, DuckDBWarehouseConfig
from dbt_ml.credentials import CredentialReference


def _md_config(**overrides: object) -> DuckDBWarehouseConfig:
    raw: dict[str, object] = {
        "type": "duckdb",
        "schema": "analytics",
        "path": "md:economic_data",
        "token": "{{ env_var('MOTHERDUCK_TOKEN') }}",
    }
    raw.update(overrides)
    cfg = parse_warehouse_config(raw)
    assert isinstance(cfg, DuckDBWarehouseConfig)
    return cfg


# ─── config surface ─────────────────────────────────────────────────────────


def test_md_path_is_motherduck_and_reports_remote_shape() -> None:
    cfg = _md_config()
    assert cfg.is_motherduck
    assert cfg.storage_location() == "md:economic_data"
    assert cfg.catalog_name() == "economic_data"
    assert cfg.local_path() is None
    assert isinstance(cfg.token, CredentialReference)


def test_bare_md_uses_account_default_database() -> None:
    cfg = _md_config(path="md:")
    assert cfg.is_motherduck
    assert cfg.catalog_name() == "my_db"


def test_md_query_params_are_stripped_from_catalog_name() -> None:
    cfg = _md_config(path="md:economic_data?attach_mode=single")
    assert cfg.catalog_name() == "economic_data"


def test_local_path_is_not_motherduck() -> None:
    cfg = parse_warehouse_config({"type": "duckdb", "path": "./target/x.duckdb"})
    assert isinstance(cfg, DuckDBWarehouseConfig)
    assert not cfg.is_motherduck
    assert cfg.local_path() is not None


# ─── credential protection ──────────────────────────────────────────────────


def test_token_never_appears_in_repr_or_serialization() -> None:
    cfg = _md_config()
    assert "MOTHERDUCK_TOKEN" not in repr(cfg)
    dumped = cfg.model_dump()
    assert "token" not in dumped
    # storage_location / catalog_name feed the manifest target block; neither
    # carries the secret.
    assert "token" not in cfg.storage_location().lower()


def test_literal_token_is_rejected() -> None:
    with pytest.raises(AdapterError):
        parse_warehouse_config(
            {"type": "duckdb", "path": "md:x", "token": "literal-secret"}
        )


def test_token_requires_a_motherduck_path() -> None:
    with pytest.raises(AdapterError):
        parse_warehouse_config(
            {
                "type": "duckdb",
                "path": "./local.duckdb",
                "token": "{{ env_var('MOTHERDUCK_TOKEN') }}",
            }
        )


# ─── connection (deterministic fake) ────────────────────────────────────────


class _FakeResult:
    def fetchone(self) -> tuple[str]:
        return ("economic_data",)


class _FakeConnection:
    def execute(self, _sql: str) -> _FakeResult:
        return _FakeResult()


def test_connect_passes_md_uri_and_reveals_token_only_into_native_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOTHERDUCK_TOKEN", "sk-motherduck-secret")
    captured: dict[str, object] = {}

    def fake_connect(target: str, **kwargs: object) -> _FakeConnection:
        captured["target"] = target
        captured["kwargs"] = kwargs
        return _FakeConnection()

    monkeypatch.setattr("dbt_ml.adapters.duckdb.duckdb.connect", fake_connect)

    adapter = DuckDBAdapter(_md_config())
    adapter._connect()

    assert captured["target"] == "md:economic_data"
    assert captured["kwargs"] == {
        "config": {"motherduck_token": "sk-motherduck-secret"}
    }


def test_connect_without_token_lets_duckdb_read_its_own_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_connect(target: str, **kwargs: object) -> _FakeConnection:
        captured["target"] = target
        captured["kwargs"] = kwargs
        return _FakeConnection()

    monkeypatch.setattr("dbt_ml.adapters.duckdb.duckdb.connect", fake_connect)

    adapter = DuckDBAdapter(_md_config(token=None))
    adapter._connect()

    assert captured["target"] == "md:economic_data"
    # No config passed — MotherDuck falls back to its own motherduck_token env.
    assert captured["kwargs"] == {}


# ─── credential-gated integration ───────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("MOTHERDUCK_TOKEN"),
    reason="requires a live MotherDuck token in MOTHERDUCK_TOKEN",
)
def test_motherduck_capability_audit_against_live_service() -> None:
    """Verify the advertised DuckDB capabilities work over MotherDuck (#186).

    Exercises atomic full replace, incremental merge, state storage, bounded
    snapshots, SQL preflight + model materialization, and cleanup in a
    disposable schema, then removes everything it created.
    """
    import polars as pl

    from dbt_ml.adapters import StateRecord, StateScope

    database = os.environ.get("DBT_ML_MOTHERDUCK_DATABASE", "md:")
    cfg = parse_warehouse_config(
        {
            "type": "duckdb",
            "schema": "dbt_ml_ci_motherduck",
            "path": database,
            "token": "{{ env_var('MOTHERDUCK_TOKEN') }}",
        }
    )
    with create_adapter(cfg) as adapter:
        try:
            # atomic full replace
            adapter.materialize_full("md_probe", pl.DataFrame({"id": ["a", "b", "c"]}))
            assert adapter.row_count("md_probe") == 3
            adapter.materialize_full("md_probe", pl.DataFrame({"id": ["a"]}))
            assert adapter.row_count("md_probe") == 1

            # incremental keyed merge
            adapter.materialize_full("md_inc", pl.DataFrame({"id": ["a", "b"], "v": [1, 2]}))
            adapter.materialize_incremental(
                "md_inc",
                pl.DataFrame({"id": ["b", "c"], "v": [20, 30]}),
                key_col="id",
                on_schema_change="append_new_columns",
            )
            merged = {
                r["id"]: r["v"]
                for r in adapter.read_table("md_inc").iter_rows(named=True)
            }
            assert merged == {"a": 1, "b": 20, "c": 30}

            # state storage
            scope = StateScope("md_model")
            adapter.upsert_state(
                scope, [StateRecord("k1", "fp1", "cv1"), StateRecord("k2", "fp2", "cv1")]
            )
            assert set(adapter.fetch_state(scope)) == {"k1", "k2"}

            # bounded snapshot
            with adapter.table_snapshot("md_probe", batch_size=1) as snap:
                assert sum(b.num_rows for b in snap) == 1

            # SQL preflight + model materialization
            assert any(c.name == "x" for c in adapter.dry_run_sql("SELECT 1 AS x").columns)
            adapter.materialize_sql_full(
                "md_sql", "SELECT id FROM " + adapter.table_ref("md_probe")
            )
            assert adapter.row_count("md_sql") == 1
        finally:
            for table in ("md_probe", "md_inc", "md_sql"):
                adapter.drop_table(table)
            adapter.delete_state(StateScope("md_model"), ["k1", "k2"])
