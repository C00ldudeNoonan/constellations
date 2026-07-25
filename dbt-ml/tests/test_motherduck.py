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
def test_motherduck_round_trip_against_live_service() -> None:
    import polars as pl

    database = os.environ.get("DBT_ML_MOTHERDUCK_DATABASE", "md:")
    cfg = parse_warehouse_config(
        {
            "type": "duckdb",
            "schema": "dbt_ml_ci",
            "path": database,
            "token": "{{ env_var('MOTHERDUCK_TOKEN') }}",
        }
    )
    with create_adapter(cfg) as adapter:
        adapter.materialize_full(
            "motherduck_probe", pl.DataFrame({"id": [1, 2, 3]})
        )
        assert adapter.row_count("motherduck_probe") == 3
        adapter.drop_table("motherduck_probe")
