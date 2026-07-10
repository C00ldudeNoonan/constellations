from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, cast

import duckdb
import polars as pl

from ..config.profile import WarehouseConfig
from .base import AdapterError, WarehouseAdapter
from .registry import register


class DuckDBWarehouseConfig(WarehouseConfig):
    type: Literal["duckdb"] = "duckdb"
    path: Path

    def absolutize(self, project_dir: Path) -> DuckDBWarehouseConfig:
        return self.model_copy(update={"path": (project_dir / self.path).resolve()})

    def storage_location(self) -> str:
        return str(self.path)

    def catalog_name(self) -> str:
        return self.path.stem

    def local_path(self) -> Path | None:
        return self.path


@register
class DuckDBAdapter(WarehouseAdapter):
    """The reference implementation. Wraps a single DuckDB connection.

    DuckDB-specific wrinkle: the catalog name comes from the database
    filename's stem; if the schema and the catalog collide (both `dbt_ml`)
    we have to fully-qualify SQL references as `"catalog"."schema"`.
    """

    def __init__(self, config: WarehouseConfig, *, project_dir: Path | None = None) -> None:
        super().__init__(config, project_dir=project_dir)
        self._con: duckdb.DuckDBPyConnection | None = None
        self._catalog: str = ""

    @classmethod
    def adapter_type(cls) -> str:
        return "duckdb"

    @classmethod
    def config_model(cls) -> type[WarehouseConfig]:
        return DuckDBWarehouseConfig

    # ─── lifecycle ────────────────────────────────────────────────────────

    def _connect(self) -> None:
        db_path = self._resolved_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(db_path))
        row = self._con.execute("SELECT current_database()").fetchone()
        self._catalog = row[0] if row else "memory"

    def _close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def _ensure_schema(self) -> None:
        self.connection.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema_ref}")

    def _ensure_state_table(self) -> None:
        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.schema_ref}.dbt_ml_state (
                model_name VARCHAR NOT NULL,
                document_id VARCHAR NOT NULL,
                content_hash VARCHAR NOT NULL,
                code_version VARCHAR NOT NULL,
                last_run_at TIMESTAMP NOT NULL,
                PRIMARY KEY (model_name, document_id)
            )
            """
        )

    # ─── identity ────────────────────────────────────────────────────────

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            raise AdapterError("Adapter must be used as a context manager")
        return self._con

    @property
    def raw_connection(self) -> duckdb.DuckDBPyConnection:
        """The underlying warehouse driver. Handed to custom python tests."""
        return self.connection

    @property
    def catalog(self) -> str:
        if not self._catalog:
            raise AdapterError("Adapter must be used as a context manager")
        return self._catalog

    @property
    def schema_ref(self) -> str:
        return f"{self.quote_ident(self.catalog)}.{self.quote_ident(self.schema)}"

    # ─── materialization ─────────────────────────────────────────────────

    def materialize_full(self, table: str, df: pl.DataFrame) -> int:
        full = self.table_ref(table)
        self.connection.register("dbt_ml_staging", df)
        try:
            self.connection.execute(
                f"CREATE OR REPLACE TABLE {full} AS SELECT * FROM dbt_ml_staging"
            )
        finally:
            self.connection.unregister("dbt_ml_staging")
        return df.height

    def materialize_incremental(
        self,
        table: str,
        df: pl.DataFrame,
        *,
        key_col: str,
        on_schema_change: str = "fail",
    ) -> int:
        if df.height == 0:
            return 0
        full = self.table_ref(table)
        self.connection.register("dbt_ml_staging", df)
        try:
            self.connection.execute(
                f"CREATE TABLE IF NOT EXISTS {full} AS "
                f"SELECT * FROM dbt_ml_staging LIMIT 0"
            )
            insert_cols = self._reconcile_schema(table, df, on_schema_change)
            if key_col in df.columns:
                key = self.quote_ident(key_col)
                self.connection.execute(
                    f"""
                    DELETE FROM {full} AS target
                    USING dbt_ml_staging AS source
                    WHERE target.{key} = source.{key}
                    """
                )
            col_list = ", ".join(self.quote_ident(c) for c in insert_cols)
            self.connection.execute(
                f"INSERT INTO {full} BY NAME SELECT {col_list} FROM dbt_ml_staging"
            )
        finally:
            self.connection.unregister("dbt_ml_staging")
        return df.height

    def materialize_full_chunks(
        self, table: str, chunks: Iterable[pl.DataFrame]
    ) -> int:
        staging = f"dbt_ml_staging__{table}"
        staging_ref = self.table_ref(staging)
        total = 0
        first = True
        try:
            for df in chunks:
                self.connection.register("dbt_ml_staging", df)
                try:
                    if first:
                        self.connection.execute(
                            f"CREATE OR REPLACE TABLE {staging_ref} "
                            "AS SELECT * FROM dbt_ml_staging"
                        )
                        first = False
                    else:
                        insert_cols = self._reconcile_schema(
                            staging, df, "append_new_columns"
                        )
                        col_list = ", ".join(
                            self.quote_ident(c) for c in insert_cols
                        )
                        self.connection.execute(
                            f"INSERT INTO {staging_ref} BY NAME "
                            f"SELECT {col_list} FROM dbt_ml_staging"
                        )
                finally:
                    self.connection.unregister("dbt_ml_staging")
                total += df.height
            if first:
                # Empty corpus: drop the target (BigQuery's materialize_full
                # contract for an empty frame) rather than swap in nothing.
                self.drop_table(table)
                return 0
            self.connection.execute(f"DROP TABLE IF EXISTS {self.table_ref(table)}")
            self.connection.execute(
                f"ALTER TABLE {staging_ref} RENAME TO {self.quote_ident(table)}"
            )
        except Exception:
            self.connection.execute(f"DROP TABLE IF EXISTS {staging_ref}")
            raise
        return total

    def _reconcile_schema(
        self, table: str, df: pl.DataFrame, on_schema_change: str
    ) -> list[str]:
        """Compare staging columns against the existing table and apply the
        on_schema_change policy. Returns the staging columns to insert."""
        target_cols = [
            r[0]
            for r in self.connection.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_catalog = ? AND table_schema = ? AND table_name = ? "
                "ORDER BY ordinal_position",
                [self.catalog, self.schema, table],
            ).fetchall()
        ]
        new = [c for c in df.columns if c not in target_cols]
        removed = [c for c in target_cols if c not in df.columns]
        if not new and not removed:
            return list(df.columns)

        if on_schema_change == "fail":
            drift = "; ".join(
                part
                for part in (
                    f"new columns {new}" if new else "",
                    f"removed columns {removed}" if removed else "",
                )
                if part
            )
            raise AdapterError(
                f"Schema change on incremental table '{table}': {drift}. "
                "Run with --full-refresh to rebuild it, or set "
                "`on_schema_change: append_new_columns` (or `ignore`) on the model."
            )
        if on_schema_change == "append_new_columns":
            if new:
                staging_types = dict(
                    self.connection.execute(
                        "SELECT column_name, column_type FROM "
                        "(DESCRIBE SELECT * FROM dbt_ml_staging)"
                    ).fetchall()
                )
                for col in new:
                    self.connection.execute(
                        f"ALTER TABLE {self.table_ref(table)} "
                        f"ADD COLUMN {self.quote_ident(col)} {staging_types[col]}"
                    )
            return list(df.columns)
        if on_schema_change == "ignore":
            return [c for c in df.columns if c in target_cols]
        raise AdapterError(
            f"Unknown on_schema_change policy '{on_schema_change}'. "
            "Allowed: fail, ignore, append_new_columns."
        )

    def delete_rows(self, table: str, *, key_col: str, keys: list[str]) -> int:
        if not keys or table not in self.list_tables():
            return 0
        full = self.table_ref(table)
        placeholders = ", ".join("?" for _ in keys)
        cursor = self.connection.execute(
            f"DELETE FROM {full} WHERE {self.quote_ident(key_col)} IN ({placeholders})",
            keys,
        )
        deleted = cursor.fetchone()
        return int(deleted[0]) if deleted else 0

    def drop_table(self, table: str) -> None:
        self.connection.execute(f"DROP TABLE IF EXISTS {self.table_ref(table)}")

    # ─── querying ────────────────────────────────────────────────────────

    def execute(self, sql: str, params: list[Any] | None = None) -> Any:
        if params is None:
            return self.connection.execute(sql)
        return self.connection.execute(sql, params)

    def query_df(self, sql: str, params: list[Any] | None = None) -> pl.DataFrame:
        if params is None:
            return self.connection.execute(sql).pl()
        return self.connection.execute(sql, params).pl()

    def scalar(self, sql: str, params: list[Any] | None = None) -> Any:
        row = self.execute(sql, params).fetchone()
        return row[0] if row else None

    def rows(self, sql: str, params: list[Any] | None = None) -> list[tuple[Any, ...]]:
        return cast(list[tuple[Any, ...]], self.execute(sql, params).fetchall())

    def clean(self) -> str:
        """Delete the DuckDB file. Closes the connection if open."""
        if self._con is not None:
            self._close()
        path = self._resolved_path()
        if path.exists():
            path.unlink()
        return str(path)

    def list_tables(self) -> list[str]:
        rows = self.connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_catalog = ? AND table_schema = ? AND table_name != 'dbt_ml_state' "
            "ORDER BY table_name",
            [self.catalog, self.schema],
        ).fetchall()
        # `dbt_ml_test_failures__*` tables are --store-failures inspection
        # artifacts and `dbt_ml_staging__*` are in-flight full loads (#77),
        # not models; keep both out of the model namespace. (Filtered in
        # Python because `_` is a LIKE wildcard in SQL.)
        return [
            r[0]
            for r in rows
            if not r[0].startswith(("dbt_ml_test_failures__", "dbt_ml_staging__"))
        ]

    # ─── state CRUD ──────────────────────────────────────────────────────

    def fetch_state(self, model_name: str) -> dict[str, tuple[str, str]]:
        rows = self.connection.execute(
            f"SELECT document_id, content_hash, code_version "
            f"FROM {self.schema_ref}.dbt_ml_state WHERE model_name = ?",
            [model_name],
        ).fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}

    def upsert_state(
        self, model_name: str, records: list[tuple[str, str, str]]
    ) -> None:
        if not records:
            return
        self.connection.executemany(
            f"""
            INSERT INTO {self.schema_ref}.dbt_ml_state
                (model_name, document_id, content_hash, code_version, last_run_at)
            VALUES (?, ?, ?, ?, current_timestamp)
            ON CONFLICT (model_name, document_id) DO UPDATE SET
                content_hash = excluded.content_hash,
                code_version = excluded.code_version,
                last_run_at  = excluded.last_run_at
            """,
            [[model_name, doc_id, ch, cv] for doc_id, ch, cv in records],
        )

    def clear_model_state(self, model_name: str) -> None:
        self.connection.execute(
            f"DELETE FROM {self.schema_ref}.dbt_ml_state WHERE model_name = ?",
            [model_name],
        )

    def delete_state(self, model_name: str, document_ids: list[str]) -> None:
        if not document_ids:
            return
        placeholders = ", ".join("?" for _ in document_ids)
        self.connection.execute(
            f"DELETE FROM {self.schema_ref}.dbt_ml_state "
            f"WHERE model_name = ? AND document_id IN ({placeholders})",
            [model_name, *document_ids],
        )

    # ─── internals ───────────────────────────────────────────────────────

    def _resolved_path(self) -> Path:
        config = self.config
        assert isinstance(config, DuckDBWarehouseConfig)
        path = config.path
        if path.is_absolute() or self.project_dir is None:
            return path.resolve()
        return (self.project_dir / path).resolve()
