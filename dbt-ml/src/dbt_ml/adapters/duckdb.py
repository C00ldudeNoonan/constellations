from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import duckdb
import polars as pl
from pydantic import BaseModel

from ..config.profile import WarehouseConfig
from .base import (
    AdapterError,
    StateRecord,
    StateScope,
    StateValue,
    WarehouseAdapter,
    WarehouseCapability,
    validate_incremental_keys,
    validate_state_keys,
    validate_state_records,
)
from .registry import register

_STATE_TABLE = "dbt_ml_state"
_STATE_V1_COLUMNS = (
    ("model_name", "VARCHAR", "NO"),
    ("document_id", "VARCHAR", "NO"),
    ("content_hash", "VARCHAR", "NO"),
    ("code_version", "VARCHAR", "NO"),
    ("last_run_at", "TIMESTAMP", "NO"),
)
_STATE_V2_COLUMNS = (
    ("model_name", "VARCHAR", "NO"),
    ("state_scope", "VARCHAR", "NO"),
    ("target_identity", "VARCHAR", "NO"),
    ("record_key", "VARCHAR", "NO"),
    ("input_fingerprint", "VARCHAR", "NO"),
    ("code_version", "VARCHAR", "NO"),
    ("last_run_at", "TIMESTAMP", "NO"),
)
_STATE_V1_KEY = ("model_name", "document_id")
_STATE_V2_KEY = ("model_name", "state_scope", "target_identity", "record_key")


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

    @classmethod
    def capabilities(cls) -> frozenset[WarehouseCapability]:
        return frozenset(WarehouseCapability)

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
        columns = self._state_columns(_STATE_TABLE)
        if columns is None:
            self.connection.execute(self._create_state_table_sql(_STATE_TABLE))
            return

        primary_key = self._state_primary_key(_STATE_TABLE)
        if columns == _STATE_V2_COLUMNS and primary_key == _STATE_V2_KEY:
            return
        if columns == _STATE_V1_COLUMNS and primary_key == _STATE_V1_KEY:
            self._migrate_v1_state()
            return

        shape = ", ".join(name for name, _type, _nullable in columns)
        raise AdapterError(
            "Unsupported dbt_ml_state schema; expected the legacy v1 or current "
            f"v2 shape, found columns: {shape or '(none)'}. Back up the table and "
            "run --full-refresh after resolving the state schema."
        )

    def _create_state_table_sql(self, table: str) -> str:
        return f"""
            CREATE TABLE {self.schema_ref}.{self.quote_ident(table)} (
                model_name VARCHAR NOT NULL,
                state_scope VARCHAR NOT NULL,
                target_identity VARCHAR NOT NULL,
                record_key VARCHAR NOT NULL,
                input_fingerprint VARCHAR NOT NULL,
                code_version VARCHAR NOT NULL,
                last_run_at TIMESTAMP NOT NULL,
                PRIMARY KEY (
                    model_name, state_scope, target_identity, record_key
                )
            )
        """

    def _state_columns(
        self, table: str
    ) -> tuple[tuple[str, str, str], ...] | None:
        rows = self.connection.execute(
            "SELECT column_name, data_type, is_nullable "
            "FROM information_schema.columns "
            "WHERE table_catalog = ? AND table_schema = ? AND table_name = ? "
            "ORDER BY ordinal_position",
            [self.catalog, self.schema, table],
        ).fetchall()
        if not rows:
            return None
        return tuple(
            (str(name), str(data_type).upper(), str(nullable))
            for name, data_type, nullable in rows
        )

    def _state_primary_key(self, table: str) -> tuple[str, ...] | None:
        row = self.connection.execute(
            "SELECT constraint_column_names FROM duckdb_constraints() "
            "WHERE database_name = ? AND schema_name = ? AND table_name = ? "
            "AND constraint_type = 'PRIMARY KEY'",
            [self.catalog, self.schema, table],
        ).fetchone()
        if row is None:
            return None
        return tuple(str(column) for column in row[0])

    def _migrate_v1_state(self) -> None:
        old_ref = f"{self.schema_ref}.{self.quote_ident(_STATE_TABLE)}"
        migration_table = f"dbt_ml_staging__state_migration_v2__{uuid4().hex}"
        migration_ref = f"{self.schema_ref}.{self.quote_ident(migration_table)}"
        with self._transaction():
            self.connection.execute(self._create_state_table_sql(migration_table))
            source_row = self.connection.execute(
                f"SELECT COUNT(*) FROM {old_ref}"
            ).fetchone()
            if source_row is None:
                raise AdapterError("DuckDB state migration could not count v1 rows")
            source_count = int(source_row[0])
            self.connection.execute(
                f"""
                INSERT INTO {migration_ref} (
                    model_name, state_scope, target_identity, record_key,
                    input_fingerprint, code_version, last_run_at
                )
                SELECT model_name, 'materialization', 'warehouse-v1', document_id,
                       content_hash, code_version, last_run_at
                FROM {old_ref}
                """
            )
            migrated_row = self.connection.execute(
                f"SELECT COUNT(*) FROM {migration_ref}"
            ).fetchone()
            if migrated_row is None:
                raise AdapterError("DuckDB state migration could not verify v2 rows")
            migrated_count = int(migrated_row[0])
            if migrated_count != source_count:
                raise AdapterError(
                    "DuckDB state migration row-count verification failed: "
                    f"expected {source_count}, migrated {migrated_count}"
                )
            self.connection.execute(f"DROP TABLE {old_ref}")
            self.connection.execute(
                f"ALTER TABLE {migration_ref} RENAME TO {self.quote_ident(_STATE_TABLE)}"
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

    def materialize_full(
        self, table: str, df: pl.DataFrame, *, options: BaseModel | None = None
    ) -> int:
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
        options: BaseModel | None = None,
    ) -> int:
        if df.height == 0:
            return 0
        validate_incremental_keys(df, key_col)
        existing = self._table_columns(table)
        if existing is not None and key_col not in existing:
            raise AdapterError(
                f"Incremental target '{table}' is missing key column '{key_col}'"
            )
        full = self.table_ref(table)
        self.connection.register("dbt_ml_staging", df)
        try:
            self.connection.execute("BEGIN TRANSACTION")
            try:
                self.connection.execute(
                    f"CREATE TABLE IF NOT EXISTS {full} AS "
                    f"SELECT * FROM dbt_ml_staging LIMIT 0"
                )
                insert_cols = self._reconcile_schema(table, df, on_schema_change)
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
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise
            else:
                self.connection.execute("COMMIT")
        finally:
            self.connection.unregister("dbt_ml_staging")
        return df.height

    def materialize_full_chunks(
        self,
        table: str,
        chunks: Iterable[pl.DataFrame],
        *,
        options: BaseModel | None = None,
    ) -> int:
        staging = f"dbt_ml_staging__{table}__{uuid4().hex[:12]}"
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
            self.connection.execute("BEGIN TRANSACTION")
            try:
                self.connection.execute(
                    f"DROP TABLE IF EXISTS {self.table_ref(table)}"
                )
                self._rename_staging(staging_ref, table)
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise
            else:
                self.connection.execute("COMMIT")
        except BaseException as error:
            try:
                self.connection.execute(f"DROP TABLE IF EXISTS {staging_ref}")
            except Exception as cleanup_error:
                error.add_note(f"Failed to clean staging table: {cleanup_error}")
            raise
        return total

    def _rename_staging(self, staging_ref: str, table: str) -> None:
        self.connection.execute(
            f"ALTER TABLE {staging_ref} RENAME TO {self.quote_ident(table)}"
        )

    def _table_columns(self, table: str) -> list[str] | None:
        rows = self.connection.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_catalog = ? AND table_schema = ? AND table_name = ? "
            "ORDER BY ordinal_position",
            [self.catalog, self.schema, table],
        ).fetchall()
        if not rows:
            return None
        return [str(row[0]) for row in rows]

    def _reconcile_schema(
        self, table: str, df: pl.DataFrame, on_schema_change: str
    ) -> list[str]:
        """Compare staging columns against the existing table and apply the
        on_schema_change policy. Returns the staging columns to insert."""
        target_cols = self._table_columns(table) or []
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

    def delete_rows_and_state(
        self,
        table: str,
        *,
        key_col: str,
        keys: Sequence[str],
        state_scope: StateScope,
        state_record_keys: Sequence[str] | None = None,
    ) -> int:
        target_keys = list(keys)
        scoped_keys = (
            target_keys if state_record_keys is None else list(state_record_keys)
        )
        validate_state_keys(target_keys)
        validate_state_keys(scoped_keys)

        deleted_count = 0
        with self._transaction():
            if target_keys and self._table_columns(table) is not None:
                placeholders = ", ".join("?" for _ in target_keys)
                cursor = self.connection.execute(
                    f"DELETE FROM {self.table_ref(table)} "
                    f"WHERE {self.quote_ident(key_col)} IN ({placeholders})",
                    target_keys,
                )
                deleted = cursor.fetchone()
                deleted_count = int(deleted[0]) if deleted else 0
            self._delete_state_rows(state_scope, scoped_keys)
        return deleted_count

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

    def _reset_storage_for_test(self) -> str:
        """Delete the DuckDB file for isolated adapter integration teardown."""
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

    def fetch_state(self, scope: StateScope) -> dict[str, StateValue]:
        rows = self.connection.execute(
            f"SELECT record_key, input_fingerprint, code_version "
            f"FROM {self.schema_ref}.{self.quote_ident(_STATE_TABLE)} "
            "WHERE model_name = ? AND state_scope = ? AND target_identity = ?",
            [scope.model_name, scope.stage, scope.target_identity],
        ).fetchall()
        return {str(r[0]): StateValue(str(r[1]), str(r[2])) for r in rows}

    def upsert_state(
        self, scope: StateScope, records: Sequence[StateRecord]
    ) -> None:
        validate_state_records(records)
        if not records:
            return
        with self._transaction():
            self._upsert_state_rows(scope, records)

    def replace_state(
        self, scope: StateScope, records: Sequence[StateRecord]
    ) -> None:
        validate_state_records(records)
        with self._transaction():
            self._clear_state_rows(scope)
            self._insert_state_rows(scope, records)

    def clear_state(self, scope: StateScope) -> None:
        self._clear_state_rows(scope)

    def delete_state(self, scope: StateScope, record_keys: Sequence[str]) -> None:
        validate_state_keys(record_keys)
        self._delete_state_rows(scope, record_keys)

    def _upsert_state_rows(
        self, scope: StateScope, records: Sequence[StateRecord]
    ) -> None:
        self.connection.executemany(
            f"""
            INSERT INTO {self.schema_ref}.{self.quote_ident(_STATE_TABLE)} (
                model_name, state_scope, target_identity, record_key,
                input_fingerprint, code_version, last_run_at
            )
            VALUES (?, ?, ?, ?, ?, ?, current_timestamp)
            ON CONFLICT (
                model_name, state_scope, target_identity, record_key
            ) DO UPDATE SET
                input_fingerprint = excluded.input_fingerprint,
                code_version = excluded.code_version,
                last_run_at  = excluded.last_run_at
            """,
            [
                [
                    scope.model_name,
                    scope.stage,
                    scope.target_identity,
                    record.record_key,
                    record.input_fingerprint,
                    record.code_version,
                ]
                for record in records
            ],
        )

    def _insert_state_rows(
        self, scope: StateScope, records: Sequence[StateRecord]
    ) -> None:
        if not records:
            return
        self.connection.executemany(
            f"""
            INSERT INTO {self.schema_ref}.{self.quote_ident(_STATE_TABLE)} (
                model_name, state_scope, target_identity, record_key,
                input_fingerprint, code_version, last_run_at
            )
            VALUES (?, ?, ?, ?, ?, ?, current_timestamp)
            """,
            [
                [
                    scope.model_name,
                    scope.stage,
                    scope.target_identity,
                    record.record_key,
                    record.input_fingerprint,
                    record.code_version,
                ]
                for record in records
            ],
        )

    def _clear_state_rows(self, scope: StateScope) -> None:
        self.connection.execute(
            f"DELETE FROM {self.schema_ref}.{self.quote_ident(_STATE_TABLE)} "
            "WHERE model_name = ? AND state_scope = ? AND target_identity = ?",
            [scope.model_name, scope.stage, scope.target_identity],
        )

    def _delete_state_rows(
        self, scope: StateScope, record_keys: Sequence[str]
    ) -> None:
        if not record_keys:
            return
        placeholders = ", ".join("?" for _ in record_keys)
        self.connection.execute(
            f"DELETE FROM {self.schema_ref}.{self.quote_ident(_STATE_TABLE)} "
            "WHERE model_name = ? AND state_scope = ? AND target_identity = ? "
            f"AND record_key IN ({placeholders})",
            [scope.model_name, scope.stage, scope.target_identity, *record_keys],
        )

    # ─── internals ───────────────────────────────────────────────────────

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self.connection.execute("BEGIN TRANSACTION")
        try:
            yield
            self.connection.execute("COMMIT")
        except BaseException as error:
            try:
                self.connection.execute("ROLLBACK")
            except Exception as rollback_error:
                error.add_note(
                    f"Failed to roll back DuckDB transaction: {rollback_error}"
                )
            raise

    def _resolved_path(self) -> Path:
        config = self.config
        assert isinstance(config, DuckDBWarehouseConfig)
        path = config.path
        if path.is_absolute() or self.project_dir is None:
            return path.resolve()
        return (self.project_dir / path).resolve()
