from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from hashlib import blake2b
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import duckdb
import polars as pl
import pyarrow as pa
from pydantic import BaseModel

from ..config.profile import WarehouseConfig
from ..hashing import canonical_fingerprint
from ..sql_models import build_key_check_sql
from .base import (
    SERVING_LEDGER_TABLE,
    AdapterError,
    ReadPredicate,
    ReadPredicateOperator,
    SqlMaterializationResult,
    SqlRelationColumn,
    SqlRelationSchema,
    StaleStateFenceError,
    StatePage,
    StatePageReader,
    StatePageRecord,
    StatePageRequest,
    StateRecord,
    StateScope,
    StateScopeFence,
    StateValue,
    TableReadRequest,
    TableReadSnapshot,
    WarehouseAdapter,
    WarehouseCapability,
    decode_state_cursor,
    encode_state_cursor,
    plan_schema_change,
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

_READ_BINARY_OPERATORS = {
    ReadPredicateOperator.EQUAL: "=",
    ReadPredicateOperator.NOT_EQUAL: "!=",
    ReadPredicateOperator.LESS_THAN: "<",
    ReadPredicateOperator.LESS_THAN_OR_EQUAL: "<=",
    ReadPredicateOperator.GREATER_THAN: ">",
    ReadPredicateOperator.GREATER_THAN_OR_EQUAL: ">=",
}


def _duckdb_read_predicates(
    adapter: DuckDBAdapter, predicates: Sequence[ReadPredicate]
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for predicate in predicates:
        column = adapter.quote_ident(predicate.column)
        if predicate.operator in _READ_BINARY_OPERATORS:
            clauses.append(f"{column} {_READ_BINARY_OPERATORS[predicate.operator]} ?")
            params.append(predicate.value)
        elif predicate.operator is ReadPredicateOperator.IN:
            clauses.append(f"{column} IN ?")
            params.append(list(cast(tuple[Any, ...], predicate.value)))
        elif predicate.operator is ReadPredicateOperator.NOT_IN:
            clauses.append(f"{column} NOT IN ?")
            params.append(list(cast(tuple[Any, ...], predicate.value)))
        elif predicate.operator is ReadPredicateOperator.IS_NULL:
            clauses.append(f"{column} IS NULL")
        else:
            assert predicate.operator is ReadPredicateOperator.IS_NOT_NULL
            clauses.append(f"{column} IS NOT NULL")
    return (" WHERE " + " AND ".join(clauses) if clauses else "", params)


def _duckdb_arrow_batches(
    reader: pa.RecordBatchReader,
    on_complete: Any,
) -> Iterator[pa.RecordBatch]:
    digest = _arrow_reader_digest(reader.schema)
    batch: pa.RecordBatch | None = None
    failure: AdapterError | None = None
    try:
        for batch in reader:
            _update_arrow_digest(digest, batch)
            yield batch
            batch = None
        on_complete(digest.hexdigest())
    except Exception:
        batch = None
        failure = AdapterError("DuckDB table snapshot batch read failed")
    finally:
        reader.close()
    if failure is not None:
        raise failure


def _arrow_reader_digest(schema: pa.Schema) -> Any:
    digest = blake2b(digest_size=16)
    payload = schema.serialize().to_pybytes()
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest


def _update_arrow_digest(digest: Any, batch: pa.RecordBatch) -> None:
    payload = batch.serialize().to_pybytes()
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


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
        return frozenset(
            {
                WarehouseCapability.ATOMIC_FULL_REPLACE,
                WarehouseCapability.ATOMIC_KEYED_UPSERT,
                WarehouseCapability.ATOMIC_STATE_SCOPE_REPLACE,
                WarehouseCapability.CHUNKED_WRITES,
                WarehouseCapability.PAGED_STATE_RECONCILIATION,
                WarehouseCapability.SCHEMA_EVOLUTION,
                WarehouseCapability.SQL_QUERIES,
                WarehouseCapability.SQL_MODEL_MATERIALIZATION,
                WarehouseCapability.SQL_INCREMENTAL_MATERIALIZATION,
                WarehouseCapability.SQL_SCHEMA_TESTS,
                WarehouseCapability.STREAMING_TABULAR_READS,
                WarehouseCapability.TABULAR_PREDICATE_PUSHDOWN,
                WarehouseCapability.TABULAR_READS,
                WarehouseCapability.TRANSACTIONS,
                WarehouseCapability.TYPED_EMPTY_RELATIONS,
            }
        )

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

    def materialize_sql_full(
        self,
        table: str,
        select_sql: str,
        *,
        options: BaseModel | None = None,
    ) -> SqlMaterializationResult:
        full = self.table_ref(table)
        # DuckDB CREATE OR REPLACE is atomic (same guarantee materialize_full
        # relies on) and leaves the prior table intact if the SELECT errors.
        try:
            self.connection.execute(f"CREATE OR REPLACE TABLE {full} AS {select_sql}")
        except duckdb.Error as e:
            raise AdapterError(
                f"SQL model materialization for '{table}' failed: {e}"
            ) from e
        row = self.connection.execute(f"SELECT COUNT(*) FROM {full}").fetchone()
        return SqlMaterializationResult(
            relation=full, rows_written=int(row[0]) if row else 0
        )

    def dry_run_sql(self, select_sql: str) -> SqlRelationSchema:
        try:
            rel = self.connection.sql(
                f"SELECT * FROM ({select_sql}) AS _dbt_ml_dry_run LIMIT 0"
            )
            columns = tuple(
                SqlRelationColumn(name=name, data_type=str(dtype))
                for name, dtype in zip(rel.columns, rel.types, strict=True)
            )
        except duckdb.Error as e:
            raise AdapterError(f"SQL dry-run failed: {e}") from e
        return SqlRelationSchema(columns=columns)

    def relation_exists(self, table: str) -> bool:
        return self._table_columns(table) is not None

    def materialize_sql_incremental(
        self,
        table: str,
        select_sql: str,
        *,
        unique_key: str,
        on_schema_change: str = "fail",
        options: BaseModel | None = None,
    ) -> SqlMaterializationResult:
        full = self.table_ref(table)
        key = self.quote_ident(unique_key)
        staging = f"dbt_ml_sql_staging__{table}"
        try:
            # A session-scoped temp table: never persisted, never visible to
            # list_tables()/tests, and dropped in `finally` even on failure.
            self.connection.execute(
                f"CREATE OR REPLACE TEMP TABLE {staging} AS {select_sql}"
            )
            check = self.connection.execute(
                build_key_check_sql(f"SELECT * FROM {staging}", key)
            ).fetchone()
            null_count, duplicate_count = (check[0] or 0, check[1] or 0) if check else (0, 0)
            if null_count or duplicate_count:
                raise AdapterError(
                    f"Incremental SQL model '{table}' unique_key '{unique_key}' "
                    f"has {null_count} null and {duplicate_count} duplicate "
                    "value(s) in the query result."
                )

            staging_cols = self._temp_table_columns(staging)
            if unique_key not in staging_cols:
                raise AdapterError(
                    f"Incremental SQL model '{table}' query does not select its "
                    f"unique_key column '{unique_key}'"
                )
            insert_cols = self._reconcile_sql_schema(
                table, staging, staging_cols, on_schema_change, unique_key=unique_key
            )

            self.connection.execute("BEGIN TRANSACTION")
            try:
                updated_row = self.connection.execute(
                    f"SELECT COUNT(*) FROM {staging} AS s "
                    f"JOIN {full} AS t ON t.{key} = s.{key}"
                ).fetchone()
                updated = int(updated_row[0]) if updated_row else 0
                self.connection.execute(
                    f"""
                    DELETE FROM {full} AS target
                    USING {staging} AS source
                    WHERE target.{key} = source.{key}
                    """
                )
                col_list = ", ".join(self.quote_ident(c) for c in insert_cols)
                self.connection.execute(
                    f"INSERT INTO {full} BY NAME SELECT {col_list} FROM {staging}"
                )
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise
            else:
                self.connection.execute("COMMIT")

            total_row = self.connection.execute(
                f"SELECT COUNT(*) FROM {staging}"
            ).fetchone()
            total = int(total_row[0]) if total_row else 0
        except duckdb.Error as e:
            raise AdapterError(
                f"Incremental SQL model materialization for '{table}' failed: {e}"
            ) from e
        finally:
            self.connection.execute(f"DROP TABLE IF EXISTS {staging}")
        return SqlMaterializationResult(
            relation=full,
            rows_written=total,
            rows_inserted=total - updated,
            rows_updated=updated,
        )

    def _temp_table_columns(self, table: str) -> list[str]:
        rows = self.connection.execute(f"DESCRIBE {table}").fetchall()
        return [row[0] for row in rows]

    def _reconcile_sql_schema(
        self,
        table: str,
        staging: str,
        staging_cols: list[str],
        on_schema_change: str,
        *,
        unique_key: str,
    ) -> list[str]:
        """SQL-sourced analogue of `_reconcile_schema`: compares a staging
        *table name*'s columns against the existing target instead of a
        DataFrame's, since a SQL model's staged result has no Polars frame."""
        target_cols = self._table_columns(table) or []
        if unique_key not in target_cols:
            # The key changed (or was never in this target). Appending it as
            # "just another new column" would leave every existing row with a
            # NULL key — silently breaking the incremental key invariant — so
            # this is fatal under every on_schema_change policy, not only
            # `fail`.
            raise AdapterError(
                f"Incremental SQL model '{table}' unique_key '{unique_key}' does "
                "not exist in the current target table (the key may have "
                "changed). Run with --full-refresh to rebuild the target under "
                "the new key."
            )
        plan = plan_schema_change(target_cols, staging_cols, on_schema_change, table)
        if plan.columns_to_add:
            staging_types = dict(
                self.connection.execute(
                    f"SELECT column_name, column_type FROM (DESCRIBE {staging})"
                ).fetchall()
            )
            for col in plan.columns_to_add:
                self.connection.execute(
                    f"ALTER TABLE {self.table_ref(table)} "
                    f"ADD COLUMN {self.quote_ident(col)} {staging_types[col]}"
                )
        return plan.columns_to_load

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
        plan = plan_schema_change(target_cols, list(df.columns), on_schema_change, table)
        if plan.columns_to_add:
            staging_types = dict(
                self.connection.execute(
                    "SELECT column_name, column_type FROM "
                    "(DESCRIBE SELECT * FROM dbt_ml_staging)"
                ).fetchall()
            )
            for col in plan.columns_to_add:
                self.connection.execute(
                    f"ALTER TABLE {self.table_ref(table)} "
                    f"ADD COLUMN {self.quote_ident(col)} {staging_types[col]}"
                )
        return plan.columns_to_load

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
        keys: Sequence[Any],
        state_scope: StateScope,
        state_record_keys: Sequence[str] | None = None,
    ) -> int:
        target_keys = list(keys)
        scoped_keys = (
            target_keys if state_record_keys is None else list(state_record_keys)
        )
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

    def _open_table_snapshot(self, request: TableReadRequest) -> TableReadSnapshot:
        cursor = self.connection.cursor()
        reader: pa.RecordBatchReader | None = None
        params: list[Any] = []
        transaction_open = False
        failure: AdapterError | None = None
        snapshot_digest: str | None = None
        snapshot_ref: TableReadSnapshot | None = None
        try:
            cursor.execute("BEGIN TRANSACTION")
            transaction_open = True
            table_ref = self.table_ref(request.table)
            schema_reader = cursor.execute(
                f"SELECT * FROM {table_ref} LIMIT 0"
            ).to_arrow_reader(1)
            try:
                available_columns = frozenset(schema_reader.schema.names)
            finally:
                schema_reader.close()
            missing = sorted(request.referenced_columns - available_columns)
            if missing:
                raise AdapterError(
                    "Table snapshot references missing column(s): "
                    + ", ".join(missing)
                )

            where_sql, params = _duckdb_read_predicates(self, request.predicates)
            if request.key_column is not None:
                key = self.quote_ident(request.key_column)
                validation = cursor.execute(
                    "SELECT COUNT(*) FILTER (WHERE "
                    f"{key} IS NULL), COUNT({key}) - COUNT(DISTINCT {key}) "
                    f"FROM {table_ref}{where_sql}",
                    params,
                ).fetchone()
                null_count = int(validation[0]) if validation else 0
                duplicate_count = int(validation[1]) if validation else 0
                if null_count or duplicate_count:
                    params.clear()
                    raise AdapterError(
                        "Table snapshot key domain is invalid: "
                        f"{null_count} NULL and {duplicate_count} duplicate value(s)"
                    )

            projection = (
                "*"
                if request.columns is None
                else ", ".join(self.quote_ident(column) for column in request.columns)
            )
            if params:
                cursor.execute(
                    f"SELECT {projection} FROM {table_ref}{where_sql}", params
                )
            else:
                cursor.execute(f"SELECT {projection} FROM {table_ref}{where_sql}")
            reader = cursor.to_arrow_reader(request.batch_size)
            fingerprint = canonical_fingerprint(
                {
                    "adapter": self.adapter_type(),
                    "catalog": self.catalog,
                    "schema": self.schema,
                    "request": request._fingerprint_payload(),
                    "snapshot_nonce": uuid4().hex,
                },
                domain="dbt-ml-warehouse-table-snapshot",
                version=1,
            )

            def complete(digest: str) -> None:
                nonlocal snapshot_digest, snapshot_ref
                snapshot_digest = digest
                assert snapshot_ref is not None
                snapshot_ref._set_generation_fingerprint(
                    canonical_fingerprint(
                        {
                            "content_digest": digest,
                            "request": request._fingerprint_payload(),
                        },
                        domain="dbt-ml-warehouse-table-generation",
                        version=1,
                    )
                )

            def validate_unchanged() -> None:
                if not transaction_open:
                    raise AdapterError(
                        "DuckDB table snapshot transaction is no longer active"
                    )
                if snapshot_digest is None:
                    raise AdapterError("DuckDB table snapshot was not fully consumed")
                current_digest = self._current_table_digest(request)
                if current_digest != snapshot_digest:
                    raise AdapterError("DuckDB table changed during its snapshot read")

            def close() -> None:
                nonlocal transaction_open
                try:
                    if transaction_open:
                        cursor.execute("ROLLBACK")
                        transaction_open = False
                finally:
                    cursor.close()

            snapshot_ref = TableReadSnapshot(
                schema=reader.schema,
                fingerprint=fingerprint,
                batches=_duckdb_arrow_batches(reader, complete),
                validate_unchanged=validate_unchanged,
                close=close,
            )
            return snapshot_ref
        except AdapterError:
            params.clear()
            if reader is not None:
                reader.close()
            if transaction_open:
                cursor.execute("ROLLBACK")
            cursor.close()
            raise
        except Exception:
            params.clear()
            if reader is not None:
                reader.close()
            if transaction_open:
                try:
                    cursor.execute("ROLLBACK")
                except Exception:
                    pass
            cursor.close()
            failure = AdapterError("DuckDB table snapshot could not be opened")
        if failure is not None:
            raise failure
        raise AssertionError("unreachable DuckDB table snapshot state")

    def _current_table_digest(self, request: TableReadRequest) -> str:
        cursor = self.connection.cursor()
        params: list[Any] = []
        reader: pa.RecordBatchReader | None = None
        transaction_open = False
        failure: AdapterError | None = None
        result: str | None = None
        batch: pa.RecordBatch | None = None
        try:
            cursor.execute("BEGIN TRANSACTION")
            transaction_open = True
            table_ref = self.table_ref(request.table)
            where_sql, params = _duckdb_read_predicates(self, request.predicates)
            projection = (
                "*"
                if request.columns is None
                else ", ".join(self.quote_ident(column) for column in request.columns)
            )
            if params:
                cursor.execute(
                    f"SELECT {projection} FROM {table_ref}{where_sql}", params
                )
            else:
                cursor.execute(f"SELECT {projection} FROM {table_ref}{where_sql}")
            reader = cursor.to_arrow_reader(request.batch_size)
            digest = _arrow_reader_digest(reader.schema)
            for batch in reader:
                _update_arrow_digest(digest, batch)
                batch = None
            result = digest.hexdigest()
        except Exception:
            batch = None
            params.clear()
            failure = AdapterError(
                "DuckDB table snapshot generation could not be validated"
            )
        finally:
            if reader is not None:
                reader.close()
            if transaction_open:
                try:
                    cursor.execute("ROLLBACK")
                except Exception:
                    pass
            cursor.close()
        if failure is not None:
            raise failure
        assert result is not None
        return result

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

    # ─── paged state reconciliation (issue #153) ─────────────────────────

    def _fetch_state_subset(
        self, scope: StateScope, record_keys: Sequence[str]
    ) -> dict[str, StateValue]:
        placeholders = ", ".join("?" for _ in record_keys)
        rows = self.connection.execute(
            f"SELECT record_key, input_fingerprint, code_version "
            f"FROM {self.schema_ref}.{self.quote_ident(_STATE_TABLE)} "
            "WHERE model_name = ? AND state_scope = ? AND target_identity = ? "
            f"AND record_key IN ({placeholders})",
            [scope.model_name, scope.stage, scope.target_identity, *record_keys],
        ).fetchall()
        return {str(r[0]): StateValue(str(r[1]), str(r[2])) for r in rows}

    def _open_state_page_reader(self, request: StatePageRequest) -> StatePageReader:
        # A dedicated cursor holds one read transaction open for the reader's
        # lifetime, so every page observes the same MVCC snapshot even while
        # the main connection mutates state between pages.
        cursor_conn = self.connection.cursor()
        transaction_open = False
        try:
            cursor_conn.execute("BEGIN TRANSACTION")
            transaction_open = True
            state_ref = f"{self.schema_ref}.{self.quote_ident(_STATE_TABLE)}"
            absence_sql = ""
            if request.absent_from is not None:
                probe_ref = self.table_ref(request.absent_from.table)
                probe_key = self.quote_ident(request.absent_from.key_column)
                try:
                    cursor_conn.execute(f"SELECT {probe_key} FROM {probe_ref} LIMIT 0")
                except duckdb.Error:
                    raise AdapterError(
                        "State absence probe relation "
                        f"'{request.absent_from.table}."
                        f"{request.absent_from.key_column}' is unavailable"
                    ) from None
                absence_sql = (
                    f" AND NOT EXISTS (SELECT 1 FROM {probe_ref} AS probe "
                    f"WHERE probe.{probe_key} = state.record_key)"
                )
            nonce = uuid4().hex
            scope = request.scope

            def fetch(cursor_value: str | None) -> StatePage:
                last_key = decode_state_cursor(cursor_value, nonce)
                key_sql = " AND state.record_key > ?" if last_key is not None else ""
                params: list[Any] = [
                    scope.model_name,
                    scope.stage,
                    scope.target_identity,
                ]
                if last_key is not None:
                    params.append(last_key)
                try:
                    rows = cursor_conn.execute(
                        "SELECT state.record_key, state.input_fingerprint, "
                        "state.code_version, state.last_run_at "
                        f"FROM {state_ref} AS state "
                        "WHERE state.model_name = ? AND state.state_scope = ? "
                        f"AND state.target_identity = ?{key_sql}{absence_sql} "
                        "ORDER BY state.record_key LIMIT ?",
                        [*params, request.page_size],
                    ).fetchall()
                except duckdb.Error:
                    raise AdapterError("DuckDB state page read failed") from None
                records = tuple(
                    StatePageRecord(str(r[0]), str(r[1]), str(r[2]), r[3])
                    for r in rows
                )
                next_cursor = (
                    encode_state_cursor(nonce, records[-1].record_key)
                    if len(records) == request.page_size
                    else None
                )
                return StatePage(records=records, next_cursor=next_cursor)

            def close() -> None:
                nonlocal transaction_open
                try:
                    if transaction_open:
                        cursor_conn.execute("ROLLBACK")
                        transaction_open = False
                finally:
                    cursor_conn.close()

            return StatePageReader(
                page_size=request.page_size, fetch=fetch, close=close
            )
        except BaseException:
            if transaction_open:
                try:
                    cursor_conn.execute("ROLLBACK")
                except duckdb.Error:
                    pass
            cursor_conn.close()
            raise

    def _replace_state_scope(
        self,
        scope: StateScope,
        record_batches: Iterator[Sequence[StateRecord]],
        fence: StateScopeFence | None,
    ) -> int:
        total = 0
        try:
            with self._transaction():
                if fence is not None:
                    self._verify_state_fence(scope, fence)
                self._clear_state_rows(scope)
                for batch in record_batches:
                    self._insert_state_rows(scope, batch)
                    total += len(batch)
        except duckdb.ConstraintException:
            raise AdapterError(
                "State scope replacement contains duplicate record keys "
                "across batches"
            ) from None
        return total

    def _verify_state_fence(self, scope: StateScope, fence: StateScopeFence) -> None:
        # Self-assignment write instead of a plain read: touching the ledger
        # row makes this transaction conflict with any concurrent authority
        # reassignment, so a stale fence cannot commit alongside recovery.
        ledger_ref = f"{self.schema_ref}.{self.quote_ident(SERVING_LEDGER_TABLE)}"
        try:
            row = self.connection.execute(
                f"UPDATE {ledger_ref} SET fencing_token = fencing_token "
                "WHERE model_name = ? AND stage = ? AND target_identity = ? "
                "AND publication_id = ? AND fencing_token = ?",
                [
                    scope.model_name,
                    scope.stage,
                    scope.target_identity,
                    fence.publication_id,
                    fence.fencing_token,
                ],
            ).fetchone()
        except duckdb.CatalogException:
            raise StaleStateFenceError(
                "Fenced state replacement requires a serving ledger, and this "
                "schema has none"
            ) from None
        changed = int(row[0]) if row else 0
        if changed != 1:
            raise StaleStateFenceError(
                "Serving publication authority was reassigned; state scope "
                "replacement aborted without mutation"
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
