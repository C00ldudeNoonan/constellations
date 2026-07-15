"""Warehouse adapter base class.

Each adapter wraps a warehouse-specific connection and exposes a uniform
interface for the runner: connect/close, schema management, materialization
(full + incremental), querying, and incremental-state CRUD. The point is
that runner.py / manifest.py / dbt_export.py / cli.py never speak DuckDB
SQL directly — they call adapter methods.

Today: DuckDB and BigQuery. Future warehouse adapters follow the dbt-core set.
Vector stores such as LanceDB are a separate role and do not emulate this API.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import polars as pl
from pydantic import BaseModel, ValidationError

from ..config.profile import WarehouseConfig
from ..hashing import canonical_fingerprint


class AdapterError(Exception):
    pass


class AdapterCapabilityError(AdapterError):
    pass


@dataclass(frozen=True)
class StateScope:
    model_name: str
    stage: str = "materialization"
    target_identity: str = "warehouse-v1"

    def __post_init__(self) -> None:
        for field_name in ("model_name", "stage", "target_identity"):
            if not getattr(self, field_name):
                raise ValueError(f"State scope {field_name} must not be empty")
        if self.stage != "materialization" and self.target_identity == "warehouse-v1":
            raise ValueError(
                "Non-materialization state scopes require an explicit target_identity; "
                "use StateScope.for_target_descriptor() for semantic target config"
            )

    @classmethod
    def for_target_descriptor(
        cls,
        model_name: str,
        *,
        stage: str,
        descriptor: Mapping[str, Any],
    ) -> Self:
        """Build a scope from semantic, non-secret serving-target configuration.

        The caller must exclude credentials and execution-only configuration;
        only the resulting domain-separated fingerprint is retained.
        """
        target_identity = canonical_fingerprint(
            descriptor,
            domain="dbt-ml-state-target-identity",
            version=1,
        )
        return cls(model_name, stage, target_identity)


@dataclass(frozen=True)
class StateValue:
    input_fingerprint: str
    code_version: str

    def __post_init__(self) -> None:
        if not self.input_fingerprint:
            raise ValueError("State input_fingerprint must not be empty")
        if not self.code_version:
            raise ValueError("State code_version must not be empty")


@dataclass(frozen=True)
class StateRecord:
    record_key: str
    input_fingerprint: str
    code_version: str

    def __post_init__(self) -> None:
        if not self.record_key:
            raise ValueError("State record_key must not be empty")
        if not self.input_fingerprint:
            raise ValueError("State input_fingerprint must not be empty")
        if not self.code_version:
            raise ValueError("State code_version must not be empty")


def validate_state_records(records: Sequence[StateRecord]) -> None:
    keys = [record.record_key for record in records]
    if len(keys) != len(set(keys)):
        raise AdapterError("State records contain duplicate record_key values")


def validate_state_keys(record_keys: Sequence[str]) -> None:
    if any(not record_key for record_key in record_keys):
        raise AdapterError("State record keys must not be empty")
    if len(record_keys) != len(set(record_keys)):
        raise AdapterError("State record keys contain duplicate values")


class WarehouseCapability(StrEnum):
    SQL_QUERIES = "sql_queries"
    TABULAR_READS = "tabular_reads"
    SQL_SCHEMA_TESTS = "sql_schema_tests"
    ATOMIC_FULL_REPLACE = "atomic_full_replace"
    ATOMIC_KEYED_UPSERT = "atomic_keyed_upsert"
    TRANSACTIONS = "transactions"
    TYPED_EMPTY_RELATIONS = "typed_empty_relations"
    CHUNKED_WRITES = "chunked_writes"
    SCHEMA_EVOLUTION = "schema_evolution"


def validate_incremental_keys(df: pl.DataFrame, key_col: str) -> None:
    if key_col not in df.columns:
        raise AdapterError(
            f"Incremental input is missing required key column '{key_col}'"
        )
    null_count = df[key_col].null_count()
    if null_count:
        raise AdapterError(
            f"Incremental key column '{key_col}' contains {null_count} NULL value(s)"
        )
    duplicate_count = df.height - df[key_col].n_unique()
    if duplicate_count:
        raise AdapterError(
            f"Incremental key column '{key_col}' contains "
            f"{duplicate_count} duplicate value(s)"
        )


class WarehouseAdapter(ABC):
    """Lifecycle-managed warehouse driver."""

    def __init__(self, config: WarehouseConfig, *, project_dir: Path | None = None) -> None:
        self.config = config
        self.project_dir = project_dir

    # ─── classification ────────────────────────────────────────────────────

    @classmethod
    @abstractmethod
    def adapter_type(cls) -> str:
        """Short name used in profiles.yml `warehouse.type`."""

    @classmethod
    @abstractmethod
    def config_model(cls) -> type[WarehouseConfig]:
        """The WarehouseConfig subclass this adapter's profile block
        validates against."""

    @classmethod
    @abstractmethod
    def capabilities(cls) -> frozenset[WarehouseCapability]:
        """Operations and guarantees callers may rely on for this adapter."""

    @classmethod
    def supports(cls, capability: WarehouseCapability) -> bool:
        return capability in cls.capabilities()

    def require_capability(
        self, capability: WarehouseCapability, *, operation: str
    ) -> None:
        if self.supports(capability):
            return
        raise AdapterCapabilityError(
            f"Warehouse adapter '{self.adapter_type()}' does not support {operation} "
            f"(missing capability: {capability.value})"
        )

    @classmethod
    def warehouse_options_model(cls) -> type[BaseModel] | None:
        """Pydantic model validating model-level `warehouse_options` for this
        adapter (issue #91), or None when the adapter supports none. A None
        adapter ignores the block entirely — a project can carry e.g. BigQuery
        partitioning config while its dev target runs DuckDB."""
        return None

    def parse_warehouse_options(
        self, options: dict[str, Any], *, model_name: str
    ) -> BaseModel | None:
        """Validate a model's `warehouse_options` against this adapter's
        options model. Unknown keys are a config error on adapters that
        declare a model, and ignored on adapters that don't."""
        options_model = self.warehouse_options_model()
        if not options or options_model is None:
            return None
        try:
            return options_model.model_validate(options)
        except ValidationError as e:
            raise AdapterError(
                f"Model '{model_name}': invalid warehouse_options for "
                f"{self.adapter_type()}: {e}"
            ) from e

    # ─── lifecycle ────────────────────────────────────────────────────────

    def __enter__(self) -> Self:
        self._connect()
        try:
            self._ensure_schema()
            self._ensure_state_table()
        except BaseException as error:
            try:
                self._close()
            except BaseException as close_error:
                error.add_note(
                    f"Failed to close warehouse adapter after initialization "
                    f"failed: {close_error}"
                )
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._close()

    @abstractmethod
    def _connect(self) -> None: ...

    @abstractmethod
    def _close(self) -> None: ...

    @abstractmethod
    def _ensure_schema(self) -> None: ...

    @abstractmethod
    def _ensure_state_table(self) -> None: ...

    # ─── identity / SQL references ────────────────────────────────────────

    @property
    @abstractmethod
    def catalog(self) -> str:
        """Catalog name used in SQL references and emitted dbt sources."""

    @property
    def schema(self) -> str:
        return self.config.schema_name

    @property
    def schema_ref(self) -> str:
        """Quoted, fully-qualified schema reference for use in SQL."""
        self.require_capability(
            WarehouseCapability.SQL_QUERIES,
            operation="SQL schema references",
        )
        raise AdapterCapabilityError(
            f"Warehouse adapter '{self.adapter_type()}' does not implement schema_ref"
        )

    def quote_ident(self, name: str) -> str:
        """Quote a single SQL identifier. ANSI rules (double quotes, embedded
        quotes doubled); adapters with other dialects override. Every table,
        schema, catalog, and column name interpolated into SQL must pass
        through here — values stay in bound parameters."""
        return '"' + name.replace('"', '""') + '"'

    def table_ref(self, table: str) -> str:
        self.require_capability(
            WarehouseCapability.SQL_QUERIES,
            operation="SQL table references",
        )
        return f"{self.schema_ref}.{self.quote_ident(table)}"

    # ─── materialization ──────────────────────────────────────────────────

    @abstractmethod
    def materialize_full(
        self, table: str, df: pl.DataFrame, *, options: BaseModel | None = None
    ) -> int:
        """Replace `table` with `df`. Returns row count written.

        `options` is this adapter's parsed warehouse_options (from
        `parse_warehouse_options`); it shapes physical layout when the target
        table is (re)created and is None for adapters that support none."""

    @abstractmethod
    def materialize_incremental(
        self,
        table: str,
        df: pl.DataFrame,
        *,
        key_col: str,
        on_schema_change: str = "fail",
        options: BaseModel | None = None,
    ) -> int:
        """Upsert rows in `df` into `table`, keyed on `key_col`. Returns rows written.

        Columns are matched by name, never by position. When `df`'s columns
        differ from the existing table's, `on_schema_change` decides:
        `fail` raises with a hint to --full-refresh; `append_new_columns`
        adds missing columns to the table (existing rows get NULL);
        `ignore` inserts only the columns the table already has.

        `options` (parsed warehouse_options) applies only when the target
        table does not exist yet — an existing table keeps its physical
        layout until --full-refresh rebuilds it."""

    @abstractmethod
    def materialize_full_chunks(
        self,
        table: str,
        chunks: Iterable[pl.DataFrame],
        *,
        options: BaseModel | None = None,
    ) -> int:
        """Replace `table` with the concatenation of `chunks` without holding
        them all in memory (issue #77): chunks land in a staging table
        (`dbt_ml_staging__<table>`) that replaces the target once every chunk
        is in. Intra-run schema drift between chunks is unioned — new columns
        are added, missing columns fill with NULL — matching what one
        whole-run DataFrame gave for free. Returns total rows written."""

    @abstractmethod
    def delete_rows(self, table: str, *, key_col: str, keys: list[str]) -> int:
        """Delete rows from `table` where `key_col` is in `keys`. Returns the
        number of rows removed. A no-op (returns 0) if the table does not exist."""

    @abstractmethod
    def delete_rows_and_state(
        self,
        table: str,
        *,
        key_col: str,
        keys: Sequence[str],
        state_scope: StateScope,
        state_record_keys: Sequence[str] | None = None,
    ) -> int:
        """Atomically delete target rows and their scoped state when supported."""

    @abstractmethod
    def drop_table(self, table: str) -> None: ...

    # ─── querying ─────────────────────────────────────────────────────────

    def read_table(self, table: str, *, limit: int | None = None) -> pl.DataFrame:
        """Read a materialized relation without exposing SQL to core callers.

        SQL adapters inherit this implementation. A future non-SQL tabular
        warehouse can override it while retaining the same core contract.
        """
        self.require_capability(
            WarehouseCapability.TABULAR_READS,
            operation="reading materialized tables",
        )
        if limit is not None and limit < 0:
            raise AdapterError("Table read limit must be non-negative")
        suffix = f" LIMIT {limit}" if limit is not None else ""
        return self.query_df(f"SELECT * FROM {self.table_ref(table)}{suffix}")

    def row_count(self, table: str) -> int:
        """Return the relation row count through a typed core operation."""
        self.require_capability(
            WarehouseCapability.TABULAR_READS,
            operation="counting materialized table rows",
        )
        value = self.scalar(f"SELECT COUNT(*) FROM {self.table_ref(table)}")
        return int(value or 0)

    def execute(self, sql: str, params: list[Any] | None = None) -> Any:
        self.require_capability(
            WarehouseCapability.SQL_QUERIES,
            operation="executing SQL",
        )
        raise AdapterCapabilityError(
            f"Warehouse adapter '{self.adapter_type()}' does not implement execute"
        )

    def query_df(self, sql: str, params: list[Any] | None = None) -> pl.DataFrame:
        self.require_capability(
            WarehouseCapability.SQL_QUERIES,
            operation="querying SQL into a DataFrame",
        )
        raise AdapterCapabilityError(
            f"Warehouse adapter '{self.adapter_type()}' does not implement query_df"
        )

    def scalar(self, sql: str, params: list[Any] | None = None) -> Any:
        """First column of first row, or None."""
        self.require_capability(
            WarehouseCapability.SQL_QUERIES,
            operation="querying SQL scalar values",
        )
        raise AdapterCapabilityError(
            f"Warehouse adapter '{self.adapter_type()}' does not implement scalar"
        )

    def rows(
        self, sql: str, params: list[Any] | None = None
    ) -> list[tuple[Any, ...]]:
        self.require_capability(
            WarehouseCapability.SQL_QUERIES,
            operation="querying SQL rows",
        )
        raise AdapterCapabilityError(
            f"Warehouse adapter '{self.adapter_type()}' does not implement rows"
        )

    @abstractmethod
    def list_tables(self) -> list[str]: ...

    # ─── incremental state CRUD ───────────────────────────────────────────

    @abstractmethod
    def fetch_state(self, scope: StateScope) -> dict[str, StateValue]:
        """Return state values keyed by the stable record identity in `scope`."""

    @abstractmethod
    def upsert_state(
        self, scope: StateScope, records: Sequence[StateRecord]
    ) -> None: ...

    @abstractmethod
    def replace_state(
        self, scope: StateScope, records: Sequence[StateRecord]
    ) -> None:
        """Atomically replace the complete state snapshot for `scope`."""

    @abstractmethod
    def clear_state(self, scope: StateScope) -> None: ...

    @abstractmethod
    def delete_state(self, scope: StateScope, record_keys: Sequence[str]) -> None:
        """Remove state rows for `record_keys` within exactly `scope`."""
