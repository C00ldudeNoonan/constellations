"""Warehouse adapter base class.

Each adapter wraps a warehouse-specific connection and exposes a uniform
interface for the runner: connect/close, schema management, materialization
(full + incremental), querying, and incremental-state CRUD. The point is
that runner.py / manifest.py / dbt_export.py / cli.py never speak DuckDB
SQL directly — they call adapter methods.

Today: DuckDB. v0.2.2: LanceDB. Beyond: Postgres / Snowflake / BigQuery /
Databricks / Redshift, matching the dbt-core set.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import polars as pl

from ..config.profile import WarehouseConfig


class AdapterError(Exception):
    pass


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

    # ─── lifecycle ────────────────────────────────────────────────────────

    def __enter__(self) -> Self:
        self._connect()
        self._ensure_schema()
        self._ensure_state_table()
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
    @abstractmethod
    def schema_ref(self) -> str:
        """Quoted, fully-qualified schema reference for use in SQL."""

    def quote_ident(self, name: str) -> str:
        """Quote a single SQL identifier. ANSI rules (double quotes, embedded
        quotes doubled); adapters with other dialects override. Every table,
        schema, catalog, and column name interpolated into SQL must pass
        through here — values stay in bound parameters."""
        return '"' + name.replace('"', '""') + '"'

    def table_ref(self, table: str) -> str:
        return f"{self.schema_ref}.{self.quote_ident(table)}"

    # ─── materialization ──────────────────────────────────────────────────

    @abstractmethod
    def materialize_full(self, table: str, df: pl.DataFrame) -> int:
        """Replace `table` with `df`. Returns row count written."""

    @abstractmethod
    def materialize_incremental(
        self,
        table: str,
        df: pl.DataFrame,
        *,
        key_col: str,
        on_schema_change: str = "fail",
    ) -> int:
        """Upsert rows in `df` into `table`, keyed on `key_col`. Returns rows written.

        Columns are matched by name, never by position. When `df`'s columns
        differ from the existing table's, `on_schema_change` decides:
        `fail` raises with a hint to --full-refresh; `append_new_columns`
        adds missing columns to the table (existing rows get NULL);
        `ignore` inserts only the columns the table already has."""

    @abstractmethod
    def delete_rows(self, table: str, *, key_col: str, keys: list[str]) -> int:
        """Delete rows from `table` where `key_col` is in `keys`. Returns the
        number of rows removed. A no-op (returns 0) if the table does not exist."""

    @abstractmethod
    def drop_table(self, table: str) -> None: ...

    # ─── querying ─────────────────────────────────────────────────────────

    @abstractmethod
    def execute(self, sql: str, params: list[Any] | None = None) -> Any: ...

    @abstractmethod
    def query_df(self, sql: str, params: list[Any] | None = None) -> pl.DataFrame: ...

    @abstractmethod
    def scalar(self, sql: str, params: list[Any] | None = None) -> Any:
        """First column of first row, or None."""

    @abstractmethod
    def rows(self, sql: str, params: list[Any] | None = None) -> list[tuple[Any, ...]]: ...

    @abstractmethod
    def list_tables(self) -> list[str]: ...

    # ─── lifecycle-bypass operations ──────────────────────────────────────

    @abstractmethod
    def clean(self) -> str:
        """Remove everything this adapter has materialized. Returns a human-readable
        description of what was removed. Implementations handle their own short-lived
        connection if needed — does not require __enter__ to be called first."""

    # ─── incremental state CRUD ───────────────────────────────────────────

    @abstractmethod
    def fetch_state(self, model_name: str) -> dict[str, tuple[str, str]]:
        """Return {document_id: (content_hash, code_version)} for `model_name`."""

    @abstractmethod
    def upsert_state(
        self, model_name: str, records: list[tuple[str, str, str]]
    ) -> None: ...

    @abstractmethod
    def clear_model_state(self, model_name: str) -> None: ...

    @abstractmethod
    def delete_state(self, model_name: str, document_ids: list[str]) -> None:
        """Remove state rows for the given `document_ids` under `model_name`."""
