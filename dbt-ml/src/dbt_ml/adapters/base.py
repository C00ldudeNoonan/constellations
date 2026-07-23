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
from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from math import isfinite
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import polars as pl
import pyarrow as pa
from pydantic import BaseModel, ValidationError

from ..config.profile import WarehouseConfig
from ..hashing import canonical_fingerprint


class AdapterError(Exception):
    pass


class AdapterConfigError(AdapterError):
    """Adapter validation failure carrying only diagnostic-safe details."""

    def __init__(
        self,
        message: str,
        validation_details: Iterable[Mapping[str, Any]],
    ) -> None:
        super().__init__(message)
        self.validation_details = tuple(
            {
                "loc": tuple(
                    part if isinstance(part, str | int) else "<unknown>"
                    for part in detail.get("loc", ())
                ),
                "msg": str(detail.get("msg", "Invalid value")),
                "type": str(detail.get("type", "value_error")),
            }
            for detail in validation_details
        )


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


# Serving-ledger table name shared with retrieval.coordination: fenced state
# replacement must verify a publication claim in the same warehouse that owns
# the state rows, without adapters importing retrieval code.
SERVING_LEDGER_TABLE = "dbt_ml_serving_ledger"

_MAX_STATE_CURSOR_CHARS = 8192
_MAX_STATE_PAGE_SIZE = 100_000
_MAX_STATE_SUBSET_KEYS = 100_000


class StaleStateFenceError(AdapterError):
    """A fenced state mutation observed a reassigned serving-ledger claim."""


@dataclass(frozen=True)
class StateScopeFence:
    """Serving-ledger claim a fenced state replacement must still hold.

    Values come from an acquired `PublishLease`; the adapter refuses the
    replacement unless the ledger row for the scope still carries exactly
    this publication id and fencing token.
    """

    publication_id: str
    fencing_token: int

    def __post_init__(self) -> None:
        if not self.publication_id:
            raise AdapterError("State scope fence publication_id must not be empty")
        if self.fencing_token < 1:
            raise AdapterError("State scope fence fencing_token must be positive")


@dataclass(frozen=True)
class StateAbsenceProbe:
    """Restrict ordered state iteration to keys absent from one relation.

    The probe relation lives in the same warehouse as the state table, so the
    adapter can evaluate absence without materializing either key domain.
    """

    table: str
    key_column: str

    def __post_init__(self) -> None:
        if not self.table:
            raise AdapterError("State absence probe table must not be empty")
        if not self.key_column:
            raise AdapterError("State absence probe key_column must not be empty")


@dataclass(frozen=True)
class StatePageRecord:
    """Projected state row surfaced by ordered paged iteration."""

    record_key: str
    input_fingerprint: str
    code_version: str
    committed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.record_key:
            raise AdapterError("State page record_key must not be empty")
        if not self.input_fingerprint:
            raise AdapterError("State page input_fingerprint must not be empty")
        if not self.code_version:
            raise AdapterError("State page code_version must not be empty")


@dataclass(frozen=True)
class StatePage:
    """One bounded, key-ordered page of scoped state records.

    `next_cursor` is an opaque adapter value scoped to the open reader's
    snapshot; None marks the final page.
    """

    records: tuple[StatePageRecord, ...]
    next_cursor: str | None


@dataclass(frozen=True)
class StatePageRequest:
    scope: StateScope
    page_size: int
    absent_from: StateAbsenceProbe | None

    def __post_init__(self) -> None:
        if not 1 <= self.page_size <= _MAX_STATE_PAGE_SIZE:
            raise AdapterError(
                f"State page_size must be between 1 and {_MAX_STATE_PAGE_SIZE}"
            )


def validate_state_cursor(cursor: str | None) -> None:
    if cursor is None:
        return
    if not cursor or len(cursor) > _MAX_STATE_CURSOR_CHARS:
        raise AdapterError("State page cursor is invalid for this reader")


def encode_state_cursor(nonce: str, last_key: str) -> str:
    """Build an opaque keyset cursor bound to one reader snapshot."""
    payload = urlsafe_b64encode(last_key.encode("utf-8")).decode("ascii")
    cursor = f"{nonce}:{payload}"
    validate_state_cursor(cursor)
    return cursor


def decode_state_cursor(cursor: str | None, nonce: str) -> str | None:
    """Recover the keyset position, rejecting foreign or malformed cursors."""
    if cursor is None:
        return None
    validate_state_cursor(cursor)
    prefix, sep, payload = cursor.partition(":")
    if not sep or prefix != nonce:
        raise AdapterError(
            "State page cursor does not belong to this reader snapshot"
        )
    try:
        return urlsafe_b64decode(payload.encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        raise AdapterError("State page cursor is malformed") from None


class StatePageReader:
    """Ordered, snapshot-consistent paged access to one state scope.

    Ordering is strict ascending Unicode code-point order on `record_key`,
    which both DuckDB (binary collation) and BigQuery (code-point STRING
    comparison) provide natively and Python `str` comparison matches, so core
    merge logic can validate it. Each page is deterministic for its cursor:
    retrying a cursor within the open reader returns the same page, and a
    cursor from another reader or snapshot is rejected rather than silently
    skipping or repeating records.
    """

    def __init__(
        self,
        *,
        page_size: int,
        fetch: Callable[[str | None], StatePage],
        close: Callable[[], None],
    ) -> None:
        self._page_size = page_size
        self._fetch = fetch
        self._close_callback = close
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def fetch_page(self, cursor: str | None = None) -> StatePage:
        if self._closed:
            raise AdapterError("State page reader is closed")
        validate_state_cursor(cursor)
        page = self._fetch(cursor)
        if len(page.records) > self._page_size:
            raise AdapterError("State page exceeded the requested page size")
        validate_state_cursor(page.next_cursor)
        previous_key: str | None = None
        for record in page.records:
            if previous_key is not None and record.record_key <= previous_key:
                raise AdapterError(
                    "State page records are not strictly ordered by record_key"
                )
            previous_key = record.record_key
        return page

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_callback()


class WarehouseCapability(StrEnum):
    SQL_QUERIES = "sql_queries"
    SQL_MODEL_MATERIALIZATION = "sql_model_materialization"
    SQL_INCREMENTAL_MATERIALIZATION = "sql_incremental_materialization"
    TABULAR_READS = "tabular_reads"
    SQL_SCHEMA_TESTS = "sql_schema_tests"
    ATOMIC_FULL_REPLACE = "atomic_full_replace"
    ATOMIC_KEYED_UPSERT = "atomic_keyed_upsert"
    TRANSACTIONS = "transactions"
    TYPED_EMPTY_RELATIONS = "typed_empty_relations"
    CHUNKED_WRITES = "chunked_writes"
    SCHEMA_EVOLUTION = "schema_evolution"
    STREAMING_TABULAR_READS = "streaming_tabular_reads"
    TABULAR_PREDICATE_PUSHDOWN = "tabular_predicate_pushdown"
    PAGED_STATE_RECONCILIATION = "paged_state_reconciliation"
    ATOMIC_STATE_SCOPE_REPLACE = "atomic_state_scope_replace"


@dataclass(frozen=True)
class SqlRelationColumn:
    name: str
    data_type: str


@dataclass(frozen=True)
class SqlRelationSchema:
    """Output schema of a compiled SELECT, from an adapter dry-run."""

    columns: tuple[SqlRelationColumn, ...]


@dataclass(frozen=True)
class SqlMaterializationResult:
    """Receipt from materializing a SQL model via CTAS/replace, or a keyed
    merge/upsert for `materialization: incremental` (issue #142). For a merge,
    `rows_written` is the total staged/source row count; `rows_inserted` and
    `rows_updated` are a best-effort split where the adapter can distinguish
    them, else both stay `None`."""

    relation: str
    rows_written: int
    job_metadata: dict[str, Any] = field(default_factory=dict)
    rows_inserted: int | None = None
    rows_updated: int | None = None


class ReadPredicateOperator(StrEnum):
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "le"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "ge"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


ReadScalar = str | int | float | bool | date | datetime


@dataclass(frozen=True, repr=False)
class ReadPredicate:
    column: str
    operator: ReadPredicateOperator
    value: ReadScalar | tuple[ReadScalar, ...] | None = None

    def __post_init__(self) -> None:
        if not self.column:
            raise AdapterError("Read predicate column must not be empty")
        if not isinstance(self.operator, ReadPredicateOperator):
            raise AdapterError("Read predicate operator is invalid")
        null_operator = self.operator in {
            ReadPredicateOperator.IS_NULL,
            ReadPredicateOperator.IS_NOT_NULL,
        }
        membership_operator = self.operator in {
            ReadPredicateOperator.IN,
            ReadPredicateOperator.NOT_IN,
        }
        if null_operator:
            if self.value is not None:
                raise AdapterError(
                    f"Read predicate operator '{self.operator.value}' does not accept a value"
                )
            return
        if self.value is None:
            raise AdapterError(
                f"Read predicate operator '{self.operator.value}' requires a value"
            )
        if membership_operator:
            if not isinstance(self.value, tuple) or not self.value:
                raise AdapterError(
                    f"Read predicate operator '{self.operator.value}' requires a non-empty tuple"
                )
            if any(not _is_read_scalar(item) for item in self.value):
                raise AdapterError("Read predicate tuple contains an unsupported value")
            first_type = type(self.value[0])
            if any(type(item) is not first_type for item in self.value[1:]):
                raise AdapterError("Read predicate tuple values must share one type")
            return
        if isinstance(self.value, tuple):
            raise AdapterError(
                f"Read predicate operator '{self.operator.value}' requires a scalar value"
            )
        if not _is_read_scalar(self.value):
            raise AdapterError("Read predicate contains an unsupported value")

    def __repr__(self) -> str:
        value = "" if self.value is None else ", value=<redacted>"
        return (
            f"ReadPredicate(column={self.column!r}, operator={self.operator.value!r}{value})"
        )

    def _fingerprint_payload(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "operator": self.operator.value,
            "value": self.value,
        }


def _is_read_scalar(value: Any) -> bool:
    if not isinstance(value, str | int | float | bool | date | datetime):
        return False
    return not isinstance(value, float) or isfinite(value)


@dataclass(frozen=True)
class TableReadRequest:
    table: str
    columns: tuple[str, ...] | None
    batch_size: int
    predicates: tuple[ReadPredicate, ...]
    key_column: str | None

    def __post_init__(self) -> None:
        if not self.table:
            raise AdapterError("Table read name must not be empty")
        if not 1 <= self.batch_size <= 100_000:
            raise AdapterError("Table read batch_size must be between 1 and 100000")
        if self.columns is not None:
            if not self.columns or any(not column for column in self.columns):
                raise AdapterError("Table read columns must not be empty")
            if len(self.columns) != len(set(self.columns)):
                raise AdapterError("Table read columns contain duplicate names")
        if self.key_column == "":
            raise AdapterError("Table read key_column must not be empty")
        if (
            self.columns is not None
            and self.key_column is not None
            and self.key_column not in self.columns
        ):
            raise AdapterError("Table read key_column must be included in columns")

    @property
    def referenced_columns(self) -> frozenset[str]:
        columns = {predicate.column for predicate in self.predicates}
        if self.columns is not None:
            columns.update(self.columns)
        if self.key_column is not None:
            columns.add(self.key_column)
        return frozenset(columns)

    def _fingerprint_payload(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "columns": self.columns,
            "predicates": [
                predicate._fingerprint_payload() for predicate in self.predicates
            ],
            "key_column": self.key_column,
        }


class ReadOrdering(StrEnum):
    UNSPECIFIED = "unspecified"


def _empty_record_batch(schema: pa.Schema) -> pa.RecordBatch:
    return pa.RecordBatch.from_arrays(
        [pa.array([], type=field.type) for field in schema],
        schema=schema,
    )


class TableReadSnapshot:
    """One-shot bounded Arrow stream tied to one immutable warehouse snapshot."""

    def __init__(
        self,
        *,
        schema: pa.Schema,
        fingerprint: str,
        batches: Iterator[pa.RecordBatch],
        validate_unchanged: Callable[[], None],
        close: Callable[[], None],
        ordering: ReadOrdering = ReadOrdering.UNSPECIFIED,
        generation_fingerprint: str | None = None,
    ) -> None:
        _validate_read_fingerprint(fingerprint)
        if generation_fingerprint is not None:
            _validate_read_fingerprint(generation_fingerprint)
        self.schema = schema
        self.fingerprint = fingerprint
        self._generation_fingerprint = generation_fingerprint
        self.ordering = ordering
        self._batches = batches
        self._validate_callback = validate_unchanged
        self._close_callback = close
        self._started = False
        self._exhausted = False
        self._closed = False

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def generation_fingerprint(self) -> str | None:
        return self._generation_fingerprint

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        if self._closed:
            raise AdapterError("Table read snapshot is closed")
        if self._started:
            raise AdapterError("Table read snapshot batches can only be consumed once")
        self._started = True
        try:
            for ordinal, batch in enumerate(self._batches):
                if not batch.schema.equals(self.schema, check_metadata=False):
                    batch = _empty_record_batch(batch.schema)
                    raise AdapterError(
                        "Warehouse returned an unstable schema while reading table batches "
                        f"at batch {ordinal}"
                    )
                yield batch
            self._exhausted = True
        except BaseException as error:
            try:
                self.close()
            except BaseException:
                error.add_note("Failed to close table read snapshot after batch failure")
            raise

    def validate_unchanged(self) -> None:
        if self._closed:
            raise AdapterError("Table read snapshot is closed")
        self._validate_callback()

    def _set_generation_fingerprint(self, fingerprint: str) -> None:
        _validate_read_fingerprint(fingerprint)
        self._generation_fingerprint = fingerprint

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_batches = getattr(self._batches, "close", None)
        try:
            if callable(close_batches):
                close_batches()
        except BaseException as error:
            try:
                self._close_callback()
            except BaseException:
                error.add_note("Failed to close table read snapshot resources")
            raise
        else:
            self._close_callback()


def _validate_read_fingerprint(fingerprint: str) -> None:
    if len(fingerprint) != 32:
        raise AdapterError(
            "Table read snapshot fingerprint must be 32 hexadecimal characters"
        )
    try:
        int(fingerprint, 16)
    except ValueError:
        raise AdapterError(
            "Table read snapshot fingerprint must be 32 hexadecimal characters"
        ) from None


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

    # ─── SQL model materialization (#143) ─────────────────────────────────

    def materialize_sql_full(
        self,
        table: str,
        select_sql: str,
        *,
        options: BaseModel | None = None,
    ) -> SqlMaterializationResult:
        """Replace `table` with the result of the compiled `select_sql` via an
        adapter-owned CTAS/replace — rows never leave the warehouse. `select_sql`
        is a validated single SELECT with refs already compiled to quoted
        relations; the adapter owns target quoting, staging, atomic replacement,
        and physical layout from `options`. Availability is gated on the
        `SQL_MODEL_MATERIALIZATION` capability."""
        raise AdapterError(
            f"Adapter '{self.adapter_type()}' does not support SQL model "
            "materialization (SQL_MODEL_MATERIALIZATION)."
        )

    def dry_run_sql(self, select_sql: str) -> SqlRelationSchema:
        """Validate `select_sql` against the real dialect without materializing
        rows and return its output schema. Raises AdapterError on invalid SQL."""
        raise AdapterError(
            f"Adapter '{self.adapter_type()}' does not support SQL dry-run."
        )

    def relation_exists(self, table: str) -> bool:
        """True if `table` currently exists in the target schema. Used to
        decide, per run, whether an incremental SQL model's `is_incremental()`
        branch is active (issue #142)."""
        raise AdapterError(
            f"Adapter '{self.adapter_type()}' cannot check relation existence."
        )

    def materialize_sql_incremental(
        self,
        table: str,
        select_sql: str,
        *,
        unique_key: str,
        on_schema_change: str = "fail",
        options: BaseModel | None = None,
    ) -> SqlMaterializationResult:
        """Merge/upsert the result of `select_sql` into the existing `table`,
        keyed on `unique_key`. Only called when `table` already exists and the
        run is not a full refresh — the runner compiles `select_sql` from the
        model's `is_incremental()` branch and calls `materialize_sql_full`
        instead on first run or `--full-refresh`. The adapter validates
        `unique_key` is non-null and unique in `select_sql`'s result via an
        in-warehouse check (never pulling rows into Python) before mutating the
        target, and applies `on_schema_change` exactly as
        `materialize_incremental` does for DataFrame-sourced models. Availability
        is gated on the `SQL_INCREMENTAL_MATERIALIZATION` capability."""
        raise AdapterError(
            f"Adapter '{self.adapter_type()}' does not support incremental SQL "
            "model materialization (SQL_INCREMENTAL_MATERIALIZATION)."
        )

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
        keys: Sequence[Any],
        state_scope: StateScope,
        state_record_keys: Sequence[str] | None = None,
    ) -> int:
        """Atomically delete target rows and their scoped state when supported."""

    @abstractmethod
    def drop_table(self, table: str) -> None: ...

    # ─── querying ─────────────────────────────────────────────────────────

    @contextmanager
    def table_snapshot(
        self,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        batch_size: int = 10_000,
        predicate: ReadPredicate | Sequence[ReadPredicate] | None = None,
        key_column: str | None = None,
    ) -> Iterator[TableReadSnapshot]:
        """Open a bounded, projected, immutable relation snapshot.

        Predicates are combined with logical AND. Ordering is unspecified; a
        consumer needing deterministic reconciliation must use a separately
        advertised ordered-read capability.
        """
        self.require_capability(
            WarehouseCapability.STREAMING_TABULAR_READS,
            operation="streaming materialized table reads",
        )
        if predicate is None:
            predicates: tuple[ReadPredicate, ...] = ()
        elif isinstance(predicate, ReadPredicate):
            predicates = (predicate,)
        else:
            predicates = tuple(predicate)
        request = TableReadRequest(
            table=table,
            columns=tuple(columns) if columns is not None else None,
            batch_size=batch_size,
            predicates=predicates,
            key_column=key_column,
        )
        if request.predicates:
            self.require_capability(
                WarehouseCapability.TABULAR_PREDICATE_PUSHDOWN,
                operation="typed table predicate pushdown",
            )
        snapshot = self._open_table_snapshot(request)
        try:
            yield snapshot
            if snapshot.exhausted:
                snapshot.validate_unchanged()
        except BaseException as error:
            try:
                snapshot.close()
            except BaseException:
                error.add_note(
                    "Failed to close table read snapshot after operation failure"
                )
            raise
        else:
            snapshot.close()

    def _open_table_snapshot(self, request: TableReadRequest) -> TableReadSnapshot:
        self.require_capability(
            WarehouseCapability.STREAMING_TABULAR_READS,
            operation="streaming materialized table reads",
        )
        raise AdapterCapabilityError(
            f"Warehouse adapter '{self.adapter_type()}' does not implement "
            "streaming materialized table reads"
        )

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

    # ─── paged state reconciliation (issue #153) ──────────────────────────

    def fetch_state_subset(
        self, scope: StateScope, record_keys: Sequence[str]
    ) -> dict[str, StateValue]:
        """Return state values for exactly `record_keys` within `scope`.

        Bounded point lookups for one publication batch; memory residency is
        proportional to `record_keys`, never to the total scope size."""
        self.require_capability(
            WarehouseCapability.PAGED_STATE_RECONCILIATION,
            operation="bounded state key lookups",
        )
        validate_state_keys(record_keys)
        if len(record_keys) > _MAX_STATE_SUBSET_KEYS:
            raise AdapterError(
                f"State subset lookups are bounded to {_MAX_STATE_SUBSET_KEYS} keys"
            )
        if not record_keys:
            return {}
        return self._fetch_state_subset(scope, record_keys)

    def _fetch_state_subset(
        self, scope: StateScope, record_keys: Sequence[str]
    ) -> dict[str, StateValue]:
        raise AdapterCapabilityError(
            f"Warehouse adapter '{self.adapter_type()}' does not implement "
            "bounded state key lookups"
        )

    @contextmanager
    def state_page_reader(
        self,
        scope: StateScope,
        *,
        page_size: int,
        absent_from: StateAbsenceProbe | None = None,
    ) -> Iterator[StatePageReader]:
        """Open ordered, snapshot-consistent paged iteration over `scope`.

        Records stream in strict ascending `record_key` order across pages.
        All pages observe one immutable snapshot of the state scope (and of
        the `absent_from` relation when given), so interleaved mutations —
        including the caller's own deletions — cannot skip or repeat records.
        A snapshot that cannot be maintained fails the read; it never
        degrades silently."""
        self.require_capability(
            WarehouseCapability.PAGED_STATE_RECONCILIATION,
            operation="ordered paged state reads",
        )
        request = StatePageRequest(
            scope=scope, page_size=page_size, absent_from=absent_from
        )
        reader = self._open_state_page_reader(request)
        try:
            yield reader
        finally:
            reader.close()

    def _open_state_page_reader(self, request: StatePageRequest) -> StatePageReader:
        raise AdapterCapabilityError(
            f"Warehouse adapter '{self.adapter_type()}' does not implement "
            "ordered paged state reads"
        )

    def replace_state_scope(
        self,
        scope: StateScope,
        record_batches: Iterable[Sequence[StateRecord]],
        *,
        fence: StateScopeFence | None = None,
    ) -> int:
        """Atomically replace every state row in `scope` with the streamed
        batches, returning rows written.

        Input is bounded: batches are validated and written incrementally,
        never held together in memory. The replacement commits atomically —
        on any failure (including cross-batch duplicate keys, detected
        warehouse-side) the prior scope state is fully retained. With a
        `fence`, the replacement succeeds only while the serving ledger row
        for this scope still carries the fenced publication claim; a
        reassigned claim raises StaleStateFenceError without mutating
        state."""
        self.require_capability(
            WarehouseCapability.ATOMIC_STATE_SCOPE_REPLACE,
            operation="atomic state scope replacement",
        )
        return self._replace_state_scope(
            scope, _validated_state_batches(record_batches), fence
        )

    def _replace_state_scope(
        self,
        scope: StateScope,
        record_batches: Iterator[Sequence[StateRecord]],
        fence: StateScopeFence | None,
    ) -> int:
        raise AdapterCapabilityError(
            f"Warehouse adapter '{self.adapter_type()}' does not implement "
            "atomic state scope replacement"
        )


def _validated_state_batches(
    record_batches: Iterable[Sequence[StateRecord]],
) -> Iterator[Sequence[StateRecord]]:
    for batch in record_batches:
        validate_state_records(batch)
        if batch:
            yield batch
