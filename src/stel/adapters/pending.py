"""Filtered reads for pending plans, using the adapters' snapshot predicates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import (
    STATE_TABLE,
    AdapterError,
    PendingUpstreamRequest,
    PendingUpstreamSummary,
    ReadPredicate,
    WarehouseAdapter,
    WarehouseCapability,
)
from .bigquery import BigQueryAdapter, _bigquery_read_clauses
from .duckdb import DuckDBAdapter, _duckdb_read_clauses


def _read_clauses(
    adapter: WarehouseAdapter, table: str, predicates: Sequence[ReadPredicate]
) -> tuple[list[str], list[Any]]:
    if isinstance(adapter, DuckDBAdapter):
        return _duckdb_read_clauses(adapter, predicates, qualifier="")
    if isinstance(adapter, BigQueryAdapter):
        try:
            metadata = adapter.client.get_table(adapter._table_id(table))
        except Exception:
            raise AdapterError(f"Filtered read relation '{table}' is unavailable") from None
        types = {str(field.name): str(field.field_type).upper() for field in metadata.schema}
        if missing := sorted({predicate.column for predicate in predicates} - types.keys()):
            raise AdapterError("Read filter references missing column(s): " + ", ".join(missing))
        return _bigquery_read_clauses(adapter, predicates, types, qualifier="")
    raise AdapterError("This warehouse adapter does not support filtered pending plans")


def pending_upstream(
    adapter: WarehouseAdapter,
    request: PendingUpstreamRequest,
    *,
    predicates: Sequence[ReadPredicate],
    dry_run: bool,
) -> PendingUpstreamSummary:
    if not predicates:
        return adapter.pending_upstream(request, dry_run=dry_run)
    adapter.require_capability(WarehouseCapability.SQL_QUERIES, operation="pending upstream rows")
    clauses, filter_params = _read_clauses(adapter, request.upstream, predicates)
    new_sql, _, scope_params = adapter._pending_upstream_sql(
        request, state_exists=adapter.table_column_names(STATE_TABLE) is not None
    )
    new_sql += " AND " + " AND ".join(clauses)
    params = [*scope_params, *filter_params]
    estimated = adapter.estimate_query_bytes(new_sql, params)
    if dry_run:
        return PendingUpstreamSummary(0, None, None, None, 0, estimated)
    new_rows, partition_min, partition_max, cluster_distinct = adapter.rows(new_sql, params)[0]
    # State keys outside the filter are indistinguishable from deleted ones.
    return PendingUpstreamSummary(
        new_rows=int(new_rows),
        partition_min=partition_min,
        partition_max=partition_max,
        cluster_distinct=None if cluster_distinct is None else int(cluster_distinct),
        deleted_rows=0,
        estimated_bytes=estimated,
    )


def estimate_table_read_bytes(
    adapter: WarehouseAdapter, table: str, predicates: Sequence[ReadPredicate]
) -> int | None:
    """Price the full-column scan used by pending plans with changed-row counts."""
    clauses, params = _read_clauses(adapter, table, predicates) if predicates else ([], [])
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    return adapter.estimate_query_bytes(f"SELECT * FROM {adapter.table_ref(table)}{where}", params)
