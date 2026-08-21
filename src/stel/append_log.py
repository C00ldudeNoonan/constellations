"""Append-only warehouse logs (issues #306, #329).

Two histories share one mechanism: the per-run inference usage log and the MCP
query log. Both add rows and never rewrite them, both create their relation on
first write, and both are off until an operator turns them on — so the module
that writes them is one place, not two.

Three rules hold for everything written here, and they are why this is a
narrow module rather than a convenience wrapper around `append_rows`:

**Never fail the thing being logged.** A log is observability, not the work. A
warehouse that rejects the write, a permission the operator forgot, a relation
someone renamed — none of that may turn a successful run into a failed one, or
a served MCP answer into an error. Writes are best-effort and failures are
logged at warning level, once, with the exception class rather than its text.

**Resolved identity and aggregates only.** No prompt text, no document text,
no credential values, and no credential *environment-variable names* — the
same rule artifacts follow. The one exception is the MCP query log's
`query_text`, which is user-authored, off by default, and gated behind its own
`capture_query_text` opt-in separate from `enabled`.

**Fingerprints answer most questions.** A query fingerprint tells you which
questions repeat and which return nothing without storing what anyone typed,
so it is always written and the raw text never has to be.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import polars as pl

from .config.profile import AppendLogConfig
from .hashing import canonical_fingerprint

log = logging.getLogger(__name__)

# Fingerprint domain for an MCP query string. Pinned in
# tests/test_frozen_names.py: this value is written into a persisted log, and
# a drift silently splits one question's history into two.
QUERY_FINGERPRINT_DOMAIN = "mcp-query"


# Explicit column types for each log, because the first batch must not decide
# the persisted schema (Codex review, #333). A first invocation with no LLM
# model leaves `provider` null, and polars infers `Null` — which DuckDB
# materializes as an integer column that every later row containing an actual
# provider name then fails to convert into. Combined with the best-effort
# catch, that failure is silent and the history sticks at its first row
# forever. Declaring the types up front is the fix; changing one is a schema
# change to an existing table, so treat these tuples as a contract.
RUN_LOG_SCHEMA: dict[str, Any] = {
    "invocation_id": pl.String,
    "profile_target": pl.String,
    "model_name": pl.String,
    "kind": pl.String,
    "status": pl.String,
    "provider": pl.String,
    "provider_model": pl.String,
    "provider_implementation": pl.String,
    "backend": pl.String,
    "rows_processed": pl.Int64,
    "rows_skipped": pl.Int64,
    "rows_written": pl.Int64,
    "api_calls": pl.Int64,
    "cache_hits": pl.Int64,
    "input_tokens": pl.Int64,
    "output_tokens": pl.Int64,
    "estimated_cost_usd": pl.Float64,
    "duration_seconds": pl.Float64,
    "started_at": pl.String,
    "completed_at": pl.String,
}

QUERY_LOG_SCHEMA: dict[str, Any] = {
    "logged_at": pl.String,
    "principal_id": pl.String,
    "tenant_id": pl.String,
    "model_name": pl.String,
    "mode": pl.String,
    "query_fingerprint": pl.String,
    "query_text": pl.String,
    "requested_limit": pl.Int64,
    "result_count": pl.Int64,
    "zero_results": pl.Boolean,
    "returned_chunk_ids": pl.List(pl.String),
    "top_score": pl.Float64,
    "elapsed_ms": pl.Float64,
}


class SupportsAppend(Protocol):
    """The one adapter capability a log writer needs.

    Narrower than `WarehouseAdapter` on purpose: this module appends rows and
    does nothing else with a warehouse, and the narrow type is what lets a
    test substitute a two-line double for the whole adapter.
    """

    def append_rows(self, table: str, df: pl.DataFrame) -> int: ...


def query_fingerprint(query: str) -> str:
    """Stable identity for a query string, without storing the string.

    Normalized on whitespace and case so the same question asked twice reads
    as one — the point is grouping repeats and spotting zero-result questions,
    not distinguishing typography.
    """
    normalized = " ".join(query.split()).casefold()
    return canonical_fingerprint({"query": normalized}, domain=QUERY_FINGERPRINT_DOMAIN)


def write_rows(
    adapter: SupportsAppend,
    config: AppendLogConfig | None,
    rows: list[dict[str, Any]],
    *,
    schema: dict[str, Any],
    what: str,
) -> int:
    """Append `rows` to the configured log relation. Best-effort by contract.

    Returns the number of rows written, or 0 when the log is disabled, there
    is nothing to write, or the write failed. `what` names the log in the
    warning so an operator can tell which one is misconfigured.
    """
    if config is None or not config.enabled or not rows:
        return 0
    try:
        return adapter.append_rows(
            config.relation, pl.DataFrame(rows, schema=schema)
        )
    except Exception as error:
        # Deliberately broad, and deliberately not re-raised: see the module
        # docstring. The class name locates the failure without echoing
        # warehouse text into logs.
        log.warning(
            "Could not write %s to '%s' [%s]; the run is unaffected",
            what,
            config.relation,
            type(error).__name__,
        )
        return 0


# ─── run log (issue #306) ───────────────────────────────────────────────────

# Usage keys the backends already meter. Read rather than re-measured: this is
# a durable sink for numbers stel computes anyway, not a second meter.
_USAGE_KEYS = ("api_calls", "cache_hits", "input_tokens", "output_tokens")


def _usage(metrics: Mapping[str, Any], key: str) -> int:
    value = metrics.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(int(value), 0)


def run_log_rows(
    results: Sequence[Any],
    *,
    invocation_id: str,
    started_at: str,
    completed_at: str,
    profile_target: str,
) -> list[dict[str, Any]]:
    """One row per model per invocation (issue #306).

    Identity and aggregates only: which model ran, under which resolved
    provider, how much it processed and what that cost. No prompt text, no
    document text, no credential names — the artifact rules, applied to a
    table that outlives the run.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        metrics = result.metrics or {}
        # `estimated_cost_usd` is what extraction publishes once the profile
        # sets `pricing:` — it already folds in provider-reported spend where
        # the provider gives it. `reported_cost_usd` is the per-call provider
        # figure and only stands in when no estimate was computed. Reading
        # only the latter left the column null on the normal path, which is
        # precisely the cross-run spend query this log exists for (Codex
        # review, #333).
        cost = metrics.get("estimated_cost_usd")
        if cost is None:
            cost = metrics.get("reported_cost_usd")
        rows.append(
            {
                "invocation_id": invocation_id,
                "profile_target": profile_target,
                "model_name": result.model_name,
                "kind": result.kind,
                "status": result.status
                or ("error" if result.errors else "success"),
                "provider": result.provider,
                "provider_model": result.provider_model,
                "provider_implementation": result.provider_implementation,
                "backend": result.backend,
                "rows_processed": result.documents_processed,
                "rows_skipped": result.documents_skipped,
                "rows_written": result.rows_written,
                **{key: _usage(metrics, key) for key in _USAGE_KEYS},
                "estimated_cost_usd": (
                    float(cost)
                    if isinstance(cost, int | float) and not isinstance(cost, bool)
                    else None
                ),
                "duration_seconds": result.duration_seconds,
                "started_at": started_at,
                "completed_at": completed_at,
            }
        )
    return rows
