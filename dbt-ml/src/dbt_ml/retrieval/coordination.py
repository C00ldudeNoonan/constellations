"""Warehouse-owned serving coordination for retrieval indexes (issue #152).

The warehouse that owns record-level publication state also owns a small
per-scope serving ledger plus a shared-lease table. Publication acquires an
exclusive claim through an atomic compare-and-set that mints a monotonically
increasing fencing token; queries acquire shared leases that pin the active
physical generation of a ready scope. Every later transition re-verifies the
claim, so a stale publisher cannot advance state, activate readiness, or keep
mutating after administrative recovery reassigned authority.

Coordination deliberately uses only single-statement conditional DML plus
verification reads, so it works on warehouses without multi-statement
transactions (BigQuery) as well as transactional ones (DuckDB). There is no
timeout-based lease stealing: authority moves only through the explicit
`recover()` operation, which the operator may run only after terminating the
old owner.

A warehouse fencing token alone cannot stop a partitioned process from calling
an independent remote store SDK. Store-side fencing is a separate retrieval
capability (`RetrievalStore.publisher_fence`); this module owns readiness and
warehouse-side coordination only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .base import RetrievalError

if TYPE_CHECKING:
    from ..adapters.base import StateScope, WarehouseAdapter

LEDGER_TABLE = "dbt_ml_serving_ledger"
LEASE_TABLE = "dbt_ml_serving_leases"

STATUS_PUBLISHING = "publishing"
STATUS_READY = "ready"
STATUS_FAILED = "failed"
STATUS_UNPUBLISHED = "unpublished"

RECOVERY_ERROR_CODE = "administrative_recovery"

_SAFE_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class ServingCoordinationError(RetrievalError):
    """Artifact-safe serving coordination failure."""


class ServingBusyError(ServingCoordinationError):
    """Another session holds a conflicting claim or lease."""


class ServingNotReadyError(ServingCoordinationError):
    """The scope has no ready, query-safe publication."""


class StaleServingLeaseError(ServingCoordinationError):
    """The caller's claim or pin no longer matches the current ledger."""


@dataclass(frozen=True)
class ServingLedgerEntry:
    """Safe, artifact-visible readiness metadata for one serving scope."""

    status: str
    fencing_token: int
    publication_id: str | None
    expected_code_version: str | None
    config_fingerprint: str | None
    active_generation: str | None
    safe_error_code: str | None
    rows_inserted: int
    rows_updated: int
    rows_skipped: int
    rows_deleted: int
    query_leases: int


@dataclass(frozen=True)
class PublishLease:
    """Exclusive fenced publication claim for one serving scope."""

    scope: StateScope
    publication_id: str
    fencing_token: int


@dataclass(frozen=True)
class QueryLease:
    """Shared query lease pinning one ready physical generation."""

    scope: StateScope
    lease_id: str
    fencing_token: int
    pinned_generation: str
    config_fingerprint: str


def validate_safe_error_code(code: str) -> str:
    if not _SAFE_ERROR_CODE.fullmatch(code):
        raise ServingCoordinationError(
            "Serving ledger error codes must be short lowercase identifiers"
        )
    return code


class ServingCoordinator:
    """Fenced publication readiness over the active warehouse adapter.

    All statements address one scope by the same (model_name, stage,
    target_identity) triple the record-state table uses, so readiness lives
    and dies with the scope it describes.
    """

    def __init__(self, adapter: WarehouseAdapter) -> None:
        self._adapter = adapter
        self._ensure_tables()

    # ─── table management ─────────────────────────────────────────────────

    def _ref(self, table: str) -> str:
        adapter = self._adapter
        return f"{adapter.schema_ref}.{adapter.quote_ident(table)}"

    def _ensure_tables(self) -> None:
        # STRING/BIGINT/TIMESTAMP parse on both DuckDB and BigQuery; no
        # primary keys because BigQuery cannot enforce them — uniqueness is
        # maintained by the conditional-claim protocol instead.
        self._adapter.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._ref(LEDGER_TABLE)} (
                model_name STRING NOT NULL,
                stage STRING NOT NULL,
                target_identity STRING NOT NULL,
                row_id STRING NOT NULL,
                fencing_token BIGINT NOT NULL,
                status STRING NOT NULL,
                publication_id STRING,
                expected_code_version STRING,
                config_fingerprint STRING,
                active_generation STRING,
                safe_error_code STRING,
                rows_inserted BIGINT NOT NULL,
                rows_updated BIGINT NOT NULL,
                rows_skipped BIGINT NOT NULL,
                rows_deleted BIGINT NOT NULL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP
            )
            """
        )
        self._adapter.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._ref(LEASE_TABLE)} (
                model_name STRING NOT NULL,
                stage STRING NOT NULL,
                target_identity STRING NOT NULL,
                lease_id STRING NOT NULL,
                fencing_token BIGINT NOT NULL,
                pinned_generation STRING NOT NULL,
                config_fingerprint STRING NOT NULL,
                acquired_at TIMESTAMP NOT NULL
            )
            """
        )

    def _ensure_row(self, scope: StateScope) -> None:
        """Create the scope's ledger row, healing benign creation races.

        The ledger has no enforceable unique constraint on every warehouse
        (BigQuery cannot enforce one), so two sessions racing the very first
        publication of a scope can both insert. Duplicates from that race are
        always unclaimed `unpublished` rows: the claim statement refuses to
        run against a duplicated scope, so no duplicate can ever advance.
        Collapsing to the smallest `row_id` is therefore a safe, deterministic
        election."""
        ledger = self._ref(LEDGER_TABLE)
        self._adapter.execute(
            f"""
            INSERT INTO {ledger} (
                model_name, stage, target_identity, row_id, fencing_token,
                status, rows_inserted, rows_updated, rows_skipped, rows_deleted
            )
            SELECT ?, ?, ?, ?, 0, ?, 0, 0, 0, 0 FROM (SELECT 1) AS seed
            WHERE NOT EXISTS (
                SELECT 1 FROM {ledger}
                WHERE model_name = ? AND stage = ? AND target_identity = ?
            )
            """,
            [
                scope.model_name,
                scope.stage,
                scope.target_identity,
                uuid4().hex,
                STATUS_UNPUBLISHED,
                scope.model_name,
                scope.stage,
                scope.target_identity,
            ],
        )
        self._adapter.execute(
            f"""
            DELETE FROM {ledger}
            WHERE model_name = ? AND stage = ? AND target_identity = ?
              AND status = ? AND publication_id IS NULL
              AND row_id <> (
                  SELECT MIN(row_id) FROM {ledger}
                  WHERE model_name = ? AND stage = ? AND target_identity = ?
              )
            """,
            [
                *self._scope_params(scope),
                STATUS_UNPUBLISHED,
                *self._scope_params(scope),
            ],
        )

    # ─── reads ────────────────────────────────────────────────────────────

    def _scope_params(self, scope: StateScope) -> list[Any]:
        return [scope.model_name, scope.stage, scope.target_identity]

    def _read_row(self, scope: StateScope) -> tuple[Any, ...] | None:
        rows = self._adapter.rows(
            f"""
            SELECT fencing_token, status, publication_id, expected_code_version,
                   config_fingerprint, active_generation, safe_error_code,
                   rows_inserted, rows_updated, rows_skipped, rows_deleted
            FROM {self._ref(LEDGER_TABLE)}
            WHERE model_name = ? AND stage = ? AND target_identity = ?
            """,
            self._scope_params(scope),
        )
        if not rows:
            return None
        if len(rows) > 1:
            raise ServingCoordinationError(
                "Serving ledger holds conflicting rows for one scope; "
                "run `dbt-ml serving recover` after terminating all publishers"
            )
        return rows[0]

    def _lease_count(self, scope: StateScope) -> int:
        rows = self._adapter.rows(
            f"""
            SELECT COUNT(*) FROM {self._ref(LEASE_TABLE)}
            WHERE model_name = ? AND stage = ? AND target_identity = ?
            """,
            self._scope_params(scope),
        )
        return int(rows[0][0]) if rows else 0

    def status(self, scope: StateScope) -> ServingLedgerEntry:
        row = self._read_row(scope)
        if row is None:
            return ServingLedgerEntry(
                status=STATUS_UNPUBLISHED,
                fencing_token=0,
                publication_id=None,
                expected_code_version=None,
                config_fingerprint=None,
                active_generation=None,
                safe_error_code=None,
                rows_inserted=0,
                rows_updated=0,
                rows_skipped=0,
                rows_deleted=0,
                query_leases=0,
            )
        return ServingLedgerEntry(
            status=str(row[1]),
            fencing_token=int(row[0]),
            publication_id=None if row[2] is None else str(row[2]),
            expected_code_version=None if row[3] is None else str(row[3]),
            config_fingerprint=None if row[4] is None else str(row[4]),
            active_generation=None if row[5] is None else str(row[5]),
            safe_error_code=None if row[6] is None else str(row[6]),
            rows_inserted=int(row[7]),
            rows_updated=int(row[8]),
            rows_skipped=int(row[9]),
            rows_deleted=int(row[10]),
            query_leases=self._lease_count(scope),
        )

    # ─── publication claims ───────────────────────────────────────────────

    def acquire_publish(
        self,
        scope: StateScope,
        *,
        expected_code_version: str,
        config_fingerprint: str,
    ) -> PublishLease:
        """Claim exclusive publication authority with a fresh fencing token.

        The claim is a conditional update: it succeeds only when no publisher
        owns the scope and no query lease is active. Publication IDs are
        random, so a successful claim is proven by reading back our own ID.
        """
        self._ensure_row(scope)
        publication_id = uuid4().hex
        ledger = self._ref(LEDGER_TABLE)
        leases = self._ref(LEASE_TABLE)
        self._adapter.execute(
            f"""
            UPDATE {ledger}
            SET publication_id = ?, fencing_token = fencing_token + 1,
                status = ?, expected_code_version = ?, config_fingerprint = ?,
                safe_error_code = NULL, started_at = CURRENT_TIMESTAMP,
                completed_at = NULL
            WHERE model_name = ? AND stage = ? AND target_identity = ?
              AND publication_id IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM {leases}
                  WHERE model_name = ? AND stage = ? AND target_identity = ?
              )
              AND (
                  SELECT COUNT(*) FROM {ledger}
                  WHERE model_name = ? AND stage = ? AND target_identity = ?
              ) = 1
            """,
            [
                publication_id,
                STATUS_PUBLISHING,
                expected_code_version,
                config_fingerprint,
                *self._scope_params(scope),
                *self._scope_params(scope),
                *self._scope_params(scope),
            ],
        )
        row = self._read_row(scope)
        if row is None:
            raise ServingCoordinationError("Serving ledger row disappeared during claim")
        if row[2] != publication_id:
            if row[2] is not None:
                raise ServingBusyError(
                    "Another publisher owns this serving scope; if its process "
                    "is dead, terminate it and run `dbt-ml serving recover`"
                )
            raise ServingBusyError(
                "Active query leases block publication for this serving scope"
            )
        return PublishLease(
            scope=scope,
            publication_id=publication_id,
            fencing_token=int(row[0]),
        )

    def verify_publish(self, lease: PublishLease) -> None:
        """Abort-before-I/O check that the claim is still the current fence."""
        row = self._read_row(lease.scope)
        if (
            row is None
            or row[2] != lease.publication_id
            or int(row[0]) != lease.fencing_token
        ):
            raise StaleServingLeaseError(
                "Serving publication authority was reassigned; aborting before "
                "further store I/O"
            )

    def _finish(
        self,
        lease: PublishLease,
        *,
        status: str,
        active_generation: str | None,
        safe_error_code: str | None,
        counts: tuple[int, int, int, int],
    ) -> None:
        inserted, updated, skipped, deleted = counts
        self._adapter.execute(
            f"""
            UPDATE {self._ref(LEDGER_TABLE)}
            SET status = ?, active_generation = ?, safe_error_code = ?,
                rows_inserted = ?, rows_updated = ?, rows_skipped = ?,
                rows_deleted = ?, publication_id = NULL,
                completed_at = CURRENT_TIMESTAMP
            WHERE model_name = ? AND stage = ? AND target_identity = ?
              AND publication_id = ? AND fencing_token = ?
            """,
            [
                status,
                active_generation,
                safe_error_code,
                inserted,
                updated,
                skipped,
                deleted,
                *self._scope_params(lease.scope),
                lease.publication_id,
                lease.fencing_token,
            ],
        )
        row = self._read_row(lease.scope)
        if (
            row is None
            or int(row[0]) != lease.fencing_token
            or row[1] != status
            or row[2] is not None
        ):
            raise StaleServingLeaseError(
                "Serving publication authority was reassigned before completion"
            )

    def mark_ready(
        self,
        lease: PublishLease,
        *,
        active_generation: str,
        config_fingerprint: str,
        counts: tuple[int, int, int, int],
    ) -> None:
        """Activate a generation, conditional on the fence and expected config."""
        if not active_generation:
            raise ServingCoordinationError("Ready activation requires a physical generation")
        row = self._read_row(lease.scope)
        if row is None or row[2] != lease.publication_id or int(row[0]) != lease.fencing_token:
            raise StaleServingLeaseError(
                "Serving publication authority was reassigned before activation"
            )
        if row[4] != config_fingerprint:
            raise ServingCoordinationError(
                "Ready activation does not match the claimed configuration fingerprint"
            )
        self._finish(
            lease,
            status=STATUS_READY,
            active_generation=active_generation,
            safe_error_code=None,
            counts=counts,
        )

    def mark_failed(
        self,
        lease: PublishLease,
        *,
        safe_error_code: str,
        counts: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        """Record a failed publication; the scope becomes unavailable to queries.

        An in-place incremental publish that failed midway may have mutated
        the previously ready generation, so failure always clears the active
        generation rather than reverting to it.
        """
        self._finish(
            lease,
            status=STATUS_FAILED,
            active_generation=None,
            safe_error_code=validate_safe_error_code(safe_error_code),
            counts=counts,
        )

    # ─── query leases ─────────────────────────────────────────────────────

    def acquire_query(self, scope: StateScope) -> QueryLease:
        """Acquire a shared lease pinned to the ready generation.

        The lease insert is conditioned on a publisher-free ready ledger at
        the observed fence; a post-insert verification read closes the race
        with a publisher claiming in between (the publisher's claim is itself
        conditioned on no lease rows existing).
        """
        row = self._read_row(scope)
        if row is None or row[1] == STATUS_UNPUBLISHED:
            raise ServingNotReadyError(
                "This search index has not been published; run `dbt-ml run`"
            )
        if row[2] is not None:
            raise ServingBusyError(
                "A publisher is reconciling this search index; retry after it completes"
            )
        if row[1] != STATUS_READY or row[5] is None:
            raise ServingNotReadyError(
                "This search index has no ready publication; re-run `dbt-ml run` "
                "and resolve any recorded failure first"
            )
        fencing_token = int(row[0])
        pinned_generation = str(row[5])
        config_fingerprint = "" if row[4] is None else str(row[4])
        lease_id = uuid4().hex
        ledger = self._ref(LEDGER_TABLE)
        leases = self._ref(LEASE_TABLE)
        self._adapter.execute(
            f"""
            INSERT INTO {leases} (
                model_name, stage, target_identity, lease_id, fencing_token,
                pinned_generation, config_fingerprint, acquired_at
            )
            SELECT ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP FROM (SELECT 1) AS seed
            WHERE EXISTS (
                SELECT 1 FROM {ledger}
                WHERE model_name = ? AND stage = ? AND target_identity = ?
                  AND status = ? AND publication_id IS NULL
                  AND fencing_token = ? AND active_generation = ?
            )
            """,
            [
                *self._scope_params(scope),
                lease_id,
                fencing_token,
                pinned_generation,
                config_fingerprint,
                *self._scope_params(scope),
                STATUS_READY,
                fencing_token,
                pinned_generation,
            ],
        )
        lease = QueryLease(
            scope=scope,
            lease_id=lease_id,
            fencing_token=fencing_token,
            pinned_generation=pinned_generation,
            config_fingerprint=config_fingerprint,
        )
        held = self._adapter.rows(
            f"SELECT 1 FROM {leases} WHERE lease_id = ?",
            [lease_id],
        )
        if not held:
            raise ServingBusyError(
                "A publisher claimed this search index during query admission; retry"
            )
        try:
            self.validate_query(lease)
        except ServingCoordinationError:
            self.release_query(lease)
            raise
        return lease

    def validate_query(self, lease: QueryLease) -> None:
        """Re-verify the pinned generation is still the ready, fenced one."""
        row = self._read_row(lease.scope)
        if (
            row is None
            or int(row[0]) != lease.fencing_token
            or row[1] != STATUS_READY
            or row[2] is not None
            or row[5] != lease.pinned_generation
        ):
            raise StaleServingLeaseError(
                "The pinned search-index generation is no longer active; retry the query"
            )

    def release_query(self, lease: QueryLease) -> None:
        self._adapter.execute(
            f"DELETE FROM {self._ref(LEASE_TABLE)} WHERE lease_id = ?",
            [lease.lease_id],
        )

    # ─── administrative recovery ──────────────────────────────────────────

    def recover(self, scope: StateScope, *, owner_terminated: bool) -> ServingLedgerEntry:
        """Explicitly reassign authority after every old owner is terminated.

        There is no timeout-based stealing: the operator asserts termination,
        the fence advances so any surviving zombie fails its next
        verification, and the scope is left failed until a fresh publication
        succeeds.
        """
        if not owner_terminated:
            raise ServingCoordinationError(
                "Serving recovery requires terminating the previous owner first; "
                "re-run with the owner-terminated confirmation"
            )
        ledger = self._ref(LEDGER_TABLE)
        self._adapter.execute(
            f"""
            DELETE FROM {self._ref(LEASE_TABLE)}
            WHERE model_name = ? AND stage = ? AND target_identity = ?
            """,
            self._scope_params(scope),
        )
        # Rebuild the scope as exactly one row above every observed fence, so
        # recovery also repairs a ledger corrupted by duplicate creation races.
        rows = self._adapter.rows(
            f"""
            SELECT COALESCE(MAX(fencing_token), 0) FROM {ledger}
            WHERE model_name = ? AND stage = ? AND target_identity = ?
            """,
            self._scope_params(scope),
        )
        next_fence = int(rows[0][0]) + 1 if rows else 1
        self._adapter.execute(
            f"""
            DELETE FROM {ledger}
            WHERE model_name = ? AND stage = ? AND target_identity = ?
            """,
            self._scope_params(scope),
        )
        self._adapter.execute(
            f"""
            INSERT INTO {ledger} (
                model_name, stage, target_identity, row_id, fencing_token,
                status, safe_error_code, rows_inserted, rows_updated,
                rows_skipped, rows_deleted, completed_at
            )
            SELECT ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, CURRENT_TIMESTAMP
            FROM (SELECT 1) AS seed
            """,
            [
                *self._scope_params(scope),
                uuid4().hex,
                next_fence,
                STATUS_FAILED,
                RECOVERY_ERROR_CODE,
            ],
        )
        return self.status(scope)
