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

import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ..adapters.base import SERVING_LEASE_TABLE, SERVING_LEDGER_TABLE
from .base import RetrievalError
from .publisher_identity import Liveness, PublisherIdentity, describe_publisher, local_liveness

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..adapters.base import StateScope, WarehouseAdapter

# Fenced state replacement (WarehouseAdapter.replace_state_scope) verifies
# claims against this same ledger, so the name is owned by adapters.base.
LEDGER_TABLE = SERVING_LEDGER_TABLE
# The lease table's name moved to adapters.base with #313, for the same reason
# as the ledger's: `stel migrate` plans the rename of all three persisted
# tables together, and it cannot import retrieval code to learn this one.
# Frozen for the same reason as the names there: renaming it strands the live
# leases and every publisher loses sight of who holds what.
LEASE_TABLE = SERVING_LEASE_TABLE

STATUS_PUBLISHING = "publishing"
STATUS_READY = "ready"
STATUS_FAILED = "failed"
STATUS_UNPUBLISHED = "unpublished"
# A publication failed, but the generation that was live before it is intact
# and still named, so queries keep being served from it (issue #449). The
# distinction from `failed` is what the scope can still do, not how bad the
# failure was: `degraded` always carries its `safe_error_code` too, so a
# pipeline that has been broken for days is visible in `stel serving status`
# rather than hidden behind a working endpoint.
STATUS_DEGRADED = "degraded"
# An in-place publisher holds the scope: it is writing into the collection the
# activation pointer names, so no reader may be admitted until it finishes
# (ADR-0003). Distinct from `publishing`, which is a private-generation build
# that readers of the live generation may run alongside. The pointer itself
# survives an in-place claim when the store promises an interrupted write
# leaves the collection sound (`INTERRUPTION_SAFE_MUTATION`, issue #617), so
# recovery after a crash can serve on; recording the exclusion in the status
# rather than by clearing the pointer is what lets both facts fit one row.
STATUS_PUBLISHING_IN_PLACE = "publishing_in_place"
# Each status still requires an active generation. `publishing` qualifies
# because it is the private-build claim; an in-place claim is never servable.
SERVABLE_STATUSES = (STATUS_READY, STATUS_DEGRADED, STATUS_PUBLISHING)

RECOVERY_ERROR_CODE = "administrative_recovery"

_SAFE_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

log = logging.getLogger(__name__)

# Ledger columns added after the table first shipped, with the type each was
# added as. `_ensure_tables` is CREATE TABLE IF NOT EXISTS, which leaves an
# existing table alone, so every one of these is also an ALTER on a ledger
# written by an earlier version (issues #355, #621).
_LEDGER_COLUMNS_ADDED_LATER: tuple[tuple[str, str], ...] = (
    ("active_collection", "STRING"),
    # Who holds the publish claim, and when they were last heard from (issue
    # #621). Epoch seconds rather than TIMESTAMP, so an age computed from
    # them does not depend on a warehouse session's time zone.
    ("publisher_host", "STRING"),
    ("publisher_pid", "BIGINT"),
    ("publisher_started_epoch", "BIGINT"),
    ("publisher_label", "STRING"),
    ("publisher_heartbeat_epoch", "BIGINT"),
    ("publisher_namespace", "STRING"),
    # What a long publication phase has done so far (issue #635).
    ("progress_note", "STRING"),
)


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
    # The physical collection the logical name currently resolves to (issue
    # #355). None means the unsuffixed default, which is what every row
    # written before generations existed means.
    active_collection: str | None
    safe_error_code: str | None
    # A free-text note a long publication phase writes so another
    # process can see it is alive and how far along (issue #635).
    # Diagnostic only: nothing reads it back to make a decision, and
    # the completion write clears it, so a stale note cannot outlive
    # the publication that wrote it.
    progress_note: str | None
    rows_inserted: int
    rows_updated: int
    rows_skipped: int
    rows_deleted: int
    query_leases: int
    # The process holding `publication_id`, when one does (issue #621); None
    # otherwise. Not part of any artifact: `serving_resource` does not carry
    # it, and `stel serving status` is where it is read.
    publisher: PublisherIdentity | None
    publisher_heartbeat_epoch: int | None


@dataclass(frozen=True)
class PublishLease:
    """Exclusive fenced publication claim for one serving scope."""

    scope: StateScope
    publication_id: str
    fencing_token: int
    config_fingerprint: str


@dataclass(frozen=True)
class QueryLease:
    """Shared query lease pinning one ready physical generation."""

    scope: StateScope
    lease_id: str
    fencing_token: int
    pinned_generation: str
    # Resolved at acquire time so a reader keeps querying the collection its
    # lease pinned even if activation moves the pointer underneath it.
    pinned_collection: str | None
    config_fingerprint: str
    # `STATUS_READY`, `STATUS_DEGRADED` or `STATUS_PUBLISHING` -- the only
    # statuses `acquire_query` admits. A caller that only reads
    # `pinned_generation` cannot tell a normal answer from one served because
    # the next publish failed (issue #617, ask 3): `stel serving status`
    # already says this in words, and a programmatic reader deserves the same
    # signal to decide whether to degrade instead of trusting a silent pin.
    status: str
    # Set only when `status == STATUS_DEGRADED`, mirroring the same ledger
    # column `stel serving status` reads to explain *why* -- a consumer that
    # wants to decide how to degrade needs more than "stale".
    safe_error_code: str | None


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

    def __init__(self, adapter: WarehouseAdapter, *, ensure_schema: bool) -> None:
        """`ensure_schema` False skips creating the ledger and lease tables.

        Required rather than defaulted (issue #535): the query path builds a
        coordinator per request, and each construction was two `CREATE TABLE
        IF NOT EXISTS` jobs plus a column probe -- ~1.7s per query on
        BigQuery, spent on tables the previous query had just read. Whether a
        caller is the first to touch this schema is something only the caller
        knows, and a default would quietly pick one answer for all five call
        sites.

        Pass True whenever the schema may not exist yet: every publish and
        administrative path, and the first query of a serving session. A
        served query that skips it and finds no ledger still reports "has not
        been published" from the read below, because a scope with no ledger
        row and a scope with no ledger table are the same answer to a reader.
        """
        self._adapter = adapter
        if ensure_schema:
            self._ensure_tables()
            self._ensure_ledger_columns()

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
                active_collection STRING,
                safe_error_code STRING,
                progress_note STRING,
                rows_inserted BIGINT NOT NULL,
                rows_updated BIGINT NOT NULL,
                rows_skipped BIGINT NOT NULL,
                rows_deleted BIGINT NOT NULL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                publisher_host STRING,
                publisher_pid BIGINT,
                publisher_started_epoch BIGINT,
                publisher_label STRING,
                publisher_heartbeat_epoch BIGINT,
                publisher_namespace STRING
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

    def _ensure_ledger_columns(self) -> None:
        """Add every column a ledger created by an earlier version lacks.

        `_ensure_tables` uses CREATE TABLE IF NOT EXISTS, which does nothing
        to an existing table, so a ledger written by an earlier version would
        otherwise fail every statement naming a newer column.

        `active_collection` predates issue #355, the publisher columns #621,
        `progress_note` #635. All nullable, so widening changes the meaning
        of nothing already recorded.
        """
        columns = self._adapter.table_column_names(LEDGER_TABLE)
        if columns is None:
            return
        for name, column_type in _LEDGER_COLUMNS_ADDED_LATER:
            if name in columns:
                continue
            self._adapter.execute(
                f"ALTER TABLE {self._ref(LEDGER_TABLE)} ADD COLUMN {name} {column_type}"
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

    # ─── scope re-keying (issue #355) ─────────────────────────────────────

    def rekey_scope(self, old: StateScope, new: StateScope) -> int:
        """Move the ledger row for `old` onto `new`, returning rows moved.

        Companion to `WarehouseAdapter.rekey_state_scope` for the two tables
        this class owns. Refused while any query lease is outstanding on
        either scope: a lease pins a generation under an identity that is
        about to stop existing, and rewriting it would leave the holder
        validating against a row it can no longer find. Release the leases
        (or `serving recover`) first.
        """
        if old.model_name != new.model_name or old.stage != new.stage:
            raise ServingCoordinationError(
                "Serving scope re-keying may only change target_identity"
            )
        if old.target_identity == new.target_identity:
            return 0
        if self._lease_count(old) or self._lease_count(new):
            raise ServingBusyError(
                "Refusing to re-key a serving scope with outstanding query "
                "leases; retry once readers have finished or run "
                "`stel serving recover`"
            )
        ledger = self._ref(LEDGER_TABLE)

        def _claimed_count(scope: StateScope) -> int:
            # `_ensure_row` plants an unclaimed `unpublished` placeholder the
            # first time any read touches a scope — merely having called
            # `status()` on the destination is not a collision. Only a row
            # that has actually been claimed counts as one.
            found = self._adapter.rows(
                f"""
                SELECT COUNT(*) FROM {ledger}
                WHERE model_name = ? AND stage = ? AND target_identity = ?
                  AND NOT (status = ? AND publication_id IS NULL)
                """,
                [*self._scope_params(scope), STATUS_UNPUBLISHED],
            )
            return int(found[0][0]) if found else 0

        # Source first: once the row has moved, the destination is occupied
        # *because* the migration succeeded, so checking it first would make
        # the second run of an idempotent command fail.
        count = _claimed_count(old)
        if count == 0:
            return 0
        if _claimed_count(new) > 0:
            raise ServingCoordinationError(
                "Refusing to re-key: the destination serving scope already has "
                "a published ledger row"
            )
        # Clear any placeholder at the destination so the moved row does not
        # land beside a duplicate for the same scope triple.
        self._adapter.execute(
            f"""
            DELETE FROM {ledger}
            WHERE model_name = ? AND stage = ? AND target_identity = ?
              AND status = ? AND publication_id IS NULL
            """,
            [*self._scope_params(new), STATUS_UNPUBLISHED],
        )
        self._adapter.execute(
            f"""
            UPDATE {ledger} SET target_identity = ?
            WHERE model_name = ? AND stage = ? AND target_identity = ?
            """,
            [new.target_identity, *self._scope_params(old)],
        )
        return count

    # ─── reads ────────────────────────────────────────────────────────────

    def _scope_params(self, scope: StateScope) -> list[Any]:
        return [scope.model_name, scope.stage, scope.target_identity]

    def _read_row(self, scope: StateScope) -> tuple[Any, ...] | None:
        rows = self._adapter.rows(
            f"""
            SELECT fencing_token, status, publication_id, expected_code_version,
                   config_fingerprint, active_generation, safe_error_code,
                   rows_inserted, rows_updated, rows_skipped, rows_deleted,
                   active_collection, publisher_host, publisher_pid,
                   publisher_started_epoch, publisher_label,
                   publisher_heartbeat_epoch, publisher_namespace, progress_note
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
                "run `stel serving recover` after terminating all publishers"
            )
        return rows[0]

    def _read_row_and_lease(
        self, lease: QueryLease
    ) -> tuple[tuple[Any, ...] | None, bool]:
        """The scope's ledger row plus whether this exact pin is still held.

        One statement rather than two (issue #535). The correlated subquery
        counts the caller's own lease row under the full identity the pin was
        issued with, so a lease that was released, fenced out or re-issued
        under a different generation reads as not held -- the same predicate
        the separate query used.
        """
        rows = self._adapter.rows(
            f"""
            SELECT fencing_token, status, publication_id, expected_code_version,
                   config_fingerprint, active_generation, safe_error_code,
                   rows_inserted, rows_updated, rows_skipped, rows_deleted,
                   active_collection, publisher_host, publisher_pid,
                   publisher_started_epoch, publisher_label,
                   publisher_heartbeat_epoch, publisher_namespace, progress_note,
                   (
                       SELECT COUNT(*) FROM {self._ref(LEASE_TABLE)}
                       WHERE model_name = ? AND stage = ? AND target_identity = ?
                         AND lease_id = ? AND fencing_token = ?
                         AND pinned_generation = ? AND config_fingerprint = ?
                   )
            FROM {self._ref(LEDGER_TABLE)}
            WHERE model_name = ? AND stage = ? AND target_identity = ?
            """,
            [
                *self._scope_params(lease.scope),
                lease.lease_id,
                lease.fencing_token,
                lease.pinned_generation,
                lease.config_fingerprint,
                *self._scope_params(lease.scope),
            ],
        )
        if not rows:
            return None, False
        if len(rows) > 1:
            raise ServingCoordinationError(
                "Serving ledger holds conflicting rows for one scope; "
                "run `stel serving recover` after terminating all publishers"
            )
        row = rows[0]
        return row[:-1], bool(row[-1])

    def _lease_count(self, scope: StateScope) -> int:
        rows = self._adapter.rows(
            f"""
            SELECT COUNT(*) FROM {self._ref(LEASE_TABLE)}
            WHERE model_name = ? AND stage = ? AND target_identity = ?
            """,
            self._scope_params(scope),
        )
        return int(rows[0][0]) if rows else 0

    def scope_exists(self, scope: StateScope) -> bool:
        """Whether this warehouse holds a ledger row for the scope at all.

        `status()` synthesizes an `unpublished` entry for a scope it has never
        seen, which reads as a settled fact about the index rather than as
        "this warehouse has no record of it". Those are different answers, and
        confusing them is what let a recovery run against the wrong target and
        report success (issue #511).
        """
        return self._read_row(scope) is not None

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
                active_collection=None,
                safe_error_code=None,
                progress_note=None,
                rows_inserted=0,
                rows_updated=0,
                rows_skipped=0,
                rows_deleted=0,
                query_leases=0,
                publisher=None,
                publisher_heartbeat_epoch=None,
            )
        return ServingLedgerEntry(
            status=str(row[1]),
            fencing_token=int(row[0]),
            publication_id=None if row[2] is None else str(row[2]),
            expected_code_version=None if row[3] is None else str(row[3]),
            config_fingerprint=None if row[4] is None else str(row[4]),
            active_generation=None if row[5] is None else str(row[5]),
            active_collection=None if row[11] is None else str(row[11]),
            safe_error_code=None if row[6] is None else str(row[6]),
            progress_note=None if row[18] is None else str(row[18]),
            rows_inserted=int(row[7]),
            rows_updated=int(row[8]),
            rows_skipped=int(row[9]),
            rows_deleted=int(row[10]),
            query_leases=self._lease_count(scope),
            publisher=_publisher_from_row(row),
            publisher_heartbeat_epoch=None if row[16] is None else int(row[16]),
        )

    # ─── publication claims ───────────────────────────────────────────────

    def acquire_publish(
        self,
        scope: StateScope,
        *,
        expected_code_version: str,
        config_fingerprint: str,
        preserves_active_generation: bool = False,
        excludes_readers: bool | None = None,
        expected_fencing_token: int | None = None,
        publisher: PublisherIdentity | None = None,
    ) -> PublishLease:
        """Claim exclusive publication authority with a fresh fencing token.

        The claim is a conditional update: it succeeds only when no publisher
        owns the scope. In-place writes also require no query leases. IDs are
        random, so a successful claim is proven by reading back our own ID.

        Two facts about the publish are recorded on the row, because a
        publisher that dies leaves no other record of its intent and
        `recover()` cannot ask it.

        `preserves_active_generation` says whether the generation serving
        queries before this claim is still sound if this publisher stops at
        any point. True for a private generation build, which writes where
        nothing is reading, and for an in-place publish on a store that
        promises an interrupted write leaves its collection whole
        (`INTERRUPTION_SAFE_MUTATION`, issue #617). False clears
        `active_generation` immediately, so a crash recovers to `failed` and
        serves nothing rather than serving a collection the dead publisher
        may have half-rewritten (issue #449 review).

        `excludes_readers` says whether readers are refused for the claim's
        duration, and the claim refused while any are pinned: true for every
        in-place publish, which mutates the collection readers would resolve
        to (ADR-0003); false for a private build, whose readers keep the
        untouched generation, including leases acquired before this claim. It
        defaults to the complement of `preserves_active_generation`, which is
        the coupling every caller had before #617; an in-place publish on an
        interruption-safe store is the one combination that has to say both.
        A private build's configuration stays in the ledger until activation;
        the lease carries the pending configuration.

        The default is the fail-closed one. A caller that says nothing gets
        in-place semantics on an untrusted store, so forgetting these
        arguments gives up availability rather than serving something corrupt.

        `publisher` is who is making the claim, recorded on the row so that a
        publisher that dies can still be identified (issue #621). It defaults
        to the calling process, which is right for every caller but a test
        standing in for one; the first heartbeat is the claim itself.
        """
        if excludes_readers is None:
            excludes_readers = not preserves_active_generation
        if publisher is None:
            publisher = PublisherIdentity.current()
        self._ensure_row(scope)
        publication_id = uuid4().hex
        ledger = self._ref(LEDGER_TABLE)
        leases = self._ref(LEASE_TABLE)
        retain_generation = (
            "" if preserves_active_generation else ", active_generation = NULL"
        )
        query_guard = (
            f"""
                AND NOT EXISTS (
                    SELECT 1 FROM {leases}
                    WHERE model_name = ? AND stage = ? AND target_identity = ?
                )"""
            if excludes_readers else ""
        )
        query_params = self._scope_params(scope) if excludes_readers else []
        planning_guard = "" if expected_fencing_token is None else "AND fencing_token = ?"
        planning_params = [] if expected_fencing_token is None else [expected_fencing_token]
        fingerprint_assignment = (
            "CASE WHEN active_generation IS NOT NULL THEN config_fingerprint ELSE ? END"
            if preserves_active_generation else "?"
        )
        self._adapter.execute(
            f"""
            UPDATE {ledger}
            SET publication_id = ?, fencing_token = fencing_token + 1,
                status = ?, expected_code_version = ?,
                config_fingerprint = {fingerprint_assignment},
                safe_error_code = NULL, started_at = CURRENT_TIMESTAMP,
                completed_at = NULL{retain_generation},
                publisher_host = ?, publisher_pid = ?,
                publisher_started_epoch = ?, publisher_label = ?,
                publisher_heartbeat_epoch = ?, publisher_namespace = ?
            WHERE model_name = ? AND stage = ? AND target_identity = ?
              AND publication_id IS NULL
              {planning_guard}
              {query_guard}
              AND (
                  SELECT COUNT(*) FROM {ledger}
                  WHERE model_name = ? AND stage = ? AND target_identity = ?
              ) = 1
            """,
            [
                publication_id,
                STATUS_PUBLISHING_IN_PLACE if excludes_readers else STATUS_PUBLISHING,
                expected_code_version,
                config_fingerprint,
                publisher.host,
                publisher.pid,
                publisher.started_epoch,
                publisher.label,
                int(time.time()),
                publisher.namespace,
                *self._scope_params(scope),
                *planning_params,
                *query_params,
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
                    "is dead, terminate it and run `stel serving recover`"
                )
            if expected_fencing_token is not None and int(row[0]) != expected_fencing_token:
                raise ServingBusyError(
                    "Serving generation changed while planning publication; retry the run"
                )
            raise ServingBusyError(
                "Active query leases block publication for this serving scope"
            )
        return PublishLease(
            scope=scope,
            publication_id=publication_id,
            fencing_token=int(row[0]),
            config_fingerprint=config_fingerprint,
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

    def heartbeat(self, lease: PublishLease) -> None:
        """Record that this publisher is still working, fenced like every write.

        The page loop calls it once per page (issue #621), so `stel serving
        status` can show how long since the holder was heard from, and an
        operator can tell a publisher dead for forty minutes from one mid-way
        through a page. It is a conditional update on the claim, so a fenced-out
        publisher learns here that authority moved, exactly as `verify_publish`
        would tell it. One DML per page: on BigQuery that is about a second
        against a page that takes minutes, and it is what the question costs.

        A silent stretch is not proof of death: the index build after the last
        page is one long call with no page to beat on, and so is a resumed
        build's index drop. `recover` therefore never acts on heartbeat age.
        """
        self._adapter.execute(
            f"""
            UPDATE {self._ref(LEDGER_TABLE)}
            SET publisher_heartbeat_epoch = ?
            WHERE model_name = ? AND stage = ? AND target_identity = ?
              AND publication_id = ? AND fencing_token = ?
            """,
            [
                int(time.time()),
                *self._scope_params(lease.scope),
                lease.publication_id,
                lease.fencing_token,
            ],
        )
        self.verify_publish(lease)

    def _finish(
        self,
        lease: PublishLease,
        *,
        status: str,
        active_generation: str | None,
        safe_error_code: str | None,
        counts: tuple[int, int, int, int],
        active_collection: str | None,
        config_fingerprint: str | None = None,
    ) -> None:
        inserted, updated, skipped, deleted = counts
        # The fingerprint belongs to the generation. Private-build claims
        # leave the old pair intact; activation replaces both together.
        restore_fingerprint = (
            "" if config_fingerprint is None else ", config_fingerprint = ?"
        )
        fingerprint_param = [] if config_fingerprint is None else [config_fingerprint]
        self._adapter.execute(
            f"""
            UPDATE {self._ref(LEDGER_TABLE)}
            SET status = ?, active_generation = ?, active_collection = ?,
                safe_error_code = ?,
                rows_inserted = ?, rows_updated = ?, rows_skipped = ?,
                rows_deleted = ?, publication_id = NULL,
                progress_note = NULL,
                completed_at = CURRENT_TIMESTAMP{restore_fingerprint},
                publisher_host = NULL, publisher_pid = NULL,
                publisher_started_epoch = NULL, publisher_label = NULL,
                publisher_heartbeat_epoch = NULL, publisher_namespace = NULL
            WHERE model_name = ? AND stage = ? AND target_identity = ?
              AND publication_id = ? AND fencing_token = ?
            """,
            [
                status,
                active_generation,
                active_collection,
                safe_error_code,
                inserted,
                updated,
                skipped,
                deleted,
                *fingerprint_param,
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

    def record_progress(self, lease: PublishLease, note: str) -> None:
        """Publish a progress note for this scope, for another process to read.

        `serving activate`'s re-stamp phase can run for hours, and for all of
        it `serving status` showed the *previous* publish's counts and
        `status: publishing` — so the only way to tell the command was alive
        was `INFORMATION_SCHEMA.JOBS` (issue #635). This is the line that
        answers "is it stuck?" from another terminal.

        Fenced like every other ledger write: a publisher whose authority was
        reassigned must not keep narrating progress over the one that replaced
        it. Unlike the other writes it does **not** raise when the fence has
        moved — a lost fence is discovered by the next `verify_publish`, which
        is the call that is allowed to stop the work, and failing here would
        turn a diagnostic into a new way for a publish to die.

        Best-effort in the same spirit as the append-only logs: the note is
        observability, and a warehouse that rejects it must not fail the
        publication it describes.
        """
        try:
            self._adapter.execute(
                f"""
                UPDATE {self._ref(LEDGER_TABLE)}
                SET progress_note = ?
                WHERE model_name = ? AND stage = ? AND target_identity = ?
                  AND publication_id = ? AND fencing_token = ?
                """,
                [
                    note,
                    *self._scope_params(lease.scope),
                    lease.publication_id,
                    lease.fencing_token,
                ],
            )
        except Exception as error:
            # The class name only, like the rest of this module: warehouse
            # text may carry SQL and must not reach logs or artifacts.
            log.warning(
                "Could not record publication progress [%s]; the publication "
                "is unaffected",
                type(error).__name__,
            )

    def mark_ready(
        self,
        lease: PublishLease,
        *,
        active_generation: str,
        config_fingerprint: str,
        counts: tuple[int, int, int, int],
        active_collection: str | None = None,
    ) -> None:
        """Activate a generation, conditional on the fence and expected config.

        `active_collection` is the physical collection the logical name should
        resolve to from now on (issue #355) — this is the atomic activation
        itself, a fenced warehouse row update. None keeps the pre-generation
        meaning: resolve to the unsuffixed default.
        """
        if not active_generation:
            raise ServingCoordinationError("Ready activation requires a physical generation")
        row = self._read_row(lease.scope)
        if row is None or row[2] != lease.publication_id or int(row[0]) != lease.fencing_token:
            raise StaleServingLeaseError(
                "Serving publication authority was reassigned before activation"
            )
        if lease.config_fingerprint != config_fingerprint:
            raise ServingCoordinationError(
                "Ready activation does not match the claimed configuration fingerprint"
            )
        self._finish(
            lease,
            status=STATUS_READY,
            active_generation=active_generation,
            active_collection=active_collection,
            safe_error_code=None,
            counts=counts,
            config_fingerprint=config_fingerprint,
        )

    def mark_failed(
        self,
        lease: PublishLease,
        *,
        safe_error_code: str,
        counts: tuple[int, int, int, int] = (0, 0, 0, 0),
        active_collection: str | None = None,
        active_generation: str | None = None,
        config_fingerprint: str | None = None,
    ) -> None:
        """Record a failed publication, retaining the previous generation when
        the failure cannot have left it unsound.

        Both pointers carry the same asymmetry, for the same reason (issues
        #355, #449). A private generation build writes where nothing is
        reading, so a failure leaves the previously-active generation
        untouched and still correct. An in-place publish writes into the
        collection the pointer names; on a store that promises an interrupted
        write leaves every row whole and the collection readable
        (`INTERRUPTION_SAFE_MUTATION`, issue #617), the generation is still
        sound, only staler than the publish meant it to be, and the publication
        state already says which rows. Both of those paths pass both existing
        pointers back, and the scope stays servable as `degraded`. On any other
        store an in-place failure may have corrupted what was live: both
        pointers must go, and the scope becomes unavailable to queries — the
        default.

        Clearing them when the generation is sound would drop a healthy
        generation out of resolution and force a full re-embed — the cost #355
        exists to avoid — and, because queries admit only on a named
        generation, would also take a working index offline until the next
        successful publish, which on a large corpus is hours away (#449) and,
        when no publish can currently finish, indefinitely (#617).

        Retaining a generation requires `config_fingerprint` — the one that
        generation was published under. The claim overwrote the ledger's with
        the configuration this failed publish was building for, and query
        admission pins whatever the ledger holds, so retaining the pointers
        without it would advertise the old index under the new configuration.
        """
        if active_generation and config_fingerprint is None:
            raise ServingCoordinationError(
                "Retaining an active generation requires the configuration "
                "fingerprint it was published under"
            )
        self._finish(
            lease,
            status=STATUS_DEGRADED if active_generation else STATUS_FAILED,
            active_generation=active_generation or None,
            active_collection=active_collection,
            safe_error_code=validate_safe_error_code(safe_error_code),
            counts=counts,
            config_fingerprint=config_fingerprint if active_generation else None,
        )

    # ─── query leases ─────────────────────────────────────────────────────

    def _raise_if_rekey_pending(
        self, scope: StateScope, legacy_scope: StateScope
    ) -> None:
        """Name the one-time re-key when the ledger row is only under the old key.

        Costs one extra read, and only on a path that is already failing. The
        alternative — re-keying here, under a query — would have a read
        silently mutate the ledger it is reading, which is the publisher's job
        and needs the publish lease that a query does not hold.
        """
        if legacy_scope.target_identity == scope.target_identity:
            return
        if self._read_row(legacy_scope) is None:
            return
        raise ServingNotReadyError(
            f"Search index '{scope.model_name}' still has its publication state "
            "under the pre-0.13 serving key, where nothing looks for it. Run "
            f"`stel serving migrate-scope {scope.model_name}` to move it — the "
            "rows and embeddings are intact and it does not republish or "
            "re-embed anything."
        )

    def acquire_query(
        self, scope: StateScope, *, legacy_scope: StateScope | None = None
    ) -> QueryLease:
        """Acquire a shared lease pinned to the ready generation.

        The insert is conditioned on the observed active generation and fence.
        Admission rechecks both after insertion, closing a race with cutover
        before any store I/O. Once admitted, the durable pin survives private
        builds and cutover; in-place publishers are excluded until it drains.

        `legacy_scope` only changes what the failure *says* (issue #413). An
        index published before #355 re-keyed the serving scope keeps its ledger
        row under the old key, and a miss on the new key is indistinguishable
        here from never having been published — so the caller passes the old
        key and gets told to re-key instead of to re-embed.
        """
        row = self._read_row(scope)
        if row is None and legacy_scope is not None:
            self._raise_if_rekey_pending(scope, legacy_scope)
        if row is None or row[1] == STATUS_UNPUBLISHED:
            raise ServingNotReadyError(
                "This search index has not been published; run `stel run`"
            )
        # A publisher holds the scope and readers may not join it: an in-place
        # publish by its status, whichever store it runs on (issue #617), or a
        # first publish with no generation to serve yet.
        if row[2] is not None and (row[1] == STATUS_PUBLISHING_IN_PLACE or row[5] is None):
            raise ServingBusyError(
                "A publisher is reconciling this search index; retry after it completes"
            )
        if row[1] not in SERVABLE_STATUSES or row[5] is None:
            raise ServingNotReadyError(
                "This search index has no ready publication; re-run `stel run` "
                "and resolve any recorded failure first"
            )
        fencing_token = int(row[0])
        status = str(row[1])
        pinned_generation = str(row[5])
        pinned_collection = None if row[11] is None else str(row[11])
        config_fingerprint = "" if row[4] is None else str(row[4])
        safe_error_code = (
            str(row[6]) if status == STATUS_DEGRADED and row[6] is not None else None
        )
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
                  AND status IN (?, ?, ?)
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
                *SERVABLE_STATUSES,
                fencing_token,
                pinned_generation,
            ],
        )
        held = self._adapter.rows(
            f"SELECT 1 FROM {leases} WHERE lease_id = ?",
            [lease_id],
        )
        if not held:
            raise ServingBusyError(
                "A publisher claimed this search index during query admission; retry"
            )
        # The insert's own WHERE EXISTS is what actually admits the lease,
        # not the read above it -- so a status this stale could still be
        # true when the fencing token and generation it checked are, if a
        # generation-preserving publish (issue #617) moved from `publishing`
        # to `degraded` in between. Both keep the same fencing token and
        # active generation, so the insert would still succeed with a status
        # this lease has not seen (Codex review, #641). Re-read after
        # confirming admission rather than trust the pre-admission snapshot.
        current = self._read_row(scope)
        if (
            current is not None
            and int(current[0]) == fencing_token
            and current[5] is not None
            and str(current[5]) == pinned_generation
        ):
            status = str(current[1])
            safe_error_code = (
                str(current[6]) if status == STATUS_DEGRADED and current[6] is not None
                else None
            )
        lease = QueryLease(
            scope=scope,
            lease_id=lease_id,
            fencing_token=fencing_token,
            pinned_generation=pinned_generation,
            pinned_collection=pinned_collection,
            config_fingerprint=config_fingerprint,
            status=status,
            safe_error_code=safe_error_code,
        )
        try:
            self.validate_query(lease, require_active=True)
        except ServingCoordinationError:
            self.release_query(lease)
            raise
        return lease

    def validate_query(self, lease: QueryLease, *, require_active: bool = False) -> None:
        """Verify the durable pin, which survives private-generation cutover.

        In-place publishers cannot acquire while any pin exists. Private
        publishers never mutate pinned collections, and retirement waits for
        all pins to drain. Recovery deletes pins, fencing out zombie readers.

        The ledger read and the held-check travel in one statement (issue
        #535). They were two, and on BigQuery a statement is a query job with
        a floor near a second regardless of what it reads -- one job carrying
        both facts measured at 0.85s against 0.81s + 0.81s for the pair. The
        facts are the same and so is the verdict below; only the round trip
        is gone. Reading both at one timestamp is also marginally stronger
        than reading them at two.
        """
        row, held = self._read_row_and_lease(lease)
        if (
            row is None
            or not held
            or int(row[0]) < lease.fencing_token
            or row[1] not in SERVABLE_STATUSES
            or row[5] is None
            or (
                require_active
                and (int(row[0]) != lease.fencing_token or row[5] != lease.pinned_generation)
            )
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
        and the fence advances so any surviving zombie fails its next
        verification. Recovery grants nobody publication authority — a new
        publisher still has to claim the scope in the ordinary way.

        The scope is left `degraded` rather than `failed` when a previously
        active generation survives, so queries keep being served from it while
        the recorded failure stays visible (issue #449). Only a scope with no
        servable generation is left `failed`: one whose claim cleared the
        pointer because the store could not promise an interrupted in-place
        write leaves the collection sound, or one that never had a generation.
        A crashed in-place publish on an interruption-safe store kept its
        pointer at claim time and serves on from here (issue #617).

        Without `owner_terminated`, recovery proceeds in exactly one case: the
        row names a publisher in *this* PID namespace on *this* kernel boot
        whose PID and process start time no longer match any running process
        (issue #621). That is provable
        from here, and it is the common case -- an orchestrator-launched build
        on the same host as the operator. Every other case is refused, and the
        refusal names what the ledger knows about the owner, so the operator's
        confirmation is informed rather than a guess. Heartbeat age is shown
        but never acted on: a publisher inside an index build has no page to
        beat on and is not dead.
        """
        if not owner_terminated:
            self._refuse_unless_owner_provably_dead(scope)
        ledger = self._ref(LEDGER_TABLE)
        # Recovery rebuilds the row, so the activation pointers have to be
        # read before it is deleted and carried across (issues #355, #449).
        # Losing them strands a generation-served index: `active_collection`
        # is the only record of which physical collection is live, and
        # `active_generation` is what a query admits on, so dropping either
        # takes a healthy index offline until the next successful publish —
        # hours, on a large corpus. `config_fingerprint` comes along because
        # query admission pins it into the lease.
        #
        # All three come from one row rather than independently: pairing a
        # generation with another row's collection would name a pointer pair
        # that was never live together.
        #
        # That row is specifically the highest fence — the one that got
        # furthest — and its generation is taken as-is rather than searched
        # for. Falling back to an older row's generation would resurrect one
        # that the in-place publish recorded on the newest row has since
        # rewritten, which is exactly what the claim cleared it to prevent.
        #
        # Read tolerantly: recovery also repairs a ledger corrupted by
        # duplicate creation races, and `_read_row` refuses exactly that case.
        pointer = self._adapter.rows(
            f"""
            SELECT active_generation, active_collection, config_fingerprint
            FROM {ledger}
            WHERE model_name = ? AND stage = ? AND target_identity = ?
            ORDER BY fencing_token DESC
            LIMIT 1
            """,
            self._scope_params(scope),
        )
        # Both, or neither: a generation whose configuration fingerprint did
        # not survive cannot be admitted to a query anyway, so degrading on it
        # would advertise a scope that then refuses every read.
        if pointer and pointer[0][0] is not None and pointer[0][2] is not None:
            active_generation: str | None = str(pointer[0][0])
            active_collection = None if pointer[0][1] is None else str(pointer[0][1])
            config_fingerprint: str | None = str(pointer[0][2])
        else:
            # No servable generation survives -- a fail-closed in-place claim
            # already cleared it, or the scope never had one. Fall back to any
            # surviving collection pointer so a republish and the retirement
            # sweep still know what is out there.
            legacy = self._adapter.rows(
                f"""
                SELECT active_collection FROM {ledger}
                WHERE model_name = ? AND stage = ? AND target_identity = ?
                  AND active_collection IS NOT NULL
                ORDER BY fencing_token DESC
                LIMIT 1
                """,
                self._scope_params(scope),
            )
            active_generation = None
            active_collection = str(legacy[0][0]) if legacy else None
            config_fingerprint = None
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
                status, safe_error_code, active_generation, active_collection,
                config_fingerprint, rows_inserted,
                rows_updated, rows_skipped, rows_deleted, completed_at
            )
            SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, CURRENT_TIMESTAMP
            FROM (SELECT 1) AS seed
            """,
            [
                *self._scope_params(scope),
                uuid4().hex,
                next_fence,
                STATUS_DEGRADED if active_generation else STATUS_FAILED,
                RECOVERY_ERROR_CODE,
                active_generation,
                active_collection,
                config_fingerprint,
            ],
        )
        return self.status(scope)

    def _refuse_unless_owner_provably_dead(self, scope: StateScope) -> None:
        """Let recovery through without confirmation only for a dead local owner."""
        entry = self.status(scope)
        owner = entry.publisher
        if owner is not None and local_liveness(owner) is Liveness.DEAD:
            log.info(
                "%s: recovering without confirmation; the recorded publisher "
                "(host=%s pid=%d) is provably not running in this PID namespace",
                scope.model_name,
                owner.host,
                owner.pid,
            )
            return
        recorded = (
            describe_publisher(
                owner, heartbeat_epoch=entry.publisher_heartbeat_epoch, now_epoch=int(time.time())
            )
            if owner is not None
            else "no publisher is recorded on the row"
        )
        raise ServingCoordinationError(
            "Serving recovery requires terminating the previous owner first; "
            f"the ledger records the publisher as {recorded}. Confirm it is "
            "gone everywhere it could be running, then re-run with the "
            "owner-terminated confirmation"
        )


def _publisher_from_row(row: tuple[Any, ...]) -> PublisherIdentity | None:
    """The publisher the row records, or None when no claim is held.

    Keyed on `publication_id` rather than on the identity columns: a finished
    publish clears both, but a row written by a version that predates the
    identity columns can hold a claim with nothing recorded about its holder.
    """
    if row[2] is None or row[12] is None or row[13] is None:
        return None
    return PublisherIdentity(
        host=str(row[12]),
        pid=int(row[13]),
        started_epoch=None if row[14] is None else int(row[14]),
        label=None if row[15] is None else str(row[15]),
        namespace=None if row[17] is None else str(row[17]),
    )
