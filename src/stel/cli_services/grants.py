"""Operator administration of the grants relation (issue #577).

`WarehouseGrantStore` reads an operator-owned relation of
`(subject_id, attribute, value)` and compiles it into the policy filters a
governed read runs under. That half shipped with #396. Nothing wrote it: an
operator administered authorization by hand-writing SQL, and stel -- the
enforcement point -- could not report on its own policy.

This module is the write and inspect half. It follows
`retrieval/coordination.py` rather than inventing a mechanism: a stel-owned
relation created with `CREATE TABLE IF NOT EXISTS`, addressed by parameterized
statements through `adapter.execute`.

Three rules shape everything here.

**A grant value is never interpolated into SQL.** It is the field an attacker
would most like to control, and it arrives from a command line. Every
statement binds it as a parameter; the only interpolated identifier is the
relation name, which comes from the operator's own flag and is validated as a
node name before use.

**A mutation names its target or refuses.** Issue #511 established that a
serving command defaulting silently onto a target renders a confident answer
about the wrong store. Writing an authorization row into the wrong warehouse
is not a usability problem -- it grants access in a place nobody is looking.
So the mutating operations require an explicit `--target`, exactly as
`serving recover` does, and every operation reports the relation and warehouse
it resolved.

**The reserved identity attribute is not a policy value.** ADR-0010 put
`warehouse_identity` in this relation so it inherits operator ownership and
the revocation TTL, but it names the principal a read *connects as* rather
than a value to filter rows by. Reaching it through the generic grant path
would let an operator revoking what looks like a filter value silently change
a connection identity. It gets its own operations, and the generic path
refuses the name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..adapters import create_adapter
from ..adapters.base import WarehouseCapability
from ..append_log import audit_relation_for, write_audit_rows
from ..config import load_project
from ..config.identifiers import validate_node_name
from ..env import GRANTS_ACTOR_ENV, read_env
from ..mcp_server.grants import (
    ATTRIBUTE_COLUMN,
    SUBJECT_COLUMN,
    VALUE_COLUMN,
    WAREHOUSE_IDENTITY_ATTRIBUTE,
)
from ..profile import resolve_profile
from .context import ConfigClickError

if TYPE_CHECKING:
    from ..adapters.base import WarehouseAdapter
    from ..profile import ResolvedProfile

# The relation `stel mcp serve --grants-relation` is most likely to name.
# There is no default on the server flag, deliberately -- a server that
# authorizes from a relation nobody named would be a surprising default. This
# is only what the admin commands assume when the operator does not say, and
# every command prints the relation it used, so a mismatch with the server's
# flag is visible rather than silent.
DEFAULT_GRANTS_RELATION = "stel_grants"


@dataclass(frozen=True)
class GrantRow:
    """One row of the grants relation, as administered."""

    subject_id: str
    attribute: str
    value: str


@dataclass(frozen=True)
class GrantsReport:
    """Rows, plus which relation and warehouse they came from.

    The context is not decoration. An empty result is equally true of a
    subject with no grants and of a relation the server never reads, and those
    have opposite meanings -- the first denies a caller, the second silently
    fails to (issue #511).
    """

    rows: tuple[GrantRow, ...]
    relation: str
    target: str
    warehouse: str
    # Rows the statement actually changed. None for a read.
    rows_affected: int | None = None


def _resolve(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    relation: str,
) -> tuple[ResolvedProfile, str]:
    """Resolve the profile and validate the relation name.

    The project's models are deliberately *not* contract-validated here.
    Grants describe subjects, not models, and an operator whose project has a
    broken model must still be able to revoke someone's access -- making
    authorization administration depend on an unrelated model compiling would
    put the fix behind the emergency.
    """
    project_config, _sources, _models = load_project(project_dir)
    try:
        checked = validate_node_name(relation, kind="Grants relation")
    except ValueError as error:
        raise ConfigClickError(str(error)) from error
    resolved = resolve_profile(
        project_config, project_dir, target=target, profiles_dir=profiles_dir
    )
    return resolved, checked


def _describe(resolved: ResolvedProfile) -> str:
    return (
        f"{resolved.warehouse.type} {resolved.warehouse.storage_location()}".strip()
    )


def _require_explicit_target(
    target: str | None, *, resolved: ResolvedProfile, relation: str, what: str
) -> None:
    """Refuse a mutation that did not name its target.

    Resolution above is a read, so this refuses before anything is written --
    and it can name the target the caller would otherwise have got, which is
    what makes the re-run obvious rather than a guess.
    """
    if target is not None:
        return
    raise ConfigClickError(
        f"'stel grants {what}' requires an explicit --target: it changes who "
        "can read governed context, so it must not act on a target nobody "
        f"named. This profile would have used '{resolved.target_name}' "
        f"(relation '{relation}' in {_describe(resolved)}). Re-run with "
        f"--target {resolved.target_name} to confirm that is the one you mean."
    )


def _require_value(value: str, *, what: str) -> str:
    """Grants must be non-empty strings.

    `_grant_from_row` refuses a blank on the read side because it is
    "ambiguous between 'no grant' and 'grant everything'". Refusing here too
    means that ambiguity cannot be written in the first place, and the
    operator hears about it while they are still looking at the command.
    """
    checked = value.strip()
    if not checked:
        raise ConfigClickError(f"A grant's {what} must not be blank")
    return checked


def _refuse_reserved(attribute: str) -> None:
    if attribute == WAREHOUSE_IDENTITY_ATTRIBUTE:
        raise ConfigClickError(
            f"'{WAREHOUSE_IDENTITY_ATTRIBUTE}' is reserved: it names the "
            "warehouse principal a subject's reads execute as, not a value to "
            "filter rows by (ADR-0010). Granting it here would give one row "
            "two meanings, so use 'stel grants identity set' instead -- which "
            "also replaces the existing principal rather than adding a second "
            "one the server would refuse."
        )


def _ref(adapter: WarehouseAdapter, relation: str) -> str:
    return f"{adapter.schema_ref}.{adapter.quote_ident(relation)}"


def _actor() -> str:
    """Who to record as having made a change (issue #580).

    `STEL_GRANTS_ACTOR` first, so a provisioning job can name itself rather
    than logging whatever OS user its runner happens to use. Otherwise the OS
    user, which `getuser` can fail to determine at all on a host with no
    passwd entry and no environment -- unknown is a better record than a
    crash, and a failed audit lookup must not be the thing that stops a
    revocation.

    Advisory in every case, and the docs say so: anyone who can run this
    command can set the variable. It distinguishes actors, it does not
    authenticate them.
    """
    named = read_env(GRANTS_ACTOR_ENV)
    if named and named.strip():
        return named.strip()
    try:
        import getpass

        return getpass.getuser()
    except Exception:
        return "unknown"


def _audit(
    adapter: WarehouseAdapter,
    *,
    relation: str,
    target: str,
    action: str,
    subject: str,
    attribute: str,
    value: str | None,
    rows_affected: int,
) -> None:
    """Record one applied change. Raises if it cannot be recorded.

    Called *after* the statement, not before, for two reasons: `rows_affected`
    is only knowable afterwards, and a log written first would record changes
    that then failed to apply. The cost is the opposite failure -- a change
    applied and not recorded -- which is why `write_audit_rows` raises instead
    of warning, and why its message says the change did land.
    """
    from datetime import UTC, datetime

    write_audit_rows(
        adapter,
        audit_relation_for(relation),
        [
            {
                "logged_at": datetime.now(UTC).isoformat(),
                "action": action,
                "subject_id": subject,
                "attribute": attribute,
                "value": value,
                "rows_affected": rows_affected,
                "actor": _actor(),
                "profile_target": target,
            }
        ],
    )


def _ensure_relation(adapter: WarehouseAdapter, relation: str) -> None:
    """Create the grants relation if it does not exist.

    STRING NOT NULL on all three columns, matching the serving ledger's types
    so the statement parses on both DuckDB and BigQuery. No primary key:
    BigQuery cannot enforce one, and uniqueness here is maintained by the
    conditional insert in `grant` rather than by the warehouse.
    """
    adapter.require_capability(
        WarehouseCapability.SQL_QUERIES,
        operation="administering the grants relation",
    )
    adapter.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_ref(adapter, relation)} (
            {SUBJECT_COLUMN} STRING NOT NULL,
            {ATTRIBUTE_COLUMN} STRING NOT NULL,
            {VALUE_COLUMN} STRING NOT NULL
        )
        """
    )


def _read_rows(
    adapter: WarehouseAdapter, relation: str, subject: str | None
) -> tuple[GrantRow, ...]:
    sql = (
        f"SELECT {SUBJECT_COLUMN}, {ATTRIBUTE_COLUMN}, {VALUE_COLUMN} "
        f"FROM {_ref(adapter, relation)}"
    )
    params: list[Any] = []
    if subject is not None:
        sql += f" WHERE {SUBJECT_COLUMN} = ?"
        params.append(subject)
    sql += f" ORDER BY {SUBJECT_COLUMN}, {ATTRIBUTE_COLUMN}, {VALUE_COLUMN}"
    return tuple(
        GrantRow(str(row[0]), str(row[1]), str(row[2]))
        for row in adapter.rows(sql, params or None)
    )


def list_grants(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    relation: str = DEFAULT_GRANTS_RELATION,
    subject: str | None = None,
) -> GrantsReport:
    """Every grant, or every grant for one subject.

    A read, so it does not demand an explicit target: showing the wrong
    warehouse's grants is recoverable by looking again, and requiring a flag
    to answer "what is configured here" would make the safe operation the
    awkward one.
    """
    resolved, checked = _resolve(
        project_dir, profiles_dir=profiles_dir, target=target, relation=relation
    )
    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        _ensure_relation(adapter, checked)
        rows = _read_rows(adapter, checked, subject)
    return GrantsReport(
        rows=rows,
        relation=checked,
        target=resolved.target_name,
        warehouse=_describe(resolved),
    )


@dataclass(frozen=True)
class AuditEntry:
    """One recorded change to the grants relation."""

    logged_at: str
    action: str
    subject_id: str
    attribute: str
    value: str | None
    rows_affected: int
    actor: str
    profile_target: str


@dataclass(frozen=True)
class AuditReport:
    entries: tuple[AuditEntry, ...]
    relation: str
    target: str
    warehouse: str
    # False when no change has ever been recorded here, which is not the same
    # as "no changes were made" -- it is also what a wrong target looks like.
    relation_exists: bool


_AUDIT_COLUMNS = (
    "logged_at",
    "action",
    "subject_id",
    "attribute",
    "value",
    "rows_affected",
    "actor",
    "profile_target",
)


def grant_history(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    relation: str = DEFAULT_GRANTS_RELATION,
    subject: str | None = None,
    limit: int = 50,
) -> AuditReport:
    """Recorded changes, newest first (issue #580).

    This is the question a hard delete costs: the grants relation holds only
    the present tense, and `who could see this in August` is unanswerable from
    it. A read, so it does not demand an explicit target.
    """
    resolved, checked = _resolve(
        project_dir, profiles_dir=profiles_dir, target=target, relation=relation
    )
    audit = audit_relation_for(checked)
    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        adapter.require_capability(
            WarehouseCapability.SQL_QUERIES, operation="reading the grants audit log"
        )
        # Not created here: the audit relation is created by the first change
        # written to it, and creating it on a read would make "nothing has
        # ever happened" and "this is the wrong warehouse" look identical
        # forever after.
        if adapter.table_column_names(audit) is None:
            return AuditReport(
                entries=(),
                relation=audit,
                target=resolved.target_name,
                warehouse=_describe(resolved),
                relation_exists=False,
            )
        sql = f"SELECT {', '.join(_AUDIT_COLUMNS)} FROM {_ref(adapter, audit)}"
        params: list[Any] = []
        if subject is not None:
            sql += " WHERE subject_id = ?"
            params.append(subject)
        sql += f" ORDER BY logged_at DESC LIMIT {int(limit)}"
        rows = adapter.rows(sql, params or None)
    return AuditReport(
        entries=tuple(
            AuditEntry(
                logged_at=str(row[0]),
                action=str(row[1]),
                subject_id=str(row[2]),
                attribute=str(row[3]),
                value=None if row[4] is None else str(row[4]),
                rows_affected=int(row[5] or 0),
                actor=str(row[6]),
                profile_target=str(row[7]),
            )
            for row in rows
        ),
        relation=audit,
        target=resolved.target_name,
        warehouse=_describe(resolved),
        relation_exists=True,
    )


def grant(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    subject: str,
    attribute: str,
    value: str,
    relation: str = DEFAULT_GRANTS_RELATION,
) -> GrantsReport:
    """Permit one value of one policy attribute for one subject.

    Idempotent. Re-granting what a subject already holds writes nothing rather
    than adding a duplicate row: `_granted_values` collects rows into the tuple
    a filter compiles from, so a duplicate would render as `IN ('x', 'x')` --
    harmless, but it makes `stel grants show` misreport, and it is the kind of
    drift that accumulates silently under a provisioning script.
    """
    subject = _require_value(subject, what="subject")
    attribute = _require_value(attribute, what="attribute")
    value = _require_value(value, what="value")
    _refuse_reserved(attribute)
    resolved, checked = _resolve(
        project_dir, profiles_dir=profiles_dir, target=target, relation=relation
    )
    _require_explicit_target(target, resolved=resolved, relation=checked, what="grant")
    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        _ensure_relation(adapter, checked)
        table = _ref(adapter, checked)
        # Asked before the insert, because the insert is conditional and
        # cannot report whether it added anything -- DuckDB and BigQuery do
        # not agree on what `execute` returns for a DML statement.
        already_held = bool(
            adapter.scalar(
                f"SELECT COUNT(*) FROM {table} WHERE {SUBJECT_COLUMN} = ? "
                f"AND {ATTRIBUTE_COLUMN} = ? AND {VALUE_COLUMN} = ?",
                [subject, attribute, value],
            )
        )
        adapter.execute(
            f"""
            INSERT INTO {table} (
                {SUBJECT_COLUMN}, {ATTRIBUTE_COLUMN}, {VALUE_COLUMN}
            )
            SELECT ?, ?, ? FROM (SELECT 1) AS seed
            WHERE NOT EXISTS (
                SELECT 1 FROM {table}
                WHERE {SUBJECT_COLUMN} = ? AND {ATTRIBUTE_COLUMN} = ?
                  AND {VALUE_COLUMN} = ?
            )
            """,
            [subject, attribute, value, subject, attribute, value],
        )
        rows = _read_rows(adapter, checked, subject)
        # Audited only when a row was actually written. The insert is
        # conditional, so re-granting what a subject already holds changes
        # nothing -- and an audit log that records non-changes stops being
        # readable as the history of who gained access when.
        written = 0 if already_held else 1
        if written:
            _audit(
                adapter,
                relation=checked,
                target=resolved.target_name,
                action="grant",
                subject=subject,
                attribute=attribute,
                value=value,
                rows_affected=written,
            )
    return GrantsReport(
        rows=rows,
        relation=checked,
        target=resolved.target_name,
        warehouse=_describe(resolved),
        rows_affected=written,
    )


def _delete(
    adapter: WarehouseAdapter,
    relation: str,
    subject: str,
    attribute: str,
    value: str | None,
) -> int:
    """Delete matching rows, returning how many there were.

    Counted before the delete rather than read from the statement's result:
    `execute` returns whatever the driver hands back, and DuckDB and BigQuery
    do not agree on what that is for a DELETE. A count is a portable answer,
    and the operator needs one -- "revoked" against zero matching rows usually
    means a typo in the subject, and reporting that as success hides it.
    """
    table = _ref(adapter, relation)
    where = f"WHERE {SUBJECT_COLUMN} = ? AND {ATTRIBUTE_COLUMN} = ?"
    params: list[Any] = [subject, attribute]
    if value is not None:
        where += f" AND {VALUE_COLUMN} = ?"
        params.append(value)
    count = adapter.scalar(f"SELECT COUNT(*) FROM {table} {where}", list(params))
    adapter.execute(f"DELETE FROM {table} {where}", list(params))
    return int(count or 0)


def revoke(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    subject: str,
    attribute: str,
    value: str | None = None,
    relation: str = DEFAULT_GRANTS_RELATION,
) -> GrantsReport:
    """Remove one value, or every value of one attribute, for one subject.

    A hard delete. The row goes away, which keeps the request-path read a
    plain equality scan -- the alternative, validity windows resolved on every
    read, would put a question nobody asks mid-request into the latency path
    #519 and #528 spent real effort clearing. The history that costs is
    recovered by an append-only audit log, which is a separate relation and a
    separate change.

    Revocation is not immediate. `WarehouseGrantStore` caches per subject, so
    a removed grant keeps applying for up to `--grant-ttl-seconds`; the command
    edge says so rather than leaving the operator to discover it.
    """
    subject = _require_value(subject, what="subject")
    attribute = _require_value(attribute, what="attribute")
    _refuse_reserved(attribute)
    if value is not None:
        value = _require_value(value, what="value")
    resolved, checked = _resolve(
        project_dir, profiles_dir=profiles_dir, target=target, relation=relation
    )
    _require_explicit_target(target, resolved=resolved, relation=checked, what="revoke")
    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        _ensure_relation(adapter, checked)
        affected = _delete(adapter, checked, subject, attribute, value)
        rows = _read_rows(adapter, checked, subject)
        # A revoke that matched nothing removed no access, so there is nothing
        # to record. It is still reported to the caller, because zero matches
        # usually means a typo.
        if affected:
            _audit(
                adapter,
                relation=checked,
                target=resolved.target_name,
                action="revoke",
                subject=subject,
                attribute=attribute,
                value=value,
                rows_affected=affected,
            )
    return GrantsReport(
        rows=rows,
        relation=checked,
        target=resolved.target_name,
        warehouse=_describe(resolved),
        rows_affected=affected,
    )


def set_identity(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    subject: str,
    principal: str,
    relation: str = DEFAULT_GRANTS_RELATION,
) -> GrantsReport:
    """Set the warehouse principal a subject's governed reads execute as.

    Replaces rather than appends, because `GrantWarehouseIdentityResolver`
    treats a second `warehouse_identity` row as a configuration error: a
    subject "cannot legitimately execute as two principals at once". An
    appending `set` would therefore break the subject's reads entirely on the
    second call, which is the opposite of what the operator asked for.
    """
    subject = _require_value(subject, what="subject")
    principal = _require_value(principal, what="principal")
    resolved, checked = _resolve(
        project_dir, profiles_dir=profiles_dir, target=target, relation=relation
    )
    _require_explicit_target(
        target, resolved=resolved, relation=checked, what="identity set"
    )
    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        _ensure_relation(adapter, checked)
        table = _ref(adapter, checked)
        # Delete-then-insert, not an update: the subject may hold no row yet,
        # and BigQuery has no portable upsert for a table with no key.
        _delete(adapter, checked, subject, WAREHOUSE_IDENTITY_ATTRIBUTE, None)
        adapter.execute(
            f"""
            INSERT INTO {table} (
                {SUBJECT_COLUMN}, {ATTRIBUTE_COLUMN}, {VALUE_COLUMN}
            )
            VALUES (?, ?, ?)
            """,
            [subject, WAREHOUSE_IDENTITY_ATTRIBUTE, principal],
        )
        rows = _read_rows(adapter, checked, subject)
        # Always recorded, even when the principal is unchanged: unlike a
        # grant, this rewrites the row every time, and "was it re-set during
        # the incident" is a question the history should be able to answer.
        _audit(
            adapter,
            relation=checked,
            target=resolved.target_name,
            action="identity_set",
            subject=subject,
            attribute=WAREHOUSE_IDENTITY_ATTRIBUTE,
            value=principal,
            rows_affected=1,
        )
    return GrantsReport(
        rows=rows,
        relation=checked,
        target=resolved.target_name,
        warehouse=_describe(resolved),
        rows_affected=1,
    )


def clear_identity(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    subject: str,
    relation: str = DEFAULT_GRANTS_RELATION,
) -> GrantsReport:
    """Remove a subject's warehouse identity.

    This denies the subject's governed reads outright rather than falling back
    to the operator's credentials -- "a missing row must never read as
    'unprotected'" (ADR-0010). Worth saying plainly, because "clear" reads
    like a relaxation and is the opposite.
    """
    subject = _require_value(subject, what="subject")
    resolved, checked = _resolve(
        project_dir, profiles_dir=profiles_dir, target=target, relation=relation
    )
    _require_explicit_target(
        target, resolved=resolved, relation=checked, what="identity clear"
    )
    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        _ensure_relation(adapter, checked)
        affected = _delete(adapter, checked, subject, WAREHOUSE_IDENTITY_ATTRIBUTE, None)
        rows = _read_rows(adapter, checked, subject)
        if affected:
            _audit(
                adapter,
                relation=checked,
                target=resolved.target_name,
                action="identity_clear",
                subject=subject,
                attribute=WAREHOUSE_IDENTITY_ATTRIBUTE,
                value=None,
                rows_affected=affected,
            )
    return GrantsReport(
        rows=rows,
        relation=checked,
        target=resolved.target_name,
        warehouse=_describe(resolved),
        rows_affected=affected,
    )
