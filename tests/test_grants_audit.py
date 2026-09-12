"""The grants audit log (issue #580).

`stel grants revoke` is a hard delete, so the grants relation holds only the
present tense. These tests cover the history that answers who was entitled to
what and since when -- and, more importantly, the one place where this log's
contract is the *inverse* of the other two: it raises where they warn.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from stel.append_log import (
    GRANT_AUDIT_SCHEMA,
    AuditLogError,
    audit_relation_for,
    write_audit_rows,
)
from stel.cli_services.grants import (
    DEFAULT_GRANTS_RELATION,
    clear_identity,
    grant,
    grant_history,
    revoke,
    set_identity,
)
from stel.env import ENV_VARS, GRANTS_ACTOR_ENV
from stel.mcp_server.grants import WAREHOUSE_IDENTITY_ATTRIBUTE

TARGET = "dev"


@pytest.fixture
def project(tmp_path: Path, example_project_dir: Path) -> Path:
    destination = tmp_path / "proj"
    shutil.copytree(
        example_project_dir,
        destination,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    return destination


def _history(project_dir: Path, **kwargs: Any) -> Any:
    return grant_history(project_dir, profiles_dir=None, target=TARGET, **kwargs)


def _grant(project_dir: Path, subject: str, attribute: str, value: str) -> Any:
    return grant(
        project_dir,
        profiles_dir=None,
        target=TARGET,
        subject=subject,
        attribute=attribute,
        value=value,
    )


# ─── what gets recorded ─────────────────────────────────────────────────────


def test_a_grant_is_recorded(project: Path) -> None:
    _grant(project, "analyst@example.com", "tenant_id", "acme")
    entries = _history(project).entries
    assert len(entries) == 1
    assert entries[0].action == "grant"
    assert entries[0].subject_id == "analyst@example.com"
    assert entries[0].attribute == "tenant_id"
    assert entries[0].value == "acme"
    assert entries[0].rows_affected == 1
    assert entries[0].profile_target == TARGET


def test_a_revoke_is_recorded_with_the_rows_it_removed(project: Path) -> None:
    for value in ("acme", "globex"):
        _grant(project, "analyst@example.com", "tenant_id", value)
    revoke(
        project,
        profiles_dir=None,
        target=TARGET,
        subject="analyst@example.com",
        attribute="tenant_id",
    )
    revoked = [e for e in _history(project).entries if e.action == "revoke"]
    assert len(revoked) == 1
    assert revoked[0].rows_affected == 2
    # Null, not an empty string: the revoke named no value, and "every value"
    # is not the same claim as "the value ''".
    assert revoked[0].value is None


def test_identity_changes_are_recorded(project: Path) -> None:
    set_identity(
        project,
        profiles_dir=None,
        target=TARGET,
        subject="analyst@example.com",
        principal="reader@p.iam",
    )
    clear_identity(
        project, profiles_dir=None, target=TARGET, subject="analyst@example.com"
    )
    actions = [e.action for e in _history(project).entries]
    assert actions == ["identity_clear", "identity_set"]
    recorded = {e.action: e for e in _history(project).entries}
    assert recorded["identity_set"].attribute == WAREHOUSE_IDENTITY_ATTRIBUTE
    assert recorded["identity_set"].value == "reader@p.iam"
    assert recorded["identity_clear"].value is None


# ─── what deliberately does not get recorded ────────────────────────────────


def test_a_grant_that_changed_nothing_is_not_recorded(project: Path) -> None:
    # The insert is conditional, so the second call writes no row. Recording
    # it would make the history stop reading as "when access changed".
    _grant(project, "analyst@example.com", "tenant_id", "acme")
    report = _grant(project, "analyst@example.com", "tenant_id", "acme")
    assert report.rows_affected == 0
    assert len(_history(project).entries) == 1


def test_a_revoke_that_matched_nothing_is_not_recorded(project: Path) -> None:
    _grant(project, "analyst@example.com", "tenant_id", "acme")
    revoke(
        project,
        profiles_dir=None,
        target=TARGET,
        subject="typo@example.com",
        attribute="tenant_id",
    )
    assert [e.action for e in _history(project).entries] == ["grant"]


def test_clearing_an_absent_identity_is_not_recorded(project: Path) -> None:
    clear_identity(
        project, profiles_dir=None, target=TARGET, subject="nobody@example.com"
    )
    assert _history(project).relation_exists is False


def test_setting_the_same_identity_twice_is_recorded_twice(project: Path) -> None:
    # Unlike a grant, `set` rewrites the row every time -- and "was it re-set
    # during the incident" is a question the history should answer.
    for _ in range(2):
        set_identity(
            project,
            profiles_dir=None,
            target=TARGET,
            subject="analyst@example.com",
            principal="reader@p.iam",
        )
    assert len(_history(project).entries) == 2


# ─── reading the history ────────────────────────────────────────────────────


def test_history_is_newest_first(project: Path) -> None:
    _grant(project, "a@example.com", "tenant_id", "one")
    _grant(project, "a@example.com", "tenant_id", "two")
    entries = _history(project).entries
    assert [e.value for e in entries] == ["two", "one"]


def test_history_filters_by_subject(project: Path) -> None:
    _grant(project, "a@example.com", "tenant_id", "acme")
    _grant(project, "b@example.com", "tenant_id", "globex")
    entries = _history(project, subject="b@example.com").entries
    assert [e.subject_id for e in entries] == ["b@example.com"]


def test_history_respects_its_limit(project: Path) -> None:
    for value in ("one", "two", "three"):
        _grant(project, "a@example.com", "tenant_id", value)
    assert len(_history(project, limit=2).entries) == 2


def test_reading_history_does_not_create_the_relation(project: Path) -> None:
    """A read must not manufacture the thing it is reporting on.

    Creating it here would make "nothing has ever happened" and "this is the
    wrong warehouse" indistinguishable from the second read onwards -- which
    is exactly the #511 failure this whole command set is shaped around.
    """
    assert _history(project).relation_exists is False
    assert _history(project).relation_exists is False
    assert _history(project).relation == audit_relation_for(DEFAULT_GRANTS_RELATION)


# ─── the actor ──────────────────────────────────────────────────────────────


def test_the_actor_env_var_is_recorded(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(GRANTS_ACTOR_ENV, "provisioning-job")
    _grant(project, "analyst@example.com", "tenant_id", "acme")
    assert _history(project).entries[0].actor == "provisioning-job"


def test_a_blank_actor_env_var_falls_back(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An empty variable is how a shell exports "unset" by accident; it must
    # not write a blank actor into the record.
    monkeypatch.setenv(GRANTS_ACTOR_ENV, "   ")
    _grant(project, "analyst@example.com", "tenant_id", "acme")
    assert _history(project).entries[0].actor.strip()


def test_the_actor_var_is_declared(project: Path) -> None:
    # env.py exists so the set is enumerable; a variable read but undeclared
    # is the drift that module prevents.
    assert GRANTS_ACTOR_ENV in ENV_VARS


# ─── the inverted contract ──────────────────────────────────────────────────


class _RefusingAdapter:
    """An adapter whose appends always fail."""

    def append_rows(self, table: str, df: pl.DataFrame) -> int:
        raise RuntimeError("warehouse said no")


def test_an_unwritable_audit_row_raises_rather_than_warning() -> None:
    """The whole point of #580's contract, in one assertion.

    `write_rows` swallows and warns, because a run must not fail over its own
    telemetry. An audit trail that does the same has holes exactly where the
    warehouse was struggling, and reads as complete.
    """
    with pytest.raises(AuditLogError):
        write_audit_rows(_RefusingAdapter(), "stel_grants_audit", [_row()])


def test_the_failure_says_the_change_was_already_applied() -> None:
    # The operator has to know both facts: the access changed, and it was not
    # recorded. Saying only the second reads as "nothing happened".
    with pytest.raises(AuditLogError, match="was applied"):
        write_audit_rows(_RefusingAdapter(), "stel_grants_audit", [_row()])


def test_the_failure_does_not_echo_warehouse_text() -> None:
    # The class name locates the failure; the warehouse's own message is not
    # repeated, matching what the rest of append_log does.
    with pytest.raises(AuditLogError) as caught:
        write_audit_rows(_RefusingAdapter(), "stel_grants_audit", [_row()])
    assert "RuntimeError" in str(caught.value)
    assert "warehouse said no" not in str(caught.value)


def test_a_failed_audit_write_is_not_swallowed_by_the_service(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The service must not catch what the log raises; a grant that could not
    # be recorded has to reach the operator.
    def boom(*_args: Any, **_kwargs: Any) -> int:
        raise AuditLogError("could not record")

    monkeypatch.setattr("stel.cli_services.grants.write_audit_rows", boom)
    with pytest.raises(AuditLogError):
        _grant(project, "analyst@example.com", "tenant_id", "acme")


def test_no_rows_is_not_an_error() -> None:
    # Nothing to record is not a failure to record.
    assert write_audit_rows(_RefusingAdapter(), "stel_grants_audit", []) == 0


def _row() -> dict[str, Any]:
    return {
        "logged_at": "2026-09-11T00:00:00+00:00",
        "action": "grant",
        "subject_id": "analyst@example.com",
        "attribute": "tenant_id",
        "value": "acme",
        "rows_affected": 1,
        "actor": "tester",
        "profile_target": TARGET,
    }


def test_the_audit_schema_declares_every_column_the_writer_sends() -> None:
    # The other two logs declare explicit types because the first batch must
    # not decide the persisted schema. A column sent but undeclared would be
    # inferred, which is the same bug.
    assert set(_row()) == set(GRANT_AUDIT_SCHEMA)


def test_a_null_value_does_not_decide_the_column_type() -> None:
    # A first write whose only row is a revoke-everything has `value` null.
    # Without the declared schema polars infers Null, and every later row
    # carrying an actual value fails to convert -- silently, in a log.
    row = _row() | {"value": None, "action": "revoke"}
    frame = pl.DataFrame([row], schema=GRANT_AUDIT_SCHEMA)
    assert frame.schema["value"] == pl.String
