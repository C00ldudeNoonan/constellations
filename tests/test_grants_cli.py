"""Administering the grants relation (issue #577).

The read half shipped with #396 and is tested in `test_mcp_grants.py`. These
tests cover the write half, and the seam between them: rows written here have
to be exactly the rows `WarehouseGrantStore` and `GrantAuthorizationProvider`
expect, so the round-trip below reads back through the real consumer rather
than asserting on the table.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

import stel.cli as cli
from stel.adapters import create_adapter
from stel.adapters.base import ReadPredicate, WarehouseIdentity
from stel.cli_services.context import ConfigClickError
from stel.cli_services.grants import (
    DEFAULT_GRANTS_RELATION,
    clear_identity,
    grant,
    list_grants,
    revoke,
    set_identity,
)
from stel.config import load_project
from stel.mcp_server.authorization import AuthorizationError, PolicyAttribute, Principal
from stel.mcp_server.grants import (
    WAREHOUSE_IDENTITY_ATTRIBUTE,
    GrantAuthorizationProvider,
    GrantWarehouseIdentityResolver,
    WarehouseGrantStore,
)
from stel.profile import resolve_profile

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


class _AdapterRowReader:
    """The narrow `GrantRowReader` the store needs, over a real warehouse.

    `WarehouseContextRepository` wants a `SearchSession`, which is a serving
    concern this has nothing to do with. `GrantRowReader` exists as a separate
    Protocol precisely so a caller can supply rows from somewhere else, so
    supplying them straight from the adapter is using the seam, not evading it.
    """

    def __init__(self, project_dir: Path) -> None:
        self._project_dir = project_dir
        project_config, _sources, _models = load_project(project_dir)
        self._resolved = resolve_profile(
            project_config, project_dir, target=TARGET, profiles_dir=None
        )

    def read_rows(
        self,
        relation: str,
        *,
        identity: WarehouseIdentity,
        predicates: Sequence[ReadPredicate],
        max_rows: int,
        columns: Sequence[str] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        subject = predicates[0].value
        with create_adapter(
            self._resolved.warehouse, project_dir=self._project_dir
        ) as adapter:
            table = f"{adapter.schema_ref}.{adapter.quote_ident(relation)}"
            # `columns=None` means every column, which is what the store asks
            # for since #582 -- naming `operator` in a projection would break
            # a relation written before it existed.
            names = (
                list(columns)
                if columns
                else sorted(adapter.table_column_names(relation) or ())
            )
            rows = adapter.rows(
                f"SELECT {', '.join(names)} FROM {table} WHERE subject_id = ?",
                [subject],
            )
        return tuple(dict(zip(names, row, strict=True)) for row in rows)


def _store(project_dir: Path) -> WarehouseGrantStore:
    # ttl 0 is not allowed, and a cached answer would defeat a test that
    # revokes and re-reads -- so each assertion builds its own store.
    return WarehouseGrantStore(
        _AdapterRowReader(project_dir), relation=DEFAULT_GRANTS_RELATION
    )


def _values(project_dir: Path, subject: str, attribute: str) -> tuple[str, ...]:
    return tuple(
        row.value
        for row in _store(project_dir).grants_for(subject)
        if row.attribute == attribute
    )


# ─── the round trip ─────────────────────────────────────────────────────────


def test_granted_rows_compile_into_the_filter_the_server_applies(
    project: Path,
) -> None:
    """A grant written here has to reach the consumer as a policy filter.

    This is the seam the whole issue exists to close, so it is asserted
    through `GrantAuthorizationProvider` rather than against the table.
    """
    grant(
        project,
        profiles_dir=None,
        target=TARGET,
        subject="analyst@example.com",
        attribute="tenant_id",
        value="acme",
    )
    provider = GrantAuthorizationProvider(_store(project))
    filters = provider.search_policy_filters(
        Principal(subject_id="analyst@example.com"),
        access="governed",
        attributes=[PolicyAttribute("tenant_id", "string")],
    )
    assert len(filters) == 1
    assert filters[0].field == "tenant_id"
    assert filters[0].value == "acme"

    # And revoking it takes the caller back to refused, rather than to an
    # unfiltered read -- the direction that matters.
    revoke(
        project,
        profiles_dir=None,
        target=TARGET,
        subject="analyst@example.com",
        attribute="tenant_id",
        value="acme",
    )
    with pytest.raises(AuthorizationError):
        GrantAuthorizationProvider(_store(project)).search_policy_filters(
            Principal(subject_id="analyst@example.com"),
            access="governed",
            attributes=[PolicyAttribute("tenant_id", "string")],
        )


def test_two_values_for_one_attribute_compile_to_in(project: Path) -> None:
    for value in ("acme", "globex"):
        grant(
            project,
            profiles_dir=None,
            target=TARGET,
            subject="analyst@example.com",
            attribute="tenant_id",
            value=value,
        )
    filters = GrantAuthorizationProvider(_store(project)).search_policy_filters(
        Principal(subject_id="analyst@example.com"),
        access="governed",
        attributes=[PolicyAttribute("tenant_id", "string")],
    )
    value = filters[0].value
    # A tuple, not a scalar: two grants have to compile to a set membership
    # test rather than an equality against one of them.
    assert isinstance(value, tuple)
    assert set(value) == {"acme", "globex"}


# ─── idempotence and revocation shape ───────────────────────────────────────


def test_granting_twice_writes_one_row(project: Path) -> None:
    # A duplicate would render as IN ('acme', 'acme'): harmless to the query,
    # wrong in `show`, and the shape a provisioning script accumulates.
    for _ in range(2):
        grant(
            project,
            profiles_dir=None,
            target=TARGET,
            subject="analyst@example.com",
            attribute="tenant_id",
            value="acme",
        )
    assert _values(project, "analyst@example.com", "tenant_id") == ("acme",)


def test_revoke_without_a_value_removes_every_value(project: Path) -> None:
    for value in ("acme", "globex"):
        grant(
            project,
            profiles_dir=None,
            target=TARGET,
            subject="analyst@example.com",
            attribute="tenant_id",
            value=value,
        )
    report = revoke(
        project,
        profiles_dir=None,
        target=TARGET,
        subject="analyst@example.com",
        attribute="tenant_id",
    )
    assert report.rows_affected == 2
    assert _values(project, "analyst@example.com", "tenant_id") == ()


def test_revoking_nothing_reports_zero_rather_than_success(project: Path) -> None:
    # A typo in the subject looks exactly like this, so the count has to be
    # honest for the command edge to be able to say so.
    report = revoke(
        project,
        profiles_dir=None,
        target=TARGET,
        subject="nobody@example.com",
        attribute="tenant_id",
    )
    assert report.rows_affected == 0


# ─── the reserved identity attribute ────────────────────────────────────────


def test_generic_grant_refuses_the_reserved_identity_attribute(project: Path) -> None:
    with pytest.raises(ConfigClickError, match="reserved"):
        grant(
            project,
            profiles_dir=None,
            target=TARGET,
            subject="analyst@example.com",
            attribute=WAREHOUSE_IDENTITY_ATTRIBUTE,
            value="reader@project.iam.gserviceaccount.com",
        )


def test_generic_revoke_refuses_the_reserved_identity_attribute(project: Path) -> None:
    # Revoke matters as much as grant: removing what looks like a filter value
    # would silently change which principal the subject connects as.
    with pytest.raises(ConfigClickError, match="reserved"):
        revoke(
            project,
            profiles_dir=None,
            target=TARGET,
            subject="analyst@example.com",
            attribute=WAREHOUSE_IDENTITY_ATTRIBUTE,
        )


def test_setting_an_identity_twice_leaves_exactly_one(project: Path) -> None:
    """Two identity rows are a configuration error, not two permissions.

    `GrantWarehouseIdentityResolver` refuses outright when a subject holds
    more than one, so an appending `set` would break the subject's reads on
    the second call.
    """
    for principal in ("first@p.iam.gserviceaccount.com", "second@p.iam.gserviceaccount.com"):
        set_identity(
            project,
            profiles_dir=None,
            target=TARGET,
            subject="analyst@example.com",
            principal=principal,
        )
    resolver = GrantWarehouseIdentityResolver(_store(project))
    identity = resolver.identity_for(Principal(subject_id="analyst@example.com"))
    assert identity == WarehouseIdentity("second@p.iam.gserviceaccount.com")


def test_clearing_an_identity_denies_rather_than_falls_back(project: Path) -> None:
    set_identity(
        project,
        profiles_dir=None,
        target=TARGET,
        subject="analyst@example.com",
        principal="reader@p.iam.gserviceaccount.com",
    )
    report = clear_identity(
        project, profiles_dir=None, target=TARGET, subject="analyst@example.com"
    )
    assert report.rows_affected == 1
    # Not OPERATOR_IDENTITY: a missing row must never read as unprotected.
    with pytest.raises(AuthorizationError):
        GrantWarehouseIdentityResolver(_store(project)).identity_for(
            Principal(subject_id="analyst@example.com")
        )


def test_clearing_an_identity_the_subject_never_had_reports_zero(
    project: Path,
) -> None:
    report = clear_identity(
        project, profiles_dir=None, target=TARGET, subject="nobody@example.com"
    )
    assert report.rows_affected == 0


def test_setting_an_identity_leaves_policy_grants_alone(project: Path) -> None:
    # `set` replaces, and the delete it does must be scoped to the reserved
    # attribute -- not to everything the subject holds.
    grant(
        project,
        profiles_dir=None,
        target=TARGET,
        subject="analyst@example.com",
        attribute="tenant_id",
        value="acme",
    )
    set_identity(
        project,
        profiles_dir=None,
        target=TARGET,
        subject="analyst@example.com",
        principal="reader@p.iam.gserviceaccount.com",
    )
    assert _values(project, "analyst@example.com", "tenant_id") == ("acme",)


# ─── refusing to act on a target nobody named ───────────────────────────────


@pytest.mark.parametrize(
    ("operation", "kwargs"),
    [
        (grant, {"subject": "a", "attribute": "tenant_id", "value": "acme"}),
        (revoke, {"subject": "a", "attribute": "tenant_id"}),
        (set_identity, {"subject": "a", "principal": "reader@p.iam"}),
        (clear_identity, {"subject": "a"}),
    ],
)
def test_mutations_refuse_without_an_explicit_target(
    project: Path, operation: Any, kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ConfigClickError, match="requires an explicit --target"):
        operation(project, profiles_dir=None, target=None, **kwargs)


def test_the_refusal_names_the_target_it_would_have_used(project: Path) -> None:
    # Naming the default is what makes the re-run obvious rather than a guess
    # (issue #511).
    with pytest.raises(ConfigClickError, match=TARGET):
        grant(
            project,
            profiles_dir=None,
            target=None,
            subject="a",
            attribute="tenant_id",
            value="acme",
        )


def test_reads_do_not_require_an_explicit_target(project: Path) -> None:
    # Listing the wrong warehouse's grants is recoverable by looking again,
    # so the safe operation must not be the awkward one.
    report = list_grants(project, profiles_dir=None, target=None)
    assert report.rows == ()
    assert report.relation == DEFAULT_GRANTS_RELATION
    assert report.target == TARGET


# ─── input validation ───────────────────────────────────────────────────────


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_grant_values_are_refused(project: Path, blank: str) -> None:
    # `_grant_from_row` refuses a blank on the read side as "ambiguous between
    # 'no grant' and 'grant everything'". It must not be writable either.
    with pytest.raises(ConfigClickError, match="must not be blank"):
        grant(
            project,
            profiles_dir=None,
            target=TARGET,
            subject="analyst@example.com",
            attribute="tenant_id",
            value=blank,
        )


def test_a_relation_name_that_is_not_an_identifier_is_refused(project: Path) -> None:
    # The relation is the one interpolated identifier in this module, so it is
    # validated rather than trusted.
    with pytest.raises(ConfigClickError):
        list_grants(
            project, profiles_dir=None, target=TARGET, relation="grants; DROP TABLE x"
        )


def test_a_value_containing_sql_is_stored_verbatim(project: Path) -> None:
    # Parameter binding, not escaping: the value round-trips unchanged and no
    # second statement runs.
    hostile = "acme'; DROP TABLE stel_grants; --"
    grant(
        project,
        profiles_dir=None,
        target=TARGET,
        subject="analyst@example.com",
        attribute="tenant_id",
        value=hostile,
    )
    assert _values(project, "analyst@example.com", "tenant_id") == (hostile,)


# ─── the constants the CLI duplicates ───────────────────────────────────────


def test_cli_literals_match_their_source_of_truth() -> None:
    # cli.py duplicates both as literals so a Click default does not import
    # the adapter stack on every `stel --help`. Pinned here so a rename fails
    # a test rather than drifting.
    assert cli._DEFAULT_GRANTS_RELATION == DEFAULT_GRANTS_RELATION
    assert cli._WAREHOUSE_IDENTITY == WAREHOUSE_IDENTITY_ATTRIBUTE
