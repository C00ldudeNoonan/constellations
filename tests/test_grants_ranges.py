"""Interval entitlements (issue #582).

ALE-73 wanted a tier to select "which date ranges". Grants could only compile
`EQUAL`, `IN` and `ARRAY_CONTAINS_ANY`, so a window had no representation.

The constraint that shapes all of this: every store combines search filters
with AND (`" AND ".join(clauses)` in both `duckdb.py` and `lancedb.py`). Two
intervals for one attribute therefore cannot express the union an operator
would read them as, which is why they are refused rather than resolved.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from stel.cli_services.context import ConfigClickError
from stel.cli_services.grants import (
    GrantsReport,
    grant,
    grant_history,
    list_grants,
    revoke,
)
from stel.mcp_server.authorization import PolicyAttribute, Principal
from stel.mcp_server.grants import (
    OPERATOR_BETWEEN,
    OPERATOR_EQUAL,
    WAREHOUSE_IDENTITY_ATTRIBUTE,
    Grant,
    GrantAuthorizationProvider,
    GrantConfigurationError,
    GrantWarehouseIdentityResolver,
    StaticGrantStore,
    parse_interval,
)
from stel.search import SearchFilter, SearchFilterOperator

TARGET = "dev"
SUBJECT = "analyst@example.com"
DATE = PolicyAttribute("filing_date", "date")


def _provider(*grants: Grant) -> GrantAuthorizationProvider:
    return GrantAuthorizationProvider(StaticGrantStore(grants))


def _between(value: str, attribute: str = "filing_date") -> Grant:
    return Grant(SUBJECT, attribute, value, OPERATOR_BETWEEN)


def _filters(
    *grants: Grant, attribute: PolicyAttribute = DATE
) -> tuple[SearchFilter, ...]:
    return _provider(*grants).search_policy_filters(
        Principal(subject_id=SUBJECT), access="governed", attributes=[attribute]
    )


# ─── compiling an interval ──────────────────────────────────────────────────


def test_a_closed_interval_compiles_to_a_bounded_pair() -> None:
    filters = _filters(_between("2024-01-01/2025-12-31"))
    assert [(f.operator, f.value) for f in filters] == [
        (SearchFilterOperator.GREATER_THAN_OR_EQUAL, "2024-01-01"),
        (SearchFilterOperator.LESS_THAN_OR_EQUAL, "2025-12-31"),
    ]


def test_an_open_upper_bound_emits_only_a_lower_filter() -> None:
    # No sentinel upper bound: the store sees exactly the constraint written.
    filters = _filters(_between("2024-01-01/.."))
    assert [(f.operator, f.value) for f in filters] == [
        (SearchFilterOperator.GREATER_THAN_OR_EQUAL, "2024-01-01")
    ]


def test_an_open_lower_bound_emits_only_an_upper_filter() -> None:
    filters = _filters(_between("../2025-12-31"))
    assert [(f.operator, f.value) for f in filters] == [
        (SearchFilterOperator.LESS_THAN_OR_EQUAL, "2025-12-31")
    ]


def test_equality_grants_are_unchanged_by_the_operator_column() -> None:
    filters = _filters(Grant(SUBJECT, "filing_date", "2024-01-01"))
    assert filters[0].operator is SearchFilterOperator.EQUAL
    assert filters[0].value == "2024-01-01"


# ─── the recheck has to know the same rule ──────────────────────────────────


def _can_read(value: str, *grants: Grant) -> bool:
    return _provider(*grants).can_read(
        Principal(subject_id=SUBJECT),
        {"authorization_resolved": True, "filing_date": value},
        attributes=[DATE],
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-06-01", True),
        ("2024-01-01", True),  # lower bound is inclusive
        ("2025-12-31", True),  # upper bound is inclusive
        ("2023-12-31", False),
        ("2026-01-01", False),
    ],
)
def test_the_recheck_applies_the_interval(value: str, expected: bool) -> None:
    """The second look must understand ranges too.

    It exists so a store that ignored a filter cannot leak a row. A recheck
    that only understood equality would admit every row a range filter was
    supposed to exclude.
    """
    assert _can_read(value, _between("2024-01-01/2025-12-31")) is expected


def test_the_recheck_refuses_a_row_with_no_value() -> None:
    assert (
        _provider(_between("2024-01-01/.."))
        .can_read(
            Principal(subject_id=SUBJECT),
            {"authorization_resolved": True},
            attributes=[DATE],
        )
        is False
    )


# ─── what cannot be expressed, and is refused rather than guessed ───────────


def test_two_intervals_for_one_attribute_are_a_configuration_error() -> None:
    # Filters AND, so these would narrow to their overlap rather than permit
    # either -- the opposite of how an operator reads two grants.
    with pytest.raises(GrantConfigurationError, match="more than one interval"):
        _filters(_between("2024-01-01/2024-12-31"), _between("2020-01-01/2020-12-31"))


def test_an_interval_beside_a_literal_is_a_configuration_error() -> None:
    with pytest.raises(GrantConfigurationError, match="both an interval and a literal"):
        _filters(_between("2024-01-01/.."), Grant(SUBJECT, "filing_date", "2019-01-01"))


def test_an_interval_on_an_array_attribute_is_refused() -> None:
    # A set of groups has no order for an interval to cut.
    with pytest.raises(GrantConfigurationError, match="array"):
        _filters(
            _between("a/b", attribute="access_groups"),
            attribute=PolicyAttribute("access_groups", "array[string]"),
        )


def test_an_interval_on_the_warehouse_identity_is_refused() -> None:
    # Resolving it to a bound would pick a principal nobody wrote down.
    resolver = GrantWarehouseIdentityResolver(
        StaticGrantStore(
            [Grant(SUBJECT, WAREHOUSE_IDENTITY_ATTRIBUTE, "a/b", OPERATOR_BETWEEN)]
        )
    )
    with pytest.raises(GrantConfigurationError, match="interval"):
        resolver.identity_for(Principal(subject_id=SUBJECT))


def test_an_interval_and_a_literal_on_different_attributes_coexist() -> None:
    # The refusal is per attribute, not global.
    filters = _provider(
        _between("2024-01-01/.."), Grant(SUBJECT, "tenant_id", "acme")
    ).search_policy_filters(
        Principal(subject_id=SUBJECT),
        access="governed",
        attributes=[DATE, PolicyAttribute("tenant_id", "string")],
    )
    assert len(filters) == 2


# ─── parsing ────────────────────────────────────────────────────────────────


def test_a_value_with_no_separator_is_refused() -> None:
    with pytest.raises(GrantConfigurationError, match="not an interval"):
        parse_interval("2024-01-01", attribute="filing_date")


def test_an_interval_open_at_both_ends_is_refused() -> None:
    # Permits everything, which nobody writes on purpose and which would be
    # indistinguishable from a correctly bounded grant in a listing.
    with pytest.raises(GrantConfigurationError, match="open at both ends"):
        parse_interval("../..", attribute="filing_date")


def test_an_inverted_interval_is_refused() -> None:
    with pytest.raises(GrantConfigurationError, match="lower bound"):
        parse_interval("2025-01-01/2024-01-01", attribute="filing_date")


def test_an_unknown_operator_is_refused_rather_than_defaulted() -> None:
    """A typo must not silently become an equality.

    `betwen` defaulted to `eq` would compile a literal equality against
    '2024-01-01/2025-12-31', which matches no row -- a silent denial that
    looks exactly like a correct empty grant set.
    """
    from stel.mcp_server.grants import _grant_from_row

    with pytest.raises(GrantConfigurationError, match="operator"):
        _grant_from_row(
            {
                "subject_id": SUBJECT,
                "attribute": "filing_date",
                "value": "2024-01-01/2025-12-31",
                "operator": "betwen",
            },
            "stel_grants",
        )


@pytest.mark.parametrize("absent", [{}, {"operator": None}, {"operator": "  "}])
def test_an_absent_operator_reads_as_equality(absent: dict[str, object]) -> None:
    # Every row written before #582 has no operator, and meant eq.
    from stel.mcp_server.grants import _grant_from_row

    parsed = _grant_from_row(
        {
            "subject_id": SUBJECT,
            "attribute": "tenant_id",
            "value": "acme",
            **absent,
        },
        "stel_grants",
    )
    assert parsed.operator == OPERATOR_EQUAL


# ─── the write path ─────────────────────────────────────────────────────────


@pytest.fixture
def project(tmp_path: Path, example_project_dir: Path) -> Path:
    destination = tmp_path / "proj"
    shutil.copytree(
        example_project_dir,
        destination,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    return destination


def _grant_interval(project_dir: Path, value: str) -> GrantsReport:
    return grant(
        project_dir,
        profiles_dir=None,
        target=TARGET,
        subject=SUBJECT,
        attribute="filing_date",
        interval=value,
    )


def test_an_interval_round_trips_through_the_relation(project: Path) -> None:
    _grant_interval(project, "2024-01-01/2025-12-31")
    rows = list_grants(project, profiles_dir=None, target=TARGET).rows
    assert len(rows) == 1
    assert rows[0].operator == OPERATOR_BETWEEN
    assert rows[0].value == "2024-01-01/2025-12-31"


def test_a_malformed_interval_is_refused_at_write_time(project: Path) -> None:
    # Validated here rather than at query time: an entitlement that fails on
    # the serving path surfaces as a refused caller with no obvious cause.
    with pytest.raises(ConfigClickError, match="not an interval"):
        _grant_interval(project, "2024-01-01")


def test_granting_a_value_and_an_interval_together_is_refused(project: Path) -> None:
    with pytest.raises(ConfigClickError, match="exactly one"):
        grant(
            project,
            profiles_dir=None,
            target=TARGET,
            subject=SUBJECT,
            attribute="filing_date",
            value="2024-01-01",
            interval="2024-01-01/..",
        )


def test_granting_neither_is_refused(project: Path) -> None:
    with pytest.raises(ConfigClickError, match="exactly one"):
        grant(
            project,
            profiles_dir=None,
            target=TARGET,
            subject=SUBJECT,
            attribute="filing_date",
        )


def test_granting_the_same_interval_twice_writes_one_row(project: Path) -> None:
    _grant_interval(project, "2024-01-01/..")
    report = _grant_interval(project, "2024-01-01/..")
    assert report.rows_affected == 0
    assert len(list_grants(project, profiles_dir=None, target=TARGET).rows) == 1


def test_an_interval_and_a_literal_of_the_same_text_are_distinct_rows(
    project: Path,
) -> None:
    # The operator is part of a grant's identity: the same text means
    # different things under eq and between, so one must not suppress the
    # other's insert.
    grant(
        project,
        profiles_dir=None,
        target=TARGET,
        subject=SUBJECT,
        attribute="tenant_id",
        value="a/b",
    )
    grant(
        project,
        profiles_dir=None,
        target=TARGET,
        subject=SUBJECT,
        attribute="tenant_id",
        interval="a/b",
    )
    assert len(list_grants(project, profiles_dir=None, target=TARGET).rows) == 2


def test_an_interval_grant_is_recorded_distinctly_in_the_history(
    project: Path,
) -> None:
    # "granted 2024" and "granted everything from 2024 onward" are different
    # entitlements, and the value alone does not say which.
    _grant_interval(project, "2024-01-01/..")
    entries = grant_history(project, profiles_dir=None, target=TARGET).entries
    assert [e.action for e in entries] == ["grant_between"]


def test_an_interval_is_revoked_by_its_written_form(project: Path) -> None:
    _grant_interval(project, "2024-01-01/2025-12-31")
    report = revoke(
        project,
        profiles_dir=None,
        target=TARGET,
        subject=SUBJECT,
        attribute="filing_date",
        value="2024-01-01/2025-12-31",
    )
    assert report.rows_affected == 1


# ─── upgrading a relation written before #582 ───────────────────────────────


def test_a_three_column_relation_is_widened_and_its_rows_still_read(
    project: Path,
) -> None:
    """The grants relation is persisted user data; widening must not lose it.

    `CREATE TABLE IF NOT EXISTS` does nothing to an existing table, so without
    an explicit ALTER every statement naming `operator` would fail against a
    relation written by an earlier stel.
    """
    from stel.adapters import create_adapter
    from stel.config import load_project
    from stel.profile import resolve_profile

    project_config, _sources, _models = load_project(project)
    resolved = resolve_profile(
        project_config, project, target=TARGET, profiles_dir=None
    )
    with create_adapter(resolved.warehouse, project_dir=project) as adapter:
        table = f"{adapter.schema_ref}.{adapter.quote_ident('stel_grants')}"
        adapter.execute(
            f"CREATE TABLE {table} (subject_id STRING NOT NULL, "
            "attribute STRING NOT NULL, value STRING NOT NULL)"
        )
        adapter.execute(
            f"INSERT INTO {table} VALUES (?, ?, ?)", [SUBJECT, "tenant_id", "acme"]
        )

    rows = list_grants(project, profiles_dir=None, target=TARGET).rows
    assert len(rows) == 1
    # Null operator, and it means what it always meant.
    assert rows[0].value == "acme"
    assert rows[0].operator == OPERATOR_EQUAL

    # And the widened relation now takes an interval.
    _grant_interval(project, "2024-01-01/..")
    assert len(list_grants(project, profiles_dir=None, target=TARGET).rows) == 2
