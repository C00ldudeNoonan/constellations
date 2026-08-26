"""Operator-owned grants as the authorization source (issue #392)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from stel.adapters.base import ReadPredicate
from stel.mcp_server.authorization import (
    AuthorizationError,
    ClaimAuthorizationProvider,
    PolicyAttribute,
    Principal,
)
from stel.mcp_server.grants import (
    Grant,
    GrantAuthorizationProvider,
    GrantConfigurationError,
    StaticGrantStore,
    WarehouseGrantStore,
)
from stel.mcp_server.service import ContextService, _authorizing_tenant
from stel.search import SearchFilterOperator

TENANT = PolicyAttribute("tenant_id", "string")
GROUP = PolicyAttribute("access_group", "string")


def _principal(**overrides: Any) -> Principal:
    payload: dict[str, Any] = {
        "subject_id": "alice",
        "tenant_id": "acme",
        "access_groups": ("analysts",),
    }
    payload.update(overrides)
    return Principal(**payload)


# ─── the property this exists for ───────────────────────────────────────────


def test_the_caller_cannot_grant_themselves_a_tenant() -> None:
    """The whole point. Claims arrive with the request; grants do not.

    With claim-derived policy, a caller who can set `X-Stel-Tenant-Id` picks
    their own tenant. Here the header is not consulted — only `subject_id`,
    which is what the transport authenticated.
    """
    store = StaticGrantStore([Grant("alice", "tenant_id", "acme")])
    provider = GrantAuthorizationProvider(store)

    forged = _principal(tenant_id="globex", policy_claims={"tenant_id": "globex"})
    (filter_,) = provider.search_policy_filters(
        forged, access="governed", attributes=[TENANT]
    )

    assert filter_.value == "acme"


def test_a_subject_with_no_grants_is_refused_not_defaulted() -> None:
    """No grant is not 'no restriction'. Falling open here would make an
    unknown caller the most privileged one."""
    provider = GrantAuthorizationProvider(StaticGrantStore([]))

    with pytest.raises(AuthorizationError, match="no grant"):
        provider.search_policy_filters(
            _principal(), access="governed", attributes=[TENANT]
        )


def test_a_row_outside_the_grant_is_refused_on_recheck() -> None:
    """The second look. If a store ignored the filter, the row still cannot
    escape — the same grants decide both times."""
    provider = GrantAuthorizationProvider(
        StaticGrantStore([Grant("alice", "tenant_id", "acme")])
    )

    assert provider.can_read(
        _principal(),
        {"authorization_resolved": True, "tenant_id": "acme"},
        attributes=[TENANT],
    )
    assert not provider.can_read(
        _principal(),
        {"authorization_resolved": True, "tenant_id": "globex"},
        attributes=[TENANT],
    )


def test_a_row_missing_the_attribute_is_refused() -> None:
    """An absent value is not a public one."""
    provider = GrantAuthorizationProvider(
        StaticGrantStore([Grant("alice", "tenant_id", "acme")])
    )

    assert not provider.can_read(
        _principal(), {"authorization_resolved": True}, attributes=[TENANT]
    )


def test_an_unresolved_row_is_refused_however_the_grants_read() -> None:
    provider = GrantAuthorizationProvider(
        StaticGrantStore([Grant("alice", "tenant_id", "acme")])
    )

    assert not provider.can_read(
        _principal(), {"tenant_id": "acme"}, attributes=[TENANT]
    )


# ─── filter compilation ─────────────────────────────────────────────────────


def test_several_grants_for_one_attribute_compile_to_IN() -> None:
    provider = GrantAuthorizationProvider(
        StaticGrantStore(
            [Grant("alice", "access_group", "analysts"), Grant("alice", "access_group", "admins")]
        )
    )

    (filter_,) = provider.search_policy_filters(
        _principal(), access="governed", attributes=[GROUP]
    )

    assert filter_.operator is SearchFilterOperator.IN
    assert filter_.value == ("analysts", "admins")


def test_every_required_attribute_must_be_granted() -> None:
    """Partial grants must not compile to a partial filter: one unfiltered
    attribute is an unfiltered read of that dimension."""
    provider = GrantAuthorizationProvider(
        StaticGrantStore([Grant("alice", "tenant_id", "acme")])
    )

    with pytest.raises(AuthorizationError, match="no grant"):
        provider.search_policy_filters(
            _principal(), access="governed", attributes=[TENANT, GROUP]
        )


def test_public_access_needs_no_grant() -> None:
    provider = GrantAuthorizationProvider(StaticGrantStore([]))
    assert (
        provider.search_policy_filters(_principal(), access="public", attributes=[TENANT])
        == ()
    )


# ─── the warehouse-backed store ─────────────────────────────────────────────


class _FakeRepository:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._rows = rows
        self.reads: list[tuple[str, Sequence[ReadPredicate]]] = []

    def read_rows(
        self,
        relation: str,
        *,
        predicates: Sequence[ReadPredicate],
        max_rows: int,
        columns: Sequence[str] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        self.reads.append((relation, predicates))
        subject = predicates[0].value
        return tuple(row for row in self._rows if row["subject_id"] == subject)


def _rows() -> list[dict[str, str]]:
    return [
        {"subject_id": "alice", "attribute": "tenant_id", "value": "acme"},
        {"subject_id": "bob", "attribute": "tenant_id", "value": "globex"},
    ]


def test_grants_are_read_per_subject_and_scoped_to_that_subject() -> None:
    repository = _FakeRepository(_rows())
    store = WarehouseGrantStore(repository, relation="ops.grants")

    assert store.grants_for("alice") == (Grant("alice", "tenant_id", "acme"),)
    assert store.grants_for("bob") == (Grant("bob", "tenant_id", "globex"),)
    # Scoped in the query, not filtered after: an unscoped read would pull
    # every tenant's grants into the request path.
    assert all(read[1][0].column == "subject_id" for read in repository.reads)


def test_grants_are_cached_and_the_ttl_bounds_revocation() -> None:
    """The cache is a revocation delay, so the test states it as one."""
    now = {"t": 0.0}
    repository = _FakeRepository(_rows())
    store = WarehouseGrantStore(
        repository, relation="ops.grants", ttl_seconds=60.0, clock=lambda: now["t"]
    )

    assert store.grants_for("alice")
    assert store.grants_for("alice")
    assert len(repository.reads) == 1, "a second call inside the TTL must not re-read"

    repository._rows = []  # grant revoked in the warehouse
    now["t"] = 30.0
    assert store.grants_for("alice"), "still cached, still granted"

    now["t"] = 61.0
    assert store.grants_for("alice") == (), "revoked once the TTL lapses"


def test_a_blank_grant_value_is_refused_rather_than_guessed() -> None:
    """Blank is ambiguous between 'no grant' and 'grant everything', and one
    of those readings is a breach."""
    repository = _FakeRepository(
        [{"subject_id": "alice", "attribute": "tenant_id", "value": "  "}]
    )
    store = WarehouseGrantStore(repository, relation="ops.grants")

    with pytest.raises(GrantConfigurationError, match="no usable"):
        store.grants_for("alice")


def test_a_grant_relation_missing_a_column_fails_loudly() -> None:
    """Silently granting nothing would be indistinguishable from a correct
    empty grant set, so a drifted relation has to say so."""
    repository = _FakeRepository(
        [{"subject_id": "alice", "attribute": "tenant_id"}]
    )
    store = WarehouseGrantStore(repository, relation="ops.grants")

    with pytest.raises(GrantConfigurationError, match="no usable 'value'"):
        store.grants_for("alice")


def test_from_project_refuses_both_authorization_and_grants_relation(
    tmp_path: Path,
) -> None:
    """Both supplied is ambiguous, and guessing would silently pick a policy.

    The check must fire before any project or warehouse work, so the operator
    sees the misconfiguration rather than a downstream failure.
    """
    with pytest.raises(ValueError, match="not both"):
        ContextService.from_project(
            tmp_path / "no-such-project",
            authorization=ClaimAuthorizationProvider(),
            grants_relation="ops.grants",
        )


# ─── review follow-ups (PR #396) ────────────────────────────────────────────


def test_malformed_grant_row_is_a_configuration_error_not_a_denial() -> None:
    """Schema drift must not masquerade as "this subject has no grants".

    `AuthorizationError` is caught by the service as an ordinary denial, so
    raising it here would show the operator an empty catalog with no reason.
    """

    class _BlankValueRepository:
        def read_rows(
            self,
            relation: str,
            *,
            predicates: Sequence[ReadPredicate],
            max_rows: int,
            columns: Sequence[str] | None = None,
        ) -> tuple[Mapping[str, Any], ...]:
            return ({"subject_id": "alice", "attribute": "tenant_id", "value": "  "},)

    store = WarehouseGrantStore(_BlankValueRepository(), relation="ops.grants")

    with pytest.raises(GrantConfigurationError):
        store.grants_for("alice")

    # The distinction is the whole point: a denial handler must not swallow it.
    assert not issubclass(GrantConfigurationError, AuthorizationError)


def test_audit_log_tenant_comes_from_grants_not_the_forged_claim() -> None:
    """A caller who asserts another tenant must not have their served query
    filed under it — audit trails are read exactly when that is suspected."""
    provider = GrantAuthorizationProvider(
        StaticGrantStore([Grant("alice", "tenant_id", "acme")])
    )

    filters = provider.search_policy_filters(
        _principal(tenant_id="globex"), access="governed", attributes=[TENANT]
    )

    assert _authorizing_tenant(filters) == "acme"


def test_authorizing_tenant_is_none_when_no_single_honest_value() -> None:
    provider = GrantAuthorizationProvider(
        StaticGrantStore(
            [Grant("alice", "tenant_id", "acme"), Grant("alice", "tenant_id", "globex")]
        )
    )

    filters = provider.search_policy_filters(
        _principal(), access="governed", attributes=[TENANT]
    )

    assert _authorizing_tenant(filters) is None
    assert _authorizing_tenant(()) is None
