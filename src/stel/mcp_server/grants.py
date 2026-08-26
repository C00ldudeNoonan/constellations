"""Operator-owned grants as the authorization source (issue #392).

`ClaimAuthorizationProvider` derives a caller's policy values from the claims
the caller arrives with. Over stdio that is right — the operator sets the
environment and is the principal. Over a network it means **the policy is
whatever the transport stamped on the request**: change
`X-Stel-Access-Groups` and you change what you can read, and the only thing
standing between a caller and another tenant is a correctly configured proxy.

This module moves the answer to an operator-controlled store keyed by subject.
Groups and tenants are looked up, never carried. A forged access-group header
then buys nothing, because the header is not consulted — which is what turns
the trusted-proxy resolver from the whole security model into just an
authentication step.

**What is still true**: stel remains the enforcement point. A grant store
makes policy central and auditable; it does not make the warehouse refuse a
query stel should not have issued. Per-tenant credentials are the layer that
does that, and this composes with them rather than replacing them.
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..adapters.base import ReadPredicate, ReadPredicateOperator
from ..search import SearchFilter, SearchFilterOperator
from .authorization import AuthorizationError, PolicyAttribute, Principal

# Columns the grants relation must provide. Named rather than inferred so a
# relation that drifted fails with the missing column instead of silently
# granting nothing — which would read as "this caller may see nothing" and be
# indistinguishable from a correct empty grant set.
SUBJECT_COLUMN = "subject_id"
ATTRIBUTE_COLUMN = "attribute"
VALUE_COLUMN = "value"
GRANT_COLUMNS = (SUBJECT_COLUMN, ATTRIBUTE_COLUMN, VALUE_COLUMN)

# A grant read per request would put a warehouse round trip in the latency of
# every search. Grants change on human timescales, so a short TTL is the right
# trade — but it is a *ceiling on how long a revocation takes to take effect*,
# which is the number an operator actually needs to reason about.
DEFAULT_GRANT_TTL_SECONDS = 60.0

# Bound on rows read for one subject: a runaway grants relation should fail
# rather than pull an unbounded result into the request path.
MAX_GRANT_ROWS = 1000


class GrantConfigurationError(Exception):
    """The grants relation itself is malformed.

    Deliberately not an `AuthorizationError`. The service treats that as an
    ordinary denial — `not_found_or_denied` for one resource, a silent skip in
    `list_context_models` — which would render schema drift in the grants
    relation indistinguishable from a subject legitimately having no grants.
    The operator would see an empty catalog and no reason for it.
    """


@dataclass(frozen=True, slots=True)
class Grant:
    """One value a subject is permitted for one policy attribute."""

    subject_id: str
    attribute: str
    value: str


class GrantStore(Protocol):
    def grants_for(self, subject_id: str) -> tuple[Grant, ...]: ...


class GrantRowReader(Protocol):
    """The one repository capability a grant store needs.

    Narrower than `ContextRepository` on purpose: reading grants is a plain
    row read, and depending on the full repository protocol would make every
    unrelated method a prerequisite for supplying grants from somewhere else.
    """

    def read_rows(
        self,
        relation: str,
        *,
        predicates: Sequence[ReadPredicate],
        max_rows: int,
        columns: Sequence[str] | None = None,
    ) -> tuple[Mapping[str, Any], ...]: ...


class StaticGrantStore:
    """Grants fixed at construction, for tests and single-tenant deployments."""

    def __init__(self, grants: Sequence[Grant]) -> None:
        self._by_subject: dict[str, tuple[Grant, ...]] = {}
        for grant in grants:
            self._by_subject.setdefault(grant.subject_id, ())
            self._by_subject[grant.subject_id] += (grant,)

    def grants_for(self, subject_id: str) -> tuple[Grant, ...]:
        return self._by_subject.get(subject_id, ())


class WarehouseGrantStore:
    """Grants read from an operator-controlled warehouse relation.

    The warehouse is where stel already keeps operator-owned state, so grants
    are auditable, queryable, and changed by the same mechanisms as everything
    else — no new store, no new credential, no network hop.

    Cached per subject with a TTL. The cache is the revocation delay: a grant
    removed from the relation stops applying within `ttl_seconds`, and an
    operator who needs it immediate restarts the server.
    """

    def __init__(
        self,
        repository: GrantRowReader,
        *,
        relation: str,
        ttl_seconds: float = DEFAULT_GRANT_TTL_SECONDS,
        clock: Any = time.monotonic,
    ) -> None:
        if not relation:
            raise ValueError("grant relation must not be empty")
        if ttl_seconds <= 0:
            raise ValueError("grant ttl_seconds must be positive")
        self._repository = repository
        self._relation = relation
        self._ttl = ttl_seconds
        self._clock = clock
        self._cache: dict[str, tuple[float, tuple[Grant, ...]]] = {}

    def grants_for(self, subject_id: str) -> tuple[Grant, ...]:
        now = self._clock()
        cached = self._cache.get(subject_id)
        if cached is not None and now < cached[0]:
            return cached[1]
        grants = self._read(subject_id)
        self._cache[subject_id] = (now + self._ttl, grants)
        return grants

    def _read(self, subject_id: str) -> tuple[Grant, ...]:
        rows = self._repository.read_rows(
            self._relation,
            predicates=[
                ReadPredicate(SUBJECT_COLUMN, ReadPredicateOperator.EQUAL, subject_id)
            ],
            max_rows=MAX_GRANT_ROWS,
            columns=list(GRANT_COLUMNS),
        )
        return tuple(_grant_from_row(row, self._relation) for row in rows)


def _grant_from_row(row: Mapping[str, Any], relation: str) -> Grant:
    values = []
    for column in GRANT_COLUMNS:
        value = row.get(column)
        if not isinstance(value, str) or not value.strip():
            raise GrantConfigurationError(
                f"Grant relation '{relation}' has a row with no usable "
                f"'{column}'. Grants must be non-empty strings; a blank one is "
                "ambiguous between 'no grant' and 'grant everything'."
            )
        values.append(value.strip())
    return Grant(*values)


class GrantAuthorizationProvider:
    """Compile policy from operator-held grants rather than caller claims.

    Deliberately ignores `principal.policy_claims`, `access_groups`, and
    `tenant_id`. Those arrive with the request; consulting them would mean the
    caller still decides their own authorization, which is the property this
    exists to remove. The only thing taken from the principal is
    `subject_id` — who the transport authenticated.
    """

    def __init__(self, store: GrantStore) -> None:
        self._store = store

    def search_policy_filters(
        self,
        principal: Principal,
        *,
        access: str,
        attributes: Sequence[PolicyAttribute],
    ) -> tuple[SearchFilter, ...]:
        if access == "public":
            return ()
        granted = _granted_values(self._store, principal.subject_id)
        filters: list[SearchFilter] = []
        for attribute in attributes:
            if attribute.data_type == "array[string]":
                raise AuthorizationError(
                    "The active retrieval store cannot compile array-valued "
                    "policy claims"
                )
            values = granted.get(attribute.name, ())
            if not values:
                raise AuthorizationError(
                    "The caller has no grant for every required policy attribute"
                )
            filters.append(
                SearchFilter(
                    attribute.name,
                    SearchFilterOperator.EQUAL
                    if len(values) == 1
                    else SearchFilterOperator.IN,
                    values[0] if len(values) == 1 else values,
                )
            )
        if not filters:
            raise AuthorizationError(
                "The governed context model has no enforceable policy attributes"
            )
        return tuple(filters)

    def can_read(
        self,
        principal: Principal,
        row: Mapping[str, Any],
        *,
        attributes: Sequence[PolicyAttribute] = (),
    ) -> bool:
        """Recheck a returned row against the same grants.

        The search filters should already have excluded it; this is the second
        look the design asks for, so a store that ignored a filter cannot leak
        a row. A row that carries no value for a required attribute is refused
        rather than allowed — an absent value is not a public one.
        """
        if row.get("authorization_resolved") is not True:
            return False
        if row.get("is_public") is True:
            return True
        if not attributes:
            return False
        granted = _granted_values(self._store, principal.subject_id)
        for attribute in attributes:
            allowed = granted.get(attribute.name, ())
            value = row.get(attribute.name)
            if not isinstance(value, str) or value not in allowed:
                return False
        return True


def _granted_values(store: GrantStore, subject_id: str) -> dict[str, tuple[str, ...]]:
    values: dict[str, tuple[str, ...]] = {}
    for grant in store.grants_for(subject_id):
        values[grant.attribute] = (*values.get(grant.attribute, ()), grant.value)
    return values
