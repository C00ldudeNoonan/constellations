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
from threading import Lock
from typing import Any, Protocol

from ..adapters.base import (
    OPERATOR_IDENTITY,
    ReadPredicate,
    ReadPredicateOperator,
    WarehouseIdentity,
)
from ..search import SearchFilter, SearchFilterOperator
from .authorization import (
    AuthorizationError,
    PolicyAttribute,
    Principal,
    policy_values_overlap,
)

# Columns the grants relation must provide. Named rather than inferred so a
# relation that drifted fails with the missing column instead of silently
# granting nothing — which would read as "this caller may see nothing" and be
# indistinguishable from a correct empty grant set.
SUBJECT_COLUMN = "subject_id"
ATTRIBUTE_COLUMN = "attribute"
VALUE_COLUMN = "value"
GRANT_COLUMNS = (SUBJECT_COLUMN, ATTRIBUTE_COLUMN, VALUE_COLUMN)

# How to read `value` (issue #582). Optional: a relation written before this
# existed has no such column, and every row in it means `eq` -- which is what
# it already meant, so an older relation keeps working untouched rather than
# failing as drift. That is also why the read below asks for every column
# instead of projecting `GRANT_COLUMNS`: a projection naming `operator` would
# turn a three-column relation into a configuration error on upgrade.
OPERATOR_COLUMN = "operator"
OPERATOR_EQUAL = "eq"
OPERATOR_BETWEEN = "between"
GRANT_OPERATORS = (OPERATOR_EQUAL, OPERATOR_BETWEEN)

# ISO 8601 interval notation, and ISO 8601-2's open-ended marker.
INTERVAL_SEPARATOR = "/"
INTERVAL_OPEN = ".."

# Reserved: names the warehouse principal a subject's governed reads execute
# as, rather than a value to filter rows by (issue #395, ADR-0010). It lives
# in this relation so it inherits operator ownership, the TTL that bounds
# revocation, the per-subject cache and its sweep — but it is not a policy
# attribute, and a context model that declares one by this name is refused
# rather than quietly given a second meaning.
WAREHOUSE_IDENTITY_ATTRIBUTE = "warehouse_identity"

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
    """One value a subject is permitted for one policy attribute.

    `operator` says how to read `value`: `eq` for the literal it has always
    been, `between` for a closed interval written `<lower>/<upper>`.
    """

    subject_id: str
    attribute: str
    value: str
    operator: str = OPERATOR_EQUAL


@dataclass(frozen=True, slots=True)
class GrantInterval:
    """A closed interval a `between` grant permits.

    Either bound may be absent, meaning unbounded on that side -- written
    `..` so an open end is something the operator typed rather than something
    they forgot.
    """

    lower: str | None
    upper: str | None

    def contains(self, value: str) -> bool:
        """Whether a row's value falls inside.

        String comparison, which is correct for the types this can be used on:
        ISO dates and timestamps sort lexicographically, and the filter the
        store receives compares the same strings the same way. A numeric
        policy attribute would not sort correctly here, which is why
        `parse_interval` refuses anything that is not an ISO-shaped value.
        """
        if self.lower is not None and value < self.lower:
            return False
        return not (self.upper is not None and value > self.upper)


def parse_interval(value: str, *, attribute: str) -> GrantInterval:
    """Read `<lower>/<upper>` into bounds, refusing anything ambiguous.

    Refusals rather than best guesses, because this decides what a caller may
    read: a malformed interval that silently became an unbounded one would be
    an over-grant, and an over-grant that looks like a narrowing is the
    failure mode this whole module exists to remove.
    """
    parts = value.split(INTERVAL_SEPARATOR)
    if len(parts) != 2:
        raise GrantConfigurationError(
            f"Grant for '{attribute}' has operator '{OPERATOR_BETWEEN}' but "
            f"its value {value!r} is not an interval. Write it as "
            f"'<lower>{INTERVAL_SEPARATOR}<upper>', using "
            f"'{INTERVAL_OPEN}' for an open end."
        )
    lower = None if parts[0].strip() in {INTERVAL_OPEN, ""} else parts[0].strip()
    upper = None if parts[1].strip() in {INTERVAL_OPEN, ""} else parts[1].strip()
    if lower is None and upper is None:
        # `../..` permits everything, which nobody writes on purpose and which
        # would be indistinguishable from a correctly bounded grant in a list.
        raise GrantConfigurationError(
            f"Grant for '{attribute}' is open at both ends, which permits "
            "every value. Bound at least one side, or grant the attribute "
            "without an interval if that is really the intent."
        )
    if lower is not None and upper is not None and lower > upper:
        raise GrantConfigurationError(
            f"Grant for '{attribute}' has an interval whose lower bound "
            f"{lower!r} is above its upper bound {upper!r}, which permits "
            "nothing. Swap them."
        )
    return GrantInterval(lower, upper)


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
        identity: WarehouseIdentity,
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

    The cache is swept on the same cadence, so a network deployment serving
    many distinct subjects over time does not grow this table forever: an
    entry past its TTL is reclaimed rather than merely ignored. Without this,
    every subject a token issuer or a proxy ever named would hold memory for
    the life of the process — the same unbounded-growth shape the
    per-principal rate-limit table was fixed for (issue #466), here in the
    grants cache instead.
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
        self._lock = Lock()
        self._last_sweep = clock()

    def grants_for(self, subject_id: str) -> tuple[Grant, ...]:
        now = self._clock()
        with self._lock:
            cached = self._cache.get(subject_id)
            if cached is not None and now < cached[0]:
                return cached[1]
        # The warehouse read happens outside the lock: concurrent requests for
        # different subjects (the common case under real load) must not
        # serialize on each other's I/O. A cache miss two callers hit at once
        # simply reads twice, same as before this lock existed.
        grants = self._read(subject_id)
        with self._lock:
            self._cache[subject_id] = (now + self._ttl, grants)
            self._sweep_expired(now)
        return grants

    def _sweep_expired(self, now: float) -> None:
        # Called under the lock. One TTL between sweeps, matching the cache's
        # own revocation cadence rather than adding an unrelated interval.
        if now - self._last_sweep < self._ttl:
            return
        self._last_sweep = now
        expired = [
            subject_id
            for subject_id, (expires_at, _grants) in self._cache.items()
            if expires_at <= now
        ]
        for subject_id in expired:
            del self._cache[subject_id]

    def _read(self, subject_id: str) -> tuple[Grant, ...]:
        rows = self._repository.read_rows(
            self._relation,
            # The grants relation is operator-owned, and reading it is what
            # resolves a caller's identity -- so it necessarily precedes one
            # and cannot be read under one (ADR-0010).
            identity=OPERATOR_IDENTITY,
            predicates=[
                ReadPredicate(SUBJECT_COLUMN, ReadPredicateOperator.EQUAL, subject_id)
            ],
            max_rows=MAX_GRANT_ROWS,
            # Every column, not a projection (issue #582). `operator` is
            # optional, and naming it here would make a relation written
            # before it existed fail as drift the moment stel was upgraded.
            # The three required columns are still validated, by
            # `_grant_from_row` rather than by the projection.
            columns=None,
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
    return Grant(*values, operator=_operator_from_row(row, relation))


def _operator_from_row(row: Mapping[str, Any], relation: str) -> str:
    """How to read this row's value.

    Absent or null means `eq`, which is what every row meant before #582 --
    so a relation written by an earlier stel, or a column left null by a
    hand-written insert, keeps its existing meaning instead of failing.

    An operator that is present but unrecognised is refused rather than
    defaulted to `eq`. Defaulting would turn a typo like `betwen` into a
    literal equality against `2024-01-01/2025-12-31`, which matches no row --
    a silent denial that looks exactly like a correct empty grant set.
    """
    value = row.get(OPERATOR_COLUMN)
    if value is None or (isinstance(value, str) and not value.strip()):
        return OPERATOR_EQUAL
    if not isinstance(value, str) or value.strip() not in GRANT_OPERATORS:
        raise GrantConfigurationError(
            f"Grant relation '{relation}' has a row whose "
            f"'{OPERATOR_COLUMN}' is {value!r}. Expected one of "
            f"{', '.join(GRANT_OPERATORS)}."
        )
    return value.strip()


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
        _refuse_reserved_attributes(attributes)
        granted = _granted_values(self._store, principal.subject_id)
        filters: list[SearchFilter] = []
        for attribute in attributes:
            held = granted.get(attribute.name)
            if held is None:
                raise AuthorizationError(
                    "The caller has no grant for every required policy attribute"
                )
            if held.interval is not None:
                if attribute.data_type == "array[string]":
                    raise GrantConfigurationError(
                        f"Grant for '{attribute.name}' is an interval, but the "
                        "context model declares it as array[string]. An "
                        "interval orders values; a set of groups has no order."
                    )
                # Two filters that AND together, which is what a closed
                # interval is. An unbounded side contributes no filter rather
                # than a sentinel bound, so the store sees exactly the
                # constraint the operator wrote.
                filters.extend(_interval_filters(attribute.name, held.interval))
                continue
            values = held.values
            if attribute.data_type == "array[string]":
                # Several grant rows for one attribute already collect into a
                # set, which is exactly the shape an overlap test wants
                # (issue #397).
                filters.append(
                    SearchFilter(
                        attribute.name,
                        SearchFilterOperator.ARRAY_CONTAINS_ANY,
                        values,
                    )
                )
                continue
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
        _refuse_reserved_attributes(attributes)
        granted = _granted_values(self._store, principal.subject_id)
        for attribute in attributes:
            held = granted.get(attribute.name)
            if held is None:
                return False
            value = row.get(attribute.name)
            if held.interval is not None:
                # The same rule the filter expressed, applied again here --
                # otherwise a store that ignored a range filter would have its
                # rows admitted by a recheck that only understood equality,
                # which is the leak this second look exists to stop.
                if attribute.data_type == "array[string]":
                    return False
                if not isinstance(value, str) or not held.interval.contains(value):
                    return False
                continue
            allowed = held.values
            if attribute.data_type == "array[string]":
                if not policy_values_overlap(value, allowed):
                    return False
                continue
            if not isinstance(value, str) or value not in allowed:
                return False
        return True


@dataclass(frozen=True, slots=True)
class _AttributeGrant:
    """What a subject holds for one attribute: values, or one interval."""

    values: tuple[str, ...]
    interval: GrantInterval | None


def _granted_values(store: GrantStore, subject_id: str) -> dict[str, _AttributeGrant]:
    """Collect a subject's grants per attribute.

    Several `eq` rows for one attribute still mean OR, compiled to `IN` --
    more grants, more access, the invariant this module has always had.

    An interval cannot join that. The filters a search receives are AND-ed by
    every store (`" AND ".join(clauses)` in both `duckdb.py` and
    `lancedb.py`), so two intervals for one attribute cannot express the union
    an operator would read them as, and an interval beside an equality cannot
    either. Both are refused as configuration errors rather than resolved to
    one reading: an entitlement that silently means something other than what
    was written is the failure this layer exists to remove, and
    `GrantConfigurationError` is how this module already distinguishes "the
    relation is wrong" from "this caller may see nothing" (issue #582).
    """
    values: dict[str, tuple[str, ...]] = {}
    intervals: dict[str, GrantInterval] = {}
    for grant in store.grants_for(subject_id):
        if grant.operator == OPERATOR_BETWEEN:
            if grant.attribute in intervals:
                raise GrantConfigurationError(
                    f"Subject holds more than one interval for "
                    f"'{grant.attribute}'. Search filters are combined with "
                    "AND, so two intervals would narrow to their overlap "
                    "rather than permit either. Leave exactly one, widening "
                    "its bounds if both were meant."
                )
            intervals[grant.attribute] = parse_interval(
                grant.value, attribute=grant.attribute
            )
            continue
        values[grant.attribute] = (*values.get(grant.attribute, ()), grant.value)
    for attribute in intervals:
        if attribute in values:
            raise GrantConfigurationError(
                f"Subject holds both an interval and a literal value for "
                f"'{attribute}'. Those cannot be combined: filters are AND-ed, "
                "so the result would be neither the union nor what either row "
                "says on its own. Keep one kind of grant per attribute."
            )
    return {
        attribute: _AttributeGrant(values.get(attribute, ()), intervals.get(attribute))
        for attribute in (*values, *intervals)
    }


def _interval_filters(
    attribute: str, interval: GrantInterval
) -> tuple[SearchFilter, ...]:
    filters: list[SearchFilter] = []
    if interval.lower is not None:
        filters.append(
            SearchFilter(
                attribute, SearchFilterOperator.GREATER_THAN_OR_EQUAL, interval.lower
            )
        )
    if interval.upper is not None:
        filters.append(
            SearchFilter(
                attribute, SearchFilterOperator.LESS_THAN_OR_EQUAL, interval.upper
            )
        )
    return tuple(filters)


def _refuse_reserved_attributes(attributes: Sequence[PolicyAttribute]) -> None:
    """A context model may not declare the reserved identity attribute.

    `warehouse_identity` grants say who to connect as; treating one as a row
    filter would silently give the same grant two meanings, and an operator
    revoking a filter value would be changing a connection identity without
    knowing it (ADR-0010).
    """
    for attribute in attributes:
        if attribute.name == WAREHOUSE_IDENTITY_ATTRIBUTE:
            raise GrantConfigurationError(
                f"'{WAREHOUSE_IDENTITY_ATTRIBUTE}' is reserved: it names the "
                "warehouse principal a read executes as, so it cannot also be "
                "a policy attribute. Rename the context model's attribute."
            )


class WarehouseIdentityResolver(Protocol):
    def identity_for(self, principal: Principal) -> WarehouseIdentity: ...


@dataclass(frozen=True, slots=True)
class OperatorWarehouseIdentityResolver:
    """Every read runs as the operator — the default, and what stel has always
    done. Deployments that have not opted into identity-scoped serving keep
    exactly today's behaviour, including stdio and single-tenant use."""

    def identity_for(self, principal: Principal) -> WarehouseIdentity:
        return OPERATOR_IDENTITY


class GrantWarehouseIdentityResolver:
    """The warehouse principal a subject's governed reads execute as (#395).

    Read from the reserved `warehouse_identity` grant, so the mapping is
    operator-owned and revocable on the same TTL as every other grant. Two
    outcomes are not "pick one":

    * **No grant is a denial.** There is no fallback to the operator's
      credentials — a missing row must never read as "unprotected", which is
      the silent-success failure this layer exists to remove.
    * **More than one is a configuration error, not a denial.** A subject may
      legitimately hold several `tenant_id` grants; it cannot legitimately
      execute as two principals at once. Reporting that as "denied" would
      leave the operator with no signal, exactly as `GrantConfigurationError`
      exists to avoid elsewhere in this module.
    """

    def __init__(self, store: GrantStore) -> None:
        self._store = store

    def identity_for(self, principal: Principal) -> WarehouseIdentity:
        held = _granted_values(self._store, principal.subject_id).get(
            WAREHOUSE_IDENTITY_ATTRIBUTE
        )
        if held is not None and held.interval is not None:
            # A principal name is not an ordered value, so an interval here is
            # always a mistake -- and resolving it to some bound would pick a
            # warehouse identity nobody wrote down.
            raise GrantConfigurationError(
                f"'{WAREHOUSE_IDENTITY_ATTRIBUTE}' is granted as an interval. "
                "It names the principal a read connects as, not a range of "
                "them; grant it a single value."
            )
        values = () if held is None else held.values
        if not values:
            raise AuthorizationError(
                "The caller has no warehouse identity grant"
            )
        if len(set(values)) > 1:
            raise GrantConfigurationError(
                f"Subject has {len(set(values))} '{WAREHOUSE_IDENTITY_ATTRIBUTE}' "
                "grants and can execute as only one principal. Leave exactly "
                "one row for this subject."
            )
        return WarehouseIdentity(values[0])
