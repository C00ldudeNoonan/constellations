"""Array-valued policy attributes compile to a set-overlap filter (issue #397).

`access_groups` is documented as `array[string]` in the agent-context
contract, and until this landed no provider could compile it: governed models
declaring the documented shape failed closed and vanished from the catalog.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
import pytest

from stel.mcp_server.authorization import (
    AuthorizationError,
    ClaimAuthorizationProvider,
    PolicyAttribute,
    Principal,
)
from stel.mcp_server.grants import Grant, GrantAuthorizationProvider, StaticGrantStore
from stel.retrieval.base import (
    RetrievalError,
    RetrievalFeature,
    RetrievalPredicate,
    RetrievalPredicateOperator,
)
from stel.retrieval.lancedb import _compile_predicates
from stel.search import SearchFilter, SearchFilterOperator

GROUPS = PolicyAttribute("access_groups", "array[string]")
TENANT = PolicyAttribute("tenant_id", "string")


def _principal(**overrides: Any) -> Principal:
    payload: dict[str, Any] = {
        "subject_id": "alice",
        "tenant_id": "acme",
        "access_groups": ("analysts", "ops"),
    }
    payload.update(overrides)
    return Principal(**payload)


# ─── the shape that used to be rejected outright ────────────────────────────


def test_claim_provider_compiles_array_groups_instead_of_refusing() -> None:
    (filter_,) = ClaimAuthorizationProvider().search_policy_filters(
        _principal(), access="governed", attributes=[GROUPS]
    )

    assert filter_.operator is SearchFilterOperator.ARRAY_CONTAINS_ANY
    assert filter_.value == ("analysts", "ops")


def test_grant_provider_compiles_array_groups_from_grants() -> None:
    provider = GrantAuthorizationProvider(
        StaticGrantStore(
            [
                Grant("alice", "access_groups", "analysts"),
                Grant("alice", "access_groups", "ops"),
            ]
        )
    )

    (filter_,) = provider.search_policy_filters(
        _principal(access_groups=("forged",)), access="governed", attributes=[GROUPS]
    )

    assert filter_.operator is SearchFilterOperator.ARRAY_CONTAINS_ANY
    # Granted, not claimed: the forged group is not consulted.
    assert filter_.value == ("analysts", "ops")


def test_a_single_group_still_compiles_to_a_set_not_an_equality() -> None:
    """A one-element grant must not become `column = 'analysts'`; the column
    is a list, so equality against a scalar is not the same question."""
    provider = GrantAuthorizationProvider(
        StaticGrantStore([Grant("alice", "access_groups", "analysts")])
    )

    (filter_,) = provider.search_policy_filters(
        _principal(), access="governed", attributes=[GROUPS]
    )

    assert filter_.operator is SearchFilterOperator.ARRAY_CONTAINS_ANY
    assert filter_.value == ("analysts",)


def test_no_grant_for_an_array_attribute_is_still_a_refusal() -> None:
    provider = GrantAuthorizationProvider(StaticGrantStore([]))

    with pytest.raises(AuthorizationError):
        provider.search_policy_filters(
            _principal(), access="governed", attributes=[GROUPS]
        )


# ─── the row recheck ────────────────────────────────────────────────────────


def _grants_provider() -> GrantAuthorizationProvider:
    return GrantAuthorizationProvider(
        StaticGrantStore([Grant("alice", "access_groups", "analysts")])
    )


# Both providers, because the recheck and the prefilter have to agree and
# nothing else forces them to. A provider that compiles a correct overlap
# prefilter and then rejects the rows it selected returns nothing at all --
# which is what the claim provider did until this was caught in review.
@pytest.mark.parametrize("provider_name", ["claims", "grants"])
@pytest.mark.parametrize(
    ("row_groups", "expected"),
    [
        (["analysts"], True),
        (["analysts", "admins"], True),
        (["admins"], False),
        ([], False),
        (None, False),
        ("analysts", False),
    ],
)
def test_can_read_intersects_array_groups(
    provider_name: str,
    row_groups: Any,
    expected: bool,
) -> None:
    """An absent or empty list is refused, matching the scalar rule: a row
    carrying no value for a required attribute is not thereby public. A bare
    string is refused too rather than being treated as a one-element list."""
    provider: Any = (
        ClaimAuthorizationProvider() if provider_name == "claims" else _grants_provider()
    )
    # The claim provider reads the caller's own groups; the grant provider
    # reads the operator's. Both are ("analysts",)-equivalent here so one
    # expectation covers both.
    principal = _principal(access_groups=("analysts",))
    row = {
        "authorization_resolved": True,
        "is_public": False,
        "access_groups": row_groups,
    }

    assert provider.can_read(principal, row, attributes=[GROUPS]) is expected


def test_a_row_the_prefilter_selects_is_not_rejected_by_the_recheck() -> None:
    """The two halves must agree. A correct prefilter paired with a recheck
    that rejects everything yields an empty result set, which looks like
    "no matching documents" rather than like a bug."""
    provider = ClaimAuthorizationProvider()
    principal = _principal(access_groups=("analysts", "ops"))
    row_groups = ["analysts", "admins"]

    (filter_,) = provider.search_policy_filters(
        principal, access="governed", attributes=[GROUPS]
    )
    assert isinstance(filter_.value, tuple)
    selected_by_prefilter = bool(set(row_groups) & set(filter_.value))

    assert selected_by_prefilter
    assert provider.can_read(
        principal,
        {
            "authorization_resolved": True,
            "is_public": False,
            "access_groups": row_groups,
        },
        attributes=[GROUPS],
    )


# ─── the store contract ─────────────────────────────────────────────────────


def test_predicate_compiles_to_array_has_any() -> None:
    clause = _compile_predicates(
        [
            RetrievalPredicate(
                "access_groups",
                RetrievalPredicateOperator.ARRAY_CONTAINS_ANY,
                ("analysts", "ops"),
            )
        ]
    )

    assert clause == "array_has_any(access_groups, ['analysts', 'ops'])"


def test_overlap_predicate_rejects_an_empty_set() -> None:
    """An empty set means "matches nothing", which is indistinguishable
    downstream from a filter that was dropped. Callers must be explicit."""
    with pytest.raises(RetrievalError, match="non-empty tuple"):
        RetrievalPredicate(
            "access_groups", RetrievalPredicateOperator.ARRAY_CONTAINS_ANY, ()
        )


def test_lancedb_declares_the_capability() -> None:
    from stel.retrieval.lancedb import LanceDBStore

    features = LanceDBStore.capabilities().features
    assert RetrievalFeature.ARRAY_CONTAINMENT_FILTERS in features


def test_compiled_clause_actually_filters_in_lancedb(tmp_path: Path) -> None:
    """The clause is only correct if the engine agrees. Runs the compiled SQL
    against a real LanceDB table rather than asserting on a string alone."""
    db = lancedb.connect(str(tmp_path))
    table = db.create_table(
        "rows",
        pa.table(
            {
                "id": pa.array(["a", "b", "c"]),
                "access_groups": pa.array(
                    [["analysts", "ops"], ["admins"], []],
                    type=pa.list_(pa.string()),
                ),
            }
        ),
    )
    clause = _compile_predicates(
        [
            RetrievalPredicate(
                "access_groups",
                RetrievalPredicateOperator.ARRAY_CONTAINS_ANY,
                ("analysts",),
            )
        ]
    )
    assert clause is not None

    matched = table.search().where(clause).limit(10).to_list()

    assert sorted(row["id"] for row in matched) == ["a"]


# ─── the filter must match the attribute's shape ────────────────────────────


def test_search_filter_rejects_an_empty_overlap_set() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        SearchFilter("access_groups", SearchFilterOperator.ARRAY_CONTAINS_ANY, ())


def test_a_store_without_the_capability_fails_closed() -> None:
    """The dangerous outcome is not "unsupported", it is "filter dropped".

    A store that cannot express overlap must refuse the query, because
    silently omitting a policy prefilter turns a governed model from unusable
    into unfiltered.
    """
    from stel.retrieval.base import RetrievalCapabilities, RetrievalCapabilityError

    capabilities = RetrievalCapabilities(
        features=frozenset({RetrievalFeature.METADATA_FILTERING}),
        distance_metrics=frozenset({"cosine"}),
        consistency_modes=frozenset({"strong"}),
        max_batch_size=10,
        max_id_bytes=None,
        max_dimensions=None,
    )

    with pytest.raises(RetrievalCapabilityError, match="array_containment_filters"):
        capabilities.require(
            {RetrievalFeature.ARRAY_CONTAINMENT_FILTERS: "array-valued attribute filters"},
            store_type="fake",
        )


def test_scalar_and_array_operators_cannot_be_swapped() -> None:
    """Both directions are silent failures if allowed through: overlap against
    a scalar column asks an unanswerable question, and a scalar comparison
    against a list can evaluate false for every row and look like no matches."""
    from stel.config import SearchAttributeConfig
    from stel.search import SearchError, _reject_operator_type_mismatch

    scalar = SearchAttributeConfig(name="tenant_id", data_type="string")
    array = SearchAttributeConfig(name="access_groups", data_type="array[string]")

    with pytest.raises(SearchError, match="requires an array-valued field"):
        _reject_operator_type_mismatch(
            SearchFilter("tenant_id", SearchFilterOperator.ARRAY_CONTAINS_ANY, ("a",)),
            scalar,
        )
    with pytest.raises(SearchError, match="must be filtered with"):
        _reject_operator_type_mismatch(
            SearchFilter("access_groups", SearchFilterOperator.EQUAL, "a"),
            array,
        )
