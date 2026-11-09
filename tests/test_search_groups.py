"""A segmented search group's declaration, and what the compiler rejects (ADR-0037).

Issue #771. A group is only as sound as its coverage: a gap loses rows from
search, an overlap serves a row twice, and a member on another embedding makes
rankings incomparable. Every one of those is decidable from configuration, so
each is a compile-time refusal rather than a corpus discovered missing later.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import date
from typing import Any

import pytest

from stel.adapters.base import ReadPredicate, ReadPredicateOperator
from stel.config.model import ModelConfig, SearchConfig
from stel.config.project import ProjectConfig
from stel.config.search_partition import SearchPartitionConfig
from stel.retrieval.registry import collection_config_fingerprint
from stel.search_groups import (
    SearchGroupError,
    group_availability,
    group_manifest_entries,
    group_version,
    model_partition_predicates,
    partition_predicates,
    validate_search_groups,
)

_EMBEDDING = {
    "provider": "fixture",
    "model": "deterministic-2d-v1",
    "provider_contract_version": 2,
    "provider_implementation": "tests:v1",
    "semantic_config_fingerprint": "deterministic-2d-v1",
    "dimensions": 2,
}


def _search(partition: dict[str, Any] | None, *, collection: str) -> dict[str, Any]:
    search: dict[str, Any] = {
        "access": "public",
        "store": "primary",
        "collection": collection,
        "id_field": "chunk_id",
        "document_id_field": "document_id",
        "chunk_id_field": "chunk_id",
        "text_fields": ["text"],
        "vector": {
            "field": "embedding",
            "dimensions": 2,
            "metric": "cosine",
            "search": "exact",
            "embedding": dict(_EMBEDDING),
        },
        "full_text": {"fields": ["text"]},
        "attributes": [
            {
                "name": "filing_year",
                "data_type": "integer",
                "nullable": True,
                "filter_role": "user",
            },
        ],
        "query": {"modes": ["vector", "text", "filter"], "consistency": "strong"},
    }
    if partition is not None:
        search["partition"] = partition
    return search


def _year(name: str, **bounds: Any) -> dict[str, Any]:
    return {
        "name": name,
        "depends_on": ["ref('embedding_rows')"],
        "materialization": "incremental",
        "search": _search(
            {"field": "filing_year", "type": "integer", **bounds}, collection=name
        ),
    }


def _models() -> dict[str, dict[str, Any]]:
    """Three year ranges with open ends, plus the null member: a whole domain."""
    return {
        "old": _year("old", upper=2023),
        "mid": _year("mid", lower=2023, upper=2025),
        "new": _year("new", lower=2025),
        "none": _year("none", nulls=True),
    }


def _validate(
    models: dict[str, dict[str, Any]],
    *,
    members: tuple[str, ...] | None = None,
    group_name: str = "filings",
    extra: tuple[dict[str, Any], ...] = (),
) -> None:
    configs = [ModelConfig.model_validate(raw) for raw in (*models.values(), *extra)]
    project = ProjectConfig.model_validate(
        {
            "name": "demo",
            "search_groups": [
                {"name": group_name, "members": members or tuple(models)},
            ],
        }
    )
    validate_search_groups(
        project, configs, taken_names={config.name for config in configs}
    )


def test_a_group_that_covers_its_whole_domain_is_accepted() -> None:
    _validate(_models())


def _members_of(models: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    return tuple(models)


def _mutate(
    change: Callable[[dict[str, dict[str, Any]]], None],
) -> dict[str, dict[str, Any]]:
    models = deepcopy(_models())
    change(models)
    return models


def _set_bound(model: str, key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def apply(models: dict[str, dict[str, Any]]) -> None:
        partition = models[model]["search"]["partition"]
        if value is None:
            partition.pop(key, None)
        else:
            partition[key] = value

    return apply


def _set_search(model: str, key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def apply(models: dict[str, dict[str, Any]]) -> None:
        models[model]["search"][key] = value

    return apply


def _set_vector_embedding(member: str, **changes: Any) -> Callable[[dict[str, Any]], None]:
    def apply(models: dict[str, dict[str, Any]]) -> None:
        models[member]["search"]["vector"]["embedding"].update(changes)

    return apply


def _drop(model: str) -> Callable[[dict[str, Any]], None]:
    def apply(models: dict[str, dict[str, Any]]) -> None:
        del models[model]

    return apply


@pytest.mark.parametrize(
    ("change", "message"),
    [
        # Coverage: the domain is every value and null, each in exactly one member.
        (_set_bound("mid", "lower", 2024), "with no member"),
        (_set_bound("mid", "upper", 2024), "with no member"),
        (_set_bound("mid", "lower", 2022), "overlap on filing_year"),
        (_set_bound("old", "lower", 2000), "values below 2000"),
        (_set_bound("new", "upper", 2030), "from 2030 up"),
        (_set_bound("old", "upper", None), "open end claims every value"),
        (_drop("none"), "no member for rows where filing_year is null"),
        # Null rows belong to one member, or two would each publish them.
        (
            lambda models: models.update(also_none=_year("also_none", nulls=True)),
            "2 members for null filing_year",
        ),
        # One logical corpus: the members must be interchangeable but for bounds.
        (_set_search("mid", "store", "secondary"), "share one store"),
        (_set_vector_embedding("mid", model="another-model"), "one embedding identity"),
        (
            _set_search("mid", "query", {"modes": ["vector"], "consistency": "strong"}),
            "its query modes",
        ),
        (_set_search("mid", "text_fields", ["text", "title"]), "`text_fields`"),
        (_set_search("mid", "collection", "old"), "publish to one collection"),
    ],
)
def test_a_group_the_compiler_cannot_serve_whole_is_rejected(
    change: Callable[[dict[str, Any]], None], message: str
) -> None:
    models = _mutate(change)
    members = tuple(models)
    with pytest.raises(SearchGroupError, match=message):
        _validate(models, members=members)


def test_members_must_partition_on_one_typed_field() -> None:
    models = _mutate(lambda m: m["mid"]["search"]["partition"].update(field="other"))
    with pytest.raises(SearchGroupError, match="one typed field"):
        _validate(models)

    models = _mutate(
        lambda m: m["mid"]["search"]["partition"].update(
            type="string", lower="2023", upper="2025"
        )
    )
    with pytest.raises(SearchGroupError, match="one typed field"):
        _validate(models)


def test_members_must_divide_one_upstream() -> None:
    models = _mutate(lambda m: m["mid"].update(depends_on=["ref('elsewhere')"]))
    with pytest.raises(SearchGroupError, match="one upstream relation"):
        _validate(models)


def test_the_partition_field_must_be_filterable_in_every_member() -> None:
    """A query's filter on it is enforced inside each member, so each stores it."""
    models = _mutate(lambda m: m["mid"]["search"].update(attributes=[]))
    with pytest.raises(SearchGroupError, match=r"under `search.attributes`"):
        _validate(models)


def test_the_partition_type_must_match_the_attribute_it_filters() -> None:
    """Issue #771 review: ranges that tile strings ("10", "2") overlap as integers.
    The warehouse evaluates the predicates in the column's type, so a mismatch is
    refused rather than left to publish a row from two members."""

    def mistyped(models: dict[str, dict[str, Any]]) -> None:
        for model in models.values():
            partition = model["search"]["partition"]
            partition["type"] = "string"
            for key in ("lower", "upper"):
                if key in partition:
                    partition[key] = str(partition[key])

    with pytest.raises(SearchGroupError, match=r"`data_type: integer`"):
        _validate(_mutate(mistyped))


def test_a_range_with_no_bounds_still_excludes_nulls() -> None:
    """Issue #771 review: an empty predicate reads the whole relation, nulls
    included, so the fully open range would publish what the null member does."""
    everything = SearchPartitionConfig(field="filing_year", type="integer")
    assert partition_predicates(everything) == (
        ReadPredicate("filing_year", ReadPredicateOperator.IS_NOT_NULL),
    )
    models = {
        "all": _year("all"),
        "none": _year("none", nulls=True),
    }
    _validate(models)


def test_the_partition_field_must_admit_the_null_member_rows() -> None:
    """The null member publishes rows whose field is null, which a non-nullable
    attribute refuses mid-publish; the compiler says so before it gets there."""

    def not_nullable(models: dict[str, dict[str, Any]]) -> None:
        models["mid"]["search"]["attributes"][0]["nullable"] = False

    with pytest.raises(SearchGroupError, match=r"`nullable: true`"):
        _validate(_mutate(not_nullable))


def test_members_must_agree_on_the_policy_contract() -> None:
    def governed(models: dict[str, dict[str, Any]]) -> None:
        search = models["mid"]["search"]
        search["access"] = "governed"
        search["attributes"].append(
            {"name": "tenant", "data_type": "string", "filter_role": "policy"}
        )

    with pytest.raises(SearchGroupError, match="access mode"):
        _validate(_mutate(governed))


def test_a_member_must_declare_its_partition() -> None:
    models = _mutate(lambda m: m["mid"]["search"].pop("partition"))
    with pytest.raises(SearchGroupError, match=r"declares no `search\.partition`") as raised:
        _validate(models)
    assert raised.value.model == "mid"


def test_a_partition_nothing_serves_is_rejected() -> None:
    """A partition outside any group publishes a slice nothing can query."""
    stray = _year("stray", lower=2020)
    with pytest.raises(SearchGroupError, match=r"belongs to no `search_groups:`") as raised:
        _validate(_models(), extra=(stray,))
    assert raised.value.model == "stray"


def test_a_model_belongs_to_one_group() -> None:
    models = _models()
    configs = [ModelConfig.model_validate(raw) for raw in models.values()]
    project = ProjectConfig.model_validate(
        {
            "name": "demo",
            "search_groups": [
                {"name": "filings", "members": list(models)},
                {"name": "filings_again", "members": ["old", "mid"]},
            ],
        }
    )
    with pytest.raises(SearchGroupError, match="one partition of one group"):
        validate_search_groups(project, configs, taken_names=set(models))


@pytest.mark.parametrize("name", ["old", "embedding_rows"])
def test_a_serving_name_is_its_own(name: str) -> None:
    """The catalog lists one resource per name: a group may not shadow a model."""
    models = _models()
    configs = [ModelConfig.model_validate(raw) for raw in models.values()]
    project = ProjectConfig.model_validate(
        {"name": "demo", "search_groups": [{"name": name, "members": list(models)}]}
    )
    with pytest.raises(SearchGroupError, match="name of a source or model"):
        validate_search_groups(project, configs, taken_names={"embedding_rows", *models})


def test_a_group_member_must_be_a_search_model() -> None:
    models = _models()
    with pytest.raises(SearchGroupError, match="not a `search:` model"):
        _validate(models, members=(*models, "ghost"))


def test_a_model_may_still_declare_no_group_at_all() -> None:
    """Nothing about an ordinary search model changes: no group, no partition."""
    plain = {
        "name": "plain",
        "depends_on": ["ref('embedding_rows')"],
        "materialization": "incremental",
        "search": _search(None, collection="plain"),
    }
    config = ModelConfig.model_validate(plain)
    project = ProjectConfig.model_validate({"name": "demo"})
    validate_search_groups(project, [config], taken_names={"plain"})
    assert model_partition_predicates(config) == ()


def test_a_partition_cannot_be_a_range_and_the_null_member_at_once() -> None:
    with pytest.raises(ValueError, match="takes no bounds"):
        SearchPartitionConfig(field="filing_year", type="integer", nulls=True, lower=2020)


def test_a_range_runs_from_a_lower_bound_to_a_higher_one() -> None:
    with pytest.raises(ValueError, match="must be below upper"):
        SearchPartitionConfig(field="filing_year", type="integer", lower=2025, upper=2025)


def test_bounds_take_the_declared_type() -> None:
    """A `for_each` template can only hold a bound as a string."""
    dated = SearchPartitionConfig.model_validate(
        {"field": "filed", "type": "date", "lower": "2024-01-01", "upper": date(2025, 1, 1)}
    )
    assert dated.lower == date(2024, 1, 1)
    assert SearchPartitionConfig.model_validate(
        {"field": "filing_year", "type": "integer", "lower": "2024"}
    ).lower == 2024
    with pytest.raises(ValueError, match="ISO date"):
        SearchPartitionConfig.model_validate(
            {"field": "filed", "type": "date", "lower": "last spring"}
        )
    with pytest.raises(ValueError, match="integer"):
        SearchPartitionConfig.model_validate(
            {"field": "filing_year", "type": "integer", "lower": True}
        )


def test_a_partition_is_the_predicates_the_read_and_the_probe_both_take() -> None:
    mid = SearchPartitionConfig(field="filing_year", type="integer", lower=2023, upper=2025)
    assert partition_predicates(mid) == (
        ReadPredicate("filing_year", ReadPredicateOperator.GREATER_THAN_OR_EQUAL, 2023),
        ReadPredicate("filing_year", ReadPredicateOperator.LESS_THAN, 2025),
    )
    first = SearchPartitionConfig(field="filing_year", type="integer", upper=2023)
    assert partition_predicates(first) == (
        ReadPredicate("filing_year", ReadPredicateOperator.LESS_THAN, 2023),
    )
    assert partition_predicates(
        SearchPartitionConfig(field="filing_year", type="integer", nulls=True)
    ) == (ReadPredicate("filing_year", ReadPredicateOperator.IS_NULL),)
    assert partition_predicates(None) == ()


def test_an_absent_partition_leaves_every_existing_identity_alone() -> None:
    """The reason the field is dropped from the dump when unset: every descriptor
    and fingerprint starts from it, and a `partition: null` there would re-key
    every search index published before groups existed (the `freshness` rule)."""
    plain = SearchConfig.model_validate(_search(None, collection="plain"))
    assert "partition" not in plain.model_dump(mode="python")
    assert "partition" not in plain.model_dump(mode="json")


def test_a_changed_bound_changes_the_index_identity() -> None:
    """A membership migration must not be reused silently: the bound reaches the
    fingerprint, so a collection cut under old bounds is no longer the same index."""

    def fingerprint(**bounds: Any) -> str:
        search = SearchConfig.model_validate(
            _search({"field": "filing_year", "type": "integer", **bounds}, collection="m")
        )
        return collection_config_fingerprint(
            search.model_dump(mode="python"), store_type="lancedb"
        )

    assert fingerprint(lower=2023, upper=2025) == fingerprint(lower=2023, upper=2025)
    assert fingerprint(lower=2023, upper=2025) != fingerprint(lower=2023, upper=2026)
    plain = SearchConfig.model_validate(_search(None, collection="m"))
    assert fingerprint(lower=2023, upper=2025) != collection_config_fingerprint(
        plain.model_dump(mode="python"), store_type="lancedb"
    )


def _entry() -> dict[str, Any]:
    models = _models()
    configs = [ModelConfig.model_validate(raw) for raw in models.values()]
    project = ProjectConfig.model_validate(
        {"name": "demo", "search_groups": [{"name": "filings", "members": list(models)}]}
    )
    published = {
        name: {"config_fingerprint": f"fp-{name}", "logical_collection": name}
        for name in models
    }
    (entry,) = group_manifest_entries(project, configs, published=published)
    return entry


def test_the_manifest_records_membership_and_bounds() -> None:
    entry = _entry()
    assert entry["name"] == "filings"
    assert entry["field"] == "filing_year"
    assert entry["type"] == "integer"
    by_model = {member["model"]: member for member in entry["members"]}
    assert by_model["mid"]["partition"] == {
        "field": "filing_year",
        "type": "integer",
        "lower": 2023,
        "upper": 2025,
        "nulls": False,
    }
    assert by_model["none"]["partition"]["nulls"] is True
    assert by_model["mid"]["config_fingerprint"] == "fp-mid"


def test_a_changed_bound_is_a_new_group_version() -> None:
    entry = _entry()
    moved = deepcopy(entry["members"])
    next(m for m in moved if m["model"] == "mid")["partition"]["upper"] = 2026
    assert group_version(entry["members"]) == entry["version"]
    assert group_version(moved) != entry["version"]


def test_a_group_is_unavailable_until_every_member_has_published_its_bounds() -> None:
    entry = _entry()
    live = {member["model"]: member["config_fingerprint"] for member in entry["members"]}
    assert group_availability(entry, live).available

    missing = {**live, "new": None}
    result = group_availability(entry, missing)
    assert not result.available
    assert result.reasons() == ("new: has not published a serving generation",)

    stale = {**live, "mid": "fp-published-under-old-bounds"}
    result = group_availability(entry, stale)
    assert not result.available
    assert "different partition or configuration" in result.reasons()[0]
    # The members that are fine are reported fine: only one blocks the group.
    assert [m.available for m in result.members].count(True) == 3
