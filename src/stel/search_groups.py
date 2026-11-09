"""Segmented search groups: what a member reads, what must hold, what serves.

The publish half of ADR-0037 (issue #771). A group is declared in the project
file and names `search:` models that each publish one partition of a shared
upstream relation. Three things live here, none of which touches a warehouse or
a store:

- `partition_predicates`: a member's partition as the typed predicates the
  snapshot read and the stale-key probe both take, so what is read and what is
  judged present are one set (issue #764).
- `validate_search_groups`: the compile-time contract. Everything it rejects is
  decidable from configuration, so it runs before source discovery, credentials
  or any remote call.
- `group_manifest_entries` and `group_availability`: the versioned record the
  serving manifest carries, and the rule that keeps a group unavailable until
  every member has published under the bounds that record names.

The query path is a separate issue; this module only says whether a group may
serve, never how a query is answered.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .adapters.base import ReadPredicate, ReadPredicateOperator
from .config.model import ModelConfig
from .config.project import ProjectConfig
from .config.search_partition import SearchPartitionConfig, bound_sort_key
from .dag import parse_ref
from .embedding import resolve_search_embedding_identity
from .hashing import canonical_fingerprint
from .retrieval.evolution import semantic_search_config


class SearchGroupError(ValueError):
    """A group declaration the compiler rejects.

    `model` names the member the message is about when there is one, so the
    compiler can point at that model's YAML; otherwise `path` locates it in the
    project file.
    """

    def __init__(
        self,
        message: str,
        *,
        path: tuple[str | int, ...] = (),
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.model = model


def partition_predicates(
    partition: SearchPartitionConfig | None,
) -> tuple[ReadPredicate, ...]:
    """The predicates that select exactly the rows this partition owns."""
    if partition is None:
        return ()
    if partition.nulls:
        return (ReadPredicate(partition.field, ReadPredicateOperator.IS_NULL),)
    predicates: list[ReadPredicate] = []
    if partition.lower is not None:
        predicates.append(
            ReadPredicate(
                partition.field, ReadPredicateOperator.GREATER_THAN_OR_EQUAL, partition.lower
            )
        )
    if partition.upper is not None:
        predicates.append(
            ReadPredicate(partition.field, ReadPredicateOperator.LESS_THAN, partition.upper)
        )
    return tuple(predicates)


def model_partition_predicates(model: ModelConfig) -> tuple[ReadPredicate, ...]:
    """The partition a search model publishes, or none when it is no member."""
    if model.search is None:
        return ()
    return partition_predicates(model.search.partition)


def validate_search_groups(
    project: ProjectConfig,
    models: Sequence[ModelConfig],
    *,
    taken_names: set[str],
) -> None:
    """Reject a group declaration that cannot serve its whole domain.

    `taken_names` is every source and model name: a group's serving name must
    not shadow one, or the catalog would list two resources under one name.
    """
    models_by_name = {model.name: model for model in models}
    seen_groups: set[str] = set()
    member_to_group: dict[str, str] = {}
    for index, group in enumerate(project.search_groups):
        where = ("search_groups", index)
        if group.name in seen_groups:
            raise SearchGroupError(
                f"search group '{group.name}' is declared twice", path=(*where, "name")
            )
        seen_groups.add(group.name)
        if group.name in taken_names:
            raise SearchGroupError(
                f"search group '{group.name}' has the name of a source or model; "
                "its serving name must be its own",
                path=(*where, "name"),
            )
        members: list[ModelConfig] = []
        for position, name in enumerate(group.members):
            member_path = (*where, "members", position)
            model = models_by_name.get(name)
            if model is None or model.search is None:
                raise SearchGroupError(
                    f"search group '{group.name}' names '{name}', which is not a "
                    "`search:` model",
                    path=member_path,
                )
            if name in member_to_group:
                raise SearchGroupError(
                    f"model '{name}' is a member of search groups "
                    f"'{member_to_group[name]}' and '{group.name}'; a member publishes "
                    "one partition of one group",
                    path=member_path,
                )
            member_to_group[name] = group.name
            if model.search.partition is None:
                raise SearchGroupError(
                    f"model '{name}' is a member of search group '{group.name}' but "
                    "declares no `search.partition`; a member must say which rows it owns",
                    model=name,
                    path=("search",),
                )
            members.append(model)
        _validate_group(group.name, members, models_by_name, where)
    for model in models:
        search = model.search
        declared = search is not None and search.partition is not None
        if declared and model.name not in member_to_group:
            raise SearchGroupError(
                f"model '{model.name}' declares `search.partition` but belongs to no "
                "`search_groups:` entry; a partition publishes a slice that nothing serves "
                "unless a group names it",
                model=model.name,
                path=("search", "partition"),
            )


def _partition(model: ModelConfig) -> SearchPartitionConfig:
    search = model.search
    assert search is not None and search.partition is not None
    return search.partition


def _validate_group(
    name: str,
    members: Sequence[ModelConfig],
    models_by_name: Mapping[str, ModelConfig],
    where: tuple[str | int, ...],
) -> None:
    first = members[0]
    first_partition = _partition(first)
    for model in members[1:]:
        partition = _partition(model)
        if partition.field != first_partition.field or partition.type != first_partition.type:
            raise SearchGroupError(
                f"search group '{name}' partitions on one typed field, but '{first.name}' "
                f"declares {first_partition.field} ({first_partition.type}) and "
                f"'{model.name}' declares {partition.field} ({partition.type})",
                model=model.name,
                path=("search", "partition"),
            )
    for model in members:
        if _upstream(model) != _upstream(first):
            raise SearchGroupError(
                f"search group '{name}' divides one upstream relation, but '{first.name}' "
                f"reads '{_upstream(first)}' and '{model.name}' reads '{_upstream(model)}'",
                model=model.name,
                path=("depends_on",),
            )
        search = model.search
        assert search is not None
        routed = {attribute.name: attribute for attribute in search.attributes}.get(
            first_partition.field
        )
        if routed is None:
            raise SearchGroupError(
                f"search group '{name}' routes on '{first_partition.field}', so '{model.name}' "
                "must declare it under `search.attributes`: a query's filter on it is "
                "enforced inside every member",
                model=model.name,
                path=("search", "attributes"),
            )
        if not routed.nullable:
            # The group always has a member for null (below), and that member
            # publishes rows whose field is null, which a non-nullable attribute
            # refuses at publish time. Every member must agree on the contract.
            raise SearchGroupError(
                f"search group '{name}' has a member for null {first_partition.field}, so "
                f"'{model.name}' must declare that attribute `nullable: true`",
                model=model.name,
                path=("search", "attributes"),
            )
    _validate_identities(name, members, models_by_name)
    _validate_coverage(name, members, first_partition, where)


def _upstream(model: ModelConfig) -> str:
    return parse_ref((model.depends_on or [""])[0])


def _collection_identity(model: ModelConfig) -> tuple[str | None, str]:
    search = model.search
    assert search is not None
    return (search.store, search.collection or model.name)


def _validate_identities(
    name: str,
    members: Sequence[ModelConfig],
    models_by_name: Mapping[str, ModelConfig],
) -> None:
    first = members[0]
    first_search = first.search
    assert first_search is not None
    first_embedding = _embedding_identity(first, models_by_name)
    first_config = _comparable(first)
    claimed: dict[tuple[str | None, str], str] = {}
    for model in members:
        search = model.search
        assert search is not None
        collection = _collection_identity(model)
        if collection in claimed:
            raise SearchGroupError(
                f"search group '{name}' members '{claimed[collection]}' and '{model.name}' "
                f"publish to one collection ('{collection[1]}'); each member needs its own",
                model=model.name,
                path=("search", "collection"),
            )
        claimed[collection] = model.name
        if model is first:
            continue
        if search.store != first_search.store:
            raise SearchGroupError(
                f"search group '{name}' members must share one store, but '{first.name}' "
                f"uses {first_search.store or 'the default store'} and '{model.name}' "
                f"uses {search.store or 'the default store'}",
                model=model.name,
                path=("search", "store"),
            )
        if _embedding_identity(model, models_by_name) != first_embedding:
            raise SearchGroupError(
                f"search group '{name}' members must share one embedding identity: "
                f"'{model.name}' embeds differently from '{first.name}', so their vectors "
                "are not comparable",
                model=model.name,
                path=("search", "vector", "embedding"),
            )
        differing = sorted(
            key
            for key in set(first_config) | set(_comparable(model))
            if first_config.get(key) != _comparable(model).get(key)
        )
        if differing:
            raise SearchGroupError(
                f"search group '{name}' members must publish one contract, but "
                f"'{model.name}' differs from '{first.name}' in {_describe(differing)}",
                model=model.name,
                path=("search",),
            )


def _comparable(model: ModelConfig) -> dict[str, Any]:
    search = model.search
    assert search is not None
    config = semantic_search_config(search.model_dump(mode="python"))
    config.pop("partition", None)
    return config


def _embedding_identity(
    model: ModelConfig, models_by_name: Mapping[str, ModelConfig]
) -> dict[str, Any] | None:
    """What the member's vectors were embedded with, whether inherited or declared."""
    search = model.search
    assert search is not None
    if search.vector is None:
        return None
    identity = resolve_search_embedding_identity(model, models_by_name)
    if identity is not None:
        return identity.to_dict()
    declared = search.vector.embedding
    return None if declared == "inherit" else declared.model_dump(mode="json")


def _describe(keys: Sequence[str]) -> str:
    labels = {
        "access": "its access mode",
        "attributes": "its attributes and policy filters",
        "query": "its query modes",
        "vector": "its vector configuration",
    }
    return ", ".join(labels.get(key, f"`{key}`") for key in keys)


def _validate_coverage(
    name: str,
    members: Sequence[ModelConfig],
    reference: SearchPartitionConfig,
    where: tuple[str | int, ...],
) -> None:
    """Every value of the field, and null, belongs to exactly one member."""
    field = reference.field
    nulls = [model for model in members if _partition(model).nulls]
    if not nulls:
        raise SearchGroupError(
            f"search group '{name}' has no member for rows where {field} is null; "
            "declare one with `search.partition.nulls: true`, since a range never "
            "selects a null",
            path=(*where, "members"),
        )
    if len(nulls) > 1:
        raise SearchGroupError(
            f"search group '{name}' has {len(nulls)} members for null {field} "
            f"({', '.join(model.name for model in nulls)}); null rows belong to one",
            path=(*where, "members"),
        )
    ranges = sorted(
        (model for model in members if not _partition(model).nulls),
        key=lambda model: _lower_key(_partition(model)),
    )
    previous: ModelConfig | None = None
    for model in ranges:
        partition = _partition(model)
        if previous is None:
            if partition.lower is not None:
                raise SearchGroupError(
                    f"search group '{name}' leaves {field} values below {partition.lower} "
                    f"with no member; leave '{model.name}' lower bound open so it takes them",
                    model=model.name,
                    path=("search", "partition", "lower"),
                )
        else:
            _check_join(name, field, previous, model)
        previous = model
    assert previous is not None
    last = _partition(previous)
    if last.upper is not None:
        raise SearchGroupError(
            f"search group '{name}' leaves {field} values from {last.upper} up with no "
            f"member; leave '{previous.name}' upper bound open so it takes them",
            model=previous.name,
            path=("search", "partition", "upper"),
        )


def _check_join(name: str, field: str, below: ModelConfig, above: ModelConfig) -> None:
    low = _partition(below)
    high = _partition(above)
    if low.upper is None or high.lower is None:
        raise SearchGroupError(
            f"search group '{name}' members '{below.name}' and '{above.name}' overlap: "
            "an open end claims every value the next member does",
            model=above.name,
            path=("search", "partition"),
        )
    if bound_sort_key(high.lower) < bound_sort_key(low.upper):
        raise SearchGroupError(
            f"search group '{name}' members '{below.name}' and '{above.name}' overlap on "
            f"{field}: '{below.name}' runs to {low.upper} and '{above.name}' starts at "
            f"{high.lower}",
            model=above.name,
            path=("search", "partition", "lower"),
        )
    if bound_sort_key(high.lower) > bound_sort_key(low.upper):
        raise SearchGroupError(
            f"search group '{name}' leaves {field} from {low.upper} to {high.lower} with no "
            f"member: '{below.name}' ends there and '{above.name}' starts later",
            model=above.name,
            path=("search", "partition", "lower"),
        )


def _lower_key(partition: SearchPartitionConfig) -> tuple[int, int, int, str]:
    if partition.lower is None:
        return (0, 0, 0, "")
    return (1, *bound_sort_key(partition.lower))


def group_version(members: Sequence[Mapping[str, Any]]) -> str:
    """A digest of the partition map, so a changed bound is a new group version."""
    return canonical_fingerprint(
        {
            "members": sorted(
                (
                    {"model": member["model"], "partition": member["partition"]}
                    for member in members
                ),
                key=lambda member: str(member["model"]),
            )
        },
        domain="stel-search-group",
        version=1,
    )


def group_manifest_entries(
    project: ProjectConfig,
    models: Sequence[ModelConfig],
    *,
    published: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """The groups as the serving manifest records them.

    Membership and bounds are written down here so a query reads them from the
    artifact that was compiled, never from project YAML as it stands later.
    `published` maps each member to its compiled `config_fingerprint` and
    `logical_collection`, the identity a member's serving ledger must show
    before the group may serve.
    """
    models_by_name = {model.name: model for model in models}
    entries: list[dict[str, Any]] = []
    for group in project.search_groups:
        members = [
            {
                "model": name,
                "logical_collection": published[name]["logical_collection"],
                "config_fingerprint": published[name]["config_fingerprint"],
                "partition": _partition(models_by_name[name]).identity(),
            }
            for name in group.members
        ]
        reference = _partition(models_by_name[group.members[0]])
        entries.append(
            {
                "name": group.name,
                "field": reference.field,
                "type": reference.type,
                "version": group_version(members),
                "members": members,
            }
        )
    return entries


@dataclass(frozen=True)
class MemberAvailability:
    model: str
    available: bool
    reason: str | None


@dataclass(frozen=True)
class GroupAvailability:
    name: str
    available: bool
    members: tuple[MemberAvailability, ...]

    def reasons(self) -> tuple[str, ...]:
        return tuple(
            f"{member.model}: {member.reason}"
            for member in self.members
            if member.reason is not None
        )


def group_availability(
    entry: Mapping[str, Any], active: Mapping[str, str | None]
) -> GroupAvailability:
    """Whether every member serves the bounds the manifest names.

    `active` maps a member to the `config_fingerprint` its serving ledger holds
    for the generation that is live, or None when nothing is live. A member is
    available only if that fingerprint is the one compiled into the manifest;
    the partition rides in it, so a member still holding rows cut under old
    bounds reads as unavailable rather than serving a stale membership. One
    unavailable member makes the group unavailable: serving the others would
    answer from a partial corpus without saying so.
    """
    statuses: list[MemberAvailability] = []
    for member in entry["members"]:
        model = str(member["model"])
        live = active.get(model)
        if live is None:
            statuses.append(
                MemberAvailability(model, False, "has not published a serving generation")
            )
        elif live != member["config_fingerprint"]:
            statuses.append(
                MemberAvailability(
                    model,
                    False,
                    "published under a different partition or configuration than "
                    f"group version {entry['version']}; republish it",
                )
            )
        else:
            statuses.append(MemberAvailability(model, True, None))
    return GroupAvailability(
        name=str(entry["name"]),
        available=all(status.available for status in statuses),
        members=tuple(statuses),
    )
