"""Search-config change classification and the pre-#344 re-stamp (issue #344).

Two claims are under test. First, that fields which only change execution
cadence no longer invalidate a published index — the bug this issue opened on,
where tuning `batch_size` demanded a full re-embed. Second, that a change which
*does* matter is named rather than merely detected, so the operator learns
which field forced a rebuild instead of being told "something changed".
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from stel.config import SearchConfig
from stel.config.model import ModelConfig, SearchVectorConfig
from stel.config.project import ProjectConfig
from stel.embedding import effective_search_config
from stel.execution.contracts import RunError
from stel.execution.search import (
    _scalar_index_fields,
    _validate_collection_schema,
    _verify_collection_config,
)
from stel.retrieval import (
    ChangeKind,
    CollectionMetadata,
    CollectionSpec,
    IndexedRow,
    classify_changes,
    classify_descriptor_changes,
    collection_config_fingerprint,
    descriptor_json,
    legacy_collection_config_fingerprint,
)
from stel.retrieval.registry import collection_descriptor
from stel.versioning import compute_model_code_version

_BASE: dict[str, Any] = {
    "access": "public",
    "store": None,
    "collection": None,
    "id_field": "chunk_id",
    "document_id_field": "document_id",
    "chunk_id_field": "chunk_id",
    "text_fields": ("text",),
    "return_text_fields": ("text",),
    "vector": {"field": "embedding", "dimensions": 2, "metric": "cosine"},
    "full_text": {"fields": ("text",)},
    "attributes": (
        {"name": "category", "data_type": "string", "filter_role": "user"},
    ),
    "display_fields": ("title",),
    "query": {"modes": {"vector", "text"}},
    "on_index_change": "fail",
    "batch_size": 1000,
    "index_options": {},
}


def _config(**overrides: Any) -> dict[str, Any]:
    return {**_BASE, **overrides}


def _fingerprint(**overrides: Any) -> str:
    return collection_config_fingerprint(_config(**overrides), store_type="lancedb")


def _search(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = collection_descriptor(
        _config(**overrides), store_type="lancedb"
    )["search"]
    return payload


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 500),
        ("index_options", {"nprobes": 20}),
        ("on_index_change", "rebuild"),
    ],
)
def test_cadence_fields_do_not_invalidate_a_published_index(
    field: str, value: Any
) -> None:
    """The reported bug. `batch_size` changes how many rows a publish sends per
    call and `index_options` is not read by any code at all, yet both used to
    change the fingerprint and force a blue/green rebuild plus a full re-embed.

    `on_index_change` is the sharper one: it is the field that decides how to
    react to a config change, so having it inside the fingerprint meant
    adopting a non-default policy tripped the very gate it was adopting."""
    assert _fingerprint(**{field: value}) == _fingerprint()


@pytest.mark.parametrize(
    "field,value",
    [
        ("id_field", "other_id"),
        ("vector", {"field": "embedding", "dimensions": 3, "metric": "cosine"}),
        ("vector", {"field": "embedding", "dimensions": 2, "metric": "dot"}),
        ("text_fields", ("text", "title")),
        ("access", "governed"),
    ],
)
def test_semantic_fields_still_invalidate(field: str, value: Any) -> None:
    """The complement: narrowing the descriptor must not have narrowed it so
    far that a real change slips through."""
    assert _fingerprint(**{field: value}) != _fingerprint()


def test_an_added_attribute_is_compatible_and_named() -> None:
    """Widening a collection with a new filterable attribute leaves every row
    already written valid under its own definition."""
    (change,) = classify_changes(
        _search(),
        _search(
            attributes=(
                {"name": "category", "data_type": "string", "filter_role": "user"},
                {"name": "symbol", "data_type": "string", "filter_role": "user"},
            )
        ),
    )
    assert change.field == "attributes"
    assert change.kind is ChangeKind.COMPATIBLE
    assert "symbol" in change.detail


def test_a_redefined_attribute_requires_a_rebuild() -> None:
    """Prevents the dangerous confusion: changing an existing attribute's type
    or filter role is not additive. Rows already indexed carry the old meaning,
    and widening cannot reinterpret them."""
    (change,) = classify_changes(
        _search(),
        _search(
            attributes=(
                {"name": "category", "data_type": "int", "filter_role": "user"},
            )
        ),
    )
    assert change.kind is ChangeKind.REBUILD_REQUIRED
    assert "category" in change.detail


def test_a_removed_attribute_requires_a_rebuild() -> None:
    (change,) = classify_changes(_search(), _search(attributes=()))
    assert change.kind is ChangeKind.REBUILD_REQUIRED
    assert "category" in change.detail


def test_a_changed_vector_dimension_names_the_field() -> None:
    """The whole point of classification: the operator is told which field
    forced the rebuild, not just that the digest moved."""
    (change,) = classify_changes(
        _search(),
        _search(vector={"field": "embedding", "dimensions": 3, "metric": "cosine"}),
    )
    assert change.field == "vector"
    assert change.kind is ChangeKind.REBUILD_REQUIRED


def test_a_wider_projection_is_compatible_but_dropping_one_is_not() -> None:
    (added,) = classify_changes(_search(), _search(display_fields=("title", "url")))
    assert added.kind is ChangeKind.COMPATIBLE
    (dropped,) = classify_changes(_search(), _search(display_fields=()))
    assert dropped.kind is ChangeKind.REBUILD_REQUIRED


def test_an_identical_config_classifies_as_no_change() -> None:
    assert classify_changes(_search(), _search()) == []


class _RecordingStore:
    """Captures whether the collection was re-stamped rather than rebuilt."""

    def __init__(self) -> None:
        self.restamped: list[str] = []
        self.evolved: list[tuple[str, list[str]]] = []

    def restamp_collection(self, spec: CollectionSpec) -> None:
        self.restamped.append(spec.descriptor)

    def evolve_collection(self, spec: CollectionSpec, added: Any) -> None:
        self.evolved.append((spec.descriptor, list(added)))
        self.restamp_collection(spec)


def _spec(**overrides: Any) -> CollectionSpec:
    config = _config(**overrides)
    return CollectionSpec(
        logical_name="context",
        physical_name="proj__dev__context",
        id_field="chunk_id",
        text_fields=("text",),
        full_text_fields=("text",),
        attribute_fields=("category",),
        scalar_index_fields=("category",),
        display_fields=("title",),
        vector_field="embedding",
        vector_dimensions=2,
        distance_metric="cosine",
        vector_search="exact",
        vector_index=None,
        config_fingerprint=collection_config_fingerprint(config, store_type="lancedb"),
        descriptor=descriptor_json(
            collection_descriptor(config, store_type="lancedb")
        ),
        legacy_config_fingerprint=legacy_collection_config_fingerprint(
            config, store_type="lancedb"
        ),
        arrow_schema=None,
    )


def _metadata(
    *, descriptor: str | None, config_fingerprint: str | None
) -> CollectionMetadata:
    return CollectionMetadata(
        physical_name="proj__dev__context",
        config_fingerprint=config_fingerprint,
        descriptor=descriptor,
        physical_generation="gen",
        row_count=2,
        schema=None,
    )


def test_a_pre_344_collection_is_restamped_not_rebuilt() -> None:
    """The migration. A collection published before the descriptor existed
    carries only the legacy digest. Recomputing that digest proves the config
    is unchanged, so the stamp is rewritten in place — no rebuild, no re-embed,
    and no operator action. Getting this wrong would inflict exactly the cost
    this issue exists to remove."""
    spec = _spec()
    store = _RecordingStore()
    existing = _metadata(
        descriptor=None, config_fingerprint=spec.legacy_config_fingerprint
    )

    _verify_collection_config(store, existing, spec)

    assert store.restamped == [spec.descriptor]


def test_a_pre_344_collection_whose_config_really_changed_still_fails() -> None:
    """The complement: the re-stamp must fire only on positive proof that
    nothing changed, never as a way to wave through a real change."""
    store = _RecordingStore()
    existing = _metadata(descriptor=None, config_fingerprint="some-other-digest")

    with pytest.raises(RunError, match="cannot be named"):
        _verify_collection_config(store, existing, _spec())

    assert store.restamped == []


def test_a_matching_descriptor_neither_fails_nor_restamps() -> None:
    spec = _spec()
    store = _RecordingStore()
    existing = _metadata(
        descriptor=spec.descriptor, config_fingerprint=spec.config_fingerprint
    )

    _verify_collection_config(store, existing, spec)

    assert store.restamped == []


def test_a_rebuild_change_names_the_field_in_the_error() -> None:
    spec = _spec(vector={"field": "embedding", "dimensions": 3, "metric": "cosine"})
    stored = _spec()
    existing = _metadata(
        descriptor=stored.descriptor, config_fingerprint=stored.config_fingerprint
    )

    with pytest.raises(RunError, match="requires a rebuild: vector"):
        _verify_collection_config(_RecordingStore(), existing, spec)


def test_an_additive_change_says_so_rather_than_demanding_a_rebuild() -> None:
    """`fail` is still the only policy, so this raises — but it must not tell
    the operator to rebuild a collection that could serve the change."""
    spec = _spec(
        attributes=(
            {"name": "category", "data_type": "string", "filter_role": "user"},
            {"name": "symbol", "data_type": "string", "filter_role": "user"},
        )
    )
    stored = _spec()
    existing = _metadata(
        descriptor=stored.descriptor, config_fingerprint=stored.config_fingerprint
    )

    with pytest.raises(RunError, match="can serve the change") as error:
        _verify_collection_config(_RecordingStore(), existing, spec)
    assert "added ['symbol']" in str(error.value)
    assert "rebuild" not in str(error.value).split("on_index_change")[0]


# ─── Codex review regressions ────────────────────────────────────────────────


def _real_spec(tmp_path: Path, **overrides: Any) -> tuple[Any, CollectionSpec]:
    """A real LanceDB store plus a spec whose arrow schema it can create.

    The re-stamp tests above use a recording fake, which is precisely how the
    fingerprint bug below survived them: a fake that does not model the stored
    stamp cannot show that re-stamping leaves it stale.
    """
    import pyarrow as pa

    from stel.retrieval import parse_store_config
    from stel.retrieval.lancedb import LanceDBStore

    config = parse_store_config({"type": "lancedb", "path": str(tmp_path / "lance")})
    store = LanceDBStore(config, project_name="demo", target_name="dev", alias="primary")
    spec = _spec(**overrides)
    schema = pa.schema(
        [pa.field("chunk_id", pa.string()), pa.field("text", pa.string())]
    )
    return store, replace(spec, arrow_schema=schema)


def test_a_restamped_collection_reports_its_new_fingerprint(tmp_path: Path) -> None:
    """Re-stamping has to update the collection's *identity*, not just add a
    descriptor beside a stale one.

    Post-publication validation compares the stored fingerprint against the
    spec's. If the re-stamp left the pre-#344 digest in place, every legacy
    collection would publish its rows, advance state, then fail that check —
    and fail it again on every retry, because the descriptor now short
    circuits the re-stamp branch. A permanent block inflicted by the very
    migration meant to avoid one (Codex review P1).

    The collection is built the way v0.10.0 built one — schema-level stamp
    only, no field metadata — so this exercises the real upgrade, not a
    re-stamp of something already current.
    """
    import lancedb
    import pyarrow as pa

    from stel.retrieval.lancedb import (
        _CONFIG_KEY,
        _CONTRACT_KEY,
        _OWNER_KEY,
        _OWNER_VALUE,
    )

    store, spec = _real_spec(tmp_path)
    legacy_schema = spec.arrow_schema.with_metadata(
        {
            _OWNER_KEY: _OWNER_VALUE,
            _CONTRACT_KEY: b"1",
            _CONFIG_KEY: spec.legacy_config_fingerprint.encode(),
        }
    )
    db = lancedb.connect(str(tmp_path / "lance"))
    db.create_table(spec.physical_name, schema=legacy_schema)
    db.open_table(spec.physical_name).add(
        [{"chunk_id": "a", "text": "x"}]
    )
    del pa

    with store:
        before = store.inspect_collection(spec.physical_name)
        assert before is not None
        assert before.descriptor is None, "fixture must look pre-#344"
        assert before.config_fingerprint == spec.legacy_config_fingerprint

        store.restamp_collection(spec)
        written = store.inspect_collection(spec.physical_name)

    assert written is not None
    assert written.descriptor == spec.descriptor
    # The assertion that was missing: identity, not just the descriptor. Without
    # it the post-publication check compares a v1 stamp against a v2 spec.
    assert written.config_fingerprint == spec.config_fingerprint
    assert written.row_count == 1, "re-stamping must not touch rows"


def test_a_created_collection_reports_the_current_fingerprint(tmp_path: Path) -> None:
    """The same invariant on the create path, so the two cannot drift."""
    store, spec = _real_spec(tmp_path)
    with store:
        store.create_collection(spec)
        written = store.inspect_collection(spec.physical_name)

    assert written is not None
    assert written.config_fingerprint == spec.config_fingerprint


def test_a_store_implementation_bump_requires_a_rebuild() -> None:
    """The descriptor's contract fields invalidate a collection on their own.

    Classifying only the `search` mapping dropped them, so a store
    implementation bump produced an empty change list and was reported as an
    additive change the existing collection could serve — the opposite of the
    design doc's "whole-index invalidation" (Codex review P2).
    """
    stored = {
        "contract_version": 2,
        "store_type": "lancedb",
        "store_implementation": "old",
        "search": {"id_field": "chunk_id"},
    }
    current = {**stored, "store_implementation": "new"}

    changes = classify_descriptor_changes(stored, current)

    assert [c.field for c in changes] == ["store_implementation"]
    assert all(c.kind is ChangeKind.REBUILD_REQUIRED for c in changes)


def test_a_contract_change_reaches_the_publish_gate_as_a_rebuild() -> None:
    """End of the same path: the operator must be told to rebuild, not told
    the existing collection could serve it."""
    spec = _spec()
    stored = json.loads(spec.descriptor)
    stored["store_implementation"] = "previous-implementation"
    existing = _metadata(
        descriptor=json.dumps(stored), config_fingerprint="stale-digest"
    )

    with pytest.raises(RunError, match="requires a rebuild: store_implementation"):
        _verify_collection_config(_RecordingStore(), existing, spec)


# ─── on_index_change: online (issue #344, second pass) ───────────────────────


class _EvolvingStore(_RecordingStore):
    """Records widening as well as re-stamping."""

    def __init__(self) -> None:
        super().__init__()
        self.evolved: list[tuple[str, list[str]]] = []

    def evolve_collection(self, spec: CollectionSpec, added: Any) -> None:
        self.evolved.append((spec.descriptor, list(added)))


def _schema_with(*names: str) -> Any:
    import pyarrow as pa

    return pa.schema([pa.field(name, pa.string()) for name in names])


def test_online_gate_refuses_to_widen_the_live_collection() -> None:
    """The planner must choose a private generation before this gate."""
    stored = _spec()
    widened = replace(
        _spec(
            attributes=(
                {"name": "category", "data_type": "string", "filter_role": "user"},
                {"name": "section", "data_type": "string", "filter_role": "user"},
            )
        ),
        arrow_schema=_schema_with("chunk_id", "text", "section"),
    )
    store = _EvolvingStore()
    existing = _metadata(
        descriptor=stored.descriptor, config_fingerprint=stored.config_fingerprint
    )
    existing = replace(existing, schema=_schema_with("chunk_id", "text"))

    with pytest.raises(RunError, match="private generation"):
        _verify_collection_config(store, existing, widened, policy="online")
    assert store.evolved == []


def test_online_still_refuses_a_rebuild_change() -> None:
    """A capability flag cannot make an incompatible change safe — the design
    doc says so outright, and the rows already written really are invalid."""
    store = _EvolvingStore()
    stored = _spec()
    changed = _spec(
        vector={"field": "embedding", "dimensions": 384, "metric": "cosine"}
    )
    existing = _metadata(
        descriptor=stored.descriptor, config_fingerprint=stored.config_fingerprint
    )

    with pytest.raises(RunError, match="requires a rebuild: vector"):
        _verify_collection_config(store, existing, changed, policy="online")

    assert store.evolved == []


def test_fail_policy_points_at_online_for_an_additive_change() -> None:
    """Under the default policy an additive change still stops the run, but the
    error now names the mode that would have applied it."""
    store = _EvolvingStore()
    stored = _spec()
    widened = _spec(display_fields=("title", "author"))
    existing = _metadata(
        descriptor=stored.descriptor, config_fingerprint=stored.config_fingerprint
    )

    with pytest.raises(RunError, match=r"on_index_change: `?online"):
        _verify_collection_config(store, existing, widened, policy="fail")

    assert store.evolved == []


def test_evolving_a_real_collection_adds_the_column_and_keeps_rows(
    tmp_path: Path,
) -> None:
    """End to end against LanceDB: the column appears, the rows survive, and
    the collection's descriptor advances so the next run sees no change."""
    import pyarrow as pa

    store, spec = _real_spec(tmp_path)
    with store:
        store.create_collection(spec)
        # Two rows that must survive the widening untouched.
        store.upsert(
            spec.physical_name,
            [
                IndexedRow("a", {"chunk_id": "a", "text": "x"}, "fp-a"),
                IndexedRow("b", {"chunk_id": "b", "text": "y"}, "fp-b"),
            ],
            id_field=spec.id_field,
            mutation_digest="digest-1",
        )

        widened = replace(
            spec,
            arrow_schema=pa.schema(
                [
                    pa.field("chunk_id", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("section", pa.string()),
                ]
            ),
            descriptor=spec.descriptor.replace('"text_fields"', '"text_fields2"'),
        )
        store.evolve_collection(widened, ["section"])
        after = store.inspect_collection(spec.physical_name)

    assert after is not None
    assert set(after.schema.names) == {"chunk_id", "text", "section"}
    assert after.row_count == 2, "widening must not rewrite or drop rows"
    # Re-stamped as part of evolving, so the next run does not re-evolve.
    assert after.descriptor == widened.descriptor


def test_a_second_run_after_an_evolution_still_validates(tmp_path: Path) -> None:
    """The run *after* a successful widening must not reject the collection.

    Evolving re-stamps the descriptor, so the next run sees no config change
    and falls through to the schema comparison. LanceDB appends added columns
    and creates them nullable — exactly the differences the evolution run
    tolerates — so an order- and nullability-sensitive comparison there fails
    permanently on a collection that was widened correctly (Codex review P1).
    """
    import pyarrow as pa

    store, spec = _real_spec(tmp_path)
    # Declared order puts the new attribute ahead of the display field, while
    # `add_columns` can only append it — the mismatch a strict compare rejects.
    declared = pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("section", pa.string()),
            pa.field("title", pa.string()),
        ]
    )
    with store:
        store.create_collection(
            replace(
                spec,
                arrow_schema=pa.schema(
                    [
                        pa.field("chunk_id", pa.string()),
                        pa.field("text", pa.string()),
                        pa.field("title", pa.string()),
                    ]
                ),
            )
        )
        widened = replace(spec, arrow_schema=declared)
        store.evolve_collection(widened, ["section"])
        after = store.inspect_collection(spec.physical_name)

    assert after is not None
    # The next run finds a matching descriptor, so it evolves nothing and goes
    # straight to schema validation — which must accept the collection it just
    # widened. This is the assertion that fails if that check is stricter than
    # the one used during the widening.
    assert after.descriptor == widened.descriptor
    assert _verify_collection_config(store, after, widened, policy="online") is False
    _validate_collection_schema(after.schema, widened)

    assert not after.schema.equals(declared, check_metadata=False), (
        "fixture must reproduce the ordering difference the bug turns on"
    )


# ─── Vector search strategy is an index build, not a rebuild (issue #461) ─────

_EXACT_VECTOR = {"field": "embedding", "dimensions": 2, "metric": "cosine", "search": "exact"}
_APPROX_VECTOR = {**_EXACT_VECTOR, "search": "approximate"}


def test_switching_vector_search_strategy_is_compatible() -> None:
    """The change this issue is about. `exact` -> `approximate` builds an ANN
    index over vectors that are already published; nothing about a stored row
    changes, so demanding a new collection name and a re-embed charged hours of
    provider time for an index flag."""
    changes = classify_changes(
        _search(vector=_EXACT_VECTOR), _search(vector=_APPROX_VECTOR)
    )

    assert [change.kind for change in changes] == [ChangeKind.COMPATIBLE]
    assert "index build" in changes[0].detail


def test_switching_back_to_exact_is_also_compatible() -> None:
    """Symmetry matters: dropping the index is no more invalidating than
    building it, and an asymmetric rule would be a trap in one direction."""
    changes = classify_changes(
        _search(vector=_APPROX_VECTOR), _search(vector=_EXACT_VECTOR)
    )

    assert [change.kind for change in changes] == [ChangeKind.COMPATIBLE]


@pytest.mark.parametrize(
    "field,value",
    [("dimensions", 4), ("metric", "euclidean"), ("field", "vec")],
)
def test_other_vector_changes_still_require_a_rebuild(field: str, value: Any) -> None:
    """The complement, and the reason this is a classifier rather than a blanket
    exemption: every other vector sub-field changes what a stored row contains
    or what its distances mean."""
    changes = classify_changes(
        _search(vector=_EXACT_VECTOR),
        _search(vector={**_EXACT_VECTOR, field: value}),
    )

    assert [change.kind for change in changes] == [ChangeKind.REBUILD_REQUIRED]


def test_adding_a_vector_to_a_collection_without_one_requires_a_rebuild() -> None:
    """`None` -> a vector config is not a strategy change; the rows carry no
    vector column at all."""
    changes = classify_changes(
        _search(vector=None), _search(vector=_EXACT_VECTOR)
    )

    assert [change.kind for change in changes] == [ChangeKind.REBUILD_REQUIRED]


def test_online_gate_refuses_to_restamp_the_live_strategy() -> None:
    """A strategy change must not publish its new stamp before index readiness."""
    import pyarrow as pa

    schema = pa.schema([pa.field("chunk_id", pa.string())])
    stored = _spec(vector=_EXACT_VECTOR)
    spec = replace(_spec(vector=_APPROX_VECTOR), arrow_schema=schema)
    store = _RecordingStore()
    existing = replace(
        _metadata(
            descriptor=stored.descriptor,
            config_fingerprint=stored.config_fingerprint,
        ),
        schema=schema,
    )

    with pytest.raises(RunError, match="private generation"):
        _verify_collection_config(store, existing, spec, policy="online")
    assert store.evolved == []
    assert store.restamped == []


def test_fail_names_the_strategy_switch_and_points_at_online() -> None:
    """Under the default policy it still raises, but what it says is the whole
    point: the operator learns the change is servable in place rather than
    being sent to rebuild 3.6M rows under a new collection name."""
    stored = _spec(vector=_EXACT_VECTOR)
    spec = _spec(vector=_APPROX_VECTOR)
    existing = _metadata(
        descriptor=stored.descriptor, config_fingerprint=stored.config_fingerprint
    )

    with pytest.raises(RunError) as error:
        _verify_collection_config(_RecordingStore(), existing, spec)

    message = str(error.value)
    assert "on_index_change: online" in message
    assert "requires a rebuild" not in message


# ─── The ANN index type is an index-build field too (issue #476) ──────────────


def _vector_model(**vector: Any) -> ModelConfig:
    return ModelConfig(
        name="ctx",
        depends_on=["ref('chunks')"],
        materialization="incremental",
        search={
            "access": "public",
            "id_field": "chunk_id",
            "document_id_field": "document_id",
            "chunk_id_field": "chunk_id",
            "text_fields": ["text"],
            "return_text_fields": ["text"],
            "full_text": {"fields": ["text"]},
            "query": {"modes": ["vector", "text"]},
            "vector": {
                "field": "embedding",
                "dimensions": 2,
                "metric": "cosine",
                "embedding": {
                    "provider": "fixture",
                    "model": "deterministic-2d-v1",
                    "provider_contract_version": 2,
                    "provider_implementation": "tests:v1",
                    "semantic_config_fingerprint": "deterministic-2d-v1",
                    "dimensions": 2,
                },
                **vector,
            },
        },
    )


def test_the_default_index_type_leaves_every_existing_stamp_untouched(
    tmp_path: Path,
) -> None:
    """The trap this field could have sprung: a default that surfaced in any
    dump of the config would reclassify every published index as changed on
    the first run after upgrade — ADR-0002's "charge everybody once". Absent
    means default, at the model itself: a config that never mentions `index`
    and one that spells out the default are byte-identical in the raw dump, in
    every collection stamp, and in the code version — the last of which is
    hashed from a raw dump in `versioning.py`, which is exactly where a
    consumer-level omission would have leaked."""
    implicit_model = _vector_model(search="approximate")
    explicit_model = _vector_model(search="approximate", index="ivf_hnsw_flat")
    assert implicit_model.search is not None and explicit_model.search is not None

    assert "index" not in implicit_model.search.model_dump()["vector"]
    assert implicit_model.search.model_dump() == explicit_model.search.model_dump()
    assert implicit_model.search.model_dump(mode="json") == explicit_model.search.model_dump(
        mode="json"
    )

    implicit = effective_search_config(implicit_model, [])
    explicit = effective_search_config(explicit_model, [])
    assert implicit == explicit
    for derive in (
        lambda payload: collection_config_fingerprint(payload, store_type="lancedb"),
        lambda payload: descriptor_json(collection_descriptor(payload, store_type="lancedb")),
        lambda payload: legacy_collection_config_fingerprint(payload, store_type="lancedb"),
    ):
        assert derive(implicit) == derive(explicit)

    project = ProjectConfig(name="p")
    assert compute_model_code_version(
        implicit_model, project, tmp_path
    ) == compute_model_code_version(explicit_model, project, tmp_path)


def test_a_chosen_index_type_survives_every_serialization() -> None:
    """The complement: the omission is of the *default* only. A deliberate
    choice is written in both dump modes and round-trips."""
    chosen = SearchVectorConfig(
        field="embedding", dimensions=2, search="approximate", index="ivf_pq", embedding="inherit"
    )

    assert chosen.model_dump()["index"] == "ivf_pq"
    assert chosen.model_dump(mode="json")["index"] == "ivf_pq"
    assert SearchVectorConfig.model_validate(chosen.model_dump()) == chosen


def test_a_chosen_index_type_is_recorded_and_classifies_as_an_index_build() -> None:
    """Only a deliberate choice enters the descriptor, and switching it is the
    same kind of change as `exact` -> `approximate`: a build over vectors that
    are already published, not a re-embed."""
    chosen = effective_search_config(
        _vector_model(search="approximate", index="ivf_pq"), []
    )
    assert chosen["vector"]["index"] == "ivf_pq"

    stored = collection_descriptor(
        effective_search_config(_vector_model(search="approximate"), []),
        store_type="lancedb",
    )["search"]
    current = collection_descriptor(chosen, store_type="lancedb")["search"]
    changes = classify_changes(stored, current)

    assert [change.kind for change in changes] == [ChangeKind.COMPATIBLE]
    assert "index None -> 'ivf_pq'" in changes[0].detail
    assert "index build" in changes[0].detail


def test_switching_mode_and_index_type_together_is_one_compatible_change() -> None:
    """`exact` -> `approximate` plus a chosen type is the whole #473 retest in
    one config edit; it must still be a single compatible change."""
    stored = collection_descriptor(
        effective_search_config(_vector_model(search="exact"), []), store_type="lancedb"
    )["search"]
    current = collection_descriptor(
        effective_search_config(_vector_model(search="approximate", index="ivf_pq"), []),
        store_type="lancedb",
    )["search"]

    assert [c.kind for c in classify_changes(stored, current)] == [ChangeKind.COMPATIBLE]


def test_an_index_type_change_alongside_a_dimension_change_still_rebuilds() -> None:
    """The index-build exemption is exact: bundling it with a change to what a
    row contains does not launder the latter."""
    changes = classify_changes(
        _search(vector=_APPROX_VECTOR),
        _search(vector={**_APPROX_VECTOR, "index": "ivf_pq", "dimensions": 4}),
    )

    assert [change.kind for change in changes] == [ChangeKind.REBUILD_REQUIRED]


@pytest.mark.parametrize("index", ["ivf_pq", "ivf_hnsw_flat"])
def test_an_index_type_under_exact_search_is_refused_at_config_time(index: str) -> None:
    """`exact` builds no index, so an `index` written under it would be accepted
    and silently do nothing — a flag that looks like a decision and is not.
    Judged on whether the key was written: spelling out the default is still a
    choice nothing will honor (Codex review, #477)."""
    with pytest.raises(ValidationError, match="only applies to search: approximate"):
        _vector_model(search="exact", index=index)
# ─── the merge key is indexed (issue #475) ─────────────────────────────────


def _search_config(**overrides: Any) -> SearchConfig:
    payload: dict[str, Any] = {
        "id_field": "chunk_id",
        "text_fields": ("text",),
        "query": {"modes": ("vector",)},
        "vector": {
            "field": "embedding",
            "dimensions": 2,
            "metric": "cosine",
            "search": "exact",
            "embedding": "inherit",
        },
    }
    payload.update(overrides)
    return SearchConfig.model_validate(payload)


def test_the_merge_key_is_among_the_indexed_scalar_fields() -> None:
    """`upsert` merges on `id_field` every incremental publish. Unindexed, that
    join-key predicate is a full column scan and the ack scans the column
    again — two O(table) passes per page (issue #475)."""
    fields = _scalar_index_fields(_search_config())

    assert fields == ("chunk_id",)


def test_the_merge_key_comes_first_and_filterable_attributes_follow() -> None:
    fields = _scalar_index_fields(
        _search_config(
            attributes=(
                {"name": "category", "data_type": "string", "filter_role": "user"},
                {"name": "ignored", "data_type": "string"},
                {"name": "published_at", "data_type": "timestamp", "sortable": True},
            )
        )
    )

    # `ignored` has no filter role and is not sortable, so it earns no index.
    assert fields == ("chunk_id", "category", "published_at")


def test_an_id_field_that_is_also_an_attribute_is_indexed_once() -> None:
    """Asking for the same BTree twice would rebuild it twice per publish."""
    fields = _scalar_index_fields(
        _search_config(
            attributes=(
                {"name": "chunk_id", "data_type": "string", "filter_role": "user"},
            )
        )
    )

    assert fields == ("chunk_id",)
