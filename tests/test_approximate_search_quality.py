"""Filtering and ranking under an approximate vector index (issue #520).

#520 reported that an attribute filter, worth a ~60x speedup under `exact`,
bought nothing once the same collection moved to `approximate`, and asked
whether stel filters before or after the search. It filters before: the
queries carry `prefilter=True` and LanceDB resolves them through the
attribute's scalar index. The first test pins that, because the difference is
invisible in a timing but obvious in the results — a postfilter takes the k
nearest rows *overall* and then discards the ones that do not match, so a
selective filter returns almost nothing.

What the investigation did turn up is that an `ivf_pq` index answers from
compressed codes, so the order and the `_distance` it reports are both
approximations. Measured on a 100k-row collection, a filtered query returned
recall@10 of 0.49 against numpy-computed ground truth, with the reported score
off by 0.40 in cosine units; `refine_factor: 10` took that to recall 1.00 and
an exact score. The second test pins the mechanism at a size the suite can
afford: with `refine_factor` set, the scores that come back are the true
distances rather than quantized ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from stel.retrieval import (
    CollectionSpec,
    IndexedRow,
    LanceDBConfig,
    LanceDBStore,
    RetrievalPredicate,
    RetrievalPredicateOperator,
    StoreRole,
)

DIM = 16
ROWS = 600
SYMBOLS = 30
TARGET = "SYM_07"
COLLECTION = "chunks"


def _vectors() -> np.ndarray:
    """Two well-separated groups, so a filter's matches are nowhere near the
    query. The target symbol's rows sit on a different axis from every other
    row; a postfilter would therefore return an empty page for it."""
    rng = np.random.default_rng(7)
    vectors = np.zeros((ROWS, DIM), dtype=np.float32)
    for index in range(ROWS):
        axis = 1 if index % SYMBOLS == 7 else 0
        base = np.zeros(DIM, dtype=np.float32)
        base[axis] = 1.0
        vectors[index] = base + rng.normal(scale=0.05, size=DIM).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


VECTORS = _vectors()
SYMBOL_OF = [f"SYM_{index % SYMBOLS:02d}" for index in range(ROWS)]


def _spec() -> CollectionSpec:
    return CollectionSpec(
        logical_name="ctx",
        physical_name=COLLECTION,
        id_field="id",
        text_fields=(),
        full_text_fields=(),
        attribute_fields=("symbol",),
        scalar_index_fields=("symbol",),
        display_fields=(),
        vector_field="embedding",
        vector_dimensions=DIM,
        distance_metric="cosine",
        vector_search="approximate",
        vector_index="ivf_pq",
        config_fingerprint="cfg-approx",
        descriptor=json.dumps({"vector_search": "approximate"}),
        legacy_config_fingerprint="legacy",
        row_fingerprint="row-fp",
        arrow_schema=pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("symbol", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), DIM)),
            ]
        ),
    )


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """One published, ivf_pq-indexed collection for the whole module: building
    and training the index is the expensive part and none of these tests
    mutate it."""
    path: Path = tmp_path_factory.mktemp("approx")
    handle = LanceDBStore(
        LanceDBConfig(type="lancedb", path=str(path / "lance")),
        project_name="proj",
        target_name="dev",
        alias="default",
        role=StoreRole.PUBLISH,
    )
    rows = [
        IndexedRow(
            str(index),
            {
                "id": str(index),
                "symbol": SYMBOL_OF[index],
                "embedding": VECTORS[index].tolist(),
            },
            f"fp-{index}",
        )
        for index in range(ROWS)
    ]
    with handle:
        handle.create_collection(_spec())
        handle.upsert(COLLECTION, rows, id_field="id", mutation_digest="d1")
        handle.ensure_indexes(_spec())
        yield handle


def _query_vector() -> list[float]:
    """Aimed squarely at the *non*-target group."""
    vector = np.zeros(DIM, dtype=np.float32)
    vector[0] = 1.0
    return vector.tolist()


def _ids(table: pa.Table) -> list[str]:
    return table.column("id").to_pylist()


def _search(store: Any, *, predicates: tuple[RetrievalPredicate, ...] = (), **extra: Any):
    return store.vector_search(
        COLLECTION,
        _query_vector(),
        vector_field="embedding",
        limit=10,
        columns=["id", "symbol"],
        predicates=predicates,
        **extra,
    )


def test_the_index_is_approximate_and_the_attribute_is_scalar_indexed(store: Any) -> None:
    """The fixture is only meaningful if both indexes exist."""
    table = store._open_owned_table(COLLECTION)
    kinds = {tuple(index.columns): index.index_type for index in table.list_indices()}
    assert kinds[("embedding",)] == "IvfPq"
    assert kinds[("symbol",)] == "BTree"


def test_a_selective_filter_still_returns_a_full_page(store: Any) -> None:
    """The prefilter pin, and #520's actual question.

    The query points at the non-target group, so the ten nearest rows overall
    contain no target row at all. A postfilter would intersect those ten with
    the filter and return nothing; a prefilter searches within the filtered
    rows and returns a full page of them.
    """
    unfiltered = _search(store)
    assert TARGET not in unfiltered.column("symbol").to_pylist()

    filtered = _search(
        store, predicates=(RetrievalPredicate("symbol", RetrievalPredicateOperator.EQUAL, TARGET),)
    )
    assert filtered.num_rows == 10, "a selective filter returned a short page"
    assert set(filtered.column("symbol").to_pylist()) == {TARGET}
    assert not set(_ids(filtered)) & set(_ids(unfiltered))


def test_refine_factor_returns_true_distances_not_quantized_ones(store: Any) -> None:
    """`ivf_pq` scores from compressed codes; refine re-ranks against the
    stored vectors. Ground truth is computed in numpy so the assertion does
    not rest on LanceDB agreeing with itself."""
    predicates = (RetrievalPredicate("symbol", RetrievalPredicateOperator.EQUAL, TARGET),)
    refined = _search(store, predicates=predicates, refine_factor=10)
    # Guard against a vacuous pass: an empty page would satisfy every
    # assertion below without checking anything.
    assert refined.num_rows == 10

    query = np.array(_query_vector(), dtype=np.float32)
    query /= np.linalg.norm(query)
    for row_id, reported in zip(
        _ids(refined), refined.column("_distance").to_pylist(), strict=True
    ):
        stored = VECTORS[int(row_id)]
        true_distance = 1.0 - float(stored @ query / np.linalg.norm(stored))
        assert reported == pytest.approx(true_distance, abs=1e-4)

    # And the rows are ordered by that true distance.
    distances = refined.column("_distance").to_pylist()
    assert distances == sorted(distances)


def test_refine_factor_is_optional_and_absent_by_default(store: Any) -> None:
    """Nothing changes for a collection that does not ask for it: the same
    query without the knob still answers, so enabling it stays a deliberate
    choice rather than a new requirement."""
    predicates = (RetrievalPredicate("symbol", RetrievalPredicateOperator.EQUAL, TARGET),)
    plain = _search(store, predicates=predicates)
    assert plain.num_rows == 10
    assert set(plain.column("symbol").to_pylist()) == {TARGET}
