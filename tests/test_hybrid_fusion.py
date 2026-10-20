"""Hybrid fusion at document grain (issue #701, ADR-0036).

The queries go through `_execute_query` with a store double that returns
fixed legs, so the dispatch from a search config to its fusion is pinned
along with the fusion itself.
"""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa

from stel.config.model import (
    ModelConfig,
    SearchConfig,
    SearchFullTextConfig,
    SearchQueryConfig,
    SearchVectorConfig,
)
from stel.retrieval.base import DistanceMetric, RetrievalPredicate
from stel.search import SearchRequest, _execute_query, _ScoredRow
from stel.timing import PhaseTimings

Leg = Sequence[tuple[str, str | None]]  # (chunk_id, document_id), best first


class _LegStore:
    """Returns each leg as given, best first, in the shape a store returns."""

    def __init__(self, *, vector: Leg, text: Leg) -> None:
        self._vector = vector
        self._text = text

    def vector_search(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        vector_field: str,
        metric: DistanceMetric,
        limit: int,
        columns: Sequence[str] | None = None,
        predicates: Sequence[RetrievalPredicate] = (),
        refine_factor: int | None = None,
        nprobes: int | None = None,
    ) -> pa.Table:
        return _table(self._vector[:limit], "_distance", ascending=True)

    def text_search(
        self,
        collection: str,
        query: str,
        *,
        text_field: str,
        limit: int,
        columns: Sequence[str] | None = None,
        predicates: Sequence[RetrievalPredicate] = (),
    ) -> pa.Table:
        return _table(self._text[:limit], "_score", ascending=False)


def _table(leg: Leg, score_column: str, *, ascending: bool) -> pa.Table:
    scores = [float(i) if ascending else float(len(leg) - i) for i in range(len(leg))]
    return pa.table(
        {
            "chunk_id": [chunk for chunk, _ in leg],
            "document_id": pa.array([document for _, document in leg], type=pa.string()),
            score_column: scores,
        }
    )


def _hybrid(store: _LegStore, *, document_id_field: str | None) -> list[_ScoredRow]:
    model = ModelConfig(
        name="chunk_search",
        materialization="incremental",
        search=SearchConfig(
            id_field="chunk_id",
            document_id_field=document_id_field,
            text_fields=("text",),
            vector=SearchVectorConfig(field="embedding", dimensions=2, embedding="inherit"),
            full_text=SearchFullTextConfig(fields=("text",)),
            query=SearchQueryConfig(modes=frozenset({"hybrid"})),
        ),
    )
    return _execute_query(
        model,
        store,
        "chunk_search",
        SearchRequest(model="chunk_search", query="q", vector=(1.0, 0.0), limit=10),
        (1.0, 0.0),
        (),
        80,
        ("chunk_id", "document_id"),
        PhaseTimings(),
    )


def _boilerplate(order: Sequence[int]) -> list[tuple[str, str | None]]:
    """Ten chunks from five other documents, in the given order."""
    return [(f"x{i}", f"X{i // 2}") for i in order]


def test_a_document_first_in_both_legs_through_different_chunks_ranks_first() -> None:
    """#701: chunk-grain RRF put every chunk ranked middling in both legs
    above a document that was first in both, through a different chunk in
    each, so the document fell out of the top five. abbv_annual_report
    scored recall 1.0 from vector and from text, and 0.0 from hybrid."""
    store = _LegStore(
        vector=[("d_vector", "D"), *_boilerplate([3, 7, 0, 9, 4, 1, 8, 2, 6, 5])],
        text=[("d_text", "D"), *_boilerplate([5, 2, 8, 0, 6, 9, 1, 4, 7, 3])],
    )

    fused = _hybrid(store, document_id_field="document_id")

    assert fused[0].values["document_id"] == "D"
    assert fused[0].score == 1.0


def test_each_document_contributes_its_best_chunk_before_any_its_second() -> None:
    """Interleaved, not grouped: a record-grain caller (MCP `search_context`)
    sees one chunk per document first. A chunk scores its document's fused
    score, so A's second chunk outscores B's first; rank is the order."""
    store = _LegStore(
        vector=[("a1", "A"), ("b1", "B"), ("a2", "A"), ("b2", "B")],
        text=[("a1", "A"), ("b1", "B"), ("a2", "A"), ("b2", "B")],
    )

    fused = _hybrid(store, document_id_field="document_id")

    assert [row.record_id for row in fused] == ["a1", "b1", "a2", "b2"]
    assert fused[2].score > fused[1].score
    assert fused[2].contributing_ranks == {"vector": 3, "text": 3}


def test_without_a_document_id_field_fusion_stays_at_chunk_grain() -> None:
    store = _LegStore(
        vector=[("d_vector", "D"), *_boilerplate([3, 7, 0, 9, 4, 1, 8, 2, 6, 5])],
        text=[("d_text", "D"), *_boilerplate([5, 2, 8, 0, 6, 9, 1, 4, 7, 3])],
    )

    fused = _hybrid(store, document_id_field=None)

    assert {row.record_id for row in fused[:10]} == {f"x{i}" for i in range(10)}


def test_a_row_without_a_document_id_is_its_own_document() -> None:
    """Even when its record id equals another row's document id."""
    store = _LegStore(
        vector=[("A", None), ("b1", "B"), ("a1", "A")],
        text=[("b1", "B"), ("a1", "A"), ("A", None)],
    )

    fused = _hybrid(store, document_id_field="document_id")

    # Ranked apart: B (2nd, 1st), the orphan (1st, 3rd), A (3rd, 2nd). Merged
    # into document A, the orphan would lift it to tie B and lead.
    assert [row.record_id for row in fused] == ["b1", "A", "a1"]
