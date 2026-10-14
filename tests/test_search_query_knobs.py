"""The query-time knobs a `search:` model declares reach the store (issue #684).

`refine_factor` (issue #520) and `nprobes` are read from the model's
`search.vector` block and passed to the store on every vector query, the
vector half of a hybrid query included. The store tests pin that the store
applies what it is given; this pins that the search layer gives it what the
model declared, so a knob set in YAML cannot be silently dropped between the
two.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from stel.config.model import ModelConfig
from stel.search import SearchMode, SearchRequest, _execute_query
from stel.timing import PhaseTimings


def _model(**vector: Any) -> ModelConfig:
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
                "search": "approximate",
                "index": "ivf_pq",
                "embedding": "inherit",
                **vector,
            },
        },
    )


class _RecordingStore:
    """Answers both halves of a hybrid query and records what each was asked."""

    def __init__(self) -> None:
        self.vector_calls: list[dict[str, Any]] = []

    def vector_search(self, collection: str, vector: Any, **kwargs: Any) -> pa.Table:
        self.vector_calls.append(kwargs)
        return pa.Table.from_pylist([{"chunk_id": "a", "text": "t", "_distance": 0.1}])

    def text_search(self, collection: str, query: str, **kwargs: Any) -> pa.Table:
        return pa.Table.from_pylist([{"chunk_id": "a", "text": "t", "_score": 1.0}])


def _run(model: ModelConfig, mode: SearchMode) -> list[dict[str, Any]]:
    store = _RecordingStore()
    rows = _execute_query(
        model,
        store,
        "chunks",
        SearchRequest(model="ctx", query="inflation", mode=mode, limit=5),
        (0.6, 0.8),
        (),
        5,
        ("chunk_id", "text"),
        PhaseTimings(),
    )
    assert rows, "an empty result would make the call assertions vacuous"
    return store.vector_calls


def test_declared_knobs_reach_the_store_in_vector_and_hybrid_mode() -> None:
    model = _model(nprobes=100, refine_factor=30)
    for mode in (SearchMode.VECTOR, SearchMode.HYBRID):
        (call,) = _run(model, mode)
        assert call["nprobes"] == 100, mode
        assert call["refine_factor"] == 30, mode


def test_undeclared_knobs_are_passed_as_unset_not_invented() -> None:
    """The store's default is what every published collection was measured
    at; the search layer must not substitute a number of its own."""
    (call,) = _run(_model(), SearchMode.HYBRID)
    assert call["nprobes"] is None
    assert call["refine_factor"] is None
