"""The checks `stel serving activate` makes before it claims anything (issue #615).

Pure functions over the collection's stamp and the upstream count, so each
refusal is pinned without a store or a project: the error text is what an
operator acts on, and each one has to name the two values that disagree.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from stel.execution.activation import activation_refusal
from stel.retrieval import CollectionMetadata, CollectionSpec


def _spec(config_fingerprint: str = "cfg-current") -> CollectionSpec:
    return CollectionSpec(
        logical_name="chunk_search",
        physical_name="proj__prod__chunk_search__g12fbc",
        id_field="chunk_id",
        text_fields=("text",),
        full_text_fields=("text",),
        attribute_fields=(),
        scalar_index_fields=("chunk_id",),
        display_fields=(),
        vector_field="embedding",
        vector_dimensions=768,
        distance_metric="cosine",
        vector_search="approximate",
        vector_index="ivf_pq",
        config_fingerprint=config_fingerprint,
        descriptor="{}",
        legacy_config_fingerprint="legacy",
        row_fingerprint=config_fingerprint,
        arrow_schema=pa.schema([]),
    )


def _existing(*, rows: int, config_fingerprint: str = "cfg-current") -> CollectionMetadata:
    return CollectionMetadata(
        physical_name="proj__prod__chunk_search__g12fbc",
        config_fingerprint=config_fingerprint,
        descriptor="{}",
        physical_generation="17b61b4a",
        row_count=rows,
        schema=pa.schema([]),
    )


def test_a_complete_matching_generation_is_not_refused() -> None:
    assert (
        activation_refusal(
            _existing(rows=3_644_778),
            _spec(),
            upstream_rows=3_644_778,
            collection="proj__prod__chunk_search__g12fbc",
        )
        is None
    )


def test_a_missing_collection_is_named() -> None:
    reason = activation_refusal(None, _spec(), upstream_rows=10, collection="nowhere")
    assert reason is not None
    assert "'nowhere' does not exist" in reason


def test_another_configuration_is_refused_before_row_counts_are_compared() -> None:
    """The fingerprint check comes first: a collection built for another
    configuration is wrong however many rows it has, and activating it would
    answer queries with an index never built for them."""
    reason = activation_refusal(
        _existing(rows=10, config_fingerprint="cfg-old"),
        _spec(),
        upstream_rows=10,
        collection="c",
    )
    assert reason is not None
    assert "different configuration" in reason
    assert "Republish" in reason


@pytest.mark.parametrize("held", [3_644_778 - 1, 1_825_000])
def test_a_generation_behind_the_upstream_is_not_refused(held: int) -> None:
    """One filing short or half the corpus short: both are rows the next
    incremental run publishes, and refusing would send the operator to a
    resume that re-reads the corpus to add them. The command reports the gap
    instead; this check only has to let it through."""
    assert (
        activation_refusal(
            _existing(rows=held), _spec(), upstream_rows=3_644_778, collection="c"
        )
        is None
    )


def test_a_generation_holding_rows_the_upstream_does_not_is_refused() -> None:
    """Rows the upstream does not have cannot be told, by a count, from another
    relation's collection, so this direction stays a refusal with both numbers."""
    reason = activation_refusal(
        _existing(rows=3_644_778), _spec(), upstream_rows=3_644_777, collection="c"
    )
    assert reason is not None
    assert "holds 3644778 row(s)" in reason
    assert "upstream relation has only 3644777" in reason
    assert "Resume the publish" in reason
