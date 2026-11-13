"""Group results are fused by document rank, not by native score (#778)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from stel.search import SearchError, SearchProvenance, SearchResult, _fuse_group_results


def _result(record: str, document: str, member: str, score: float) -> SearchResult:
    return SearchResult(
        record_id=record,
        document_id=document,
        chunk_id=record,
        rank=1,
        score=score,
        raw_score=score,
        raw_score_kind="distance",
        text={},
        metadata={},
        display={},
        contributing_ranks={},
        provenance=SearchProvenance(
            project="p", model=member, unique_id=member, target="t", store_type="fake",
            logical_collection=member, physical_collection=member, generation=member,
            upstream="upstream", embedding=None,
        ),
    )


def test_document_rank_fusion_is_stable_and_preserves_member_generation() -> None:
    ranked = {
        "a": [_result("a1", "doc-a", "a", 0.01), _result("a2", "doc-a", "a", 99)],
        "b": [_result("b1", "doc-b", "b", 999), _result("b2", "doc-a", "b", 9999)],
    }
    result = _fuse_group_results(ranked, limit=3)
    assert [item.record_id for item in result] == ["a1", "b1", "a2"]
    assert [item.rank for item in result] == [1, 2, 3]
    assert [item.provenance.generation for item in result] == ["a", "b", "a"]
    assert all(item.raw_score is None for item in result)
    reversed_input = dict(reversed(list(ranked.items())))
    assert result == _fuse_group_results(reversed_input, limit=3)


def test_missing_document_id_fails_instead_of_fusing_at_chunk_grain() -> None:
    hit = replace(_result("a1", "doc-a", "a", 1), document_id=None)
    with pytest.raises(SearchError, match="without a document ID"):
        _fuse_group_results({"a": [hit]}, limit=1)
