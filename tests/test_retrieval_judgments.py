"""Candidate retrieval judgments derived from the transcript corpus (#380).

These rows are #329 phase 3's raw material, and rule 2 is the constraint that
shapes them: they are candidates a human promotes, never goldens, and the
distinctions between "cited", "returned but not cited", and "matched nothing"
have to survive into the relation or a reviewer cannot tell evidence from
absence of evidence.
"""
from __future__ import annotations

import json
from typing import Any, cast

import polars as pl
import pytest

from stel.transcripts.transforms.retrieval_judgments import (
    ID_SPACE,
    declared_dependencies,
    declared_incremental_contract,
    run,
    validate_options,
)

_CTX_A = "a" * 32
_CTX_B = "b" * 32
_FP = "f" * 32


class _Ctx:
    """The subset of TransformContext this transform reads."""

    def __init__(self, **options: Any) -> None:
        self.options = options


def _call(**overrides: Any) -> dict[str, Any]:
    call = {
        "model": "context_search",
        "query_fingerprint": _FP,
        "query_text": None,
        "returned_context_ids": [_CTX_A, _CTX_B],
        "returned_chunk_ids": ["c" * 32, "d" * 32],
        "cited_context_ids": [_CTX_A],
        "zero_results": False,
        "error_code": None,
    }
    call.update(overrides)
    return call


def _frame(*calls: dict[str, Any], as_json: bool = False) -> pl.DataFrame:
    exchanges = [
        {
            "ordinal": 0,
            "heading": "[0] why",
            "started_at": "2026-08-20T14:00:00Z",
            "context_calls": list(calls),
        }
    ]
    return pl.DataFrame(
        {
            "document_id": ["doc-1"],
            "session_id": ["sess-1"],
            "harness": ["claude-code"],
            # The json backend scalarizes nested lists to a JSON string; an
            # in-memory frame keeps them as objects. Both reach the transform.
            "exchanges": [json.dumps(exchanges) if as_json else exchanges],
        }
    )


def _run(frame: pl.DataFrame) -> pl.DataFrame:
    context = cast(Any, _Ctx(transcripts="raw_transcripts"))
    return run({"raw_transcripts": frame}, context)


# ─── what a row means ───────────────────────────────────────────────────────


@pytest.mark.parametrize("as_json", [False, True])
def test_cited_and_uncited_returns_are_distinguished(as_json: bool) -> None:
    out = _run(_frame(_call(), as_json=as_json))

    by_id = {row["context_id"]: row for row in out.iter_rows(named=True)}
    assert by_id[_CTX_A]["judgment"] == "cited"
    assert by_id[_CTX_B]["judgment"] == "returned_not_cited"
    assert {row["id_space"] for row in out.iter_rows(named=True)} == {ID_SPACE}
    assert by_id[_CTX_A]["session_id"] == "sess-1"
    assert by_id[_CTX_A]["harness"] == "claude-code"
    assert by_id[_CTX_A]["exchange_ordinal"] == 0
    assert by_id[_CTX_A]["context_model"] == "context_search"
    assert by_id[_CTX_A]["query_fingerprint"] == _FP


def test_a_zero_result_query_is_the_only_negative() -> None:
    out = _run(_frame(_call(returned_context_ids=[], cited_context_ids=[], zero_results=True)))

    rows = out.to_dicts()
    assert len(rows) == 1
    assert rows[0]["judgment"] == "zero_result"
    assert rows[0]["context_id"] is None


def test_a_failed_call_yields_no_candidates() -> None:
    """A denied or timed-out search returned nothing because it failed. It is
    neither a judgment nor a negative, so it must not appear at all."""
    out = _run(
        _frame(
            _call(
                returned_context_ids=[],
                cited_context_ids=[],
                zero_results=False,
                error_code="not_found_or_denied",
            )
        )
    )

    assert out.height == 0
    assert out.columns == list(_run(_frame(_call())).columns)


def test_a_call_without_a_fingerprint_is_skipped() -> None:
    # No fingerprint means nothing to join a promoted golden back to.
    assert _run(_frame(_call(query_fingerprint=None))).height == 0


def test_query_text_travels_only_when_the_corpus_captured_it() -> None:
    assert _run(_frame(_call())).to_dicts()[0]["query_text"] is None
    captured = _run(_frame(_call(query_text="how does state advance")))
    assert captured.to_dicts()[0]["query_text"] == "how does state advance"


# ─── identity and provenance ────────────────────────────────────────────────


def test_candidate_ids_are_stable_and_distinct_per_row() -> None:
    first = _run(_frame(_call()))
    second = _run(_frame(_call()))

    assert first.to_dicts() == second.to_dicts()
    assert first["candidate_id"].n_unique() == first.height


def test_a_reissued_query_in_one_exchange_keeps_distinct_child_keys() -> None:
    """`candidate_id` is the incremental contract's `child_key`, which
    `replace_children` upserts on. An agent that reissues a query within one
    exchange judges the same ids again, so without the call ordinal in the
    identity those rows collide onto one key — a duplicate upsert key, not a
    deduplication.
    """
    out = _run(_frame(_call(), _call()))

    assert out.height == 4
    assert out["candidate_id"].n_unique() == out.height
    assert out["call_ordinal"].to_list() == [0, 0, 1, 1]


def test_the_same_id_judged_differently_gets_a_distinct_row() -> None:
    out = _run(_frame(_call(), _call(cited_context_ids=[])))

    assert out["candidate_id"].n_unique() == out.height


def test_every_row_carries_the_provenance_a_reviewer_asks_for() -> None:
    out = _run(_frame(_call()))

    for row in out.iter_rows(named=True):
        assert row["session_id"] and row["harness"]
        assert row["source_document_id"] == "doc-1"
        assert row["observed_at"] == "2026-08-20T14:00:00Z"


# ─── contract ───────────────────────────────────────────────────────────────


def test_options_are_validated_and_dependencies_declared() -> None:
    assert declared_dependencies({"transcripts": "raw_transcripts"}) == (
        "raw_transcripts",
    )
    with pytest.raises(ValueError, match="unknown options"):
        validate_options({"transcripts": "t", "nope": 1})
    with pytest.raises(ValueError, match="transcripts"):
        validate_options({})


def test_the_incremental_contract_treats_a_session_as_the_parent() -> None:
    contract = declared_incremental_contract({"transcripts": "raw_transcripts"})

    assert contract.parent_key == "source_document_id"
    assert contract.child_key == "candidate_id"
    assert contract.parent_source == "raw_transcripts"
    assert contract.parent_source_key == "document_id"
    contract.validate_against(["raw_transcripts"])


def test_an_exchange_without_context_calls_contributes_nothing() -> None:
    frame = pl.DataFrame(
        {
            "document_id": ["doc-1"],
            "session_id": ["sess-1"],
            "harness": ["codex"],
            "exchanges": [[{"ordinal": 0, "heading": "[0] hi", "started_at": None}]],
        }
    )

    assert _run(frame).height == 0
