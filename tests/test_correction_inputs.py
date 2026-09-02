"""Exchange-grain rows for the correction classifier (#456, #329 phase 3).

The `eval:` half of phase 3 needs labels attached to *records*, and this is
the step that decides which records an exchange could possibly be correcting.
Every test here is about a decision the module makes on the reviewer's behalf,
because those are the ones that quietly produce useless ground truth.
"""
from __future__ import annotations

import json
from typing import Any, cast

import polars as pl
import pytest

from stel.transcripts.transforms import correction_inputs


class _Ctx:
    """The subset of TransformContext this transform reads."""

    def __init__(self, **options: Any) -> None:
        self.options = options


def _call(**overrides: Any) -> dict[str, Any]:
    call = {
        "query_fingerprint": "f" * 32,
        "context_model": "context_search",
        "returned_context_ids": ["ctx-1", "ctx-2"],
        "cited_context_ids": ["ctx-2"],
        "error_code": None,
    }
    call.update(overrides)
    return call


def _document(**overrides: Any) -> dict[str, Any]:
    document = {
        "document_id": "doc-1",
        "session_id": "sess-1",
        "harness": "claude-code",
        "text": (
            "## [1] what form type is this filing\n"
            "It is a 10-Q.\n\n"
            "## [2] no, it is a 10-K\n"
            "You are right, it is a 10-K.\n"
        ),
        "exchanges": [
            {
                "ordinal": 1,
                "heading": "[1] what form type is this filing",
                "started_at": "2026-09-02T10:00:00",
                "context_calls": [_call()],
            },
            {
                "ordinal": 2,
                "heading": "[2] no, it is a 10-K",
                "started_at": "2026-09-02T10:01:00",
                "context_calls": [_call()],
            },
        ],
    }
    document.update(overrides)
    return document


def _run(*documents: dict[str, Any]) -> pl.DataFrame:
    frame = pl.DataFrame(documents)
    ctx = cast(Any, _Ctx(transcripts="raw_transcripts"))
    return correction_inputs.run({"raw_transcripts": frame}, ctx)


# ─── the prose a classifier reads ───────────────────────────────────────────


def test_each_exchange_carries_its_own_prose() -> None:
    """Sliced by heading, so a correction in exchange 2 is not read as
    context for exchange 1 -- the classifier is asked about one turn."""
    rows = _run(_document()).sort("exchange_ordinal")

    assert rows["exchange_ordinal"].to_list() == [1, 2]
    assert "It is a 10-Q." in rows["exchange_text"][0]
    assert "10-K" not in rows["exchange_text"][0]
    assert "You are right, it is a 10-K." in rows["exchange_text"][1]


def test_a_fingerprint_only_corpus_yields_nothing() -> None:
    """#329 rule 1: text exists here only because a harness chose to capture
    it. Without prose there is nothing to classify, and emitting empty rows
    would invite a classifier to guess from headings alone."""
    rows = _run(_document(text=""))

    assert rows.height == 0


# ─── which records a correction can attach to ───────────────────────────────


def test_an_exchange_with_no_context_call_is_not_emitted() -> None:
    """The constraint that shapes this design: `eval.expected` joins on a key,
    and a correction with no record to attach to cannot become one. The agent
    had to retrieve something to be corrected about it."""
    document = _document()
    document["exchanges"][0]["context_calls"] = []

    rows = _run(document)

    assert rows["exchange_ordinal"].to_list() == [2]


def test_cited_ids_come_before_merely_returned_ones() -> None:
    """Both are candidate subjects -- an agent can be wrong about a chunk it
    never named -- but the cited one is the better guess, so it leads."""
    rows = _run(_document())

    ids = json.loads(rows["candidate_context_ids"][0])
    assert ids == ["ctx-2", "ctx-1"]


def test_a_failed_call_contributes_no_ids() -> None:
    """A denied or timed-out search returned nothing to be wrong about."""
    document = _document()
    document["exchanges"][0]["context_calls"] = [_call(error_code="denied")]

    rows = _run(document)

    assert rows["exchange_ordinal"].to_list() == [2]


def test_the_id_space_is_recorded_on_every_row() -> None:
    """Promotion has to reconcile against the target index's `id_field`
    rather than assume it, exactly as the retrieval half does (#380 c3)."""
    rows = _run(_document())

    assert set(rows["id_space"].to_list()) == {"context_id"}


# ─── contracts ──────────────────────────────────────────────────────────────


def test_rows_are_keyed_per_exchange_and_stable() -> None:
    first = _run(_document())
    second = _run(_document())

    assert first["input_id"].to_list() == second["input_id"].to_list()
    assert len(set(first["input_id"].to_list())) == first.height


def test_the_incremental_contract_replaces_a_session_as_a_unit() -> None:
    contract = correction_inputs.declared_incremental_contract(
        {"transcripts": "raw_transcripts"}
    )

    assert contract.parent_key == "source_document_id"
    assert contract.child_key == "input_id"
    assert contract.parent_source == "raw_transcripts"


def test_options_are_validated() -> None:
    with pytest.raises(ValueError, match="unknown options"):
        correction_inputs.validate_options({"transcripts": "x", "nope": 1})
    with pytest.raises(ValueError, match="requires `transcripts:`"):
        correction_inputs.validate_options({})


def test_list_columns_survive_a_json_scalarizing_backend() -> None:
    """Extraction backends disagree about nested types: the json backend
    scalarizes to a string while an in-memory frame keeps lists. Both reach
    this transform, so both have to work."""
    document = _document()
    document["exchanges"] = json.dumps(document["exchanges"])

    rows = _run(document)

    assert rows.height == 2
    assert json.loads(rows["candidate_context_ids"][0]) == ["ctx-2", "ctx-1"]
