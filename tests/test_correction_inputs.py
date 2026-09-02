"""Exchange-pair rows for the correction classifier (#456, #329 phase 3).

Documents are built through the **real converter** rather than hand-written.
The first version of these tests wrote the rendered text by hand, and got it
wrong in the one way that mattered: the converter renders a single-line human
prompt *as* the `## [n]` heading and nowhere else, so slicing from after the
heading fed the classifier assistant prose only — the one turn that can hold a
correction was the one turn it could not see. A hand-made fixture agreed with
the bug (PR #458 review). These do not.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

from stel.transcripts.convert import build_document
from stel.transcripts.events import (
    AssistantProse,
    ContextCall,
    ParsedSession,
    ToolCall,
    UserTurn,
)
from stel.transcripts.transforms import correction_inputs

_AT = datetime(2026, 9, 2, 10, 0, tzinfo=UTC)


class _Ctx:
    """The subset of TransformContext this transform reads."""

    def __init__(self, **options: Any) -> None:
        self.options = options


def _search(*, returned: tuple[str, ...], error_code: str | None = None) -> ToolCall:
    return ToolCall(
        name="mcp__stel-context__search_context",
        args_fingerprint="a" * 32,
        files=(),
        ok=error_code is None,
        result_bytes=100,
        timestamp=_AT,
        context=ContextCall(
            model="context_search",
            query_fingerprint="f" * 32,
            query_text=None,
            returned_context_ids=returned,
            returned_chunk_ids=(),
            cited_context_ids=(),
            zero_results=not returned,
            error_code=error_code,
        ),
    )


def _session(*events: Any) -> ParsedSession:
    return ParsedSession(
        harness="claude-code",
        session_id="sess-1",
        source_path=Path("sess-1.jsonl"),
        project_path=None,
        git_branch=None,
        events=tuple(events),
    )


def _rows(session: ParsedSession) -> pl.DataFrame:
    document = build_document(session)
    assert document is not None
    payload = document.model_dump(mode="json")
    payload["document_id"] = "doc-1"
    frame = pl.DataFrame([payload])
    ctx = cast(Any, _Ctx(transcripts="raw_transcripts"))
    return correction_inputs.run({"raw_transcripts": frame}, ctx)


def _corrected_session(**overrides: Any) -> ParsedSession:
    """The shape #456 is about: a claim, then a human correcting it."""
    search = overrides.get("search", _search(returned=("ctx-1", "ctx-2")))
    return _session(
        UserTurn(text="what form type is this filing", timestamp=_AT),
        search,
        AssistantProse(text="It is a 10-Q.", timestamp=_AT),
        UserTurn(text="no, it is a 10-K", timestamp=_AT),
        AssistantProse(text="You are right.", timestamp=_AT),
    )


# ─── the classifier must be able to see the correction ──────────────────────


def test_the_human_correction_reaches_the_classifier_input() -> None:
    """The bug this file exists to prevent. The converter renders a
    single-line prompt as the heading only, so an input that drops headings
    contains the agent's claim and nothing the human said — and no prompt,
    however well written, can find a correction that is not in its input."""
    rows = _rows(_corrected_session())

    assert rows.height == 1
    text = rows["exchange_text"][0]
    assert "no, it is a 10-K" in text, text
    # And the claim being corrected, so the classifier can judge rather than
    # guess from an assertion alone.
    assert "It is a 10-Q." in text


def test_a_pair_carries_both_exchanges_in_order() -> None:
    rows = _rows(_corrected_session())

    text = rows["exchange_text"][0]
    assert text.index("what form type") < text.index("no, it is a 10-K")
    # The converter numbers exchanges from zero; the pair is (N, N+1).
    answered = rows["answered_exchange_ordinal"][0]
    assert rows["exchange_ordinal"][0] == answered + 1


# ─── the ids belong to the exchange that made the claim ─────────────────────


def test_ids_come_from_the_answering_exchange_not_the_correcting_one() -> None:
    """A correction refers to records retrieved *before* it. Taking ids from
    the correcting exchange would key the label to whatever the agent looked
    up after being told it was wrong (PR #458 review)."""
    session = _session(
        UserTurn(text="what form type is this filing", timestamp=_AT),
        _search(returned=("answered-ctx",)),
        AssistantProse(text="It is a 10-Q.", timestamp=_AT),
        UserTurn(text="no, it is a 10-K", timestamp=_AT),
        _search(returned=("looked-up-after",)),
        AssistantProse(text="You are right.", timestamp=_AT),
    )

    rows = _rows(session)

    ids = json.loads(rows["candidate_context_ids"][0])
    assert ids == ["answered-ctx"]
    assert "looked-up-after" not in ids


def test_a_correction_survives_when_the_follow_up_searches_nothing() -> None:
    """The other half of the same bug: reading ids from the correcting
    exchange dropped the candidate entirely whenever the human's correction
    prompted no new search — which is the common case."""
    rows = _rows(_corrected_session())

    assert rows.height == 1
    assert json.loads(rows["candidate_context_ids"][0]) == ["ctx-1", "ctx-2"]


def test_an_answer_that_retrieved_nothing_yields_no_pair() -> None:
    """`eval:` joins on a key, so a correction with no record behind the claim
    cannot become ground truth."""
    session = _session(
        UserTurn(text="what form type is this filing", timestamp=_AT),
        AssistantProse(text="It is a 10-Q.", timestamp=_AT),
        UserTurn(text="no, it is a 10-K", timestamp=_AT),
    )

    assert _rows(session).height == 0


def test_a_failed_search_is_not_a_retrieval() -> None:
    """A denied call returned nothing to be wrong about."""
    session = _corrected_session(
        search=_search(returned=("ctx-1",), error_code="denied")
    )

    assert _rows(session).height == 0


# ─── contracts ──────────────────────────────────────────────────────────────


def test_the_id_space_is_recorded_on_every_row() -> None:
    rows = _rows(_corrected_session())

    assert set(rows["id_space"].to_list()) == {"context_id"}


def test_rows_are_keyed_per_pair_and_stable() -> None:
    first = _rows(_corrected_session())
    second = _rows(_corrected_session())

    assert first["input_id"].to_list() == second["input_id"].to_list()
    assert len(set(first["input_id"].to_list())) == first.height


def test_a_single_exchange_session_has_no_pair() -> None:
    session = _session(
        UserTurn(text="what form type is this filing", timestamp=_AT),
        _search(returned=("ctx-1",)),
        AssistantProse(text="It is a 10-Q.", timestamp=_AT),
    )

    assert _rows(session).height == 0


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
    """The json extraction backend scalarizes nested lists to strings while an
    in-memory frame keeps them; both reach this transform."""
    document = build_document(_corrected_session())
    assert document is not None
    payload = document.model_dump(mode="json")
    payload["document_id"] = "doc-1"
    payload["exchanges"] = json.dumps(payload["exchanges"])
    ctx = cast(Any, _Ctx(transcripts="raw_transcripts"))

    rows = correction_inputs.run(
        {"raw_transcripts": pl.DataFrame([payload])}, ctx
    )

    assert rows.height == 1
    assert json.loads(rows["candidate_context_ids"][0]) == ["ctx-1", "ctx-2"]
