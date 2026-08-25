"""Candidate retrieval judgments derived from the transcript corpus (#380).

#329 phase 3. The transcript corpus records, per exchange, which context ids a
`search_context` call returned and which of them the assistant then named
(issue #380, `transcript/v1.1`). That is the raw material of a retrieval
judgment, and this transform reshapes it into one candidate row per
(exchange, returned id).

**Candidates, never goldens.** Nothing here is read by `retrieval_tests:` or
`eval:`, and nothing promotes itself at any confidence. That is #329 rule 2,
and it is the whole reason this relation has its own name: deriving eval
labels from the model's own behaviour and then tuning to them is
self-reinforcement. A human promotes; see the `judgment` values below for what
each row is and — more importantly — is not evidence of.

**What a row means**

- ``cited`` — the answer named this id after the call returned it. The
  strongest signal available without asking a human, and the intended input to
  a promoted golden's ``relevant_ids``.
- ``returned_not_cited`` — returned, not named. **Not** evidence of
  irrelevance: an agent may use a chunk without quoting its id, or answer from
  the first hit alone. Recorded because a reviewer promoting a query needs to
  see what else came back, never as a negative.
- ``zero_result`` — the query matched nothing. The one honest negative in the
  set, and the cheapest quality signal the corpus has.

Failed calls are excluded entirely: a denied or timed-out search returned
nothing because it failed, not because the corpus lacked a match, so it is
neither a judgment nor a negative (that is what `error_code` is for).
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from ...hashing import canonical_fingerprint
from ...transforms import IncrementalContract, TransformContext

# The id space these candidates are expressed in. `search_context` results
# carry both a `context_id` and a `chunk_id`, and a search index keys on one or
# the other (`id_field`). Promotion has to reconcile that against the target
# index rather than assume, so the space is recorded on every row instead of
# being implied (issue #380, constraint 3).
ID_SPACE = "context_id"

JUDGMENT_CITED = "cited"
JUDGMENT_RETURNED_NOT_CITED = "returned_not_cited"
JUDGMENT_ZERO_RESULT = "zero_result"

_SCHEMA: dict[str, pl.DataType] = {
    "candidate_id": pl.String(),
    "source_document_id": pl.String(),
    "session_id": pl.String(),
    "harness": pl.String(),
    "exchange_ordinal": pl.Int64(),
    "call_ordinal": pl.Int64(),
    "context_model": pl.String(),
    "query_fingerprint": pl.String(),
    "query_text": pl.String(),
    "id_space": pl.String(),
    "context_id": pl.String(),
    "judgment": pl.String(),
    "observed_at": pl.String(),
}


def validate_options(options: Mapping[str, Any]) -> None:
    unknown = sorted(set(options) - {"transcripts"})
    if unknown:
        raise ValueError(
            f"retrieval_judgments: unknown options {unknown}; expected 'transcripts'"
        )
    transcripts = options.get("transcripts")
    if not isinstance(transcripts, str) or not transcripts.strip():
        raise ValueError(
            "retrieval_judgments requires `transcripts:` naming the model that "
            "holds transcript/v1.1 rows"
        )


def declared_dependencies(options: Mapping[str, Any]) -> tuple[str, ...]:
    validate_options(options)
    return (str(options["transcripts"]),)


def declared_incremental_contract(options: Mapping[str, Any]) -> IncrementalContract:
    """One transcript document is one parent.

    A session's landing document is rewritten whole when the session grows, so
    its candidates are replaced as a unit — which is exactly the parent/child
    shape `IncrementalContract` exists for.
    """
    validate_options(options)
    return IncrementalContract(
        parent_key="source_document_id",
        child_key="candidate_id",
        parent_source=str(options["transcripts"]),
        parent_source_key="document_id",
    )


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    validate_options(ctx.options)
    frame = deps[str(ctx.options["transcripts"])]
    rows: list[dict[str, Any]] = []
    for document in frame.iter_rows(named=True):
        rows.extend(_document_candidates(document))
    return pl.DataFrame(rows, schema=_SCHEMA)


def _document_candidates(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    session_id = _text(document.get("session_id"))
    harness = _text(document.get("harness"))
    source_document_id = _text(document.get("document_id"))
    rows: list[dict[str, Any]] = []
    for exchange in _exchanges(document.get("exchanges")):
        ordinal = exchange.get("ordinal")
        if not isinstance(ordinal, int):
            continue
        observed_at = _text(exchange.get("started_at"))
        for call_ordinal, call in enumerate(_sequence(exchange.get("context_calls"))):
            if not isinstance(call, Mapping):
                continue
            rows.extend(
                _call_candidates(
                    call,
                    source_document_id=source_document_id,
                    session_id=session_id,
                    harness=harness,
                    ordinal=ordinal,
                    call_ordinal=call_ordinal,
                    observed_at=observed_at,
                )
            )
    return rows


def _call_candidates(
    call: Mapping[str, Any],
    *,
    source_document_id: str,
    session_id: str,
    harness: str,
    ordinal: int,
    call_ordinal: int,
    observed_at: str,
) -> list[dict[str, Any]]:
    # A failed call is neither a judgment nor a negative (issue #380).
    if call.get("error_code") is not None:
        return []
    fingerprint = _text(call.get("query_fingerprint"))
    if not fingerprint:
        return []
    context_model = _text(call.get("context_model") or call.get("model"))
    query_text = call.get("query_text")
    returned = [_text(value) for value in _sequence(call.get("returned_context_ids"))]
    cited = {_text(value) for value in _sequence(call.get("cited_context_ids"))}

    def row(context_id: str | None, judgment: str) -> dict[str, Any]:
        return {
            "candidate_id": canonical_fingerprint(
                {
                    "session": session_id,
                    "exchange": ordinal,
                    # The call's position within the exchange: an agent that
                    # reissues a query in one exchange makes two observations,
                    # and collapsing them onto one child key would be a
                    # duplicate upsert key, not a deduplication.
                    "call": call_ordinal,
                    "query": fingerprint,
                    "context": context_id or "",
                    "judgment": judgment,
                },
                domain="stel.retrieval-judgment-candidate",
            ),
            "source_document_id": source_document_id,
            "session_id": session_id,
            "harness": harness,
            "exchange_ordinal": ordinal,
            "call_ordinal": call_ordinal,
            "context_model": context_model or None,
            "query_fingerprint": fingerprint,
            "query_text": query_text if isinstance(query_text, str) else None,
            "id_space": ID_SPACE,
            "context_id": context_id,
            "judgment": judgment,
            "observed_at": observed_at or None,
        }

    if bool(call.get("zero_results")):
        return [row(None, JUDGMENT_ZERO_RESULT)]
    return [
        row(
            context_id,
            JUDGMENT_CITED if context_id in cited else JUDGMENT_RETURNED_NOT_CITED,
        )
        for context_id in returned
        if context_id
    ]


def _exchanges(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in _sequence(value) if isinstance(item, Mapping)]


def _sequence(value: Any) -> Sequence[Any]:
    """A list column, however the extraction backend rendered it.

    The json backend scalarizes nested lists to a JSON string, while an
    in-memory frame keeps them as lists — both reach this transform.
    """
    if isinstance(value, str):
        if not value.strip():
            return ()
        try:
            parsed = json.loads(value)
        except ValueError:
            return ()
        return parsed if isinstance(parsed, list) else ()
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _text(value: Any) -> str:
    return str(value) if value is not None else ""
