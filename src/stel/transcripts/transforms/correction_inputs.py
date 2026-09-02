"""Exchange-grain rows a correction classifier can read (issue #456).

The `eval:` half of #329 phase 3, and the mechanical part of it. Deciding
whether a human corrected the assistant is a judgement, so it belongs to an
`llm:` model in an ordinary stel project (#329 rule 3). What that model needs
first is a row per exchange carrying two things the corpus keeps in different
places:

- **the prose of that exchange**, sliced out of the session's rendered text by
  its `## [<ordinal>] ...` heading, which is the same anchor `chunk:` uses for
  heading attribution;
- **the context ids that exchange touched**, from `context_calls`
  (`transcript/v1.1`), because a correction has to attach to a *record* to
  become an `eval.expected` row and those ids are the only record identity a
  transcript reliably names.

**Why the ids are the constraint.** `eval:` scores predictions against
expected labels joined on a key. A human saying "no, that filing is a 10-K"
is worthless as ground truth unless we know which filing. The agent had to
retrieve something to be corrected about it, so the ids its context calls
returned are the candidate subjects; an exchange with no context call has
nothing to attach a label to and is not emitted. That is a real limit, not an
oversight: it means this derives labels only for records the corpus can name.

**Sensitivity.** This carries exchange prose, which exists here only because
the harness chose to capture it -- `transcript/v1` makes text optional and the
converter keeps what it was given (#329 rule 1). A fingerprint-only corpus
produces empty prose and therefore no candidates, which is the correct
outcome rather than a degraded one.

Nothing here decides anything: it shapes input. The classifier proposes, and
per #329 rule 2 a human promotes.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from ...hashing import canonical_fingerprint
from ...transforms import IncrementalContract, TransformContext

# The exchange heading the transcript renderer writes, and the anchor the
# example's `chunk:` model already splits on. Captured so a slice can be tied
# back to the ordinal its context calls are recorded under.
_HEADING = re.compile(r"^## \[(\d+)\] (.*)$", re.MULTILINE)

# The id space these candidates will be expressed in, recorded for the same
# reason #387 records it: a search index keys on `context_id` or `chunk_id`,
# and promotion has to reconcile against the target rather than assume.
ID_SPACE = "context_id"

_SCHEMA: dict[str, pl.DataType] = {
    "input_id": pl.String(),
    "source_document_id": pl.String(),
    "session_id": pl.String(),
    "harness": pl.String(),
    "exchange_ordinal": pl.Int64(),
    "heading": pl.String(),
    "exchange_text": pl.String(),
    "id_space": pl.String(),
    "candidate_context_ids": pl.String(),
    "observed_at": pl.String(),
}


def validate_options(options: Mapping[str, Any]) -> None:
    unknown = sorted(set(options) - {"transcripts"})
    if unknown:
        raise ValueError(
            f"correction_inputs: unknown options {unknown}; expected 'transcripts'"
        )
    transcripts = options.get("transcripts")
    if not isinstance(transcripts, str) or not transcripts.strip():
        raise ValueError(
            "correction_inputs requires `transcripts:` naming the model that "
            "holds transcript/v1.1 rows"
        )


def declared_dependencies(options: Mapping[str, Any]) -> tuple[str, ...]:
    validate_options(options)
    return (str(options["transcripts"]),)


def declared_incremental_contract(options: Mapping[str, Any]) -> IncrementalContract:
    """One transcript document is one parent, as in `retrieval_judgments`.

    A session's landing document is rewritten whole when the session grows, so
    its rows are replaced as a unit.
    """
    validate_options(options)
    return IncrementalContract(
        parent_key="source_document_id",
        child_key="input_id",
        parent_source=str(options["transcripts"]),
        parent_source_key="document_id",
    )


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    validate_options(ctx.options)
    frame = deps[str(ctx.options["transcripts"])]
    rows: list[dict[str, Any]] = []
    for document in frame.iter_rows(named=True):
        rows.extend(_document_rows(document))
    return pl.DataFrame(rows, schema=_SCHEMA)


def _document_rows(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    session_id = _text(document.get("session_id"))
    harness = _text(document.get("harness"))
    source_document_id = _text(document.get("document_id"))
    prose = _exchange_prose(_text(document.get("text")))
    rows: list[dict[str, Any]] = []
    for exchange in _exchanges(document.get("exchanges")):
        ordinal = exchange.get("ordinal")
        if not isinstance(ordinal, int):
            continue
        context_ids = _touched_context_ids(exchange)
        if not context_ids:
            # Nothing to attach a label to; see the module docstring.
            continue
        text = prose.get(ordinal, "")
        if not text.strip():
            # A fingerprint-only corpus, or an exchange the renderer wrote no
            # prose for. There is nothing for a classifier to read.
            continue
        rows.append(
            {
                "input_id": canonical_fingerprint(
                    {"session": session_id, "exchange": ordinal},
                    domain="stel.correction-input",
                    version=1,
                ),
                "source_document_id": source_document_id,
                "session_id": session_id,
                "harness": harness,
                "exchange_ordinal": ordinal,
                "heading": _text(exchange.get("heading")),
                "exchange_text": text,
                "id_space": ID_SPACE,
                # A JSON array rather than a list column: the same reason
                # `_sequence` exists on the sibling transform -- extraction
                # backends disagree about nested types, and a string crosses
                # every warehouse unchanged.
                "candidate_context_ids": json.dumps(context_ids),
                "observed_at": _text(exchange.get("started_at")),
            }
        )
    return rows


def _touched_context_ids(exchange: Mapping[str, Any]) -> list[str]:
    """Ids this exchange retrieved, cited ones first.

    Both are candidate subjects for a correction: an agent can be wrong about
    a chunk it read without quoting its id. Failed calls contribute nothing --
    they returned nothing to be wrong about.
    """
    cited: list[str] = []
    returned: list[str] = []
    for call in _sequence(exchange.get("context_calls")):
        if not isinstance(call, Mapping) or call.get("error_code") is not None:
            continue
        cited.extend(_text(value) for value in _sequence(call.get("cited_context_ids")))
        returned.extend(
            _text(value) for value in _sequence(call.get("returned_context_ids"))
        )
    ordered: list[str] = []
    for value in [*cited, *returned]:
        if value and value not in ordered:
            ordered.append(value)
    return ordered


def _exchange_prose(text: str) -> dict[int, str]:
    """Slice the rendered session into per-exchange prose by its headings."""
    matches = list(_HEADING.finditer(text))
    prose: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        try:
            ordinal = int(match.group(1))
        except ValueError:  # pragma: no cover - the pattern only matches digits
            continue
        prose[ordinal] = text[match.end() : end].strip()
    return prose


def _exchanges(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in _sequence(value) if isinstance(item, Mapping)]


def _sequence(value: Any) -> Sequence[Any]:
    """A list column, however the extraction backend rendered it."""
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
