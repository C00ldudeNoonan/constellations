"""Exchange-pair rows a correction classifier can read (issue #456).

The `eval:` half of #329 phase 3, and the mechanical part of it. Deciding
whether a human corrected the assistant is a judgement, so it belongs to an
`llm:` model in an ordinary stel project (#329 rule 3). Getting that model the
right *input* is this module's whole job, and the corpus makes it non-obvious
in two ways that both have to be handled together (PR #458 review).

**A correction spans two exchanges.** An exchange is one human prompt plus the
assistant's answer to it, so a human correcting an answer does so in the
*next* prompt. The claim being corrected is in exchange N; the correction is
in exchange N+1. A row here is therefore a pair, carrying the rendered text of
both, and the classifier is told that `## [n]` headings are the human's turn.

**The ids belong to the earlier exchange.** `context_calls` on an exchange are
the searches made while answering *that* prompt, so the records behind a wrong
answer were retrieved in exchange N -- not in N+1, where the correction lives.
Reading ids from the correcting exchange would key a label to whatever the
agent looked up *after* being corrected, or drop the correction entirely when
the follow-up searched nothing.

**Why ids are the constraint at all.** `eval:` scores predictions against
expected labels joined on a key. A human saying "no, that filing is a 10-K" is
worthless as ground truth unless we know which filing. So a pair is emitted
only when exchange N retrieved something: that is a real limit on what can be
derived, not an oversight.

**Sensitivity.** This carries exchange prose, which exists here only because
the harness chose to capture it -- `transcript/v1` makes text optional and the
converter keeps what it was given (#329 rule 1). A fingerprint-only corpus
produces no rows rather than degraded ones.

Nothing here decides anything: it shapes input. The classifier proposes, and
per #329 rule 2 a human promotes.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from itertools import pairwise
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
    # The exchange whose prompt may hold the correction.
    "exchange_ordinal": pl.Int64(),
    # The exchange that made the claim, and whose searches the ids come from.
    "answered_exchange_ordinal": pl.Int64(),
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
    sections = _exchange_sections(_text(document.get("text")))
    exchanges = [
        exchange
        for exchange in _exchanges(document.get("exchanges"))
        if isinstance(exchange.get("ordinal"), int)
    ]
    exchanges.sort(key=lambda item: int(item["ordinal"]))

    rows: list[dict[str, Any]] = []
    for answered, correcting in pairwise(exchanges):
        answered_ordinal = int(answered["ordinal"])
        correcting_ordinal = int(correcting["ordinal"])
        # Ids come from the exchange that produced the claim, not the one that
        # corrects it. See the module docstring.
        context_ids = _touched_context_ids(answered)
        if not context_ids:
            continue
        text = "\n\n".join(
            part
            for part in (
                sections.get(answered_ordinal, ""),
                sections.get(correcting_ordinal, ""),
            )
            if part.strip()
        )
        if not text.strip():
            # A fingerprint-only corpus: nothing for a classifier to read.
            continue
        rows.append(
            {
                "input_id": canonical_fingerprint(
                    {
                        "session": session_id,
                        "answered": answered_ordinal,
                        "correcting": correcting_ordinal,
                    },
                    domain="stel.correction-input",
                    version=1,
                ),
                "source_document_id": source_document_id,
                "session_id": session_id,
                "harness": harness,
                "exchange_ordinal": correcting_ordinal,
                "answered_exchange_ordinal": answered_ordinal,
                "heading": _text(correcting.get("heading")),
                "exchange_text": text,
                "id_space": ID_SPACE,
                # A JSON array rather than a list column: extraction backends
                # disagree about nested types, and a string crosses every
                # warehouse unchanged.
                "candidate_context_ids": json.dumps(context_ids),
                "observed_at": _text(correcting.get("started_at")),
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


def _exchange_sections(text: str) -> dict[int, str]:
    """Slice the rendered session into per-exchange sections, headings kept.

    The heading is not decoration to strip: `_build_exchange` renders the
    human's prompt *as* the `## [n] ...` line, and a single-line prompt
    appears nowhere else. Slicing from after the heading -- which this did
    first -- handed the classifier assistant prose only, so the one turn that
    can contain a correction was the one turn it could not see (PR #458
    review).
    """
    matches = list(_HEADING.finditer(text))
    sections: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        # The pattern captures `\d+`, so this cannot fail to parse.
        sections[int(match.group(1))] = text[match.start() : end].strip()
    return sections


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
