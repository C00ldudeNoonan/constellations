"""Draft a promoted golden set from candidate judgments (issue #380).

The last piece of #329 phase 3. #388 established that promotion's artifact is
a reviewed file rather than a warehouse write; this is the drafting aid that
issue named, and the distinction it preserves is the whole point:

    a command that writes rows decides;
    a command that writes a diff proposes.

So nothing here promotes anything. It turns candidate rows into the *shape* of
a golden set and hands it to a human, who reads it, fixes it, and merges it
like any other change. Every guard below exists to stop a draft from being
mistaken for a decision.

**Only `cited` becomes a label.** `returned_not_cited` is explicitly not
evidence of irrelevance (an agent may use a chunk without naming its id), and
`zero_result` asserts nothing a golden can hold — a query that should have
matched something needs a human to say *what*. Both are reported as skipped
rather than dropped, because a reviewer deciding what to promote needs to see
what was left behind.

**Query text is auto-filled but never assumed.** #380's constraint 2: the
corpus records only a fingerprint unless the operator opted into capturing
text, and `retrieval_tests` replays each query through `search()`. Where the
corpus captured the text, transcribing it is more faithful than asking a
reviewer to remember it, so it is filled in and shown for confirmation. Where
it did not, the row carries `UNCONFIRMED_QUERY_TEXT`, which the contract
refuses to load — an unreviewed draft fails loudly instead of running as a
test that asks the wrong question.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from .contract import (
    GOLDEN_SET_VERSION,
    UNCONFIRMED_QUERY_TEXT,
    GoldenSetFile,
    PromotedQuery,
    PromotionError,
    PromotionEvidence,
)

JUDGMENT_CITED = "cited"


@dataclass(frozen=True)
class CandidateRow:
    """One candidate judgment, as the transcript corpus produced it."""

    session_id: str
    harness: str | None
    context_model: str
    query_fingerprint: str
    query_text: str | None
    id_space: str
    context_id: str | None
    judgment: str


@dataclass(frozen=True)
class DraftedQuery:
    """A drafted query and what the reviewer still owes it."""

    query_id: str
    query_text: str
    text_from_corpus: bool
    relevant_ids: tuple[str, ...]
    sessions: tuple[str, ...]


@dataclass(frozen=True)
class SkippedQuery:
    query_fingerprint: str
    reason: str


@dataclass(frozen=True)
class Draft:
    """A drafted golden set, plus what it declined to draft and why."""

    golden_set: GoldenSetFile
    # The one context model these ids belong to. A golden set is checked
    # against a single index, so a draft that spanned two would carry ids
    # that cannot exist in either (see `_resolve_context_model`).
    context_model: str
    drafted: tuple[DraftedQuery, ...]
    skipped: tuple[SkippedQuery, ...]

    @property
    def needs_text(self) -> tuple[DraftedQuery, ...]:
        return tuple(query for query in self.drafted if not query.text_from_corpus)


def _resolve_id_space(rows: list[CandidateRow]) -> str:
    spaces = sorted({row.id_space for row in rows if row.id_space})
    if not spaces:
        raise PromotionError(
            "Candidate rows carry no id_space; promotion cannot check them "
            "against the target index's id_field"
        )
    if len(spaces) > 1:
        # Silently picking one is the failure mode #380 constraint 3 is about:
        # ids in the wrong space match nothing and the eval reports zero
        # recall as though retrieval were broken.
        raise PromotionError(
            "Candidate rows mix id spaces "
            f"({', '.join(spaces)}); draft one space at a time"
        )
    return spaces[0]


def _resolve_context_model(rows: list[CandidateRow]) -> str:
    """The single index these candidates judge, or refuse to guess.

    `query_fingerprint` hashes the query string alone, so the same question
    asked of two context models shares one fingerprint. Grouping on it without
    this check would merge ids from two different indexes into one golden set,
    and the `id_space` guard cannot catch that — two indexes commonly key on
    the same space. The result would be a set whose ids simply do not exist in
    the index it is run against, reported as zero recall (PR #451 review).
    """
    models = sorted({row.context_model for row in rows if row.context_model})
    if not models:
        raise PromotionError(
            "Candidate rows name no context model; promotion cannot tell "
            "which index these ids belong to"
        )
    if len(models) > 1:
        raise PromotionError(
            "Candidate rows span more than one context model "
            f"({', '.join(models)}). A golden set is checked against one "
            "index, so draft one at a time: pass --context-model."
        )
    return models[0]


def _query_id(fingerprint: str) -> str:
    return f"q-{fingerprint[:12]}"


def draft_golden_set(
    rows: list[CandidateRow],
    *,
    promoted_by: str,
    promoted_at: date,
    context_model: str | None = None,
) -> Draft:
    """Shape candidate rows into a golden set for a human to review.

    Grouped by `query_fingerprint` because that is the identity the corpus and
    the MCP query log agree on — the join key #329 called the linchpin. That
    key says nothing about *which* index answered, so `context_model` narrows
    the rows first and a corpus spanning several is refused rather than
    merged.
    """
    if not rows:
        raise PromotionError("No candidate judgments to draft from")
    if context_model is not None:
        rows = [row for row in rows if row.context_model == context_model]
        if not rows:
            raise PromotionError(
                f"No candidate judgments for context model '{context_model}'"
            )
    model = _resolve_context_model(rows)
    id_space = _resolve_id_space(rows)

    grouped: dict[str, list[CandidateRow]] = defaultdict(list)
    for row in rows:
        if not row.query_fingerprint:
            # A candidate that cannot be joined back to the query it judged
            # cannot be promoted, and #387 already asserts none are produced.
            continue
        grouped[row.query_fingerprint].append(row)

    drafted: list[DraftedQuery] = []
    skipped: list[SkippedQuery] = []
    queries: list[PromotedQuery] = []
    for fingerprint in sorted(grouped):
        group = grouped[fingerprint]
        cited = sorted(
            {
                row.context_id
                for row in group
                if row.judgment == JUDGMENT_CITED and row.context_id
            }
        )
        if not cited:
            observed = sorted({row.judgment for row in group})
            skipped.append(
                SkippedQuery(
                    query_fingerprint=fingerprint,
                    reason=(
                        "no cited id to promote; observed "
                        f"{', '.join(observed)}. A reviewer must say what "
                        "should have matched"
                    ),
                )
            )
            continue

        captured = [row.query_text for row in group if row.query_text]
        text_from_corpus = bool(captured)
        query_text = captured[0] if captured else UNCONFIRMED_QUERY_TEXT
        sessions = tuple(sorted({row.session_id for row in group if row.session_id}))
        if not sessions:
            skipped.append(
                SkippedQuery(
                    query_fingerprint=fingerprint,
                    reason="no session recorded; a promotion must name its evidence",
                )
            )
            continue
        harnesses = sorted({row.harness for row in group if row.harness})

        query_id = _query_id(fingerprint)
        queries.append(
            PromotedQuery(
                query_id=query_id,
                query_text=query_text,
                relevant_ids=tuple(cited),
                promoted_by=promoted_by,
                promoted_at=promoted_at,
                evidence=PromotionEvidence(
                    sessions=sessions,
                    query_fingerprint=fingerprint,
                    # Only when the evidence agrees; a query seen through two
                    # harnesses is not attributable to either.
                    harness=harnesses[0] if len(harnesses) == 1 else None,
                ),
            )
        )
        drafted.append(
            DraftedQuery(
                query_id=query_id,
                query_text=query_text,
                text_from_corpus=text_from_corpus,
                relevant_ids=tuple(cited),
                sessions=sessions,
            )
        )

    if not queries:
        raise PromotionError(
            "No candidate query had a cited id to promote; there is nothing "
            "to draft"
        )
    return Draft(
        golden_set=GoldenSetFile(
            version=GOLDEN_SET_VERSION, id_space=id_space, queries=tuple(queries)
        ),
        context_model=model,
        drafted=tuple(drafted),
        skipped=tuple(skipped),
    )
