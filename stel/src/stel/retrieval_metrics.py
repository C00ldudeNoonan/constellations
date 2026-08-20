"""Deterministic ranking metrics for golden-set retrieval evaluation (#137).

Pure functions over a ranked ID list and a relevance judgment — no I/O, no
warehouse, no store. `evaluate_query` is the single per-query entry point;
`retrieval_eval.py` composes it against `search()` results and aggregates
across a golden set.

Design note: a query with an empty relevant set is reported as
`NO_RELEVANT_LABELS`, not scored as zero — silently averaging it in with
ordinary misses would understate quality on well-covered queries and mask
lambda-labeling gaps. Callers exclude `NO_RELEVANT_LABELS` queries from
aggregate means but still surface them per-query.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum


class QueryDiagnosis(StrEnum):
    OK = "ok"
    NO_RELEVANT_LABELS = "no_relevant_labels"
    EMPTY_RESULTS = "empty_results"


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    query_id: str
    diagnosis: QueryDiagnosis
    # metric_name -> {k: value}, e.g. {"recall": {5: 0.4, 10: 0.6}}
    values: Mapping[str, Mapping[int, float]]
    ranked_ids: tuple[str, ...]
    missing_ids: tuple[str, ...]  # relevant IDs that never appeared in results


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        raise ValueError("recall_at_k requires a non-empty relevant_ids set")
    top_k = set(ranked_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("precision_at_k requires k > 0")
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k)


def hit_rate_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    return 1.0 if set(ranked_ids[:k]) & relevant_ids else 0.0


def mrr_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    for position, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant_ids:
            return 1.0 / position
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
) -> float:
    """Graded NDCG@k. `relevance` maps ID -> non-negative grade; IDs absent
    from `relevance` are treated as grade 0. Binary relevance (grade 1 for
    every relevant ID) reduces to the standard unweighted NDCG."""

    def dcg(ids: Sequence[str]) -> float:
        total = 0.0
        for position, rid in enumerate(ids[:k], start=1):
            grade = relevance.get(rid, 0.0)
            if grade:
                total += grade / math.log2(position + 1)
        return total

    ideal_order = sorted(relevance, key=lambda rid: relevance[rid], reverse=True)
    ideal = dcg(ideal_order)
    if ideal == 0.0:
        raise ValueError("ndcg_at_k requires at least one relevant grade > 0")
    return dcg(ranked_ids) / ideal


_METRIC_FNS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "hit_rate": hit_rate_at_k,
    "mrr": mrr_at_k,
}


def evaluate_query(
    query_id: str,
    ranked_ids: Sequence[str],
    *,
    relevant_ids: set[str],
    graded_relevance: Mapping[str, float] | None = None,
    cutoffs: Sequence[int],
) -> QueryMetrics:
    """Compute every metric at every cutoff for one query. `graded_relevance`
    defaults to binary relevance (grade 1 for each `relevant_ids` member) when
    not supplied, so NDCG is always computable whenever the other metrics are."""
    ranked = tuple(ranked_ids)
    if not relevant_ids:
        return QueryMetrics(
            query_id=query_id,
            diagnosis=QueryDiagnosis.NO_RELEVANT_LABELS,
            values={},
            ranked_ids=ranked,
            missing_ids=(),
        )
    grades = dict(graded_relevance) if graded_relevance else dict.fromkeys(relevant_ids, 1.0)

    values: dict[str, dict[int, float]] = {name: {} for name in (*_METRIC_FNS, "ndcg")}
    for k in cutoffs:
        for name, fn in _METRIC_FNS.items():
            values[name][k] = fn(ranked, relevant_ids, k)
        values["ndcg"][k] = ndcg_at_k(ranked, grades, k)

    diagnosis = QueryDiagnosis.EMPTY_RESULTS if not ranked else QueryDiagnosis.OK
    missing = tuple(sorted(relevant_ids - set(ranked)))
    return QueryMetrics(
        query_id=query_id,
        diagnosis=diagnosis,
        values=values,
        ranked_ids=ranked,
        missing_ids=missing,
    )


def aggregate_metrics(
    per_query: Sequence[QueryMetrics], cutoffs: Sequence[int]
) -> dict[str, dict[int, float]]:
    """Mean of each `<metric>@k` across queries with `diagnosis == OK` or a
    scored diagnosis (anything with `values`) — `NO_RELEVANT_LABELS` queries
    are excluded from the mean, not averaged in as zero."""
    scored = [q for q in per_query if q.values]
    aggregate: dict[str, dict[int, float]] = {}
    for name in (*_METRIC_FNS, "ndcg"):
        aggregate[name] = {}
        for k in cutoffs:
            samples = [q.values[name][k] for q in scored if k in q.values.get(name, {})]
            aggregate[name][k] = sum(samples) / len(samples) if samples else 0.0
    return aggregate
