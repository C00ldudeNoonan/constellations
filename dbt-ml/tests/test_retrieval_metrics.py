from __future__ import annotations

import math

import pytest

from dbt_ml.retrieval_metrics import (
    QueryDiagnosis,
    aggregate_metrics,
    evaluate_query,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# ── hand-computed fixtures ───────────────────────────────────────────────────

RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "d", "z"}  # z never appears in results


def test_recall_at_k_hand_computed() -> None:
    # top-3 = {a,b,c}; hits = {b} -> 1/3 of the 3 relevant
    assert recall_at_k(RANKED, RELEVANT, 3) == pytest.approx(1 / 3)
    # top-5 = all ranked; hits = {b,d} -> 2/3
    assert recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 3)


def test_recall_at_k_requires_relevant_ids() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        recall_at_k(RANKED, set(), 5)


def test_precision_at_k_hand_computed() -> None:
    # top-3 = {a,b,c}; 1 hit / 3
    assert precision_at_k(RANKED, RELEVANT, 3) == pytest.approx(1 / 3)
    # top-1 = {a}; 0 hits
    assert precision_at_k(RANKED, RELEVANT, 1) == 0.0


def test_precision_at_k_empty_top_k_is_zero_not_divide_by_zero() -> None:
    assert precision_at_k([], RELEVANT, 5) == 0.0


def test_hit_rate_at_k() -> None:
    assert hit_rate_at_k(RANKED, RELEVANT, 1) == 0.0  # 'a' not relevant
    assert hit_rate_at_k(RANKED, RELEVANT, 2) == 1.0  # 'b' relevant


def test_mrr_at_k_hand_computed() -> None:
    # first relevant hit is 'b' at position 2 -> 1/2
    assert mrr_at_k(RANKED, RELEVANT, 5) == pytest.approx(0.5)


def test_mrr_at_k_no_hit_within_cutoff_is_zero() -> None:
    assert mrr_at_k(RANKED, RELEVANT, 1) == 0.0


def test_mrr_at_k_ties_use_returned_order_deterministically() -> None:
    # Two equally-relevant IDs at different ranks: MRR takes the first by rank,
    # not by any secondary tie-break — determinism is search()'s job upstream,
    # this just trusts the given order.
    ranked = ["x", "y", "z"]
    assert mrr_at_k(ranked, {"y", "z"}, 3) == pytest.approx(0.5)  # 'y' at position 2


def test_ndcg_at_k_binary_relevance_matches_hand_computation() -> None:
    # binary grades {b:1, d:1}; ranked = a,b,c,d,e
    # DCG@5 = 1/log2(3) [pos2=b] + 1/log2(5) [pos4=d]
    # ideal DCG@5 (b,d first) = 1/log2(2) + 1/log2(3)
    grades = {"b": 1.0, "d": 1.0}
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, grades, 5) == pytest.approx(dcg / ideal)


def test_ndcg_at_k_graded_relevance() -> None:
    # grades: b=3 (highly relevant), d=1 (marginally relevant)
    grades = {"b": 3.0, "d": 1.0}
    dcg = 3 / math.log2(3) + 1 / math.log2(5)
    ideal = 3 / math.log2(2) + 1 / math.log2(3)  # ideal order: b (grade 3), d (grade 1)
    assert ndcg_at_k(RANKED, grades, 5) == pytest.approx(dcg / ideal)


def test_ndcg_at_k_perfect_ranking_is_one() -> None:
    grades = {"a": 1.0, "b": 1.0}
    assert ndcg_at_k(["a", "b", "c"], grades, 3) == pytest.approx(1.0)


def test_ndcg_at_k_requires_positive_ideal() -> None:
    with pytest.raises(ValueError, match="at least one relevant grade"):
        ndcg_at_k(RANKED, {}, 5)


# ── per-query orchestration + edge cases ─────────────────────────────────────

def test_evaluate_query_no_relevant_labels_is_diagnosed_not_scored() -> None:
    result = evaluate_query("q1", RANKED, relevant_ids=set(), cutoffs=[5, 10])
    assert result.diagnosis == QueryDiagnosis.NO_RELEVANT_LABELS
    assert result.values == {}


def test_evaluate_query_empty_results_is_diagnosed() -> None:
    result = evaluate_query("q1", [], relevant_ids={"b"}, cutoffs=[5])
    assert result.diagnosis == QueryDiagnosis.EMPTY_RESULTS
    # Still scored — recall/precision/etc. over an empty ranking are legitimate
    # zeros, distinct from "no ground truth to score against".
    assert result.values["recall"][5] == 0.0


def test_evaluate_query_reports_missing_relevant_ids() -> None:
    result = evaluate_query("q1", RANKED, relevant_ids=RELEVANT, cutoffs=[5])
    assert result.missing_ids == ("z",)


def test_evaluate_query_binary_relevance_default() -> None:
    result = evaluate_query("q1", RANKED, relevant_ids={"b", "d"}, cutoffs=[5])
    assert result.values["recall"][5] == pytest.approx(1.0)
    assert result.values["ndcg"][5] > 0


def test_evaluate_query_all_metrics_present_at_every_cutoff() -> None:
    result = evaluate_query("q1", RANKED, relevant_ids={"b"}, cutoffs=[1, 3, 5])
    for metric in ("recall", "precision", "hit_rate", "mrr", "ndcg"):
        assert set(result.values[metric]) == {1, 3, 5}


# ── aggregation ───────────────────────────────────────────────────────────────

def test_aggregate_excludes_no_relevant_labels_queries() -> None:
    scored = evaluate_query("q1", RANKED, relevant_ids={"b"}, cutoffs=[5])
    unlabeled = evaluate_query("q2", RANKED, relevant_ids=set(), cutoffs=[5])
    agg = aggregate_metrics([scored, unlabeled], cutoffs=[5])
    # Mean is over the ONE scored query only, not diluted by the unlabeled one.
    assert agg["recall"][5] == scored.values["recall"][5]


def test_aggregate_of_no_scored_queries_is_zero_not_error() -> None:
    unlabeled = evaluate_query("q1", RANKED, relevant_ids=set(), cutoffs=[5])
    agg = aggregate_metrics([unlabeled], cutoffs=[5])
    assert agg["recall"][5] == 0.0


def test_aggregate_mean_across_two_scored_queries() -> None:
    q1 = evaluate_query("q1", ["a", "b"], relevant_ids={"a"}, cutoffs=[2])  # recall=1
    q2 = evaluate_query("q2", ["a", "b"], relevant_ids={"z"}, cutoffs=[2])  # recall=0
    agg = aggregate_metrics([q1, q2], cutoffs=[2])
    assert agg["recall"][2] == pytest.approx(0.5)
