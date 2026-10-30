"""Exact, directed canonical-triple evaluation with no I/O (issue #731).

Complete triples have set semantics: repeated evidence is one fact. Unresolved
predictions cannot match gold and each row costs a false positive; collapsing
unknown endpoints would silently guess that they denote the same entity.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RelationTriple:
    subject: str | None
    relation: str
    object: str | None
    document: str | None


@dataclass(frozen=True, slots=True)
class RelationScore:
    label: str | None
    true_positives: int
    false_positives: int
    false_negatives: int
    unmatched_subject_rows: int
    unmatched_object_rows: int
    unmatched_entity_rows: int

    @property
    def expected_triples(self) -> int:
        return self.true_positives + self.false_negatives


def score_relations(
    predictions: Iterable[RelationTriple],
    expected: Iterable[RelationTriple],
    *,
    labels: Iterable[str],
) -> tuple[RelationScore, ...]:
    """Micro scores overall and per type; zero denominators report 0.0.

    Callers validate gold endpoints and row types before invoking this pure
    core. Direction and document identity are part of exact matching. There
    is no inverse inference: project relation characteristics are not shipped.
    """
    predicted: set[RelationTriple] = set()
    unresolved: Counter[str] = Counter()
    subjects: Counter[str] = Counter()
    objects: Counter[str] = Counter()
    for triple in predictions:
        if triple.subject is None or triple.object is None:
            unresolved[triple.relation] += 1
            subjects[triple.relation] += triple.subject is None
            objects[triple.relation] += triple.object is None
        else:
            predicted.add(triple)
    gold = set(expected)
    hits = Counter(triple.relation for triple in predicted & gold)
    extra = Counter(triple.relation for triple in predicted - gold) + unresolved
    missing = Counter(triple.relation for triple in gold - predicted)
    universe = set(labels) | hits.keys() | extra.keys() | missing.keys()
    scores = tuple(
        RelationScore(
            label,
            hits[label],
            extra[label],
            missing[label],
            subjects[label],
            objects[label],
            unresolved[label],
        )
        for label in sorted(universe)
    )
    overall = RelationScore(
        None,
        sum(hits.values()),
        sum(extra.values()),
        sum(missing.values()),
        sum(subjects.values()),
        sum(objects.values()),
        sum(unresolved.values()),
    )
    return (overall, *scores)


def relation_metric_rows(scores: Sequence[RelationScore]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in scores:
        predicted = item.true_positives + item.false_positives
        expected = item.expected_triples
        denominator = 2 * item.true_positives + item.false_positives + item.false_negatives
        metrics = {
            "precision": item.true_positives / predicted if predicted else 0.0,
            "recall": item.true_positives / expected if expected else 0.0,
            "f1": 2 * item.true_positives / denominator if denominator else 0.0,
            "support": float(expected),
            "true_positives": float(item.true_positives),
            "false_positives": float(item.false_positives),
            "false_negatives": float(item.false_negatives),
            "predicted_triples": float(predicted),
            "expected_triples": float(expected),
            "unmatched_subject_rows": float(item.unmatched_subject_rows),
            "unmatched_object_rows": float(item.unmatched_object_rows),
            "unmatched_entity_rows": float(item.unmatched_entity_rows),
        }
        rows.extend(
            {"metric": metric, "label": item.label, "value": value}
            for metric, value in metrics.items()
        )
    return rows
