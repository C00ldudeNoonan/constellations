"""Deterministic single-label classification metrics (issue #309).

Pure functions over predicted/expected label pairs — no I/O, no warehouse, no
provider. `execution/eval.py` composes these against two warehouse relations
and materializes the result; everything scoreable lives here so it can be
tested without a database.

Two conventions worth stating, because both are places a quality number can
quietly lie:

**Zero denominators score 0.0, they do not vanish.** A label nothing predicted
has no precision in the mathematical sense, and a label nothing expected has no
recall. Dropping those rows would make a collapsed label disappear from the
report exactly when it most needs reading, so they are emitted as 0.0 and the
`support` row tells you which kind of zero it is. This matches scikit-learn's
`zero_division=0`.

**The label universe is declared, not observed.** Deriving it from the data
means a label the model stopped predicting silently leaves the report. Callers
pass the declared set (an `enum` field's values, #304) and it is unioned with
whatever appears, so a regression to zero shows up as `recall: 0.0` rather than
as an absence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

# Metric names emitted without a label (whole-run scalars).
OVERALL_METRICS = (
    "accuracy",
    "macro_f1",
    "evaluated_rows",
    "unmatched_rows",
    "unusable_expected_rows",
)
# Metric names emitted once per label.
PER_LABEL_METRICS = ("precision", "recall", "f1", "support")


@dataclass(frozen=True, slots=True)
class LabelScore:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    accuracy: float
    macro_f1: float
    evaluated_rows: int
    unmatched_rows: int
    # Expected rows that could not be scored at all — a null key or a null
    # label. Distinct from `unmatched_rows` (scoreable ground truth with no
    # prediction): these are defects in the labelled set itself, and folding
    # them into either bucket would hide them (Codex review, #328).
    unusable_expected_rows: int
    labels: tuple[LabelScore, ...]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def score(
    pairs: Sequence[tuple[str, str]],
    *,
    declared_labels: Iterable[str] = (),
    unmatched_rows: int = 0,
    unusable_expected_rows: int = 0,
) -> ClassificationReport:
    """Score `(predicted, expected)` pairs.

    `declared_labels` is the taxonomy the model was asked for; the report covers
    it whether or not the data uses every label. `unmatched_rows` is expected
    rows that had no prediction to join to, and `unusable_expected_rows` is
    expected rows dropped before joining (null key or null label) — both are
    carried through so the caller publishes the loss instead of silently
    inflating quality over the rows that happened to survive.
    """
    universe = set(declared_labels)
    for predicted, expected in pairs:
        universe.add(predicted)
        universe.add(expected)

    true_positive: dict[str, int] = dict.fromkeys(universe, 0)
    predicted_count: dict[str, int] = dict.fromkeys(universe, 0)
    expected_count: dict[str, int] = dict.fromkeys(universe, 0)
    correct = 0
    for predicted, expected in pairs:
        predicted_count[predicted] += 1
        expected_count[expected] += 1
        if predicted == expected:
            true_positive[predicted] += 1
            correct += 1

    scores: list[LabelScore] = []
    for label in sorted(universe):
        hits = true_positive[label]
        precision = _ratio(hits, predicted_count[label])
        recall = _ratio(hits, expected_count[label])
        denominator = precision + recall
        scores.append(
            LabelScore(
                label=label,
                precision=precision,
                recall=recall,
                f1=(2 * precision * recall / denominator) if denominator else 0.0,
                support=expected_count[label],
            )
        )

    return ClassificationReport(
        accuracy=_ratio(correct, len(pairs)),
        # Unweighted mean over labels: a collapsed rare label moves this, where
        # accuracy would hide it behind the majority class. That is the number
        # worth gating on for an imbalanced taxonomy.
        macro_f1=(
            sum(item.f1 for item in scores) / len(scores) if scores else 0.0
        ),
        evaluated_rows=len(pairs),
        unmatched_rows=unmatched_rows,
        unusable_expected_rows=unusable_expected_rows,
        labels=tuple(scores),
    )


def as_rows(report: ClassificationReport) -> list[dict[str, object]]:
    """Long format: one row per metric, so a new metric never changes the schema.

    Wide format would put every label in a column, which turns adding a label
    into a schema migration and makes `WHERE metric = 'recall'` impossible.
    """
    rows: list[dict[str, object]] = [
        {"metric": "accuracy", "label": None, "value": report.accuracy},
        {"metric": "macro_f1", "label": None, "value": report.macro_f1},
        {
            "metric": "evaluated_rows",
            "label": None,
            "value": float(report.evaluated_rows),
        },
        {
            "metric": "unmatched_rows",
            "label": None,
            "value": float(report.unmatched_rows),
        },
        {
            "metric": "unusable_expected_rows",
            "label": None,
            "value": float(report.unusable_expected_rows),
        },
    ]
    for item in report.labels:
        rows.extend(
            [
                {"metric": "precision", "label": item.label, "value": item.precision},
                {"metric": "recall", "label": item.label, "value": item.recall},
                {"metric": "f1", "label": item.label, "value": item.f1},
                {"metric": "support", "label": item.label, "value": float(item.support)},
            ]
        )
    return rows


def declared_labels_for(field_name: str, fields: Sequence[object]) -> tuple[str, ...]:
    """The declared `enum` values for `field_name`, if it declares any (#304)."""
    for field in fields:
        if getattr(field, "name", None) == field_name:
            return tuple(getattr(field, "values", ()) or ())
    return ()


def join_pairs(
    predictions: Mapping[str, str], expected: Mapping[str, str]
) -> tuple[list[tuple[str, str]], int]:
    """Pair predictions to expectations by key.

    Returns the pairs and the count of expected keys with no prediction. An
    inner join alone would report a model that stopped emitting rows as a
    smaller but equally good one, so the loss is counted rather than dropped.
    """
    pairs = [
        (predictions[key], label)
        for key, label in expected.items()
        if key in predictions
    ]
    return pairs, len(expected) - len(pairs)
