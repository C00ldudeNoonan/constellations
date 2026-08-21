"""Classification eval models (issue #309).

`golden` answers "identical or not"; this answers "which labels moved, and by
how much". The tests below pin the metric maths, the two conventions that stop
a quality number from quietly lying (zero denominators score rather than
vanish, and the label universe is declared rather than observed), and the
threshold test that makes a regression gate a build.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest

from stel.classification_metrics import as_rows, join_pairs, score
from stel.config.model import EvalConfig, ModelConfig

# ─── metric maths ───────────────────────────────────────────────────────────


def test_perfect_predictions_score_one() -> None:
    report = score([("a", "a"), ("b", "b")])

    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert {item.label for item in report.labels} == {"a", "b"}
    assert all(item.precision == 1.0 and item.recall == 1.0 for item in report.labels)


def test_accuracy_and_per_label_scores_disagree_usefully() -> None:
    # Four of five right, but one label is wrong every time it is expected.
    pairs = [("a", "a"), ("a", "a"), ("a", "a"), ("a", "b"), ("c", "c")]

    report = score(pairs)
    by_label = {item.label: item for item in report.labels}

    assert report.accuracy == 0.8
    # `a` is over-predicted: perfect recall, imperfect precision.
    assert by_label["a"].recall == 1.0
    assert by_label["a"].precision == 0.75
    # `b` collapsed entirely, and the report says so rather than omitting it.
    assert by_label["b"].recall == 0.0
    assert by_label["b"].support == 1
    # macro_f1 is dragged down by the collapsed label where accuracy is not.
    assert report.macro_f1 < report.accuracy


def test_a_declared_label_nobody_predicted_still_reports() -> None:
    """The failure mode worth catching is silent disappearance.

    A label the model stopped predicting must show as recall 0.0, not as an
    absent row that a reader has to notice is missing.
    """
    report = score([("a", "a")], declared_labels=["a", "gone_quiet"])

    by_label = {item.label: item for item in report.labels}
    assert "gone_quiet" in by_label
    assert by_label["gone_quiet"].recall == 0.0
    assert by_label["gone_quiet"].precision == 0.0
    assert by_label["gone_quiet"].support == 0


def test_zero_denominators_score_zero_rather_than_raising() -> None:
    report = score([], declared_labels=["a"])

    assert report.accuracy == 0.0
    by_label = {item.label: item for item in report.labels}
    assert by_label["a"].f1 == 0.0


def test_unmatched_expected_rows_are_counted_not_dropped() -> None:
    """An inner join alone would flatter a model that stopped emitting rows."""
    pairs, unmatched = join_pairs({"k1": "a"}, {"k1": "a", "k2": "b"})

    assert pairs == [("a", "a")]
    assert unmatched == 1

    report = score(pairs, unmatched_rows=unmatched)
    assert report.evaluated_rows == 1
    assert report.unmatched_rows == 1
    # Accuracy is over the rows actually scored; the loss is reported beside it
    # rather than folded in.
    assert report.accuracy == 1.0


def test_rows_are_long_format() -> None:
    rows = as_rows(score([("a", "a")]))

    assert {"metric", "label", "value"} == set(rows[0])
    overall = {row["metric"] for row in rows if row["label"] is None}
    assert overall == {"accuracy", "macro_f1", "evaluated_rows", "unmatched_rows"}
    per_label = {row["metric"] for row in rows if row["label"] is not None}
    assert per_label == {"precision", "recall", "f1", "support"}


# ─── config ─────────────────────────────────────────────────────────────────


def _eval_config(**overrides: Any) -> EvalConfig:
    fields: dict[str, Any] = {
        "predictions": "ref('signals')",
        "predicted_field": "signal",
        "expected": "ref('signals_labeled')",
        "expected_field": "expected_signal",
        "key": "chunk_id",
    }
    fields.update(overrides)
    return EvalConfig.model_validate(fields)


def test_eval_is_a_model_kind() -> None:
    model = ModelConfig(name="signal_eval", eval=_eval_config())

    assert model.kind_block_count == 1


def test_duplicate_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="twice"):
        _eval_config(labels=["a", "a"])


def test_empty_column_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _eval_config(key="  ")


# ─── compiler wiring ────────────────────────────────────────────────────────


def test_depends_on_is_derived_from_the_scored_relations() -> None:
    """An eval's inputs become ordinary DAG edges.

    That is what makes selectors, lineage, and ordering work without any of
    them needing to know what an eval is.
    """
    from stel.compiler import _prepare_eval

    model = ModelConfig(name="signal_eval", eval=_eval_config())
    _prepare_eval(model)

    assert model.depends_on == ["ref('signals')", "ref('signals_labeled')"]


def test_a_bare_model_name_is_accepted_like_depends_on() -> None:
    # `parse_ref` takes either form everywhere else in stel; an eval is not the
    # place to invent a stricter spelling rule.
    from stel.compiler import _prepare_eval

    model = ModelConfig(name="signal_eval", eval=_eval_config(predictions="signals"))
    _prepare_eval(model)

    assert model.depends_on == ["ref('signals')", "ref('signals_labeled')"]


def test_an_empty_relation_name_is_rejected() -> None:
    from stel.compiler import _prepare_eval

    model = ModelConfig(name="signal_eval", eval=_eval_config(predictions="  "))

    with pytest.raises(Exception, match="must name a model"):
        _prepare_eval(model)


def test_declaring_depends_on_directly_is_rejected() -> None:
    # Two sources of truth for the same edges is how they drift.
    from stel.compiler import _prepare_eval

    model = ModelConfig(
        name="signal_eval", eval=_eval_config(), depends_on=["ref('something')"]
    )

    with pytest.raises(Exception, match="must not declare `depends_on:`"):
        _prepare_eval(model)


# ─── end to end ─────────────────────────────────────────────────────────────


_PREDICTED = {
    "c1": "churn_risk",
    "c2": "pricing",
    "c3": "churn_risk",  # expected `expansion` — the one deliberate miss
    "c4": "support",
    "c5": "none",
}


def _eval_project(tmp_path: Path, *, tests: str = "") -> Path:
    project = tmp_path / "proj"
    (project / "models").mkdir(parents=True)
    (project / "sources").mkdir()
    docs = project / "data" / "docs"
    docs.mkdir(parents=True)

    (project / "stel_project.yml").write_text(
        "name: evals\nversion: '0.1.0'\nprofile: evals\n"
    )
    (project / "profiles.yml").write_text(
        "evals:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n        schema: main\n"
    )
    (project / "sources" / "src.yml").write_text(
        "version: 2\nsources:\n  - name: raw_notes\n    path: data/docs\n"
        "    file_pattern: '*.json'\n"
    )
    for key, label in _PREDICTED.items():
        (docs / f"{key}.json").write_text(json.dumps({"chunk_id": key, "signal": label}))

    # Ground truth: the predictions with c3 corrected.
    (project / "labels.sql").write_text(
        "SELECT chunk_id, CASE WHEN chunk_id = 'c3' THEN 'expansion' ELSE signal END "
        "AS expected_signal FROM {{ ref('signals') }}\n"
    )
    (project / "models" / "models.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: signals\n"
        "    source: ref('raw_notes')\n"
        "    extraction:\n      backend: json\n      options:\n"
        "        fields: [chunk_id, signal]\n"
        "    fields:\n"
        "      - name: chunk_id\n        type: string\n"
        "      - name: signal\n        type: enum\n"
        "        values: [churn_risk, expansion, pricing, support, none]\n"
        "  - name: signals_labeled\n"
        "    transform:\n      type: sql\n      path: labels.sql\n"
        "  - name: signal_eval\n"
        "    eval:\n      kind: classification\n"
        "      predictions: ref('signals')\n      predicted_field: signal\n"
        "      expected: ref('signals_labeled')\n"
        "      expected_field: expected_signal\n      key: chunk_id\n"
        f"{tests}"
    )
    return project


def _metrics(project: Path) -> dict[tuple[str, str | None], float]:
    con = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        return {
            (row[0], row[1]): row[2]
            for row in con.execute(
                "SELECT metric, label, value FROM main.signal_eval"
            ).fetchall()
        }
    finally:
        con.close()


def test_eval_model_publishes_metric_rows(tmp_path: Path) -> None:
    from stel.runner import run_project

    project = _eval_project(tmp_path)

    results = {r.model_name: r for r in run_project(project)}

    assert results["signal_eval"].kind == "eval"
    metrics = _metrics(project)
    # Four of five correct.
    assert metrics[("accuracy", None)] == 0.8
    assert metrics[("evaluated_rows", None)] == 5.0
    assert metrics[("unmatched_rows", None)] == 0.0
    # `churn_risk` was over-predicted onto c3: perfect recall, half precision.
    assert metrics[("recall", "churn_risk")] == 1.0
    assert metrics[("precision", "churn_risk")] == 0.5


def test_a_label_the_model_never_predicted_is_still_reported(tmp_path: Path) -> None:
    """`expansion` is expected once and predicted never.

    The label set comes from the predicted field's declared `enum` (#304), so
    this reports rather than disappearing.
    """
    from stel.runner import run_project

    project = _eval_project(tmp_path)
    run_project(project)

    metrics = _metrics(project)
    assert metrics[("recall", "expansion")] == 0.0
    assert metrics[("support", "expansion")] == 1.0


def test_min_metric_gates_on_one_label(tmp_path: Path) -> None:
    from stel.runner import build_project

    project = _eval_project(
        tmp_path,
        tests=(
            "    tests:\n"
            "      - min_metric: { metric: recall, label: churn_risk, min: 0.5 }\n"
            "      - min_metric: { metric: accuracy, min: 0.9 }\n"
        ),
    )

    result = build_project(project)
    tests = {
        r.column or "": r
        for r in result.test_results
        if r.test_name == "min_metric"
    }

    assert tests["recall[churn_risk]"].status == "pass"
    # Accuracy is 0.8, and the threshold says 0.9.
    assert tests["accuracy"].status == "fail"
    assert "below 0.9" in tests["accuracy"].message


def test_min_metric_fails_when_the_metric_row_is_absent(tmp_path: Path) -> None:
    """A label that stopped being reported is the regression, not a pass."""
    from stel.runner import build_project

    project = _eval_project(
        tmp_path,
        tests=(
            "    tests:\n"
            "      - min_metric: { metric: recall, label: not_a_label, min: 0.5 }\n"
        ),
    )

    result = build_project(project)
    failures = [r for r in result.test_results if r.test_name == "min_metric"]

    assert [r.status for r in failures] == ["fail"]
    assert "no recall[not_a_label] row" in failures[0].message
