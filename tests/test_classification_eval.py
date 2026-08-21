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
    assert overall == {
        "accuracy",
        "macro_f1",
        "evaluated_rows",
        "unmatched_rows",
        "unusable_expected_rows",
    }
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


def test_depends_on_is_derived_at_construction() -> None:
    """An eval's inputs become ordinary DAG edges the moment the model exists.

    Derived in ModelConfig validation rather than a compiler pass, because
    `stel ls`, manifest generation, and run_results all construct ProjectDAG
    straight from loaded models — an edge that appears only after contract
    validation would be missing there (Codex review, #328).
    """
    model = ModelConfig(name="signal_eval", eval=_eval_config())

    assert model.depends_on == ["ref('signals')", "ref('signals_labeled')"]


def test_a_bare_model_name_is_accepted_like_depends_on() -> None:
    # `parse_ref` takes either form everywhere else in stel; an eval is not the
    # place to invent a stricter spelling rule.
    model = ModelConfig(name="signal_eval", eval=_eval_config(predictions="signals"))

    assert model.depends_on == ["ref('signals')", "ref('signals_labeled')"]


def test_config_ref_parsing_agrees_with_dag_parse_ref() -> None:
    # The pattern is duplicated in config (which dag imports, so config cannot
    # import it back); this is the pin holding the two grammars together.
    from stel.config.model import _parse_ref_expression
    from stel.dag import parse_ref

    for expression in ("ref('signals')", 'ref("signals")', "signals", " padded "):
        assert _parse_ref_expression(expression) == parse_ref(expression)


def test_an_empty_relation_name_is_rejected() -> None:
    with pytest.raises(Exception, match="must name a model"):
        ModelConfig(name="signal_eval", eval=_eval_config(predictions="  "))


def test_declaring_depends_on_directly_is_rejected() -> None:
    # Two sources of truth for the same edges is how they drift.
    with pytest.raises(Exception, match="must not declare `depends_on:`"):
        ModelConfig(
            name="signal_eval", eval=_eval_config(), depends_on=["ref('something')"]
        )


def test_eval_participates_in_single_kind_validation() -> None:
    with pytest.raises(Exception, match="multiple kind blocks"):
        ModelConfig(
            name="signal_eval",
            eval=_eval_config(),
            chunk={"strategy": "recursive"},
        )


def test_public_serializers_know_the_eval_kind() -> None:
    # `stel ls` and the manifest each classify models independently of the
    # runner; `kind: unknown` there means docs and artifact consumers cannot
    # describe a valid model (Codex review, #328).
    from stel.cli import _model_kind
    from stel.runner import _model_kind_label

    model = ModelConfig(name="signal_eval", eval=_eval_config())

    assert _model_kind(model) == "eval"
    assert _model_kind_label(model) == "eval"


# ─── end to end ─────────────────────────────────────────────────────────────


_PREDICTED = {
    "c1": "churn_risk",
    "c2": "pricing",
    "c3": "churn_risk",  # expected `expansion` — the one deliberate miss
    "c4": "support",
    "c5": "none",
}


def _eval_project(
    tmp_path: Path, *, tests: str = "", materialization: str | None = None
) -> Path:
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
        + (f"    materialization: {materialization}\n" if materialization else "")
        + tests
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


# ─── review follow-ups (PR #328) ────────────────────────────────────────────


def test_unusable_expected_rows_are_counted_not_dropped(tmp_path: Path) -> None:
    """A partially malformed labelled set must not silently inflate quality.

    Rows with a null key or null label cannot be scored; dropping them without
    a trace reports metrics over the survivors as if they were the whole set.
    """
    from stel.runner import run_project

    project = _eval_project(tmp_path)
    # Two defective ground-truth rows: one null label, one null key.
    (project / "labels.sql").write_text(
        "SELECT chunk_id, CASE WHEN chunk_id = 'c3' THEN 'expansion' "
        "WHEN chunk_id = 'c5' THEN NULL ELSE signal END AS expected_signal "
        "FROM {{ ref('signals') }} UNION ALL "
        "SELECT NULL AS chunk_id, 'pricing' AS expected_signal\n"
    )

    run_project(project)

    metrics = _metrics(project)
    assert metrics[("unusable_expected_rows", None)] == 2.0
    # And the scored set shrank accordingly: c5's row is unusable, not wrong.
    assert metrics[("evaluated_rows", None)] == 4.0


def test_duplicate_keys_are_a_hard_error(tmp_path: Path) -> None:
    """Which duplicate wins would depend on warehouse row order."""
    from stel.execution.contracts import RunError
    from stel.runner import run_project

    project = _eval_project(tmp_path)
    (project / "labels.sql").write_text(
        "SELECT chunk_id, signal AS expected_signal FROM {{ ref('signals') }} "
        "UNION ALL SELECT 'c1' AS chunk_id, 'pricing' AS expected_signal\n"
    )

    with pytest.raises(RunError, match="duplicate 'chunk_id'"):
        run_project(project)


def test_min_metric_reads_only_the_latest_evaluation(tmp_path: Path) -> None:
    """A historical dip must not fail the gate forever.

    An incremental eval keeps one metric set per predictions version; the gate
    reads the newest set only, so a recovered classifier passes and a stale
    row cannot satisfy the existence check.
    """
    import duckdb as _duckdb

    from stel.runner import build_project

    project = _eval_project(
        tmp_path,
        tests=(
            "    tests:\n"
            "      - min_metric: { metric: accuracy, min: 0.9 }\n"
        ),
    )
    run_first = build_project(project)
    assert [
        r.status for r in run_first.test_results if r.test_name == "min_metric"
    ] == ["fail"]  # accuracy 0.8 on the current evaluation

    # Plant an older, better evaluation. If the gate aggregated history, the
    # planted 1.0 row would be the MIN comparison's rescuer — assert it isn't.
    con = _duckdb.connect(str(project / "target" / "db.duckdb"))
    try:
        con.execute(
            "INSERT INTO main.signal_eval "
            "SELECT 'planted', metric, label, 1.0, predictions_version, "
            "code_version, '2000-01-01T00:00:00+00:00' "
            "FROM main.signal_eval WHERE metric = 'accuracy'"
        )
    finally:
        con.close()

    from stel.adapters import create_adapter, parse_warehouse_config
    from stel.checks.runner import run_model_tests

    cfg = parse_warehouse_config(
        {
            "type": "duckdb",
            "path": str(project / "target" / "db.duckdb"),
            "schema": "main",
        }
    )
    model = ModelConfig(
        name="signal_eval",
        eval=_eval_config(),
        tests=[{"min_metric": {"metric": "accuracy", "min": 0.9}}],
    )
    with create_adapter(cfg) as adapter:
        results = run_model_tests(model, adapter)

    # Still fails: the planted historical 1.0 is not the latest evaluation.
    assert [r.status for r in results if r.test_name == "min_metric"] == ["fail"]


def test_incremental_rerun_removes_stale_metric_rows(tmp_path: Path) -> None:
    """Shrinking the label universe must not leave the removed label behind.

    Same predictions version, so the stale rows share it: an upsert alone would
    keep `expansion`'s four per-label rows forever, with an old code_version,
    contradicting the report's declared universe and feeding min_metric stale
    results.
    """
    from stel.runner import run_project

    project = _eval_project(tmp_path, materialization="incremental")
    run_project(project)
    assert ("recall", "expansion") in _metrics(project)

    # Correct the ground truth (c3 really was churn_risk) and drop the label
    # from the enum: `expansion` now exists nowhere — not declared, not in
    # either relation — so its rows must vanish rather than linger.
    (project / "labels.sql").write_text(
        "SELECT chunk_id, signal AS expected_signal FROM {{ ref('signals') }}\n"
    )
    models_yml = (project / "models" / "models.yml").read_text()
    (project / "models" / "models.yml").write_text(
        models_yml.replace(
            "values: [churn_risk, expansion, pricing, support, none]",
            "values: [churn_risk, pricing, support, none]",
        )
    )
    run_project(project)

    metrics = _metrics(project)
    assert ("recall", "expansion") not in metrics
    # Fully replaced, not duplicated: one physical row per metric.
    assert _metric_row_count(project) == len(metrics)
    assert metrics[("accuracy", None)] == 1.0


def _metric_row_count(project: Path) -> int:
    con = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        row = con.execute("SELECT COUNT(*) FROM main.signal_eval").fetchone()
        assert row is not None
        return int(row[0])
    finally:
        con.close()
