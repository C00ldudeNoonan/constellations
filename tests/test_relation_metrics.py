"""Canonical-triple scoring and validation contracts (constellations#731, ALE-425)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import polars as pl
import pyarrow as pa
import pytest
from pydantic import ValidationError

from stel import compiler
from stel.adapters import WarehouseCapability, adapter_capabilities
from stel.compiler import validate_warehouse_capabilities
from stel.config.loader import ConfigError
from stel.config.model import EvalConfig, ModelConfig, RelationEvalConfig
from stel.execution.contracts import RunError
from stel.execution.eval import run_eval_model
from stel.execution.relation_eval import read_relation_scores
from stel.relation_metrics import RelationTriple, relation_metric_rows, score_relations
from stel.versioning import compute_code_version


def _triple(subject="a", relation="owns", obj="b", document=None):
    return RelationTriple(subject, relation, obj, document)


def _metrics(predicted, expected, labels=()):
    return {
        (row["metric"], row["label"]): row["value"]
        for row in relation_metric_rows(score_relations(predicted, expected, labels=labels))
    }


def test_exact_set_matching_counts_missing_extra_and_duplicate_facts():
    """Repeated evidence must not improve recall or punish precision (#731)."""
    a, b, c = _triple(), _triple("b", "located_in", "c"), _triple("z")
    metrics = _metrics([a, a, c], [a, a, b], labels=["silent"])
    assert metrics["precision", None] == 0.5
    assert metrics["recall", None] == 0.5
    assert metrics["f1", None] == 0.5
    assert metrics["false_positives", "owns"] == 1
    assert metrics["false_negatives", "located_in"] == 1
    assert metrics["recall", "located_in"] == 0
    assert metrics["f1", "silent"] == 0
    assert metrics["support", "silent"] == 0


@pytest.mark.parametrize(
    "wrong",
    [
        _triple("b", "owns", "a"),
        _triple("a", "owned_by", "b"),
        _triple(document="other"),
    ],
)
def test_direction_relation_type_and_document_are_part_of_identity(wrong):
    """A shared endpoint pair cannot rescue a wrong assertion (#731)."""
    metrics = _metrics([wrong], [_triple()])
    assert metrics["f1", None] == 0
    assert metrics["false_positives", None] == metrics["false_negatives", None] == 1


def test_micro_scores_do_not_average_types_and_empty_predictions_keep_recall_zero():
    gold = [_triple(str(i)) for i in range(3)] + [_triple(relation="rare")]
    metrics = _metrics(gold[:3], gold)
    assert metrics["precision", None] == 1
    assert metrics["recall", None] == 0.75
    assert metrics["f1", None] == pytest.approx(6 / 7)
    empty = _metrics([], gold)
    assert empty["recall", None] == empty["precision", None] == empty["f1", None] == 0
    assert empty["false_negatives", None] == 4


def test_unresolved_endpoints_are_false_positives_and_do_not_disappear():
    """Dropping unlinked rows would report perfect precision (#731)."""
    missing_subject = _triple(None)
    metrics = _metrics(
        [
            _triple(),
            missing_subject,
            missing_subject,
            _triple(obj=None),
            _triple(None, obj=None),
        ],
        [_triple()],
    )
    assert metrics["precision", None] == 0.2
    assert metrics["recall", None] == 1
    assert metrics["false_positives", None] == 4
    assert metrics["unmatched_subject_rows", "owns"] == 3
    assert metrics["unmatched_object_rows", "owns"] == 2
    assert metrics["unmatched_entity_rows", "owns"] == 4


def _config(**overrides):
    return RelationEvalConfig.model_validate(
        {
            "kind": "relations",
            "predictions": "predictions",
            "expected": "gold",
            "subject_field": "subject",
            "relation_field": "relation",
            "object_field": "object",
            **overrides,
        }
    )


def test_relation_config_derives_edges_and_preserves_default_classification():
    config = _config()
    model = ModelConfig(name="quality", eval=config)
    assert model.depends_on == ["ref('predictions')", "ref('gold')"]
    assert model.kind_label() == "eval"
    legacy = ModelConfig.model_validate(
        {
            "name": "quality",
            "eval": {
                "predictions": "predictions",
                "expected": "gold",
                "key": "id",
                "predicted_field": "predicted",
                "expected_field": "expected",
            },
        }
    )
    assert isinstance(legacy.eval, EvalConfig)
    assert legacy.eval.model_dump() == {
        "kind": "classification",
        "predictions": "predictions",
        "expected": "gold",
        "key": "id",
        "predicted_field": "predicted",
        "expected_field": "expected",
        "labels": [],
    }


@pytest.mark.parametrize(
    "change",
    [
        {"subject_field": " "},
        {"object_field": "subject"},
        {"document_field": ""},
        {"labels": ["owns", "owns"]},
        {"labels": [""]},
        {"key": "id"},
        {"predicted_field": "label"},
        {"kind": "classification"},
        {"kind": "typo"},
    ],
)
def test_malformed_or_mixed_config_is_rejected_before_execution(change):
    with pytest.raises(ValidationError):
        _config(**change)


class FakeEvalWarehouse:
    """A projected, multi-batch table reader; no collecting read_table fallback."""

    def __init__(self, predicted: pl.DataFrame, gold: pl.DataFrame):
        self.frames = {"predictions": predicted, "gold": gold}
        self.projections: list[tuple[str, tuple[str, ...]]] = []
        self.closed: list[str] = []
        self.writes = 0

    def read_table_schema(self, table: str) -> pl.DataFrame:
        return self.frames[table].head(0)

    @contextmanager
    def table_snapshot(self, table: str, *, columns: list[str]) -> Iterator[list[pa.RecordBatch]]:
        self.projections.append((table, tuple(columns)))
        try:
            yield self.frames[table].select(columns).to_arrow().to_batches(max_chunksize=1)
        finally:
            self.closed.append(table)

    def materialize_full(self, *args, **kwargs):
        self.writes += 1
        return 0


def _frame(**overrides):
    return pl.DataFrame({"subject": ["a"], "relation": ["owns"], "object": ["b"], **overrides})


def test_projected_batches_deduplicate_across_pages_and_collect_version():
    adapter = FakeEvalWarehouse(
        pl.concat([_frame(), _frame()]).with_columns(
            pl.lit("v1").alias("code_version"), pl.lit("PRIVATE").alias("raw_text")
        ),
        _frame(raw_text=["PRIVATE"]),
    )
    scores, version = read_relation_scores(cast(Any, adapter), _config())
    assert version == "v1"
    assert scores[0].true_positives == 1
    assert adapter.projections == [
        ("predictions", ("subject", "relation", "object", "code_version")),
        ("gold", ("subject", "relation", "object")),
    ]
    assert adapter.closed == ["predictions", "gold"]


@pytest.mark.parametrize("versions, expected", [(["v1", "v2"], None), ([None, None], None)])
def test_mixed_or_absent_prediction_versions_do_not_invent_a_version(versions, expected):
    adapter = FakeEvalWarehouse(
        pl.concat([_frame(), _frame()]).with_columns(pl.Series("code_version", versions)), _frame()
    )
    _, version = read_relation_scores(cast(Any, adapter), _config())
    assert version == expected


@pytest.mark.parametrize(
    "predicted,gold,config,message",
    [
        (_frame(), _frame(subject=[None]), _config(), "nonempty strings"),
        (_frame(), _frame(object=[" "]), _config(), "nonempty strings"),
        (_frame(), _frame().head(0), _config(), "no gold triples"),
        (_frame(relation=[None]), _frame(), _config(), "nonempty strings"),
        (_frame(relation=[["PRIVATE"]]), _frame(), _config(), "must contain strings"),
        (_frame(subject=[12]), _frame(), _config(), "must contain strings"),
        (_frame().drop("object"), _frame(), _config(), "no required column"),
        (
            _frame(document=[None]),
            _frame(document=["d"]),
            _config(document_field="document"),
            "nonempty strings",
        ),
    ],
)
def test_bad_inputs_fail_before_metric_publication(tmp_path, predicted, gold, config, message):
    """Neither malformed gold nor a broken schema may replace good metrics (#731)."""
    adapter = FakeEvalWarehouse(predicted, gold)
    model = ModelConfig(name="quality", eval=config)
    with pytest.raises(RunError, match=message) as error:
        run_eval_model(
            model=model,
            models_by_name={},
            project_dir=tmp_path,
            adapter=cast(Any, adapter),
            full_refresh=False,
        )
    assert adapter.writes == 0
    assert "PRIVATE" not in str(error.value)
    assert error.value.__cause__ is None


def test_document_scope_is_explicit_and_corpus_scope_collapses_repeated_evidence():
    predicted = pl.concat([_frame(document=["d1"]), _frame(document=["d2"])])
    gold = _frame(document=["d1"])
    scoped, _ = read_relation_scores(
        cast(Any, FakeEvalWarehouse(predicted, gold)), _config(document_field="document")
    )
    corpus, _ = read_relation_scores(cast(Any, FakeEvalWarehouse(predicted, gold)), _config())
    assert scoped[0].false_positives == 1
    assert corpus[0].false_positives == 0


def test_scoring_configuration_participates_in_code_identity(tmp_path):
    def version(config):
        return compute_code_version(
            extraction=None, transform=None, eval_config=config, project_dir=tmp_path
        )

    assert (
        len(
            {
                version(_config()),
                version(_config(document_field="document")),
                version(_config(labels=["silent"])),
                version(_config(subject_field="id")),
            }
        )
        == 4
    )


def test_eval_capability_preflight_requires_streaming_reads(monkeypatch):
    available = adapter_capabilities("duckdb") - {WarehouseCapability.STREAMING_TABULAR_READS}
    monkeypatch.setattr(compiler, "adapter_capabilities", lambda _: available)
    with pytest.raises(ConfigError, match="projected relation eval input reads"):
        validate_warehouse_capabilities([ModelConfig(name="quality", eval=_config())], "duckdb")
