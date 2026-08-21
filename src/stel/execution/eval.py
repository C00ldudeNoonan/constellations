"""Classification eval models (issue #309).

Reads two already-materialized relations — predictions and labelled ground
truth — and publishes long-format metric rows. No inference, no provider, no
credentials: an eval is pure warehouse arithmetic, so it is cheap enough to run
on every change and safe to run in CI.

The metric maths lives in `classification_metrics.py`; this module is the
warehouse boundary around it.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from ..adapters import WarehouseAdapter
from ..classification_metrics import (
    as_rows,
    declared_labels_for,
    join_pairs,
    score,
)
from ..config.model import ModelConfig
from ..dag import parse_ref
from ..hashing import canonical_fingerprint
from ..versioning import compute_code_version
from .contracts import ModelRunResult, RunError
from .warehouse import warehouse_options

# Identity of one metric row, so an incremental eval upserts rather than
# duplicating. Keyed on what the metric *is* plus the version of the
# predictions it scored — re-running the same predictions replaces the row,
# and a new predictions version appends one. That is the quality time series.
_METRIC_ID_DOMAIN = "eval-metric-id"

# Column carrying the producing code version on model kinds that publish one.
_PREDICTIONS_VERSION_COLUMN = "code_version"


def _column(frame: pl.DataFrame, name: str, model: str, relation: str) -> None:
    if name not in frame.columns:
        raise RunError(
            f"Eval model '{model}': relation '{relation}' has no column "
            f"'{name}'. Available: {sorted(frame.columns)}"
        )


def _labels_by_key(
    frame: pl.DataFrame,
    *,
    key: str,
    value: str,
    model: str,
    relation: str,
) -> tuple[dict[str, str], int]:
    """Key→label mapping, plus how many rows could not enter it.

    A null key cannot be joined and a null label is not a class; those rows
    are counted and returned rather than silently dropped, because metrics
    computed over the rows that happened to survive are inflated, not smaller
    (Codex review, #328). A duplicate key is a hard error: keeping the last
    row would make support and accuracy depend on warehouse row order, which
    silently corrupts an experiment.
    """
    out: dict[str, str] = {}
    duplicates: set[str] = set()
    dropped = 0
    for row in frame.iter_rows(named=True):
        row_key = row[key]
        label = row[value]
        if row_key is None or label is None:
            dropped += 1
            continue
        row_key = str(row_key)
        if row_key in out:
            duplicates.add(row_key)
        out[row_key] = str(label)
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:5])
        raise RunError(
            f"Eval model '{model}': relation '{relation}' has "
            f"{len(duplicates)} duplicate '{key}' value(s) (e.g. {sample}). "
            "Which row would be scored depends on warehouse row order; "
            "deduplicate the relation or evaluate at a different key."
        )
    return out, dropped


def _predictions_version(frame: pl.DataFrame) -> str | None:
    """The single code version behind these predictions, when there is one.

    A relation built in one run carries one version. A mixed relation (an
    incremental model mid-migration) has no single answer, so it reports None
    rather than picking one and making the time series lie.
    """
    if _PREDICTIONS_VERSION_COLUMN not in frame.columns:
        return None
    versions = {
        value
        for value in frame[_PREDICTIONS_VERSION_COLUMN].to_list()
        if value is not None
    }
    if len(versions) != 1:
        return None
    return str(next(iter(versions)))


def run_eval_model(
    *,
    model: ModelConfig,
    models_by_name: Mapping[str, ModelConfig],
    project_dir: Path,
    adapter: WarehouseAdapter,
    full_refresh: bool,
) -> ModelRunResult:
    assert model.eval is not None
    config = model.eval

    predictions_name = parse_ref(config.predictions)
    expected_name = parse_ref(config.expected)
    predictions = adapter.read_table(predictions_name)
    expected = adapter.read_table(expected_name)

    _column(predictions, config.key, model.name, predictions_name)
    _column(predictions, config.predicted_field, model.name, predictions_name)
    _column(expected, config.key, model.name, expected_name)
    _column(expected, config.expected_field, model.name, expected_name)

    # Predictions-side null-row loss needs no separate metric: a prediction
    # that cannot be joined leaves its expected counterpart unmatched, which
    # is already counted below.
    predicted_by_key, _ = _labels_by_key(
        predictions,
        key=config.key,
        value=config.predicted_field,
        model=model.name,
        relation=predictions_name,
    )
    expected_by_key, unusable_expected = _labels_by_key(
        expected,
        key=config.key,
        value=config.expected_field,
        model=model.name,
        relation=expected_name,
    )
    if not expected_by_key:
        raise RunError(
            f"Eval model '{model.name}': '{expected_name}' has no usable rows "
            f"(every '{config.key}' or '{config.expected_field}' was null). "
            "Scoring nothing would report a perfect zero."
        )

    pairs, unmatched = join_pairs(predicted_by_key, expected_by_key)

    # The taxonomy the model was asked for, so a label it stopped predicting
    # reports recall 0.0 instead of vanishing from the report (issue #304).
    declared = list(config.labels)
    if not declared:
        upstream = models_by_name.get(predictions_name)
        if upstream is not None:
            declared = list(
                declared_labels_for(config.predicted_field, upstream.fields)
            )

    report = score(
        pairs,
        declared_labels=declared,
        unmatched_rows=unmatched,
        unusable_expected_rows=unusable_expected,
    )

    code_version = compute_code_version(
        extraction=None,
        transform=None,
        eval_config=config,
        depends_on=model.depends_on,
        project_dir=project_dir,
    )
    evaluated_at = datetime.now(UTC).isoformat()
    predictions_version = _predictions_version(predictions)

    rows: list[dict[str, Any]] = []
    for row in as_rows(report):
        rows.append(
            {
                "metric_id": canonical_fingerprint(
                    {
                        "model": model.name,
                        "metric": row["metric"],
                        "label": row["label"],
                        "predictions_version": predictions_version,
                    },
                    domain=_METRIC_ID_DOMAIN,
                ),
                **row,
                "predictions_version": predictions_version,
                "code_version": code_version,
                "evaluated_at": evaluated_at,
            }
        )

    frame = pl.DataFrame(rows)
    parsed_options = warehouse_options(adapter, model)
    if model.materialization == "full" or full_refresh:
        rows_written = adapter.materialize_full(
            model.name, frame, options=parsed_options
        )
    else:
        # A re-run of the same predictions version must fully replace that
        # version's metric set, not just upsert into it: shrink the taxonomy
        # and the removed label's rows would otherwise survive with a stale
        # code_version, contradicting the declared universe and feeding
        # min_metric stale results (Codex review, #328). History for *other*
        # versions is untouched — that is the time series.
        current_ids = set(frame["metric_id"].to_list())
        if model.name in set(adapter.list_tables()):
            existing = adapter.read_table(model.name)
            if "predictions_version" in existing.columns:
                same_version = existing.filter(
                    pl.col("predictions_version") == predictions_version
                    if predictions_version is not None
                    else pl.col("predictions_version").is_null()
                )
                stale = [
                    metric_id
                    for metric_id in same_version["metric_id"].to_list()
                    if metric_id not in current_ids
                ]
                if stale:
                    adapter.delete_rows(
                        model.name, key_col="metric_id", keys=stale
                    )
        outcome = adapter.materialize_incremental(
            model.name,
            frame,
            key_col="metric_id",
            on_schema_change=model.on_schema_change,
            options=parsed_options,
            update_when_changed=model.update_when_changed,
        )
        rows_written = outcome if isinstance(outcome, int) else outcome.rows_written

    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="eval",
        documents_processed=report.evaluated_rows,
        documents_skipped=report.unmatched_rows,
        rows_written=rows_written,
    )
