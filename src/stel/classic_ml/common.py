"""Shared classic-ML run plumbing (issue #190, Workstream B).

The run result contract, deterministic source-row assembly, training-input
provenance, and metrics projection — used by every algorithm family. Imports
no family module, so the dependency stays one-way.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from ..config.model import MLConfig
from ..dag import parse_ref
from ..hashing import HASH_DIGEST_SIZE
from .artifacts import ClassicMLArtifactPublication


@dataclass
class ClassicMLRun:
    df: pl.DataFrame
    artifact_path: Path
    artifact_version: str
    training_input: dict[str, Any]
    metrics: dict[str, Any]
    artifact_metadata: dict[str, Any]
    # Companion tables materialized as `<model>__<key>` alongside the primary
    # table (e.g. topic_model emits `topics`; cluster emits `representative_docs`).
    secondary_tables: dict[str, pl.DataFrame] = field(default_factory=dict)
    _publication: ClassicMLArtifactPublication | None = field(default=None, repr=False)

    def publish_artifact(self) -> None:
        if self._publication is not None:
            self._publication.publish()

    def discard_staged_artifact(self) -> None:
        if self._publication is not None:
            self._publication.discard()


def _canonical_row_key(row: dict[str, Any]) -> tuple[int, str, str]:
    """Warehouses return `SELECT *` in arbitrary order; training input must
    not depend on it. Order by the stable row identifier when present —
    chunk_id before document_id, since chunk models repeat document_id
    across a document's chunks — with canonical row content breaking any
    remaining ties (fully identical rows are interchangeable)."""
    content = json.dumps(row, sort_keys=True, default=str)
    for key in ("chunk_id", "document_id", "id"):
        value = row.get(key)
        if value is not None:
            return (0, str(value), content)
    return (1, content, "")


def _source_rows(
    df: pl.DataFrame,
    text_field: str,
    label_field: str | None = None,
) -> list[dict[str, Any]]:
    ordered = sorted(df.iter_rows(named=True), key=_canonical_row_key)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(ordered):
        text = "" if row[text_field] is None else str(row[text_field])
        row_id = str(row.get("document_id") or row.get("id") or index)
        payload: dict[str, Any] = {"row_index": index, "row_id": row_id, "text": text}
        if label_field is not None:
            payload["label"] = None if row[label_field] is None else str(row[label_field])
        if "document_id" in row:
            payload["document_id"] = row["document_id"]
        if "source_path" in row:
            payload["source_path"] = row["source_path"]
        rows.append(payload)
    return rows


def _training_input(depends_on: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    content = [
        {
            key: row[key]
            for key in ("row_id", "text", "label")
            if key in row
        }
        for row in rows
    ]
    raw = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return {
        "refs": [parse_ref(ref) for ref in depends_on],
        "row_count": len(rows),
        "content_hash": hashlib.blake2b(
            raw.encode(), digest_size=HASH_DIGEST_SIZE
        ).hexdigest(),
    }


# Significant digits kept for a fitted float metric (issue #600). A metric is
# a reduction over the matrix, and the order of that reduction depends on how
# many threads the BLAS/OpenMP runtime used -- which is a property of the
# machine, not of the fit. Measured on kmeans, `inertia` moved one ULP between
# a fit under one thread and the same fit under two (0x1.da28f795400edp-3 /
# ...eep-3 / ...efp-3 at one, two and four), which is far below anything the
# number is read for and far above nothing: `artifact_version` hashes the
# metadata, so one ULP changed the artifact's identity. float64 carries ~15.95
# significant digits, so 12 leaves several digits of headroom over an
# accumulation of rounding error while preserving every digit a person uses.
# The model payload has been rounded for the same reason since it was written
# (`round(float(v), 6)` on the centroids); metrics were the gap.
_METRIC_SIGNIFICANT_DIGITS = 12


def _stable_metric(value: Any) -> Any:
    """Round a metric to a precision the machine cannot move it below.

    Recursive: a metric may be a list (per-class scores) or a mapping, and a
    single unstable float anywhere under it moves the artifact version. Only
    `float` is touched -- an `int` count or a `bool` is already exact.
    """
    if isinstance(value, float):
        # `.12g` is significant digits rather than decimal places, so it
        # behaves the same for an inertia near 0.2 and a perplexity near 1e6.
        # Infinities and NaN round-trip through this unchanged.
        return float(f"{value:.{_METRIC_SIGNIFICANT_DIGITS}g}")
    if isinstance(value, dict):
        return {key: _stable_metric(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable_metric(item) for item in value]
    return value


def _project_metrics(ml: MLConfig, metrics: dict[str, Any]) -> dict[str, Any]:
    if not ml.metrics:
        return {name: _stable_metric(value) for name, value in metrics.items()}
    return {name: _stable_metric(metrics.get(name)) for name in ml.metrics}
