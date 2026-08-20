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


def _project_metrics(ml: MLConfig, metrics: dict[str, Any]) -> dict[str, Any]:
    if not ml.metrics:
        return dict(metrics)
    return {name: metrics.get(name) for name in ml.metrics}
