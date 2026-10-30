"""Projected warehouse reads for canonical relation evaluation (issue #731)."""

from __future__ import annotations

from collections.abc import Iterator

import polars as pl

from ..adapters import WarehouseAdapter
from ..config.model import RelationEvalConfig
from ..dag import parse_ref
from ..relation_metrics import RelationScore, RelationTriple, score_relations
from .contracts import RunError


def _text(value: object, *, column: str, relation: str, nullable: bool) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if nullable:
            return None
        raise RunError(
            f"Relation eval: '{relation}' column '{column}' must contain nonempty strings"
        )
    if not isinstance(value, str):
        raise RunError(f"Relation eval: '{relation}' column '{column}' must contain strings")
    return value


def _triples(
    adapter: WarehouseAdapter,
    config: RelationEvalConfig,
    relation: str,
    *,
    gold: bool,
    versions: set[str],
) -> Iterator[RelationTriple]:
    columns = [config.subject_field, config.relation_field, config.object_field]
    if config.document_field is not None:
        columns.append(config.document_field)
    schema = adapter.read_table_schema(relation)
    missing = sorted(set(columns) - set(schema.columns))
    if missing:
        raise RunError(f"Relation eval: '{relation}' has no required column(s): {missing}")
    has_version = not gold and "code_version" in schema.columns
    if has_version and "code_version" not in columns:
        columns.append("code_version")
    with adapter.table_snapshot(relation, columns=columns) as snapshot:
        for batch in snapshot:
            frame = pl.from_arrow(batch)
            assert isinstance(frame, pl.DataFrame)
            if has_version:
                versions.update(str(value) for value in frame["code_version"] if value is not None)
            for row in frame.iter_rows(named=True):
                yield _triple(row, config, relation, gold=gold)


def _triple(
    row: dict[str, object], config: RelationEvalConfig, relation: str, *, gold: bool
) -> RelationTriple:
    subject = _text(
        row[config.subject_field], column=config.subject_field, relation=relation, nullable=not gold
    )
    obj = _text(
        row[config.object_field], column=config.object_field, relation=relation, nullable=not gold
    )
    relation_type = _text(
        row[config.relation_field], column=config.relation_field, relation=relation, nullable=False
    )
    assert relation_type is not None
    document = None
    if config.document_field is not None:
        document = _text(
            row[config.document_field],
            column=config.document_field,
            relation=relation,
            nullable=False,
        )
    return RelationTriple(subject, relation_type, obj, document)


def read_relation_scores(
    adapter: WarehouseAdapter, config: RelationEvalConfig
) -> tuple[tuple[RelationScore, ...], str | None]:
    versions: set[str] = set()
    scores = score_relations(
        _triples(adapter, config, parse_ref(config.predictions), gold=False, versions=versions),
        _triples(adapter, config, parse_ref(config.expected), gold=True, versions=set()),
        labels=config.labels,
    )
    # An empty gold relation cannot establish quality. Empty predictions are
    # valid and score zero recall against the full gold universe.
    if scores[0].expected_triples == 0:
        raise RunError("Relation eval: expected relation has no gold triples to score")
    version = next(iter(versions)) if len(versions) == 1 else None
    return scores, version
