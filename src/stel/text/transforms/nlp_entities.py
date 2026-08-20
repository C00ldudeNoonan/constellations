"""Emit one stable child-table row per named entity."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from ...transforms import IncrementalContract, TransformContext
from ..nlp import NLPEntityOptions
from ._nlp import run_entities, validate_entity_options


def validate_options(options: Mapping[str, Any]) -> None:
    validate_entity_options(options)


def declared_incremental_contract(options: Mapping[str, Any]) -> IncrementalContract:
    """One entity child table per document; re-analyzing a changed document
    replaces exactly its entity rows (issue #218)."""
    parsed = NLPEntityOptions.model_validate(options)
    return IncrementalContract(
        parent_key="document_id",
        child_key="entity_id",
        parent_source_key=parsed.document_id_field,
    )


def run(
    deps: dict[str, pl.DataFrame],
    ctx: TransformContext,
) -> pl.DataFrame:
    return run_entities(deps, ctx)
