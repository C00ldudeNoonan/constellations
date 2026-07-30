"""Emit one stable child-table row per NLP token."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from ...transforms import IncrementalContract, TransformContext
from ..nlp import NLPTokenOptions
from ._nlp import run_tokens, validate_token_options


def validate_options(options: Mapping[str, Any]) -> None:
    validate_token_options(options)


def declared_incremental_contract(options: Mapping[str, Any]) -> IncrementalContract:
    """One token child table per document; re-tokenizing a changed document
    replaces exactly its token rows (issue #218)."""
    parsed = NLPTokenOptions.model_validate(options)
    return IncrementalContract(
        parent_key="document_id",
        child_key="token_id",
        parent_source_key=parsed.document_id_field,
    )


def run(
    deps: dict[str, pl.DataFrame],
    ctx: TransformContext,
) -> pl.DataFrame:
    return run_tokens(deps, ctx)
