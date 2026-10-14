"""One row per linked system: the name and description the star map shows.

Without it a star is named by whatever the corpus calls it most often
("webhooks"), not by what the vocabulary says it is.
"""
from __future__ import annotations

from typing import Any

import polars as pl

from stel.transforms import TransformContext

_SCHEMA: dict[str, pl.DataType] = {
    "canonical_id": pl.String(),
    "display_name": pl.String(),
    "description": pl.String(),
    "class": pl.String(),
}


def validate_options(options: dict[str, Any]) -> None:
    if not isinstance(options.get("vocabulary"), str):
        raise ValueError("system_names needs `vocabulary:`, the name of a declared vocabulary")


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    name = ctx.options["vocabulary"]
    if name not in ctx.vocabularies:
        raise ValueError(f"vocabulary '{name}' is not declared in stel_project.yml")
    linked = set(deps["entity_links"]["canonical_id"].drop_nulls().to_list())
    rows = [
        {
            "canonical_id": term.label,
            "display_name": term.label,
            "description": term.description,
            "class": term.entity_class,
        }
        for term in ctx.vocabularies[name].terms
        if term.label in linked
    ]
    return pl.DataFrame(rows, schema=_SCHEMA)
