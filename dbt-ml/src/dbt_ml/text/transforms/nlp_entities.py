"""Emit one stable child-table row per named entity."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from ...transforms import TransformContext
from ._nlp import run_entities, validate_entity_options


def validate_options(options: Mapping[str, Any]) -> None:
    validate_entity_options(options)


def run(
    deps: dict[str, pl.DataFrame],
    ctx: TransformContext,
) -> pl.DataFrame:
    return run_entities(deps, ctx)
