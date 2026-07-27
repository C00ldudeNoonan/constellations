"""Emit one stable child-table row per NLP token."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from ...transforms import TransformContext
from ._nlp import run_tokens, validate_token_options


def validate_options(options: Mapping[str, Any]) -> None:
    validate_token_options(options)


def run(
    deps: dict[str, pl.DataFrame],
    ctx: TransformContext,
) -> pl.DataFrame:
    return run_tokens(deps, ctx)
