"""Emit one stable child-table row per extracted keyphrase per document."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from ...transforms import IncrementalContract, TransformContext
from ._keyphrases import (
    declared_keyphrase_dependencies,
    keyphrase_incremental_contract,
    run_keyphrases,
    validate_keyphrase_options,
)


def validate_options(options: Mapping[str, Any]) -> None:
    validate_keyphrase_options(options)


def declared_dependencies(options: Mapping[str, Any]) -> tuple[str, ...]:
    return declared_keyphrase_dependencies(options)


def declared_incremental_contract(options: Mapping[str, Any]) -> IncrementalContract:
    return keyphrase_incremental_contract(options)


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    return run_keyphrases(deps, ctx)
