"""Emit one stable child-table row per relation between two entity mentions."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from ...transforms import IncrementalContract, TransformContext
from ..relations import get_relation_extractor, parse_relation_options
from ._relations import (
    declared_relation_dependencies,
    declared_relation_incremental_contract,
    run_relations,
    validate_relation_options,
)


def validate_options(options: Mapping[str, Any]) -> None:
    validate_relation_options(options)


def requires_llm(options: Mapping[str, Any]) -> bool:
    """The `model_assertion` extractor calls a governed inference provider, so a
    model using it must set `transform.uses_llm: true`. The deterministic
    extractors return False."""
    return get_relation_extractor(
        parse_relation_options(options).extractor
    ).requires_inference()


def declared_dependencies(options: Mapping[str, Any]) -> tuple[str, ...]:
    return declared_relation_dependencies(options)


def declared_incremental_contract(options: Mapping[str, Any]) -> IncrementalContract:
    return declared_relation_incremental_contract(options)


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    return run_relations(deps, ctx)
