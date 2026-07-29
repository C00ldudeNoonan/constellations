"""Emit one typed row of aggregate features per document."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from ...transforms import TransformContext
from ._document_features import (
    declared_feature_dependencies,
    run_document_features,
    validate_feature_options,
)


def validate_options(options: Mapping[str, Any]) -> None:
    validate_feature_options(options)


def declared_dependencies(options: Mapping[str, Any]) -> tuple[str, ...]:
    return declared_feature_dependencies(options)


def run(
    deps: dict[str, pl.DataFrame],
    ctx: TransformContext,
) -> pl.DataFrame:
    return run_document_features(deps, ctx)
