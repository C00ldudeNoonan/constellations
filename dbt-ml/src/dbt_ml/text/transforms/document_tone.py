"""Deterministic document-level tone/sentiment over the token child table."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from ...transforms import TransformContext
from ._tone import declared_tone_dependencies, run_tone, validate_tone_options


def validate_options(options: Mapping[str, Any]) -> None:
    validate_tone_options(options)


def declared_dependencies(options: Mapping[str, Any]) -> tuple[str, str]:
    return declared_tone_dependencies(options)


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    return run_tone(deps, ctx)
