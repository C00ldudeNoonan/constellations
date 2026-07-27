"""Link entity mentions to canonical IDs via an operator-owned alias table."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from ...transforms import TransformContext
from ._linking import run_links, validate_link_options


def validate_options(options: Mapping[str, Any]) -> None:
    validate_link_options(options)


def run(
    deps: dict[str, pl.DataFrame],
    ctx: TransformContext,
) -> pl.DataFrame:
    return run_links(deps, ctx)
