"""Render landed block rows (Notion, Confluence, Linear via an EL tool) into one
ordered markdown document per page.

YAML:

    - name: page_documents
      depends_on: [ref('notion_pages'), ref('notion_blocks')]
      transform:
        type: python
        module: stel.text.transforms.render_blocks
        options:
          pages: notion_pages          # one row per page: page_id, title, ...
          blocks: notion_blocks        # one row per block, keyed to its page
          include_fields: [parent_page_id, database_id]
      materialization: incremental
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from ...transforms import IncrementalContract, TransformContext
from ._blocks import (
    declared_render_dependencies,
    declared_render_incremental_contract,
    parse_render_blocks_options,
    run_render_blocks,
)


def validate_options(options: Mapping[str, Any]) -> None:
    parse_render_blocks_options(options)


def declared_dependencies(options: Mapping[str, Any]) -> tuple[str, str]:
    return declared_render_dependencies(options)


def declared_incremental_contract(options: Mapping[str, Any]) -> IncrementalContract:
    return declared_render_incremental_contract(options)


def run(
    deps: dict[str, pl.DataFrame],
    ctx: TransformContext,
) -> pl.DataFrame:
    return run_render_blocks(deps, ctx)
