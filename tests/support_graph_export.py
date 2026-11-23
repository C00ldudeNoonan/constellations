"""Frames and writers shared by the graph export tests (#518: a helper two test
modules need lives here, not in one of them)."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl

from stel.graph_export import (
    GraphContext,
    merge_documents,
    write_graph_export,
)


def links_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    base = {
        "document_id": "d1",
        "entity_namespace": "systems",
        "status": "matched",
        "resolver": "alias_table",
        "resolver_version": "1",
    }
    return pl.DataFrame(
        [{**base, "entity_link_id": f"l{index}", **row} for index, row in enumerate(rows)],
        schema={
            "entity_link_id": pl.String(),
            "document_id": pl.String(),
            "mention_id": pl.String(),
            "canonical_id": pl.String(),
            "entity_namespace": pl.String(),
            "status": pl.String(),
            "resolver": pl.String(),
            "resolver_version": pl.String(),
        },
    )


def relations_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    base = {
        "document_id": "d1",
        "relation_type": "co_occurs_with",
        "directed": False,
        "method": "co_occurrence",
        "status": "asserted",
        "confidence": None,
        "sentence_index": 0,
        "extractor": "co_occurrence",
        "extractor_version": "1",
    }
    return pl.DataFrame(
        [{**base, **row} for row in rows],
        schema={
            "relation_id": pl.String(),
            "document_id": pl.String(),
            "subject_mention_id": pl.String(),
            "object_mention_id": pl.String(),
            "relation_type": pl.String(),
            "directed": pl.Boolean(),
            "method": pl.String(),
            "status": pl.String(),
            "confidence": pl.Float64(),
            "sentence_index": pl.Int64(),
            "extractor": pl.String(),
            "extractor_version": pl.String(),
        },
    )


def documents(links: pl.DataFrame, relations: pl.DataFrame) -> Iterator[Any]:
    """The two tables as the warehouse hands them over: ordered by document."""
    return merge_documents(
        links.sort("document_id").iter_rows(named=True),
        relations.sort("document_id").iter_rows(named=True),
    )


def context(**kwargs: Any) -> GraphContext:
    options: dict[str, Any] = {"entity_name": "system", "vocabularies": {}, "classes": ()}
    options.update(kwargs)
    return GraphContext.create(**options)


def export(
    out: Path,
    links: pl.DataFrame,
    relations: pl.DataFrame,
    *,
    relation_model: str = "r",
    graph_format: str = "jsonl",
    **kwargs: Any,
) -> Any:
    return write_graph_export(
        documents(links, relations),
        out,
        project="p",
        linking_model="l",
        relation_model=relation_model,
        context=context(**kwargs),
        graph_format=graph_format,
    )


def snapshot(out: Path) -> dict[str, bytes]:
    return {
        path.relative_to(out).as_posix(): path.read_bytes()
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
