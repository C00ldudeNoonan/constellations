"""Close the interval of a functional relation that a later fact replaced.

A `full` step over the relation table and the entity-link table (ADR-0056). It
has no incremental contract on purpose: closing a row because another document
arrived writes outside the changed document's parent key.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import polars as pl

from ...relation_validity import (
    SUPERSESSION_VERSION,
    parse_supersede_options,
    supersede,
)
from ...transforms import TransformContext

log = logging.getLogger(__name__)


def validate_options(options: Mapping[str, Any]) -> None:
    parse_supersede_options(options)


def declared_dependencies(options: Mapping[str, Any]) -> tuple[str, ...]:
    parsed = parse_supersede_options(options)
    return (parsed.relations, parsed.entity_links)


def code_version_identity(options: Mapping[str, Any]) -> dict[str, str]:
    """The closing rule's version: this built-in module has no project-local
    file for the code-version hash to see (same reason `extract_relations`
    reports its extractor's)."""
    parse_supersede_options(options)
    return {"supersession_version": SUPERSESSION_VERSION}


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    options = parse_supersede_options(ctx.options)
    expected = {options.relations, options.entity_links}
    if set(deps) != expected:
        raise ValueError(
            "supersede_relations expects the models named by `relations` and "
            f"`entity_links` ({sorted(expected)}); got: {sorted(deps)}"
        )
    functional = frozenset(
        relation.name for relation in ctx.relations if relation.functional
    )
    if not functional:
        raise ValueError(
            "supersede_relations has no relation to act on: no relation under "
            "`relations:` declares `functional: true`"
        )
    result = supersede(
        deps[options.relations], deps[options.entity_links], functional=functional
    )
    log.info(
        "supersede_relations: closed %d row(s); left out %s; %d simultaneous "
        "conflict(s) left open",
        result.closed,
        ", ".join(f"{count} {reason}" for reason, count in result.skipped.items()),
        result.simultaneous_conflicts,
    )
    return result.frame
