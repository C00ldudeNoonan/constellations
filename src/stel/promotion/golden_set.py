"""Materialize a promoted golden-set file as a golden-set relation (#380).

The last step of #329 phase 3, and deliberately the thinnest one: the evals
need no changes, because `retrieval_tests.golden_set` already refs an ordinary
model. This transform is what turns the reviewed file into that model.

Point a model at the file and at the search index the set is for::

    depends_on: [ref('retrieval_judgment_candidates')]
    transform:
      type: python
      module: stel.promotion.golden_set
      options:
        path: golden_sets/context_search.yml
        search_model: context_search

`search_model` is not decoration — the file's declared `id_space` is checked
against that index's `id_field`, so a set promoted in the wrong id space fails
loudly instead of matching nothing and reporting a perfect zero (issue #380,
constraint 3).

`depends_on` names the candidates the set was promoted from. The rows are not
read — the file is the source of truth, and a promoted golden must survive the
corpus it came from being rotated away — but the edge keeps promotion ordered
after derivation in the DAG, and the runner requires a transform to declare
one.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import polars as pl

from ..config import load_project
from ..paths import resolve_within_project
from ..transforms import TransformContext
from .contract import PromotionError, load_golden_set

_SCHEMA: dict[str, pl.DataType] = {
    "query_id": pl.String(),
    "query_text": pl.String(),
    "mode": pl.String(),
    "relevant_ids": pl.String(),
    "required_ids": pl.String(),
    "excluded_ids": pl.String(),
    "id_space": pl.String(),
    "promoted_by": pl.String(),
    "promoted_at": pl.String(),
    "evidence_sessions": pl.String(),
    "evidence_harness": pl.String(),
    "query_fingerprint": pl.String(),
}


def validate_options(options: Mapping[str, Any]) -> None:
    unknown = sorted(set(options) - {"path", "search_model"})
    if unknown:
        raise ValueError(
            f"golden_set: unknown options {unknown}; expected 'path' and "
            "'search_model'"
        )
    for name in ("path", "search_model"):
        value = options.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"golden_set requires a non-empty `{name}:`")


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    del deps  # the file is the source of truth; see the module docstring
    validate_options(ctx.options)
    path = resolve_within_project(
        str(ctx.options["path"]),
        ctx.project_dir,
        surface="golden_set transform option `path`",
    )
    golden = load_golden_set(path)
    _check_id_space(golden.id_space, ctx, path=str(path))
    rows = [
        {
            "query_id": query.query_id,
            "query_text": query.query_text,
            "mode": query.mode,
            "relevant_ids": json.dumps(list(query.relevant_ids)),
            "required_ids": json.dumps(list(query.required_ids)),
            "excluded_ids": json.dumps(list(query.excluded_ids)),
            "id_space": golden.id_space,
            "promoted_by": query.promoted_by,
            "promoted_at": query.promoted_at.isoformat(),
            "evidence_sessions": json.dumps(list(query.evidence.sessions)),
            "evidence_harness": query.evidence.harness,
            "query_fingerprint": query.evidence.query_fingerprint,
        }
        for query in golden.queries
    ]
    return pl.DataFrame(rows, schema=_SCHEMA)


def _check_id_space(id_space: str, ctx: TransformContext, *, path: str) -> None:
    """Refuse a set promoted in an id space the target index does not key on.

    Without this the mismatch is invisible: every `relevant_id` simply fails
    to match a returned `record_id`, and the eval reports zero recall as if
    retrieval were broken rather than as if the golden set were mislabelled.
    """
    model_name = str(ctx.options["search_model"])
    _project, _sources, models = load_project(ctx.project_dir)
    target = next((model for model in models if model.name == model_name), None)
    if target is None:
        raise PromotionError(
            f"golden_set `search_model: {model_name}` is not a model in this project"
        )
    if target.search is None:
        raise PromotionError(
            f"golden_set `search_model: {model_name}` does not declare `search:`"
        )
    if target.search.id_field != id_space:
        raise PromotionError(
            f"{path} promotes ids in the '{id_space}' space, but search model "
            f"'{model_name}' keys on '{target.search.id_field}'. Every id would "
            "silently fail to match and the eval would report zero recall; "
            "re-promote in the index's id space."
        )
