"""Comparing variants of a search model on one golden set (issue #532).

Two variants of a step exist side by side (`for_each`, see the reference's
"Experimenting with variants"), and `stel eval` scores each against its
golden set, one artifact per model. Which one is better was a diff of two
JSON files by eye. This module runs the same per-model evaluation for every
model in a selection and emits one comparison: per metric, each variant's
value and its delta from the baseline; per query, the ones whose score moved,
so a regression is traceable to examples rather than to a number.

No second scorer. The numbers are `retrieval_eval.py`'s, computed once per
model; this module only lines them up. Comparability is checked before any
query runs: every model must carry the same tests, each on the same golden
set, at the same cutoffs and granularity, or the comparison is meaningless
and refused with both models named.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compiler import validate_project_contract
from .config import load_project
from .config.model import ModelConfig
from .config.project import ProjectConfig
from .dag import ProjectDAG, parse_ref
from .profile import resolve_profile
from .retrieval_eval import (
    RetrievalEvalError,
    RetrievalTestResult,
    evaluate_search_models,
)
from .versioning import compute_model_code_version

# A metric is unchanged when it differs by less than this; the means are
# floating-point sums over queries, so exact equality would report a
# "movement" that is only summation order.
_UNCHANGED = 1e-9


@dataclass(frozen=True)
class Side:
    """One model's result for one test, with the code version that produced
    the index it was scored on -- the identity a reader needs to tie a
    number back to a configuration."""

    model_name: str
    code_version: str
    result: RetrievalTestResult


@dataclass(frozen=True)
class QueryMovement:
    query_id: str
    baseline_diagnosis: str
    diagnosis: str
    #: metric -> cutoff -> variant minus baseline; only the cutoffs that moved.
    deltas: Mapping[str, Mapping[int, float]]
    baseline_ranked_ids: tuple[str, ...]
    ranked_ids: tuple[str, ...]


@dataclass(frozen=True)
class VariantComparison:
    side: Side
    #: metric -> cutoff -> variant minus baseline, for every scored cutoff.
    deltas: Mapping[str, Mapping[int, float]]
    moved: tuple[QueryMovement, ...]


@dataclass(frozen=True)
class TestComparison:
    test_name: str
    golden_set: str
    golden_set_hash: str
    granularity: str
    cutoffs: tuple[int, ...]
    baseline: Side
    variants: tuple[VariantComparison, ...]


@dataclass(frozen=True)
class RetrievalComparison:
    project: str
    baseline: str
    #: Baseline first, then the variants in comparison order.
    models: tuple[str, ...]
    tests: tuple[TestComparison, ...]
    #: Every per-model result the comparison was built from, in evaluation
    #: order, so the ordinary eval artifact can be written alongside.
    results: tuple[RetrievalTestResult, ...]

    def status(self) -> str:
        """Worst status over every side: a variant that leaks an excluded id
        is a failed comparison, not a data point."""
        statuses = {result.status for result in self.results}
        for worst in ("fail", "warn"):
            if worst in statuses:
                return worst
        return "pass"


def compare_retrieval_variants(
    project_dir: Path,
    *,
    compare: str,
    baseline: str | None,
    target: str | None,
    profiles_dir: Path | None,
) -> RetrievalComparison:
    """Evaluate every search model `compare` selects and line the results up
    against `baseline` (the first selected model when None).

    Selection order is the order the models were named when every token of
    `compare` is a plain model name, and name order otherwise (a tag or a
    graph operator has no order of its own)."""
    project, sources, models = load_project(project_dir)
    dag = validate_project_contract(project, sources, models, project_dir)
    resolved = resolve_profile(project, project_dir, target=target, profiles_dir=profiles_dir)
    models_by_name = {model.name: model for model in models}

    ordered = _ordered_selection(dag, compare)
    if len(ordered) < 2:
        raise RetrievalEvalError(
            f"--compare needs at least two search models; '{compare}' selects "
            f"{len(ordered)}: {ordered}"
        )
    if baseline is not None:
        if baseline not in ordered:
            raise RetrievalEvalError(
                f"--baseline '{baseline}' is not among the compared models {ordered}"
            )
        ordered = [baseline, *(name for name in ordered if name != baseline)]
    base_model = models_by_name[ordered[0]]
    _require_evaluable(base_model)
    for name in ordered[1:]:
        variant = models_by_name[name]
        _require_evaluable(variant)
        _require_comparable(base_model, variant)

    results = evaluate_search_models(
        project_dir,
        models,
        resolved,
        selected=set(ordered),
        target=target,
        profiles_dir=profiles_dir,
    )
    code_versions = {
        name: compute_model_code_version(
            models_by_name[name], project, project_dir, resolved=resolved
        )
        for name in ordered
    }
    by_key = {(result.model_name, result.test_name): result for result in results}

    tests: list[TestComparison] = []
    for test in base_model.retrieval_tests:
        base_result = by_key[(base_model.name, test.name)]
        base_side = Side(base_model.name, code_versions[base_model.name], base_result)
        variants: list[VariantComparison] = []
        for name in ordered[1:]:
            result = by_key[(name, test.name)]
            deltas, moved = compare_results(base_result, result)
            variants.append(
                VariantComparison(
                    side=Side(name, code_versions[name], result),
                    deltas=deltas,
                    moved=moved,
                )
            )
        tests.append(
            TestComparison(
                test_name=test.name,
                golden_set=base_result.golden_set,
                golden_set_hash=base_result.golden_set_hash,
                granularity=base_result.granularity,
                cutoffs=tuple(test.at),
                baseline=base_side,
                variants=tuple(variants),
            )
        )
    return RetrievalComparison(
        project=project.name,
        baseline=ordered[0],
        models=tuple(ordered),
        tests=tuple(tests),
        results=tuple(results),
    )


def _ordered_selection(dag: ProjectDAG, compare: str) -> list[str]:
    selected = dag.select_models(select=compare)
    tokens = compare.split()
    if all(token in selected for token in tokens):
        return list(dict.fromkeys(tokens))
    return sorted(selected)


def _require_evaluable(model: ModelConfig) -> None:
    if model.search is None:
        raise RetrievalEvalError(
            f"--compare selected '{model.name}', which is not a search model"
        )
    if not model.retrieval_tests:
        raise RetrievalEvalError(
            f"--compare selected '{model.name}', which declares no retrieval_tests"
        )


def _require_comparable(base: ModelConfig, variant: ModelConfig) -> None:
    """The same tests on the same ground truth, or the numbers are not
    comparable and the comparison is refused before a query runs."""
    base_tests = {test.name: test for test in base.retrieval_tests}
    variant_tests = {test.name: test for test in variant.retrieval_tests}
    if set(base_tests) != set(variant_tests):
        raise RetrievalEvalError(
            f"--compare: '{base.name}' declares retrieval_tests "
            f"{sorted(base_tests)} but '{variant.name}' declares "
            f"{sorted(variant_tests)}; compared models must carry the same tests"
        )
    for name, base_test in base_tests.items():
        variant_test = variant_tests[name]
        base_golden = parse_ref(base_test.golden_set)
        variant_golden = parse_ref(variant_test.golden_set)
        if base_golden != variant_golden:
            raise RetrievalEvalError(
                f"--compare: test '{name}' scores '{base.name}' against golden "
                f"set '{base_golden}' but '{variant.name}' against "
                f"'{variant_golden}'; a comparison needs one golden set"
            )
        for field_name in ("at", "granularity"):
            base_value = getattr(base_test, field_name)
            variant_value = getattr(variant_test, field_name)
            if base_value != variant_value:
                raise RetrievalEvalError(
                    f"--compare: test '{name}' sets `{field_name}: "
                    f"{_render(base_value)}` on '{base.name}' but `{field_name}: "
                    f"{_render(variant_value)}` on '{variant.name}'"
                )


def _render(value: object) -> str:
    return json.dumps(list(value)) if isinstance(value, tuple) else str(value)


def compare_results(
    baseline: RetrievalTestResult, variant: RetrievalTestResult
) -> tuple[dict[str, dict[int, float]], tuple[QueryMovement, ...]]:
    """Aggregate deltas over every scored metric and cutoff, and the queries
    whose per-query score moved. A rank change that changes no metric is not
    a movement: two irrelevant results swapping places is noise."""
    deltas: dict[str, dict[int, float]] = {}
    for metric, by_cutoff in baseline.aggregate.items():
        variant_by_cutoff = variant.aggregate.get(metric, {})
        deltas[metric] = {
            cutoff: variant_by_cutoff.get(cutoff, 0.0) - value
            for cutoff, value in by_cutoff.items()
        }

    variant_queries = {query.query_id: query for query in variant.per_query}
    moved: list[QueryMovement] = []
    for base_query in baseline.per_query:
        variant_query = variant_queries.get(base_query.query_id)
        if variant_query is None:
            continue
        query_deltas = _query_deltas(base_query.values, variant_query.values)
        if not query_deltas:
            continue
        moved.append(
            QueryMovement(
                query_id=base_query.query_id,
                baseline_diagnosis=base_query.diagnosis.value,
                diagnosis=variant_query.diagnosis.value,
                deltas=query_deltas,
                baseline_ranked_ids=base_query.ranked_ids,
                ranked_ids=variant_query.ranked_ids,
            )
        )
    return deltas, tuple(moved)


def _query_deltas(
    baseline: Mapping[str, Mapping[int, float]],
    variant: Mapping[str, Mapping[int, float]],
) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for metric in sorted(set(baseline) | set(variant)):
        base_by_cutoff = baseline.get(metric, {})
        variant_by_cutoff = variant.get(metric, {})
        for cutoff in sorted(set(base_by_cutoff) | set(variant_by_cutoff)):
            delta = variant_by_cutoff.get(cutoff, 0.0) - base_by_cutoff.get(cutoff, 0.0)
            if abs(delta) >= _UNCHANGED:
                out.setdefault(metric, {})[cutoff] = delta
    return out


RETRIEVAL_COMPARE_FILENAME = "retrieval_compare.json"
RETRIEVAL_COMPARE_ARTIFACT_VERSION = 1


def _side_dict(side: Side) -> dict[str, Any]:
    result = side.result
    return {
        "model": side.model_name,
        "code_version": side.code_version,
        "mode": result.mode,
        "status": result.status,
        "aggregate": result.aggregate,
        "policy_violations": [
            {"query_id": v.query_id, "kind": v.kind, "ids": list(v.ids)}
            for v in result.policy_violations
        ],
    }


def build_retrieval_compare_artifact(comparison: RetrievalComparison) -> dict[str, Any]:
    """The machine-readable comparison. Metrics, model names, each side's
    code_version, and query ids with the ids each side ranked -- never chunk
    text, prompt text, or anything from a profile."""
    return {
        "version": RETRIEVAL_COMPARE_ARTIFACT_VERSION,
        "project": comparison.project,
        "baseline": comparison.baseline,
        "models": list(comparison.models),
        "status": comparison.status(),
        "tests": [
            {
                "test": test.test_name,
                "golden_set": test.golden_set,
                "golden_set_hash": test.golden_set_hash,
                "granularity": test.granularity,
                "cutoffs": list(test.cutoffs),
                "baseline": _side_dict(test.baseline),
                "variants": [
                    {
                        **_side_dict(variant.side),
                        "deltas": variant.deltas,
                        "moved_queries": [
                            {
                                "query_id": movement.query_id,
                                "baseline_diagnosis": movement.baseline_diagnosis,
                                "diagnosis": movement.diagnosis,
                                "deltas": movement.deltas,
                                "baseline_ranked_ids": list(movement.baseline_ranked_ids),
                                "ranked_ids": list(movement.ranked_ids),
                            }
                            for movement in variant.moved
                        ],
                    }
                    for variant in test.variants
                ],
            }
            for test in comparison.tests
        ],
    }


def write_retrieval_compare_artifact(
    project_dir: Path, project: ProjectConfig, comparison: RetrievalComparison
) -> Path:
    target_dir = (project_dir / project.target_path).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / RETRIEVAL_COMPARE_FILENAME
    out.write_text(
        json.dumps(build_retrieval_compare_artifact(comparison), indent=2),
        encoding="utf-8",
    )
    return out


def format_comparison(comparison: RetrievalComparison) -> list[str]:
    """One table per test: a row per metric@k, a column per model, the
    variants' cells carrying their delta from the baseline. Then the moved
    queries per variant, each with the ranking on both sides."""
    lines: list[str] = []
    width = max(len(name) for name in comparison.models)
    cell = max(width, 18)
    for test in comparison.tests:
        queries = len(test.baseline.result.per_query)
        lines.append(
            f"{test.test_name}: golden_set={test.golden_set} ({queries} queries, "
            f"granularity={test.granularity}); baseline {test.baseline.model_name}"
        )
        header = f"{'metric':<14}" + "".join(f"{name:<{cell + 2}}" for name in comparison.models)
        lines.append(header.rstrip())
        lines.append("-" * len(header.rstrip()))
        for metric, by_cutoff in test.baseline.result.aggregate.items():
            for cutoff in test.cutoffs:
                row = f"{f'{metric}@{cutoff}':<14}{by_cutoff.get(cutoff, 0.0):<{cell + 2}.3f}"
                for variant in test.variants:
                    value = variant.side.result.aggregate.get(metric, {}).get(cutoff, 0.0)
                    delta = variant.deltas.get(metric, {}).get(cutoff, 0.0)
                    row += f"{f'{value:.3f} ({delta:+.3f})':<{cell + 2}}"
                lines.append(row.rstrip())
        lines.extend(_format_status_row(test, cell))
        for variant in test.variants:
            lines.extend(_format_movements(variant))
        lines.append("")
    return lines


def _format_status_row(test: TestComparison, cell: int) -> list[str]:
    row = f"{'status':<14}{test.baseline.result.status:<{cell + 2}}"
    for variant in test.variants:
        row += f"{variant.side.result.status:<{cell + 2}}"
    return [row.rstrip()]


def _format_movements(variant: VariantComparison) -> list[str]:
    name = variant.side.model_name
    if not variant.moved:
        return [f"  {name}: no query moved"]
    lines = [f"  {name}: {len(variant.moved)} query(ies) moved"]
    for movement in variant.moved:
        changes = ", ".join(
            f"{metric}@{cutoff} {delta:+.3f}"
            for metric, by_cutoff in movement.deltas.items()
            for cutoff, delta in by_cutoff.items()
        )
        lines.append(f"    {movement.query_id}: {changes}")
        lines.append(
            f"      ranked {_ids(movement.baseline_ranked_ids)} -> {_ids(movement.ranked_ids)}"
        )
    return lines


def _ids(ids: Sequence[str]) -> str:
    shown = list(ids[:5])
    suffix = f", +{len(ids) - 5}" if len(ids) > 5 else ""
    return "[" + ", ".join(shown) + suffix + "]"
