"""Golden-set retrieval evaluation orchestration (issue #137).

Reads a golden-set model's rows, executes each query through the same
`search()` API a real caller would use, scores results with
`retrieval_metrics`, applies configured thresholds, and asserts policy hard
failures (required/excluded IDs) independently of ranking-metric averaging —
a query that leaks an excluded ID must fail even if its average recall is
excellent.

See docs/architecture/semantic-retrieval.md ("Evaluation (#137)") for the
golden-set row contract and design decisions this module implements.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters import WarehouseAdapter, create_adapter
from .compiler import validate_project_contract
from .config import load_project
from .config.model import ModelConfig, RetrievalTestConfig
from .config.project import ProjectConfig
from .dag import parse_ref
from .embedding import resolve_search_embedding_identity
from .hashing import canonical_fingerprint
from .profile import ResolvedProfile, resolve_profile
from .retrieval_metrics import QueryMetrics, aggregate_metrics, evaluate_query
from .search import (
    SearchError,
    SearchFilter,
    SearchFilterOperator,
    SearchMode,
    SearchRequest,
    search,
)


class RetrievalEvalError(Exception):
    """Raised when an evaluation cannot run at all (bad golden-set row, search
    setup failure) — distinct from a query merely scoring poorly."""


@dataclass(frozen=True)
class PolicyViolation:
    query_id: str
    kind: str  # "missing_required" | "unexpected_excluded"
    ids: tuple[str, ...]


@dataclass(frozen=True)
class ThresholdStatus:
    key: str  # "<metric>_at_<k>"
    min: float
    actual: float
    severity: str
    status: str  # "pass" | "warn" | "fail"


@dataclass
class RetrievalTestResult:
    model_name: str
    test_name: str
    golden_set: str
    mode: str
    per_query: list[QueryMetrics] = field(default_factory=list)
    aggregate: dict[str, dict[int, float]] = field(default_factory=dict)
    thresholds: list[ThresholdStatus] = field(default_factory=list)
    policy_violations: list[PolicyViolation] = field(default_factory=list)
    duration_seconds: float = 0.0
    embedding_identity: dict[str, Any] | None = None
    store_provenance: dict[str, Any] | None = None
    golden_set_hash: str = ""

    @property
    def status(self) -> str:
        """Worst outcome across thresholds, with any policy violation forcing
        `fail` regardless of threshold severities (issue #137: "security
        exclusions should be hard failures, not averaged away")."""
        if self.policy_violations:
            return "fail"
        statuses = {t.status for t in self.thresholds}
        if "fail" in statuses:
            return "fail"
        if "warn" in statuses:
            return "warn"
        return "pass"


def run_retrieval_evaluation(
    project_dir: Path,
    *,
    select: str | None = None,
    exclude: str | None = None,
    target: str | None = None,
    profiles_dir: Path | None = None,
) -> list[RetrievalTestResult]:
    project, sources, models = load_project(project_dir)
    dag = validate_project_contract(project, sources, models, project_dir)
    resolved = resolve_profile(project, project_dir, target=target, profiles_dir=profiles_dir)
    models_by_name = {m.name: m for m in models}

    selected = set(dag.select_models(select=select, exclude=exclude))
    targets = [
        (model, test)
        for model in models
        if model.name in selected and model.search is not None
        for test in model.retrieval_tests
    ]

    results: list[RetrievalTestResult] = []
    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        for model, test in targets:
            results.append(
                _run_one(
                    project_dir,
                    model,
                    test,
                    models_by_name,
                    adapter,
                    resolved,
                    target=target,
                    profiles_dir=profiles_dir,
                )
            )
    return results


def _run_one(
    project_dir: Path,
    model: ModelConfig,
    test: RetrievalTestConfig,
    models_by_name: dict[str, ModelConfig],
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
    *,
    target: str | None,
    profiles_dir: Path | None,
) -> RetrievalTestResult:
    assert model.search is not None
    start = time.monotonic()
    golden_name = parse_ref(test.golden_set)
    golden_rows = adapter.read_table(golden_name).to_dicts()
    if not golden_rows:
        raise RetrievalEvalError(
            f"Retrieval test '{test.name}' golden_set '{golden_name}' has no rows"
        )

    mode = SearchMode(test.mode) if test.mode else _default_mode(model.search.query.modes)
    limit = max(test.at)

    per_query: list[QueryMetrics] = []
    violations: list[PolicyViolation] = []
    store_provenance: dict[str, Any] | None = None
    for row in golden_rows:
        query_id = _require_str(row, "query_id", golden_name)

        def _ctx(field_name: str, _query_id: str = query_id) -> str:
            return _field_context(golden_name, _query_id, field_name)

        request = SearchRequest(
            model=model.name,
            query=row.get("query_text") or None,
            vector=_parse_vector(row.get("query_vector"), context=_ctx("query_vector")),
            mode=SearchMode(row["mode"]) if row.get("mode") else mode,
            limit=limit,
            filters=_parse_filters(row.get("filters"), context=_ctx("filters")),
        )
        policy_filters = _parse_filters(row.get("policy_filters"), context=_ctx("policy_filters"))
        try:
            hits = search(
                project_dir,
                request,
                target=target,
                profiles_dir=profiles_dir,
                policy_filters=policy_filters,
            )
        except SearchError as e:
            raise RetrievalEvalError(
                f"Retrieval test '{test.name}' query '{query_id}' failed: {e}"
            ) from e
        ranked_ids = [hit.record_id for hit in hits]
        if store_provenance is None and hits:
            # Safe store identity (type, target, logical/physical collection) —
            # the same provenance a real caller sees, captured once per test
            # rather than re-derived; never includes credentials.
            store_provenance = hits[0].provenance.to_dict()

        required_ids = set(_parse_ids(row.get("required_ids"), context=_ctx("required_ids")))
        missing_required = required_ids - set(ranked_ids)
        if missing_required:
            violations.append(
                PolicyViolation(query_id, "missing_required", tuple(sorted(missing_required)))
            )
        excluded_ids = set(_parse_ids(row.get("excluded_ids"), context=_ctx("excluded_ids")))
        found_excluded = excluded_ids & set(ranked_ids)
        if found_excluded:
            violations.append(
                PolicyViolation(query_id, "unexpected_excluded", tuple(sorted(found_excluded)))
            )

        relevant_ids = set(_parse_ids(row.get("relevant_ids"), context=_ctx("relevant_ids")))
        graded = _parse_graded_relevance(
            row.get("graded_relevance"), context=_ctx("graded_relevance")
        )
        per_query.append(
            evaluate_query(
                query_id,
                ranked_ids,
                relevant_ids=relevant_ids,
                graded_relevance=graded,
                cutoffs=test.at,
            )
        )

    aggregate = aggregate_metrics(per_query, cutoffs=test.at)
    thresholds = _evaluate_thresholds(test, aggregate)
    embedding_identity = None
    if model.search.vector is not None and model.search.vector.embedding == "inherit":
        try:
            identity = resolve_search_embedding_identity(model, models_by_name)
        except Exception:
            identity = None  # identity is best-effort artifact metadata
        if identity is not None:
            embedding_identity = {
                "provider": identity.provider,
                "model": identity.model,
                "dimensions": identity.dimensions,
                "implementation": identity.implementation,
            }

    return RetrievalTestResult(
        model_name=model.name,
        test_name=test.name,
        golden_set=golden_name,
        mode=mode.value,
        per_query=per_query,
        aggregate=aggregate,
        thresholds=thresholds,
        policy_violations=violations,
        duration_seconds=round(time.monotonic() - start, 3),
        embedding_identity=embedding_identity,
        store_provenance=store_provenance,
        golden_set_hash=canonical_fingerprint(
            golden_rows, domain="dbt_ml/retrieval_eval/golden_set"
        ),
    )


def _default_mode(declared_modes: frozenset[str]) -> SearchMode:
    """`declared_modes` may also contain `"filter"`, which is not a retrieval
    mode; prefer the richest actual query mode the index supports."""
    for candidate in (SearchMode.HYBRID, SearchMode.VECTOR, SearchMode.TEXT):
        if candidate.value in declared_modes:
            return candidate
    raise RetrievalEvalError(
        "Search model declares no vector/text/hybrid query mode to evaluate"
    )


def _evaluate_thresholds(
    test: RetrievalTestConfig, aggregate: dict[str, dict[int, float]]
) -> list[ThresholdStatus]:
    statuses: list[ThresholdStatus] = []
    for key, threshold in test.thresholds.items():
        metric, _, k_text = key.rpartition("_at_")
        k = int(k_text)
        actual = aggregate.get(metric, {}).get(k, 0.0)
        if actual >= threshold.min:
            status = "pass"
        else:
            status = "warn" if threshold.severity == "warn" else "fail"
        statuses.append(
            ThresholdStatus(
                key=key,
                min=threshold.min,
                actual=actual,
                severity=threshold.severity,
                status=status,
            )
        )
    return statuses


def _require_str(row: dict[str, Any], field_name: str, golden_name: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value:
        raise RetrievalEvalError(
            f"Golden-set '{golden_name}' row is missing a non-empty '{field_name}'"
        )
    return value


def _field_context(golden_name: str, query_id: str, field_name: str) -> str:
    return f"Golden-set '{golden_name}' row '{query_id}' field '{field_name}'"


def _parse_json(value: Any, *, context: str) -> Any:
    """Parse a golden-set JSON-typed column. Malformed content is a data
    problem in the golden set, not an internal error — surface it as a
    RetrievalEvalError naming the golden set, row, and field, not a bare
    traceback."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise RetrievalEvalError(f"{context} is not valid JSON: {e}") from e
    return value  # already a native list/dict from the adapter


def _parse_ids(value: Any, *, context: str) -> list[str]:
    parsed = _parse_json(value, context=context)
    if not parsed:
        return []
    try:
        return [str(item) for item in parsed]
    except TypeError as e:
        raise RetrievalEvalError(f"{context} must be a JSON array") from e


def _parse_vector(value: Any, *, context: str) -> tuple[float, ...] | None:
    parsed = _parse_json(value, context=context)
    if not parsed:
        return None
    try:
        return tuple(float(item) for item in parsed)
    except (TypeError, ValueError) as e:
        raise RetrievalEvalError(f"{context} must be a JSON array of numbers") from e


def _parse_graded_relevance(value: Any, *, context: str) -> dict[str, float] | None:
    parsed = _parse_json(value, context=context)
    if not parsed:
        return None
    try:
        return {str(k): float(v) for k, v in parsed.items()}
    except (TypeError, ValueError, AttributeError) as e:
        raise RetrievalEvalError(
            f"{context} must be a JSON object mapping id to a numeric grade"
        ) from e


def _parse_filters(value: Any, *, context: str) -> tuple[SearchFilter, ...]:
    parsed = _parse_json(value, context=context)
    if not parsed:
        return ()
    try:
        return tuple(
            SearchFilter(
                field=item["field"],
                operator=SearchFilterOperator(item["operator"]),
                value=(
                    tuple(item["value"])
                    if isinstance(item["value"], list)
                    else item["value"]
                ),
            )
            for item in parsed
        )
    except (TypeError, KeyError, ValueError) as e:
        raise RetrievalEvalError(
            f"{context} must be a JSON array of {{field, operator, value}} objects: {e}"
        ) from e


RETRIEVAL_EVAL_FILENAME = "retrieval_eval.json"
RETRIEVAL_EVAL_ARTIFACT_VERSION = 1


def build_retrieval_eval_artifact(
    project: ProjectConfig, results: list[RetrievalTestResult]
) -> dict[str, Any]:
    """A machine-readable record of one `dbt-ml eval` run — safe for CI
    comparison and docs rendering. Never includes secrets, credential-bearing
    profile values, or raw embedding vectors."""
    return {
        "version": RETRIEVAL_EVAL_ARTIFACT_VERSION,
        "project": project.name,
        "results": [
            {
                "model": r.model_name,
                "test": r.test_name,
                "golden_set": r.golden_set,
                "golden_set_hash": r.golden_set_hash,
                "mode": r.mode,
                "store": r.store_provenance,
                "embedding": r.embedding_identity,
                "status": r.status,
                "duration_seconds": r.duration_seconds,
                "thresholds": [
                    {
                        "key": t.key,
                        "min": t.min,
                        "actual": t.actual,
                        "severity": t.severity,
                        "status": t.status,
                    }
                    for t in r.thresholds
                ],
                "policy_violations": [
                    {"query_id": v.query_id, "kind": v.kind, "ids": list(v.ids)}
                    for v in r.policy_violations
                ],
                "aggregate": r.aggregate,
                "queries": [
                    {
                        "query_id": q.query_id,
                        "diagnosis": q.diagnosis.value,
                        "ranked_ids": list(q.ranked_ids),
                        "missing_ids": list(q.missing_ids),
                        "metrics": q.values,
                    }
                    for q in r.per_query
                ],
            }
            for r in results
        ],
    }


def write_retrieval_eval_artifact(
    project_dir: Path, project: ProjectConfig, results: list[RetrievalTestResult]
) -> Path:
    target_dir = (project_dir / project.target_path).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / RETRIEVAL_EVAL_FILENAME
    payload = build_retrieval_eval_artifact(project, results)
    out.write_text(json.dumps(payload, indent=2))
    return out
