"""`stel plan`: what the next run would reprocess, before it spends anything (issue #529).

Every incremental decision stel makes compares a stored (input_fingerprint,
code_version) pair against a recomputed one. A configuration change moves
code_version, so the published state already records exactly which rows the
next run will refuse to skip -- one aggregate query per model says how many.
Input changes need source discovery or an upstream read, which this command
deliberately does not do: it connects to the warehouse, reads stel's own
state table, and touches no model table, source, or provider.

A change cascades. A re-keyed chunk model gives its embed child new ids, and
a re-run transform gives its child new input fingerprints, so a downstream
model whose own code_version is unchanged still reprocesses. The plan names
that as `cascade` with an upper bound (every published row), because the
exact count is only knowable by reading the upstream output the run will
produce.

Rows become dollars through history (issue #665): when the target keeps a
run log, each reprocessing model is priced at what its recent runs paid per
row that reached the provider. That is stel's own table too, read once.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

from .adapters import (
    AdapterError,
    ReadPredicate,
    ReadPredicateOperator,
    StateScope,
    WarehouseAdapter,
    create_adapter,
)
from .adapters.base import RETRIEVAL_PUBLISH_STAGE
from .compiler import validate_project_contract, validate_warehouse_capabilities
from .config.loader import load_project
from .config.model import ModelConfig
from .config.project import ProjectConfig
from .dag import ProjectDAG
from .embedding import EmbeddingIdentity, estimate_embed_requests
from .profile import ResolvedProfile, resolve_embedding_options, resolve_profile
from .providers import ProviderError
from .retrieval import StoreRole, create_store
from .versioning import (
    compute_model_code_version,
    describe_model_inference,
    describe_model_llm,
)

log = logging.getLogger(__name__)

PLAN_FILENAME = "plan.json"
# 2: `estimated_cost_usd` per model and `cost_source` in metadata (issue #665).
# 3: `confirm_cost_above_usd` in metadata, the target's cost guard (issue #586).
PLAN_SCHEMA_VERSION = 3

PlanStatus = Literal["new", "unchanged", "changed", "cascade", "full"]

# The batch an estimate is priced against. Providers split a batch by text
# count (Vertex issues one call per text for some models), never by content
# stel could know before reading the upstream, so a placeholder text is the
# honest input.
_PLACEHOLDER_TEXT = "x"

# Runs a model's cost per provider row is pooled over. Recent enough that a
# changed price or prompt shows up within a few runs, wide enough that one
# small run does not set the rate.
PRICED_RUNS = 5


class PlanError(RuntimeError):
    """A plan could not be produced from this project and profile."""


@dataclass(frozen=True)
class ModelPlan:
    name: str
    kind: str
    materialization: str
    status: PlanStatus
    code_version: str
    #: Rows the state table holds for this model's scope.
    state_rows: int
    #: Of those, rows whose recorded code_version differs from the current one.
    stale_rows: int
    #: Rows the next run would reprocess for reasons visible before it starts.
    rows_to_reprocess: int
    #: True when `rows_to_reprocess` is a ceiling (an upstream change re-keys
    #: or re-fingerprints this model's input), False when it is exact.
    reprocess_is_upper_bound: bool
    #: Planned upstream models whose own code_version changed and whose
    #: change reaches this one; the roots, not every model in between.
    caused_by: tuple[str, ...]
    #: Provider requests those rows imply; None for a kind that spends nothing
    #: or whose fan-out per row is not knowable ahead of the run.
    estimated_provider_calls: int | None
    provider: str | None
    provider_model: str | None
    reason: str
    #: USD those rows would cost if every one reached the provider (issue
    #: #665), priced at this model's recent cost per provider row in the run
    #: log; None without a run log, without priced history, or for a kind
    #: whose rows are not counted (`full`). An estimate, not a bound, in both
    #: directions: reuse, the response caches and in-window duplicates lower
    #: what a run pays, while longer text, a costlier prompt or model, or a
    #: price rise since those runs raise it (Codex review on #690).
    estimated_cost_usd: float | None
    #: The model's `on_code_change` policy and `reprocess_limit` (issue #530);
    #: None for a kind that carries no guard.
    reprocess_policy: str | None = None
    reprocess_limit: int | None = None
    #: Of `rows_to_reprocess`, the rows `estimated_cost_usd` was priced at:
    #: fewer for an embed model under `--reprocess-all`, whose flag-only rows
    #: reuse their vectors (issue #655). None when the model is unpriced.
    priced_rows: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "materialization": self.materialization,
            "status": self.status,
            "code_version": self.code_version,
            "state_rows": self.state_rows,
            "stale_rows": self.stale_rows,
            "rows_to_reprocess": self.rows_to_reprocess,
            "reprocess_is_upper_bound": self.reprocess_is_upper_bound,
            "caused_by": list(self.caused_by),
            "estimated_provider_calls": self.estimated_provider_calls,
            "provider": self.provider,
            "provider_model": self.provider_model,
            "estimated_cost_usd": self.estimated_cost_usd,
            "reason": self.reason,
            "reprocess_policy": self.reprocess_policy,
            "reprocess_limit": self.reprocess_limit,
            "priced_rows": self.priced_rows,
        }


@dataclass(frozen=True)
class ProjectPlan:
    models: tuple[ModelPlan, ...]
    target: dict[str, Any]
    generated_at: str
    #: The run-log relation costs were priced from, or None when the target
    #: has no run log enabled and no model could be priced.
    cost_source: str | None
    #: The target's cost-guard threshold (issue #586), or None when unset.
    confirm_cost_above_usd: float | None

    def counts(self) -> dict[str, int]:
        counts = {status: 0 for status in ("new", "unchanged", "changed", "cascade", "full")}
        for model in self.models:
            counts[model.status] += 1
        counts["total"] = len(self.models)
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": {
                "schema_version": PLAN_SCHEMA_VERSION,
                "invocation": "plan",
                "generated_at": self.generated_at,
                "target": self.target,
                "counts": self.counts(),
                "cost_source": self.cost_source,
                "confirm_cost_above_usd": self.confirm_cost_above_usd,
            },
            "models": [model.to_dict() for model in self.models],
        }


def plan_project(
    project_dir: Path,
    *,
    select: str | None = None,
    exclude: str | None = None,
    target: str | None = None,
    profiles_dir: Path | None = None,
    reprocess_all: bool = False,
) -> ProjectPlan:
    """Classify every selected model against its published state.

    Same preflight as `run`: the project contract and warehouse capabilities
    are validated before the warehouse is opened, so a bad configuration
    fails here with the same message it would fail a run with."""
    project, sources, models = load_project(project_dir)
    dag = validate_project_contract(project, sources, models, project_dir)
    resolved = resolve_profile(
        project, project_dir, target=target, profiles_dir=profiles_dir
    )
    adapter = create_adapter(resolved.warehouse, project_dir=project_dir)
    selected = dag.select_models(select=select, exclude=exclude)
    models_by_name = {model.name: model for model in models}
    planned = [models_by_name[name] for name in selected]
    validate_warehouse_capabilities(planned, adapter)
    with adapter:
        plans = plan_models(
            planned,
            dag=dag,
            project=project,
            project_dir=project_dir,
            adapter=adapter,
            resolved=resolved,
            reprocess_all=reprocess_all,
        )
        # Priced here rather than in `plan_models`: the runner's reprocess
        # guard reads that function before every run, and a cost it never
        # shows is not worth a warehouse read per invocation.
        cost_source = run_log_relation(resolved)
        if cost_source is not None:
            try:
                plans = price_plans(
                    plans,
                    adapter=adapter,
                    relation=cost_source,
                    reprocess_all=reprocess_all,
                )
            except AdapterError as error:
                # The run log is best-effort when written, so reading it is
                # too: a plan that cannot price is still a plan. The cost
                # guard does not get this leniency (Codex review on #695).
                log.warning(
                    "Could not read run log '%s' to price this plan [%s]",
                    cost_source,
                    type(error).__name__,
                )
    warehouse = resolved.warehouse
    return ProjectPlan(
        models=tuple(plans),
        cost_source=cost_source,
        confirm_cost_above_usd=resolved.confirm_cost_above_usd,
        target={
            "profile": resolved.profile_name,
            "name": resolved.target_name,
            "adapter_type": warehouse.type,
            "schema": warehouse.schema_name,
            "catalog": warehouse.catalog_name(),
            "location": warehouse.storage_location(),
        },
        generated_at=datetime.now(UTC).isoformat(),
    )


def write_plan_artifact(project_dir: Path, plan: ProjectPlan) -> Path:
    project, _, _ = load_project(project_dir)
    target_dir = (project_dir / project.target_path).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / PLAN_FILENAME
    out.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    return out


def plan_models(
    planned: list[ModelConfig],
    *,
    dag: ProjectDAG,
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
    reprocess_all: bool = False,
) -> list[ModelPlan]:
    """Plan `planned` (in execution order) against an open adapter.

    The runner calls this before the first model runs so the reprocess guard
    (issue #530) reads the same numbers `stel plan` prints."""
    # Planned models whose own code_version moved. A change reaches every
    # descendant, through cascade and full models alike, and each of those
    # descendants has the same root in its ancestry -- so naming the roots is
    # both the complete propagation rule and the answer a reader wants.
    roots: set[str] = set()
    plans: list[ModelPlan] = []
    for model in planned:
        code_version = compute_model_code_version(
            model, project, project_dir, resolved=resolved
        )
        counts = adapter.state_code_version_counts(
            _state_scope(model, project=project, resolved=resolved)
        )
        caused_by = tuple(sorted(dag.ancestors(model.name) & roots))
        plan = _classify(
            model,
            code_version=code_version,
            counts=counts,
            caused_by=caused_by,
            resolved=resolved,
            project=project,
            project_dir=project_dir,
            reprocess_all=reprocess_all,
        )
        if plan.status == "changed":
            roots.add(model.name)
        plans.append(plan)
    return plans


def _state_scope(
    model: ModelConfig, *, project: ProjectConfig, resolved: ResolvedProfile
) -> StateScope:
    """The scope the model's execution path reads its state from.

    Every warehouse-materialized kind keys state on the model name. A search
    publish keys it on the logical collection's serving descriptor, which the
    store computes from configuration alone -- no store I/O happens here."""
    search = model.search
    if search is None:
        return StateScope(model.name)
    if resolved.retrieval is None:
        raise PlanError(
            f"Search index '{model.name}' requires a configured retrieval target"
        )
    alias = search.store or resolved.retrieval.default
    store_config = resolved.retrieval.stores.get(alias)
    if store_config is None:
        raise PlanError(
            f"Search index '{model.name}' selects unknown retrieval target '{alias}'"
        )
    store = create_store(
        store_config,
        project_name=project.name,
        target_name=resolved.target_name,
        alias=alias,
        role=StoreRole.PUBLISH,
    )
    logical_collection = search.collection or model.name
    return StateScope.for_target_descriptor(
        model.name,
        stage=RETRIEVAL_PUBLISH_STAGE,
        descriptor=store.state_descriptor(logical_collection).descriptor(),
    )


def _classify(
    model: ModelConfig,
    *,
    code_version: str,
    counts: dict[str, int],
    caused_by: tuple[str, ...],
    resolved: ResolvedProfile,
    project: ProjectConfig,
    project_dir: Path,
    reprocess_all: bool,
) -> ModelPlan:
    state_rows = sum(counts.values())
    stale_rows = sum(
        count for version, count in counts.items() if version != code_version
    )
    upstream = ", ".join(caused_by)
    if model.search is None and model.materialization == "full":
        status: PlanStatus = "full"
        rows, upper_bound = 0, False
        reason = "materialization: full rebuilds from its input every run"
        if caused_by:
            reason += f"; upstream {upstream} changed, so its output changes too"
    elif state_rows == 0:
        status = "new"
        rows, upper_bound = 0, False
        reason = "no published state: the first run processes every input"
    elif reprocess_all and model.search is None:
        # A `search:` model is excluded, as it is from the unchanged-parent
        # skip: the runner does not pass the flag to `run_search_model`,
        # whose own choice is rebuild-vs-extend rather than a per-record
        # skip, and `--full-refresh` is the documented way to rebuild an
        # index. Classifying its rows as reprocessing here would promise a
        # run that cannot happen (PR #660 review).
        #
        # Not an upper bound: every published row reprocesses, which is the
        # whole request (issue #655). Reported as `changed` so the reprocess
        # guard -- reading these same plans -- sees the real number for a
        # paid model the flag does not release.
        status = "changed"
        rows, upper_bound = state_rows, False
        reason = (
            "--reprocess-all: every published row reprocesses, whatever its "
            "code_version"
        )
        if caused_by:
            reason += f"; upstream {upstream} changed as well"
    elif stale_rows > 0:
        status = "changed"
        rows, upper_bound = (state_rows, True) if caused_by else (stale_rows, False)
        reason = f"code_version differs for {stale_rows} of {state_rows} published rows"
        if caused_by:
            reason += f"; upstream {upstream} changed as well"
    elif caused_by:
        status = "cascade"
        rows, upper_bound = state_rows, True
        reason = (
            f"upstream {upstream} changed: this model's input re-keys or "
            f"re-fingerprints, up to every published row"
        )
    else:
        status = "unchanged"
        rows, upper_bound = 0, False
        reason = "code_version matches every published row; only input changes run"
    calls, provider, provider_model = _estimate_provider_calls(
        model, rows, resolved=resolved, project=project, project_dir=project_dir
    )
    policy = model.embed if model.embed is not None else model.llm
    if status == "full":
        # A full model pays for every input every run, and the plan does not
        # count inputs; reporting 0 would read as "free".
        calls = None
    return ModelPlan(
        name=model.name,
        kind=model.kind_label(),
        materialization=model.materialization,
        status=status,
        code_version=code_version,
        state_rows=state_rows,
        stale_rows=stale_rows,
        rows_to_reprocess=rows,
        reprocess_is_upper_bound=upper_bound,
        caused_by=caused_by,
        estimated_provider_calls=calls,
        provider=provider,
        provider_model=provider_model,
        reason=reason,
        estimated_cost_usd=None,
        reprocess_policy=policy.on_code_change if policy is not None else None,
        reprocess_limit=policy.reprocess_limit if policy is not None else None,
    )


def _estimate_provider_calls(
    model: ModelConfig,
    rows: int,
    *,
    resolved: ResolvedProfile,
    project: ProjectConfig,
    project_dir: Path,
) -> tuple[int | None, str | None, str | None]:
    """Requests the paid kinds would issue for `rows`.

    embed: the provider's own batch split, priced against a placeholder
    batch. llm map and `backend: llm` extraction: one request per row or
    document (batch submission bills differently and is not modeled). A
    transform that calls the LLM helper fans out per parent by its own
    code, so only its provider identity is reported."""
    if model.embed is not None:
        options = resolve_embedding_options(model.embed.provider, resolved)
        identity = EmbeddingIdentity.from_config(
            model.embed, profile_options=options.provider_options
        )
        return (
            _embed_calls(
                rows,
                batch_size=model.embed.batch_size,
                identity=identity,
                profile_options=options.provider_options,
            ),
            identity.provider,
            identity.model,
        )
    if model.llm is not None:
        descriptor = describe_model_llm(
            model, resolved=resolved, project_dir=project_dir
        )
        return (rows, *_descriptor_identity(descriptor))
    descriptor = describe_model_inference(model, project, resolved=resolved)
    if descriptor is None:
        return None, None, None
    if model.extraction is not None:
        return (rows, *_descriptor_identity(descriptor))
    return (None, *_descriptor_identity(descriptor))


def run_log_relation(resolved: ResolvedProfile) -> str | None:
    run_log = resolved.run_log
    if run_log is None or not run_log.enabled:
        return None
    return run_log.relation


def price_plans(
    plans: list[ModelPlan],
    *,
    adapter: WarehouseAdapter,
    relation: str,
    reprocess_all: bool,
) -> list[ModelPlan]:
    """Price each reprocessing model at its recent cost per provider row.

    A plan cannot know what a row costs before it is read -- an embed bills
    by the characters in it, an llm call by its tokens -- so the honest unit
    is history: what this model's last priced runs paid per row that reached
    the provider. That is the run log's `estimated_cost_usd` (the arithmetic
    `pricing:` already does after the fact) over `rows_processed -
    cache_hits`, and it uses the rates those runs were priced at.

    Rows only, never calls: a request's cost depends on how many texts it
    carries, which is the provider's batch split rather than anything the
    log records.

    Raises `AdapterError` when the run log cannot be read. Each caller
    decides what that means: `stel plan` still reports, but the cost guard
    must not read an unreadable history as a free run.
    """
    wanted = tuple(
        plan.name
        for plan in plans
        if plan.rows_to_reprocess > 0 and plan.provider is not None
    )
    if not wanted:
        return plans
    rates = _cost_per_provider_row(adapter, relation=relation, models=wanted)
    code_changed = {plan.name for plan in plans if plan.stale_rows > 0}
    return [
        _priced(
            plan,
            rows=_provider_rows(
                plan, reprocess_all=reprocess_all, code_changed=code_changed
            ),
            rate=rates[plan.name],
        )
        if plan.name in rates
        else plan
        for plan in plans
    ]


def _priced(plan: ModelPlan, *, rows: int, rate: float) -> ModelPlan:
    return replace(
        plan, priced_rows=rows, estimated_cost_usd=round(rows * rate, 6)
    )


def _provider_rows(
    plan: ModelPlan, *, reprocess_all: bool, code_changed: set[str]
) -> int:
    """How many of the rows a plan reprocesses would reach the provider.

    All of them, except for an embed model under `--reprocess-all` (issue
    #655): it reads its own vectors back by input hash, so the rows the flag
    alone puts back through the stage cost nothing. Only rows whose
    code_version moved are paid for then, or every row when an upstream
    model's code changed and may have changed their text. Pricing the flag's
    rows anyway would make the cost guard ask before the one recovery run
    that is free (#586).

    `caused_by` alone cannot say which: under the flag every upstream model
    with state plans as `changed`, so it names those too. `code_changed`
    -- the planned models with a stale row -- is the upstream change that
    can move text."""
    if (
        reprocess_all
        and plan.kind == "embed"
        and code_changed.isdisjoint(plan.caused_by)
    ):
        return plan.stale_rows
    return plan.rows_to_reprocess


_RUN_LOG_COST_COLUMNS = (
    "model_name",
    "rows_processed",
    "cache_hits",
    "estimated_cost_usd",
    "completed_at",
)


def _cost_per_provider_row(
    adapter: WarehouseAdapter, *, relation: str, models: tuple[str, ...]
) -> dict[str, float]:
    """USD per provider row over each model's last priced runs.

    A run counts when it was priced and something reached the provider: a
    run with no `pricing:` has no cost, and a run answered wholly by reuse
    has no denominator. Pooled (total cost over total rows) rather than a
    mean of per-run rates, so a 10-row run cannot outvote a 3.6M-row one.
    """
    if relation not in adapter.list_all_tables():
        return {}
    present = set(adapter.read_table_schema(relation).columns)
    if not set(_RUN_LOG_COST_COLUMNS).issubset(present):
        # A log from before a column existed is history the plan cannot read,
        # not a reason to fail one.
        return {}
    runs: dict[str, list[tuple[str, float, int]]] = {}
    with adapter.table_snapshot(
        relation,
        columns=list(_RUN_LOG_COST_COLUMNS),
        predicate=ReadPredicate("model_name", ReadPredicateOperator.IN, models),
    ) as snapshot:
        for batch in snapshot:
            frame = pl.from_arrow(batch)
            assert isinstance(frame, pl.DataFrame)
            for row in frame.iter_rows(named=True):
                priced = _priced_run(row)
                if priced is not None:
                    runs.setdefault(row["model_name"], []).append(priced)
    rates: dict[str, float] = {}
    for name, history in runs.items():
        recent = sorted(history, reverse=True)[:PRICED_RUNS]
        rates[name] = sum(cost for _, cost, _ in recent) / sum(
            rows for _, _, rows in recent
        )
    return rates


def _priced_run(row: dict[str, Any]) -> tuple[str, float, int] | None:
    cost = row["estimated_cost_usd"]
    processed = row["rows_processed"] or 0
    provider_rows = processed - (row["cache_hits"] or 0)
    if cost is None or cost <= 0 or provider_rows <= 0:
        return None
    return (row["completed_at"] or "", float(cost), provider_rows)


def _descriptor_identity(
    descriptor: dict[str, str] | None,
) -> tuple[str | None, str | None]:
    if descriptor is None:
        return None, None
    return descriptor.get("provider"), descriptor.get("model")


def _embed_calls(
    rows: int,
    *,
    batch_size: int,
    identity: EmbeddingIdentity,
    profile_options: Any,
) -> int | None:
    if rows == 0:
        return 0
    full_batches, remainder = divmod(rows, batch_size)
    try:
        per_full_batch = (
            estimate_embed_requests(
                [_PLACEHOLDER_TEXT] * batch_size,
                identity,
                profile_options=profile_options,
            )
            if full_batches
            else 0
        )
        per_remainder = (
            estimate_embed_requests(
                [_PLACEHOLDER_TEXT] * remainder,
                identity,
                profile_options=profile_options,
            )
            if remainder
            else 0
        )
    except ProviderError:
        # The provider plugin is absent or unconfigured on the planning host.
        # The run would fail on the same condition; the plan reports the rows
        # and leaves the request count unknown rather than failing to plan.
        return None
    return full_batches * per_full_batch + per_remainder


def format_plan_table(plan: ProjectPlan) -> list[str]:
    """The terminal rendering: one row per model, a reason line under any
    model that is not unchanged, and a one-line summary."""
    name_width = max([len("model"), *(len(model.name) for model in plan.models)]) + 2
    header = (
        f"{'model':<{name_width}}{'kind':<12}{'mater.':<13}{'status':<11}"
        f"{'state_rows':>11}{'reprocess':>11}{'est_calls':>11}{'est_cost':>12}"
    )
    lines = [header, "-" * len(header)]
    for model in plan.models:
        reprocess = (
            f"<={model.rows_to_reprocess}"
            if model.reprocess_is_upper_bound
            else str(model.rows_to_reprocess)
        )
        calls = (
            "-"
            if model.estimated_provider_calls is None
            else (
                f"<={model.estimated_provider_calls}"
                if model.reprocess_is_upper_bound
                else str(model.estimated_provider_calls)
            )
        )
        cost = (
            "-"
            if model.estimated_cost_usd is None
            else f"~{format_usd(model.estimated_cost_usd)}"
        )
        lines.append(
            f"{model.name:<{name_width}}{model.kind:<12}{model.materialization:<13}"
            f"{model.status:<11}{model.state_rows:>11}{reprocess:>11}{calls:>11}"
            f"{cost:>12}"
        )
        if model.status != "unchanged":
            lines.append(f"  {model.reason}")
    lines.append("")
    lines.append(_summary_line(plan))
    return lines


def _summary_line(plan: ProjectPlan) -> str:
    counts = plan.counts()
    reprocessing = counts["changed"] + counts["cascade"]
    parts = [
        f"{counts['total']} model(s) planned: "
        f"{counts['unchanged']} unchanged, {counts['changed']} changed, "
        f"{counts['cascade']} downstream of a change, {counts['new']} new, "
        f"{counts['full']} rebuilt every run."
    ]
    if reprocessing:
        exact = sum(
            m.rows_to_reprocess
            for m in plan.models
            if m.status == "changed" and not m.reprocess_is_upper_bound
        )
        bounded = sum(
            m.rows_to_reprocess for m in plan.models if m.reprocess_is_upper_bound
        )
        parts.append(
            f"Rows to reprocess: {exact} from code changes, "
            f"up to {bounded} more downstream."
        )
        if cost := _cost_sentence(plan):
            parts.append(cost)
    parts.append(
        "No source was discovered and no provider was called; "
        "changed inputs are found by the run itself."
    )
    return " ".join(parts)


def _cost_sentence(plan: ProjectPlan) -> str:
    priced = [m for m in plan.models if m.estimated_cost_usd is not None]
    # A paid model with rows to reprocess and no price is the case the reader
    # most needs told about: a blank would read as free.
    unpriced = [
        m.name
        for m in plan.models
        if m.estimated_cost_usd is None
        and m.provider is not None
        and m.rows_to_reprocess > 0
    ]
    if plan.cost_source is None:
        if not unpriced:
            return ""
        return (
            "Provider cost is not estimated: a plan prices rows from the run "
            "log's history, and this target has no run-log enabled."
        )
    sentence = ""
    if priced:
        total = sum(m.estimated_cost_usd or 0.0 for m in priced)
        sentence = (
            f"Provider cost: about {format_usd(total)}, priced at each "
            f"model's cost per provider row over its last {PRICED_RUNS} "
            f"priced runs in {plan.cost_source}. An estimate, not a cap: reuse "
            "and caches lower it, longer text or a costlier model raise it."
        )
    if unpriced:
        sentence += (
            f"{' ' if sentence else ''}No priced history in {plan.cost_source} "
            f"for {', '.join(unpriced)}."
        )
    return sentence


def format_usd(value: float) -> str:
    if value == 0:
        return "$0"
    if value < 0.01:
        return "$<0.01"
    return f"${value:,.2f}"

