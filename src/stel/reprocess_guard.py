"""The reprocess guard: refuse to spend before the first model runs (issue #530).

A search index refuses a rebuild-required change by default. Until this
module, embed and llm models had no equivalent: a changed embedding identity,
or an upstream change that re-keys their input, reprocessed the corpus at
provider prices with only the budget cap in the way -- and that cap is sized
for runaway loops, not for "you just changed the embedding model".

The guard reads the plan (`stel plan`, issue #529) rather than inspecting
state inside each execution path, which is what lets it see the cascade: an
embed model whose own configuration is unchanged still reprocesses every row
when the chunk model above it re-keys. Every planned model is classified once,
before the first one runs, and any paid model under `on_code_change: fail`
that would reprocess more than its `reprocess_limit` stops the run.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .plan import ModelPlan

# The two statuses under which published rows are paid for again. `new` is a
# first build and `full` rebuilds every run by declaration; neither is a
# surprise the guard exists to catch.
_GUARDED_STATUSES = frozenset({"changed", "cascade"})


@dataclass(frozen=True)
class Refusal:
    plan: ModelPlan

    def describe(self) -> str:
        plan = self.plan
        rows = (
            f"up to {plan.rows_to_reprocess:,}"
            if plan.reprocess_is_upper_bound
            else f"{plan.rows_to_reprocess:,}"
        )
        provider = (
            f"{plan.provider}/{plan.provider_model}"
            if plan.provider and plan.provider_model
            else plan.kind
        )
        calls = (
            f"; about {plan.estimated_provider_calls:,} provider request(s)"
            if plan.estimated_provider_calls
            else ""
        )
        return (
            f"{plan.name} ({plan.kind}, {provider}): {rows} of {plan.state_rows:,} "
            f"published rows would reprocess{calls}; reprocess_limit is "
            f"{plan.reprocess_limit}. {plan.reason}."
        )


def guard_reprocess(plans: Sequence[ModelPlan]) -> list[Refusal]:
    """The planned models `on_code_change: fail` refuses to run, in plan order."""
    return [
        Refusal(plan)
        for plan in plans
        if plan.reprocess_policy == "fail"
        and plan.reprocess_limit is not None
        and plan.status in _GUARDED_STATUSES
        and plan.rows_to_reprocess > plan.reprocess_limit
    ]


def format_refusals(refusals: Sequence[Refusal]) -> str:
    """The error a refused run exits with: every refused model and every way
    forward, so the operator does not have to go looking for either."""
    count = len(refusals)
    lines = [
        f"Refusing to start: {count} model(s) would reprocess published rows at "
        "provider cost (on_code_change: fail)."
    ]
    lines.extend(f"  {refusal.describe()}" for refusal in refusals)
    lines.append(
        "Run `stel plan` for the whole picture. To proceed: `--accept-reprocess` "
        "(reprocess these rows incrementally), `--full-refresh` (rebuild), or set "
        "`on_code_change: reprocess` or a higher `reprocess_limit` on the model."
    )
    return "\n".join(lines)

