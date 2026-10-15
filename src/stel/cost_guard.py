"""The cost guard: ask before a run whose projected cost is over a threshold (issue #586).

The reprocess guard (ADR-0008) counts rows, and a row is not a price: a
thousand llm rows can cost more than a million embed rows, and a model under
`on_code_change: reprocess`, or a run passed `--accept-reprocess` or
`--full-refresh`, is past that guard entirely. A target that sets
`confirm_cost_above_usd` has the plan priced from run-log history (ADR-0033)
before the first model runs. Over the threshold, an interactive run asks and
any other run refuses, unless `--accept-cost` said yes in advance.

The projection covers what a plan can see without reading a source: published
rows that would reprocess. New input, a first build and a `full` model's
rebuild are not in it, and a model with no priced history is named rather than
counted. This is a question asked before starting, not a cap; the run budget's
`max_cost_usd` is the cap, and it sees everything.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from .plan import PRICED_RUNS, ModelPlan, format_usd


@dataclass(frozen=True)
class CostProjection:
    threshold_usd: float
    total_usd: float
    #: Models with rows to reprocess and a price, in plan order.
    priced: tuple[ModelPlan, ...]
    #: Paid models with rows to reprocess and no priced history.
    unpriced: tuple[str, ...]
    #: The run-log relation the prices came from.
    source: str

    @property
    def exceeds_threshold(self) -> bool:
        return self.total_usd > self.threshold_usd

    def describe(self) -> list[str]:
        """The lines a prompt or a refusal shows: the total against the
        threshold, then each priced model and anything left out."""
        lines = [
            f"Projected provider cost: about {format_usd(self.total_usd)}, over "
            f"confirm_cost_above_usd ({format_usd(self.threshold_usd)})."
            if self.exceeds_threshold
            else f"Projected provider cost: about {format_usd(self.total_usd)}, within "
            f"confirm_cost_above_usd ({format_usd(self.threshold_usd)})."
        ]
        for plan in self.priced:
            rows = (
                f"up to {plan.rows_to_reprocess:,}"
                if plan.reprocess_is_upper_bound
                else f"{plan.rows_to_reprocess:,}"
            )
            lines.append(
                f"  {plan.name} ({plan.kind}): ~{format_usd(plan.estimated_cost_usd or 0.0)}"
                f" for {rows} row(s). {plan.reason}."
            )
        if self.unpriced:
            lines.append(
                f"  Not included, no priced history in {self.source}: "
                f"{', '.join(self.unpriced)}."
            )
        lines.append(
            f"Priced at each model's cost per provider row over its last "
            f"{PRICED_RUNS} priced runs in {self.source}; an estimate, not a cap. "
            "New input since the last run is not in it."
        )
        return lines


#: Asked when a projection is over its threshold; True proceeds. The CLI
#: supplies one only when a person is at the terminal to answer it.
CostConfirmation = Callable[[CostProjection], bool]


def full_refresh_rows(plans: Sequence[ModelPlan]) -> list[ModelPlan]:
    """Plans as a `--full-refresh` run would spend them: every published row
    of a paid model reaches the provider again.

    The plan itself has no full-refresh status, because the reprocess guard
    never plans one (the flag is the operator saying yes). The cost guard
    still has to price it -- it is the most expensive run there is."""
    return [
        replace(
            plan,
            rows_to_reprocess=plan.state_rows,
            reprocess_is_upper_bound=False,
            reason="--full-refresh reprocesses every published row",
        )
        if plan.provider is not None and plan.status != "full" and plan.state_rows > 0
        else plan
        for plan in plans
    ]


def project_cost(
    plans: Sequence[ModelPlan], *, threshold_usd: float, source: str
) -> CostProjection:
    """Sum the priced plans against `threshold_usd`."""
    priced = tuple(
        plan
        for plan in plans
        if plan.rows_to_reprocess > 0 and plan.estimated_cost_usd is not None
    )
    unpriced = tuple(
        plan.name
        for plan in plans
        if plan.rows_to_reprocess > 0
        and plan.provider is not None
        and plan.estimated_cost_usd is None
    )
    return CostProjection(
        threshold_usd=threshold_usd,
        total_usd=round(sum(plan.estimated_cost_usd or 0.0 for plan in priced), 6),
        priced=priced,
        unpriced=unpriced,
        source=source,
    )


def format_cost_refusal(projection: CostProjection, *, declined: bool) -> str:
    """The error a run over its threshold exits with."""
    lead = (
        "Not started: the projected provider cost was declined."
        if declined
        else "Refusing to start: the projected provider cost is over this target's "
        "confirm_cost_above_usd, and no one is at a terminal to confirm it."
    )
    return "\n".join(
        [
            lead,
            *projection.describe(),
            "Run `stel plan` for the whole picture. To proceed: `--accept-cost`, or "
            "raise confirm_cost_above_usd on the target.",
        ]
    )
