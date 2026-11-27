"""Selecting a pipeline variant by measured quality, cost and latency (ALE-286, #795).

`stel eval --compare` lines variants up on one golden set; this module takes
the next step for extraction variants. Given per-document counts and per-phase
cost evidence for each candidate it reports the observed Pareto frontier,
which candidates clear an explicit quality floor *with uncertainty*, the
cheapest feasible one, a one-shot holdout check of that frozen selection, and
an adopt/defer recommendation with a reviewable config diff.

It is a pure function of the evidence the caller supplies. It makes no
provider call, reads no document and writes nothing but the artifact it is
asked to write; it never edits a model file or a profile. The numbers it
consumes come from the existing scorers, the budget ledger (`budget.py`) and
`execution/cost.py`; an experiment's spend cap is enforced by that ledger, and
this module only records and checks the total.

What it does not claim: a point-estimate frontier describes the sample, and a
candidate whose interval overlaps another's is *tied*, not better. The result
says so rather than ranking them.
"""

from __future__ import annotations

import difflib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

VARIANT_SELECTION_ARTIFACT_VERSION = 1
VARIANT_SELECTION_FILENAME = "variant_selection.json"
MIN_BOOTSTRAP_SAMPLES = 100

CostBasis = Literal["measured", "estimated", "unknown"]
Decision = Literal["adopt", "defer"]

# Config handed to the diff must not carry credentials or the names of the
# environment variables that hold them (AGENTS.md security invariants).
_SECRET_KEY = re.compile(r"(api[_-]?key|secret|token|password|credential|_env$)", re.I)
_TIED = "statistically tied with "


class VariantSelectionError(ValueError):
    """The evidence is malformed or the candidates are not comparable."""


def _require_non_negative(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise VariantSelectionError(f"{label} must be a number")
    if not math.isfinite(value) or value < 0:
        raise VariantSelectionError(f"{label} must be finite and non-negative")


@dataclass(frozen=True)
class DocumentScore:
    """One document's scored counts. Precision is found/emitted, recall found/gold;
    the upstream scorer decides which rows (quarantined ones included) count."""

    document_id: str
    source_id: str
    gold: int
    found: int
    emitted: int

    def __post_init__(self) -> None:
        if not self.document_id or not self.source_id:
            raise VariantSelectionError("document and source identities are required")
        if any(type(v) is not int for v in (self.gold, self.found, self.emitted)):
            raise VariantSelectionError("fact counts must be integers")
        if self.gold <= 0 or not 0 <= self.found <= min(self.gold, self.emitted):
            raise VariantSelectionError(
                f"fact counts are inconsistent for document {self.document_id!r}"
            )


@dataclass(frozen=True)
class Measurement:
    """One phase's spend and elapsed time.

    `cost_usd` includes billed failures and retries; their counts are kept so a
    reader can see how much of the figure they are. Money is `measured`
    (provider-reported), `estimated` (tokens x the pricing assumption) or
    `unknown` -- never silently zero.
    """

    cost_usd: float | None
    cost_basis: CostBasis
    elapsed_seconds: float
    complete: bool
    billed_failures: int = 0
    retries: int = 0
    usage: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.complete) is not bool:
            raise VariantSelectionError("completion must be boolean")
        if self.cost_basis not in ("measured", "estimated", "unknown"):
            raise VariantSelectionError("invalid cost basis")
        if (self.cost_usd is None) != (self.cost_basis == "unknown"):
            raise VariantSelectionError("unknown cost must have an unknown basis, and only it")
        if self.cost_usd is not None:
            _require_non_negative(self.cost_usd, "cost")
        _require_non_negative(self.elapsed_seconds, "elapsed time")
        for label, value in (("billed failures", self.billed_failures), ("retries", self.retries)):
            if type(value) is not int or value < 0:
                raise VariantSelectionError(f"{label} must be a non-negative integer")


@dataclass(frozen=True)
class Candidate:
    """A configuration under test, with the identities that tie its numbers back to it."""

    name: str
    provider: str
    model: str
    implementation_identity: str
    strategy: str
    prompt_identity: str
    schema_identity: str
    config_identity: str
    corpus_identity: str
    judgment_identity: str
    pricing_assumption: str
    documents: tuple[DocumentScore, ...]
    cold: Measurement
    warm: Measurement
    changed: Measurement

    def __post_init__(self) -> None:
        identities = (
            self.name,
            self.provider,
            self.model,
            self.implementation_identity,
            self.strategy,
            self.prompt_identity,
            self.schema_identity,
            self.config_identity,
            self.corpus_identity,
            self.judgment_identity,
            self.pricing_assumption,
        )
        if not all(identities) or not self.documents:
            raise VariantSelectionError("candidate identities and documents are required")
        ids = [d.document_id for d in self.documents]
        if len(ids) != len(set(ids)):
            raise VariantSelectionError("duplicate document identity")


@dataclass(frozen=True)
class ExperimentSpend:
    """What the experiment itself cost, against the cap it was given."""

    budget_usd: float
    spent_usd: float
    basis: Literal["measured", "estimated"]

    def __post_init__(self) -> None:
        _require_non_negative(self.budget_usd, "experiment budget")
        _require_non_negative(self.spent_usd, "experiment spend")
        if self.basis not in ("measured", "estimated"):
            raise VariantSelectionError("experiment spend must be measured or estimated")
        if self.spent_usd > self.budget_usd:
            raise VariantSelectionError(
                f"experiment spend {self.spent_usd} exceeds its budget {self.budget_usd}"
            )


@dataclass(frozen=True)
class Result:
    name: str
    precision: float | None
    recall: float
    precision_interval: tuple[float, float] | None
    recall_interval: tuple[float, float] | None
    infeasible_reason: str | None
    candidate: Candidate

    @property
    def feasible(self) -> bool:
        return self.infeasible_reason is None


@dataclass(frozen=True)
class DevelopmentComparison:
    results: tuple[Result, ...]
    frontier: tuple[str, ...]
    # Pairs of frontier members whose quality intervals overlap: neither is
    # shown better on quality, so the cheaper is not "proven" to cost less for
    # the same quality either.
    tied: tuple[tuple[str, str], ...]
    selected: str | None
    sources: frozenset[str]
    seed: int
    resamples: int
    precision_floor: float
    recall_floor: float
    experiment: ExperimentSpend


@dataclass(frozen=True)
class HoldoutValidation:
    result: Result

    @property
    def passed(self) -> bool:
        return self.result.feasible


@dataclass(frozen=True)
class Recommendation:
    decision: Decision
    selected: str | None
    reasons: tuple[str, ...]
    config_diff: str


def _rates(docs: Sequence[DocumentScore]) -> tuple[float | None, float]:
    found = sum(d.found for d in docs)
    emitted = sum(d.emitted for d in docs)
    return (found / emitted if emitted else None, found / sum(d.gold for d in docs))


def _intervals(
    docs: Sequence[DocumentScore], *, seed: int, resamples: int
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Document-level percentile bootstrap, 95%. None when it cannot be stated."""
    if len(docs) < 2:
        return None, None
    rng = random.Random(seed)
    precision: list[float] = []
    recall: list[float] = []
    for _ in range(resamples):
        sample = [docs[rng.randrange(len(docs))] for _ in docs]
        p, r = _rates(sample)
        if p is None:
            # An undefined resample is not evidence that the floor is met.
            return None, None
        precision.append(p)
        recall.append(r)
    precision.sort()
    recall.sort()
    lower = int(0.025 * (resamples - 1))
    upper = math.ceil(0.975 * (resamples - 1))
    return (precision[lower], precision[upper]), (recall[lower], recall[upper])


def _evaluate(
    candidate: Candidate, *, precision_floor: float, recall_floor: float, seed: int, resamples: int
) -> Result:
    docs = sorted(candidate.documents, key=lambda d: d.document_id)
    precision, recall = _rates(docs)
    p_interval, r_interval = _intervals(docs, seed=seed, resamples=resamples)
    reason: str | None = None
    if p_interval is None or r_interval is None:
        reason = "quality uncertainty cannot be stated (fewer than two documents or no output)"
    elif p_interval[0] < precision_floor or r_interval[0] < recall_floor:
        reason = "lower bound of a quality interval is below its floor"
    elif not candidate.cold.complete or candidate.cold.cost_usd is None:
        reason = "cold build is incomplete or its cost is unknown"
    return Result(candidate.name, precision, recall, p_interval, r_interval, reason, candidate)


def _point(result: Result) -> tuple[float, float, float, float] | None:
    cold = result.candidate.cold
    if result.precision is None or not cold.complete or cold.cost_usd is None:
        return None
    return (result.precision, result.recall, -cold.cost_usd, -cold.elapsed_seconds)


def _dominates(left: Result, right: Result) -> bool:
    a, b = _point(left), _point(right)
    if a is None or b is None:
        return False
    return all(x >= y for x, y in zip(a, b, strict=True)) and a != b


def _overlap(a: tuple[float, float] | None, b: tuple[float, float] | None) -> bool:
    # An interval that cannot be stated cannot separate two candidates.
    return a is None or b is None or (a[0] <= b[1] and b[0] <= a[1])


def compare_development(
    candidates: Sequence[Candidate],
    *,
    precision_floor: float,
    recall_floor: float,
    seed: int,
    resamples: int,
    experiment: ExperimentSpend,
) -> DevelopmentComparison:
    """Compare candidates scored on the same aligned documents and judgments."""
    if len(candidates) < 2 or len({c.name for c in candidates}) != len(candidates):
        raise VariantSelectionError("at least two uniquely named candidates are required")
    for floor in (precision_floor, recall_floor):
        if isinstance(floor, bool) or not math.isfinite(floor) or not 0 <= floor <= 1:
            raise VariantSelectionError("quality floors must be finite and between zero and one")
    if type(seed) is not int or type(resamples) is not int or resamples < MIN_BOOTSTRAP_SAMPLES:
        raise VariantSelectionError(
            f"seed must be an integer and resamples at least {MIN_BOOTSTRAP_SAMPLES}"
        )
    first = candidates[0]
    reference = {d.document_id: (d.source_id, d.gold) for d in first.documents}
    results: list[Result] = []
    for candidate in candidates:
        if (candidate.corpus_identity, candidate.judgment_identity) != (
            first.corpus_identity,
            first.judgment_identity,
        ):
            raise VariantSelectionError(
                f"candidate {candidate.name!r} uses a different corpus or judgment identity "
                f"than {first.name!r}"
            )
        by_id = {d.document_id: (d.source_id, d.gold) for d in candidate.documents}
        if by_id != reference:
            raise VariantSelectionError(
                f"candidate {candidate.name!r} was scored on different documents or labels "
                f"than {first.name!r}"
            )
        results.append(
            _evaluate(
                candidate,
                precision_floor=precision_floor,
                recall_floor=recall_floor,
                seed=seed,
                resamples=resamples,
            )
        )
    measured = [r for r in results if _point(r) is not None]
    on_frontier = [r for r in measured if not any(_dominates(o, r) for o in measured)]
    tied = tuple(
        (a.name, b.name)
        for i, a in enumerate(on_frontier)
        for b in on_frontier[i + 1 :]
        if _overlap(a.precision_interval, b.precision_interval)
        and _overlap(a.recall_interval, b.recall_interval)
    )
    eligible = [r for r in results if r.feasible]
    selected = (
        min(eligible, key=lambda r: (r.candidate.cold.cost_usd or 0.0, r.name)).name
        if eligible
        else None
    )
    return DevelopmentComparison(
        tuple(results),
        tuple(r.name for r in on_frontier),
        tied,
        selected,
        frozenset(d.source_id for d in first.documents),
        seed,
        resamples,
        precision_floor,
        recall_floor,
        experiment,
    )


def validate_holdout(development: DevelopmentComparison, candidate: Candidate) -> HoldoutValidation:
    """Score only the frozen selection on sources absent from development.

    It never chooses a replacement: a failed validation is reported as failed.
    The caller must not re-run it after tuning; nothing here can enforce that.
    """
    if development.selected is None or candidate.name != development.selected:
        raise VariantSelectionError("holdout must use the frozen development selection")
    chosen = next(r for r in development.results if r.name == candidate.name).candidate
    if candidate.config_identity != chosen.config_identity:
        raise VariantSelectionError("holdout configuration differs from development")
    if candidate.judgment_identity != chosen.judgment_identity:
        raise VariantSelectionError("holdout must use the development judgment identity")
    if candidate.corpus_identity == chosen.corpus_identity:
        raise VariantSelectionError("holdout must be a different corpus version than development")
    if development.sources.intersection(d.source_id for d in candidate.documents):
        raise VariantSelectionError("holdout sources overlap development")
    return HoldoutValidation(
        _evaluate(
            candidate,
            precision_floor=development.precision_floor,
            recall_floor=development.recall_floor,
            seed=development.seed,
            resamples=development.resamples,
        )
    )


def _reject_secrets(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _SECRET_KEY.search(str(key)):
                raise VariantSelectionError(
                    "configuration for review must not carry credentials or credential names"
                )
            _reject_secrets(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _reject_secrets(item)


def _render(config: Mapping[str, Any]) -> list[str]:
    return [line + "\n" for line in json.dumps(config, indent=2, sort_keys=True).splitlines()]


def config_diff(baseline: Mapping[str, Any], selected: Mapping[str, Any], *, label: str) -> str:
    """A unified diff of two configs for review. It is text to read, not a patch
    stel applies: nothing here edits a model file or a profile."""
    _reject_secrets(baseline)
    _reject_secrets(selected)
    return "".join(
        difflib.unified_diff(_render(baseline), _render(selected), fromfile="current", tofile=label)
    )


def recommend(
    development: DevelopmentComparison,
    holdout: HoldoutValidation | None,
    *,
    baseline_config: Mapping[str, Any],
    selected_config: Mapping[str, Any],
    min_documents: int,
) -> Recommendation:
    """Adopt only a feasible, holdout-validated selection from a large enough sample.

    Anything else is a defer with the reasons, because a defer is the honest
    answer to thin evidence. A tie with another frontier member is noted, not
    held against the selection: the selection is the cheapest feasible one.
    """
    blocking: list[str] = []
    n_dev = len(development.results[0].candidate.documents)
    if development.selected is None:
        blocking.append("no candidate's quality lower bound clears both floors")
    if n_dev < min_documents:
        blocking.append(f"development sample has {n_dev} documents, fewer than {min_documents}")
    if development.selected is not None and holdout is None:
        blocking.append("the selection has not been validated on a holdout")
    if holdout is not None:
        if holdout.result.name != development.selected:
            blocking.append("holdout was not run for the selected candidate")
        elif not holdout.passed:
            blocking.append(f"holdout failed: {holdout.result.infeasible_reason}")
        n_hold = len(holdout.result.candidate.documents)
        if n_hold < min_documents:
            blocking.append(f"holdout sample has {n_hold} documents, fewer than {min_documents}")
    notes = [
        f"{_TIED}{other}: quality is not shown to differ"
        for pair in development.tied
        if development.selected in pair
        for other in pair
        if other != development.selected
    ]
    reasons = tuple(blocking + notes)
    if blocking or development.selected is None:
        return Recommendation("defer", development.selected, reasons, "")
    diff = config_diff(baseline_config, selected_config, label=development.selected)
    return Recommendation("adopt", development.selected, reasons, diff)


def _measurement_dict(m: Measurement) -> dict[str, Any]:
    return {
        "cost_usd": m.cost_usd,
        "cost_basis": m.cost_basis,
        "elapsed_seconds": m.elapsed_seconds,
        "complete": m.complete,
        "billed_failures": m.billed_failures,
        "retries": m.retries,
        "usage": dict(sorted(m.usage.items())),
    }


def _result_dict(result: Result) -> dict[str, Any]:
    c = result.candidate
    return {
        "name": c.name,
        "identity": {
            "provider": c.provider,
            "model": c.model,
            "implementation": c.implementation_identity,
            "strategy": c.strategy,
            "prompt": c.prompt_identity,
            "schema": c.schema_identity,
            "config": c.config_identity,
            "corpus": c.corpus_identity,
            "judgment": c.judgment_identity,
            "pricing_assumption": c.pricing_assumption,
        },
        "documents": len(c.documents),
        "precision": result.precision,
        "recall": result.recall,
        "precision_interval": result.precision_interval,
        "recall_interval": result.recall_interval,
        "feasible": result.feasible,
        "infeasible_reason": result.infeasible_reason,
        "cold": _measurement_dict(c.cold),
        "warm": _measurement_dict(c.warm),
        "changed": _measurement_dict(c.changed),
    }


def build_variant_selection_artifact(
    development: DevelopmentComparison,
    holdout: HoldoutValidation | None,
    recommendation: Recommendation,
) -> dict[str, Any]:
    """Counts, identities, money and elapsed time -- never document text, prompt
    text, provider responses or credentials."""
    return {
        "version": VARIANT_SELECTION_ARTIFACT_VERSION,
        "bootstrap": {
            "method": "document_percentile",
            "confidence_level": 0.95,
            "samples": development.resamples,
            "seed": development.seed,
        },
        "floors": {
            "precision": development.precision_floor,
            "recall": development.recall_floor,
        },
        "experiment": {
            "budget_usd": development.experiment.budget_usd,
            "spent_usd": development.experiment.spent_usd,
            "basis": development.experiment.basis,
        },
        "candidates": [_result_dict(r) for r in development.results],
        "frontier": list(development.frontier),
        "frontier_note": "Observed-sample point estimates; not evidence of superiority.",
        "tied": [list(pair) for pair in development.tied],
        "selected": development.selected,
        "holdout": (
            None
            if holdout is None
            else {"passed": holdout.passed, "result": _result_dict(holdout.result)}
        ),
        "recommendation": {
            "decision": recommendation.decision,
            "selected": recommendation.selected,
            "reasons": list(recommendation.reasons),
            "config_diff": recommendation.config_diff,
        },
    }


def write_variant_selection_artifact(target_dir: Path, artifact: Mapping[str, Any]) -> Path:
    """Write the artifact under `target_dir` (the project's target path). It is the
    only file this module writes."""
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / VARIANT_SELECTION_FILENAME
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return out
