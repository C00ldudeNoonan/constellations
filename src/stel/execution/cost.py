from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..config.profile import PricingConfig
from ..profile import ResolvedProfile
from ..providers import InferenceProvider


def estimate_cost(totals: dict[str, Any], pricing: PricingConfig) -> float:
    """Token totals × user-supplied USD-per-Mtok rates. Cache rates are
    optional; tokens with no configured rate contribute nothing."""
    rates = [
        ("input_tokens", pricing.input_usd_per_mtok),
        ("output_tokens", pricing.output_usd_per_mtok),
        ("cache_read_input_tokens", pricing.cache_read_usd_per_mtok),
        ("cache_creation_input_tokens", pricing.cache_write_usd_per_mtok),
    ]
    cost = sum(
        float(totals.get(key, 0)) * rate for key, rate in rates if rate is not None
    )
    return round(cost / 1_000_000, 6)


def budget_cost_estimator(
    resolved: ResolvedProfile,
    *,
    batch: bool,
    provider: InferenceProvider | None,
) -> Callable[[Mapping[str, Any]], float] | None:
    """Per-response USD estimate for spend budgets, honoring the provider's
    native-batch discount. Provider-reported cost wins when present."""
    if resolved.llm is None or resolved.llm.pricing is None:
        return None
    pricing = resolved.llm.pricing
    multiplier = (
        provider.batch_cost_multiplier if batch and provider is not None else 1.0
    )

    def _estimate(metrics: Mapping[str, Any]) -> float:
        return round(estimate_cost(dict(metrics), pricing) * multiplier, 6)

    return _estimate
