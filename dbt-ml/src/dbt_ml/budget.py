"""Enforceable execution budgets for LLM extraction (issue #149).

Budgets are declared per-model (`extraction.options.budget`) and per-run
(profiles.yml `llm.budget`). Configs validate strictly at compile time;
enforcement happens in the runner and LLM backend, always before the next
provider call so exhaustion never bills further work.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BudgetExceededError(Exception):
    """A model or run budget would be (or has been) exceeded.

    Raised before the next provider call. Message text is artifact-safe:
    limit name, scope, configured cap, and observed value only.
    """

    def __init__(self, *, scope: str, limit: str, cap: float, observed: float) -> None:
        self.scope = scope
        self.limit = limit
        self.cap = cap
        self.observed = observed
        cap_text = f"{cap:g}"
        observed_text = f"{observed:g}"
        super().__init__(
            f"{scope} budget exceeded: {limit}={observed_text} over cap {cap_text}"
        )


class LLMBudgetConfig(BaseModel):
    """Strictly typed execution caps. Absent fields are unlimited."""

    model_config = ConfigDict(extra="forbid", strict=True)

    max_documents: int | None = Field(default=None, ge=1)
    max_file_bytes: int | None = Field(default=None, ge=1)
    max_total_bytes: int | None = Field(default=None, ge=1)
    max_input_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    max_api_calls: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)


class BudgetLedger:
    """Thread-safe accumulator enforcing one LLMBudgetConfig.

    Callers precheck statically knowable limits (documents, file size,
    request bytes) before doing work, and charge measured usage (calls,
    tokens, cost) after each provider response. `ensure_headroom()` gates
    the next provider call on the accumulated totals.
    """

    def __init__(self, config: LLMBudgetConfig, *, scope: str) -> None:
        self._config = config
        self._scope = scope
        self._lock = threading.Lock()
        self._documents = 0
        self._total_bytes = 0
        self._api_calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost_usd = 0.0

    @property
    def scope(self) -> str:
        return self._scope

    def charge_documents(self, count: int) -> None:
        cap = self._config.max_documents
        with self._lock:
            self._documents += count
            if cap is not None and self._documents > cap:
                raise BudgetExceededError(
                    scope=self._scope,
                    limit="max_documents",
                    cap=cap,
                    observed=self._documents,
                )

    def check_file_bytes(self, size: int) -> None:
        cap = self._config.max_file_bytes
        if cap is not None and size > cap:
            raise BudgetExceededError(
                scope=self._scope,
                limit="max_file_bytes",
                cap=cap,
                observed=size,
            )

    def charge_bytes(self, count: int) -> None:
        cap = self._config.max_total_bytes
        with self._lock:
            self._total_bytes += count
            if cap is not None and self._total_bytes > cap:
                raise BudgetExceededError(
                    scope=self._scope,
                    limit="max_total_bytes",
                    cap=cap,
                    observed=self._total_bytes,
                )

    def ensure_headroom(self, *, next_calls: int = 1) -> None:
        """Gate the next provider call(s) on accumulated totals.

        Token and spend usage is only measurable after a response, so those
        limits stop the run at the first call that would follow the
        overrun — never billing past the cap by more than one response.
        """
        config = self._config
        with self._lock:
            checks: list[tuple[str, float | None, float]] = [
                ("max_api_calls", config.max_api_calls, self._api_calls + next_calls),
                ("max_input_tokens", config.max_input_tokens, self._input_tokens),
                ("max_output_tokens", config.max_output_tokens, self._output_tokens),
                ("max_cost_usd", config.max_cost_usd, self._cost_usd),
            ]
            for limit, cap, observed in checks:
                if cap is None:
                    continue
                exceeded = (
                    observed > cap
                    if limit == "max_api_calls"
                    else observed >= cap
                )
                if exceeded:
                    raise BudgetExceededError(
                        scope=self._scope,
                        limit=limit,
                        cap=cap,
                        observed=observed,
                    )

    def charge_usage(
        self,
        *,
        api_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record measured usage. Never raises: the overrunning call already
        happened, and `ensure_headroom()` stops the next one."""
        if not math.isfinite(cost_usd) or cost_usd < 0:
            cost_usd = 0.0
        with self._lock:
            self._api_calls += max(api_calls, 0)
            self._input_tokens += max(input_tokens, 0)
            self._output_tokens += max(output_tokens, 0)
            self._cost_usd += cost_usd

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {
                "documents": self._documents,
                "total_bytes": self._total_bytes,
                "api_calls": self._api_calls,
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "cost_usd": round(self._cost_usd, 6),
            }


class BudgetGuard:
    """Model-scope plus optional run-scope ledgers checked as one unit.

    The run ledger is shared across every model in the invocation; the model
    ledger is private to one model run. Every check consults both.
    """

    def __init__(
        self,
        model_ledger: BudgetLedger | None,
        run_ledger: BudgetLedger | None,
        *,
        cost_estimator: Callable[[Mapping[str, Any]], float] | None = None,
    ) -> None:
        self._ledgers = [
            ledger for ledger in (model_ledger, run_ledger) if ledger is not None
        ]
        self._cost_estimator = cost_estimator

    @property
    def active(self) -> bool:
        return bool(self._ledgers)

    def charge_documents(self, count: int) -> None:
        for ledger in self._ledgers:
            ledger.charge_documents(count)

    def check_file_bytes(self, size: int) -> None:
        for ledger in self._ledgers:
            ledger.check_file_bytes(size)

    def charge_bytes(self, count: int) -> None:
        for ledger in self._ledgers:
            ledger.charge_bytes(count)

    def ensure_headroom(self, *, next_calls: int = 1) -> None:
        for ledger in self._ledgers:
            ledger.ensure_headroom(next_calls=next_calls)

    def charge_usage(
        self,
        *,
        api_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        for ledger in self._ledgers:
            ledger.charge_usage(
                api_calls=api_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )

    def charge_metrics(self, metrics: Mapping[str, Any]) -> None:
        """Charge measured usage from an extraction-result metrics mapping.

        Provider-reported spend wins over the runner-supplied estimator.
        """

        def _int(key: str) -> int:
            value = metrics.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int | float):
                return 0
            return max(int(value), 0)

        cost = metrics.get("reported_cost_usd")
        if (
            isinstance(cost, bool)
            or not isinstance(cost, int | float)
            or not math.isfinite(float(cost))
            or float(cost) <= 0
        ):
            cost = (
                self._cost_estimator(metrics)
                if self._cost_estimator is not None
                else 0.0
            )
        self.charge_usage(
            api_calls=_int("api_calls"),
            input_tokens=_int("input_tokens"),
            output_tokens=_int("output_tokens"),
            cost_usd=float(cost),
        )
