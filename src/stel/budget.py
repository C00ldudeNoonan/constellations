"""Enforceable execution budgets for LLM extraction (issue #149).

Budgets are declared per-model (`extraction.options.budget`) and per-run
(profiles.yml `llm.budget`). Configs validate strictly at compile time;
enforcement happens in the runner and LLM backend, always before the next
provider call so exhaustion never bills further work.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
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
    request bytes) before doing work. Provider calls are reserved at admission;
    tokens and cost are charged from each response before another call can use
    a response-measured cap on this ledger.
    """

    def __init__(self, config: LLMBudgetConfig, *, scope: str) -> None:
        self._config = config
        self._scope = scope
        self._lock = threading.Lock()
        # Shared by every guard that wraps this ledger. The totals lock makes
        # individual operations thread-safe; this one makes check-plus-charge
        # admission atomic across models.
        self._admission = threading.Lock()
        self._documents = 0
        self._total_bytes = 0
        self._api_calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost_usd = 0.0

    @property
    def scope(self) -> str:
        return self._scope

    @property
    def has_response_measured_caps(self) -> bool:
        """Whether admission must wait for the preceding response's usage."""
        config = self._config
        return any(
            cap is not None
            for cap in (
                config.max_input_tokens,
                config.max_output_tokens,
                config.max_cost_usd,
            )
        )

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


class _ProviderCallReservation:
    """One pre-charged provider call whose measured usage is not settled yet."""

    def __init__(self, guard: BudgetGuard, *, reserved_calls: int) -> None:
        self._guard = guard
        self._reserved_calls = reserved_calls
        self._settled = False

    @property
    def settled(self) -> bool:
        return self._settled

    def settle(
        self,
        metrics: Mapping[str, Any],
        *,
        actual_calls: int | None = None,
    ) -> None:
        self.settle_many((metrics,), actual_calls=actual_calls)

    def settle_many(
        self,
        metrics: Iterable[Mapping[str, Any]],
        *,
        actual_calls: int | None = None,
    ) -> None:
        """Settle several responses without collapsing their cost semantics."""
        if self._settled:
            raise RuntimeError("provider-call budget reservation was settled twice")
        measured_calls = 0
        for item in metrics:
            measured_calls += self._guard._metric_int(item, "api_calls")
            self._guard.charge_metrics(
                {key: value for key, value in item.items() if key != "api_calls"}
            )
        if actual_calls is None:
            actual_calls = measured_calls
        self._guard.settle_calls(
            reserved=self._reserved_calls,
            actual=max(actual_calls, 0),
        )
        self._settled = True


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
        self._ledgers: list[BudgetLedger] = []
        for ledger in (model_ledger, run_ledger):
            if ledger is not None and all(existing is not ledger for existing in self._ledgers):
                self._ledgers.append(ledger)
        self._cost_estimator = cost_estimator
        # Every guard lists model then run, but identity ordering also keeps
        # lock acquisition safe if a future caller composes ledgers differently.
        self._admission_ledgers = tuple(sorted(self._ledgers, key=id))

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

    def reserve_calls(self, count: int) -> None:
        """Admit `count` provider calls and charge them in one step.

        `ensure_headroom` then `charge_*` is a check-then-act: sequential
        callers are fine because nothing runs between the two, but concurrent
        ones all pass admission against the same pre-charge total and every one
        of them proceeds. With `max_api_calls: 1` and two workers in flight,
        both are admitted and two calls are billed — the cap silently buys more
        than it says (issue #432 review).

        Reserving under one lock closes that: a call is charged when it is
        admitted, not when it returns, so the cap counts work in flight.
        `settle_calls` charges the difference once the provider says what it
        actually billed.

        Admission locks belong to ledgers, so separate model guards sharing a
        run ledger cannot admit against the same pre-charge run total.
        """
        acquired = self._acquire_admission()
        try:
            self.ensure_headroom(next_calls=count)
            self.charge_usage(api_calls=count)
        finally:
            self._release_admission(acquired)

    @contextmanager
    def provider_call(self) -> Iterator[_ProviderCallReservation]:
        with self.provider_calls(1) as reservation:
            yield reservation

    @contextmanager
    def provider_calls(self, count: int) -> Iterator[_ProviderCallReservation]:
        """Reserve real provider calls and settle their response atomically.

        API-call caps reserve known units and release their ledger admission
        locks immediately, preserving concurrency. Token and spend caps are
        only known from the response, so only those ledgers stay locked until
        usage is settled. A shared run ledger therefore coordinates every
        model guard without unnecessarily serializing model-only caps.

        Call this only after a cache miss. A cache hit has no provider call and
        must consume no call budget.
        """
        if count < 1:
            raise ValueError("provider call reservation must be at least one")

        acquired = self._acquire_admission()
        try:
            self.ensure_headroom(next_calls=count)
            self.charge_usage(api_calls=count)
            response_ledgers = [
                ledger for ledger in acquired if ledger.has_response_measured_caps
            ]
            for ledger in reversed(acquired):
                if ledger not in response_ledgers:
                    ledger._admission.release()
            acquired = response_ledgers

            reservation = _ProviderCallReservation(
                self,
                reserved_calls=count,
            )
            yield reservation
            if not reservation.settled:
                raise RuntimeError(
                    "successful provider call did not settle its budget reservation"
                )
        finally:
            self._release_admission(acquired)

    def _acquire_admission(self) -> list[BudgetLedger]:
        acquired: list[BudgetLedger] = []
        try:
            for ledger in self._admission_ledgers:
                ledger._admission.acquire()
                acquired.append(ledger)
        except BaseException:
            self._release_admission(acquired)
            raise
        return acquired

    @staticmethod
    def _release_admission(ledgers: list[BudgetLedger]) -> None:
        for ledger in reversed(ledgers):
            ledger._admission.release()

    def settle_calls(self, *, reserved: int, actual: int) -> None:
        """Charge whatever the provider billed beyond the reservation.

        Only ever adds. An over-estimate leaves the ledger conservative, which
        fails safe: the run stops early rather than spending past the cap.
        """
        if actual > reserved:
            self.charge_usage(api_calls=actual - reserved)

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
            api_calls=self._metric_int(metrics, "api_calls"),
            input_tokens=self._metric_int(metrics, "input_tokens"),
            output_tokens=self._metric_int(metrics, "output_tokens"),
            cost_usd=float(cost),
        )

    @staticmethod
    def _metric_int(metrics: Mapping[str, Any], key: str) -> int:
        value = metrics.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return 0
        return max(int(value), 0)
