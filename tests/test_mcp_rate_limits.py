"""Per-principal rate limiting on the MCP server (issue #463).

`--max-requests-per-minute` was sized for one local stdio client. On a network
transport it is one global cap every caller draws from, so a single principal
can exhaust it and starve the rest — and nothing distinguishes a noisy tenant
from an attack. These pin the shape that fixes that:

- each subject gets its own window, so one caller's flood is refused as *that
  caller's* limit while others are still served;
- the global cap stays as the ceiling above every window;
- requests with no resolvable principal share one anonymous bucket, so an
  unauthenticated flood counts against something rather than nothing;
- unset, nothing changes — the existing single-window behavior is exact.

Built directly on `ContextService` with a resolver whose identity can be
switched between calls, because `StaticPrincipalResolver` is one fixed
principal and the point here is several.
"""
from __future__ import annotations

from typing import Any

import pytest

from stel.mcp_server.authorization import (
    AuthorizationError,
    ClaimAuthorizationProvider,
    Principal,
)
from stel.mcp_server.contracts import ListContextModelsRequest, MCPErrorCode
from stel.mcp_server.service import (
    ContextServerSettings,
    ContextService,
    ContextServiceError,
    _OperationLimiter,
)
from tests.test_mcp_server import FakeRepository, FakeSearch, _artifact_catalog, _fixture_rows


class _SwitchingResolver:
    """Whoever `principal` is at call time — set it between requests."""

    def __init__(self) -> None:
        self.principal: Principal | None = Principal("alice", tenant_id="research")
        self.raise_on_resolve = False

    def resolve(self) -> Principal | None:
        if self.raise_on_resolve:
            raise AuthorizationError("resolver refused")
        return self.principal


def _service(**settings: Any) -> tuple[ContextService, _SwitchingResolver]:
    resolver = _SwitchingResolver()
    service = ContextService(
        catalog=_artifact_catalog(),
        repository=FakeRepository(_fixture_rows()),
        context_search=FakeSearch(),
        principal_resolver=resolver,
        authorization=ClaimAuthorizationProvider(),
        settings=ContextServerSettings(**settings),
    )
    return service, resolver


def _call(service: ContextService) -> Any:
    return service.list_context_models(ListContextModelsRequest())


def _as(resolver: _SwitchingResolver, subject: str) -> None:
    resolver.principal = Principal(subject, tenant_id="research")


# ─── isolation between principals ──────────────────────────────────────────


def test_one_caller_exhausting_its_share_does_not_starve_another() -> None:
    """The point of #463: alice at her limit is refused; bob is still served."""
    service, resolver = _service(
        max_requests_per_minute=10, max_requests_per_minute_per_principal=2
    )
    try:
        _as(resolver, "alice")
        assert _call(service).error is None
        assert _call(service).error is None
        third = _call(service)
        assert third.error is not None
        assert third.error.code is MCPErrorCode.BUSY
        assert third.error.retryable is True
        assert "This caller" in third.error.message

        _as(resolver, "bob")
        assert _call(service).error is None, "bob must not pay for alice's flood"
    finally:
        service.close()


def test_the_global_cap_is_still_the_ceiling_across_principals() -> None:
    """Per-principal shares do not add up past the global budget."""
    service, resolver = _service(
        max_requests_per_minute=3, max_requests_per_minute_per_principal=2
    )
    try:
        _as(resolver, "alice")
        assert _call(service).error is None
        assert _call(service).error is None
        _as(resolver, "bob")
        assert _call(service).error is None  # third overall, bob's first
        fourth = _call(service)
        assert fourth.error is not None
        assert fourth.error.code is MCPErrorCode.BUSY
        assert "context server" in fourth.error.message, (
            "bob has one of his two left; it is the global ceiling refusing him"
        )
    finally:
        service.close()


def test_a_globally_refused_request_is_not_charged_to_the_caller() -> None:
    """Both windows are decided before either is appended to, so a request the
    ceiling turns away does not also burn the caller's own share."""
    limiter = _OperationLimiter(
        max_concurrency=1,
        max_requests_per_minute=1,
        max_requests_per_minute_per_principal=5,
        timeout_seconds=5.0,
    )
    try:
        limiter.run(lambda: None, principal_key="alice")
        with pytest.raises(ContextServiceError) as refused:
            limiter.run(lambda: None, principal_key="bob")  # global ceiling
        assert refused.value.code is MCPErrorCode.BUSY
        assert "context server" in refused.value.message
        assert len(limiter._per_principal_times.get("bob", ())) == 0
    finally:
        limiter.close()


# ─── the anonymous bucket ──────────────────────────────────────────────────


def test_unauthenticated_requests_share_one_anonymous_bucket() -> None:
    """An unauthenticated flood counts against something. Each request is still
    refused as MISSING_PRINCIPAL by the operation — the bucket is about what
    the flood costs everyone else, not about letting it through."""
    service, resolver = _service(
        max_requests_per_minute=10, max_requests_per_minute_per_principal=2
    )
    try:
        resolver.principal = None
        first = _call(service)
        second = _call(service)
        assert first.error is not None and first.error.code is MCPErrorCode.MISSING_PRINCIPAL
        assert second.error is not None and second.error.code is MCPErrorCode.MISSING_PRINCIPAL
        third = _call(service)
        assert third.error is not None
        assert third.error.code is MCPErrorCode.BUSY, (
            "the third anonymous request should hit the anonymous bucket before "
            "it ever reaches principal resolution"
        )
        # A real caller is unaffected by the anonymous flood.
        _as(resolver, "carol")
        assert _call(service).error is None
    finally:
        service.close()


def test_a_resolver_that_raises_counts_as_anonymous() -> None:
    """A resolver failure is not a free pass past the limiter."""
    service, resolver = _service(
        max_requests_per_minute=10, max_requests_per_minute_per_principal=1
    )
    try:
        resolver.raise_on_resolve = True
        _call(service)
        second = _call(service)
        assert second.error is not None
        assert second.error.code is MCPErrorCode.BUSY
    finally:
        service.close()


# ─── unset means unchanged ─────────────────────────────────────────────────


def test_without_a_per_principal_cap_the_global_window_is_all_there_is() -> None:
    """Existing deployments see byte-identical behavior: no per-key state is
    kept at all, and the caller is never resolved ahead of the operation."""
    service, resolver = _service(max_requests_per_minute=2)
    try:
        _as(resolver, "alice")
        assert _call(service).error is None
        _as(resolver, "bob")
        assert _call(service).error is None
        _as(resolver, "carol")
        third = _call(service)
        assert third.error is not None and third.error.code is MCPErrorCode.BUSY
        assert "context server" in third.error.message
        assert service._limiter._per_principal_times == {}
    finally:
        service.close()


# ─── configuration ─────────────────────────────────────────────────────────


def test_a_per_principal_cap_above_the_global_one_is_rejected() -> None:
    """The global cap is the ceiling, so a larger share is unreachable and
    would only mislead an operator reading the config."""
    with pytest.raises(ValueError, match="exceeds"):
        ContextServerSettings(
            max_requests_per_minute=10, max_requests_per_minute_per_principal=11
        )


def test_a_per_principal_cap_equal_to_the_global_one_is_allowed() -> None:
    ContextServerSettings(
        max_requests_per_minute=10, max_requests_per_minute_per_principal=10
    )


# ─── the per-key table does not leak ───────────────────────────────────────


def test_idle_principals_are_swept_once_the_table_is_large(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A churn of one-off subjects must not grow the table forever. Keys whose
    window is empty after trimming are dropped once the table passes the sweep
    threshold — asserted by driving time forward rather than waiting a minute."""
    from stel.mcp_server import service as service_module

    monkeypatch.setattr(service_module, "_SWEEP_KEYS", 4)
    clock = {"now": 1000.0}
    monkeypatch.setattr(service_module, "monotonic", lambda: clock["now"])

    limiter = _OperationLimiter(
        max_concurrency=1,
        max_requests_per_minute=1000,
        max_requests_per_minute_per_principal=100,
        timeout_seconds=5.0,
    )
    try:
        for subject in ("a", "b", "c", "d", "e"):
            limiter.run(lambda: None, principal_key=subject)
        assert len(limiter._per_principal_times) == 5
        # A minute later every one of those windows is empty; the next request
        # pushes the table past the threshold and sweeps them.
        clock["now"] += 61
        limiter.run(lambda: None, principal_key="f")
        assert set(limiter._per_principal_times) == {"f"}
    finally:
        limiter.close()
