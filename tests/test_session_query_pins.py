"""Durable session pins and cooperative drain, including paused readers (#541)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier, Event, Lock
from typing import Any

import pytest

from stel.adapters import StateScope, create_adapter, parse_warehouse_config
from stel.adapters.base import AdapterError
from stel.adapters.serialized import SerializedAdapter
from stel.retrieval.coordination import (
    QUERY_DRAIN_SECONDS,
    STATUS_DEGRADED,
    ServingBusyError,
    ServingCoordinator,
    StaleServingLeaseError,
)
from stel.retrieval.query_pins import IDLE_PIN_SECONDS, PIN_SWEEP_SECONDS, SessionQueryPins
from stel.timing import PhaseTimings


@pytest.fixture
def serving(tmp_path: Any) -> Any:
    config = parse_warehouse_config({"type": "duckdb", "path": str(tmp_path / "pins.duckdb")})
    with create_adapter(config) as adapter:
        coordinator = ServingCoordinator(adapter, ensure_schema=True)
        scope = StateScope("search", stage="retrieval_publish", target_identity="test")
        publish = coordinator.acquire_publish(
            scope, expected_code_version="v1", config_fingerprint="cfg"
        )
        coordinator.mark_ready(
            publish, active_generation="g1", config_fingerprint="cfg", counts=(1, 0, 0, 0)
        )
        pins = SessionQueryPins(lambda: nullcontext(adapter))
        try:
            yield coordinator, scope, pins
        finally:
            pins.close()


def _query(coordinator: Any, scope: Any, pins: Any) -> Any:
    return pins.query(coordinator, scope, legacy_scope=scope, timings=PhaseTimings())


def _complete_query(coordinator: Any, scope: Any, pins: Any) -> None:
    with _query(coordinator, scope, pins):
        pass


def test_a_drain_never_revokes_a_paused_reader_or_lets_new_readers_join(
    serving: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#541: expiry of an admission gate cannot be expiry of the reader pin."""
    coordinator, scope, pins = serving
    now = 1000
    monkeypatch.setattr("stel.retrieval.coordination.time.time", lambda: now)
    with _query(coordinator, scope, pins) as outer:
        with _query(coordinator, scope, pins) as inner:
            assert outer.lease_id == inner.lease_id
            assert coordinator.status(scope).query_leases == 1
            with pytest.raises(ServingBusyError, match="query leases"):
                coordinator.acquire_publish(
                    scope, expected_code_version="v2", config_fingerprint="cfg"
                )
            with pytest.raises(ServingBusyError, match="drain"):
                coordinator.acquire_query(scope)
            with pytest.raises(ServingBusyError, match="drain"):
                _complete_query(coordinator, scope, pins)
            # Even after the gate expires and a cleanup runs, both holders
            # still own their generation. Time passing proves no termination.
            now += QUERY_DRAIN_SECONDS + 1
            pins.sweep(closing=True)
            coordinator.validate_query(inner)
        assert coordinator.status(scope).query_leases == 1
        coordinator.validate_query(outer)
        pins.close()
        assert coordinator.status(scope).query_leases == 1
    assert coordinator.status(scope).query_leases == 0
    coordinator.acquire_publish(scope, expected_code_version="v2", config_fingerprint="cfg")


def test_an_abandoned_drain_gate_expires_but_its_durable_pin_does_not(
    serving: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#541: a failed publisher cannot permanently disable admission."""
    coordinator, scope, pins = serving
    now = 1000
    monkeypatch.setattr("stel.retrieval.coordination.time.time", lambda: now)
    query = coordinator.acquire_query(scope)
    with pytest.raises(ServingBusyError):
        coordinator.acquire_publish(scope, expected_code_version="v2", config_fingerprint="cfg")
    now += QUERY_DRAIN_SECONDS + 1
    with _query(coordinator, scope, pins):
        assert coordinator.status(scope).query_leases == 2
    coordinator.validate_query(query)
    coordinator.release_query(query)


def test_idle_retention_restarts_on_completion_and_preserves_reuse_until_its_deadline(
    serving: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#541: pin the documented timing defaults and actual idle boundary."""
    assert (IDLE_PIN_SECONDS, PIN_SWEEP_SECONDS, QUERY_DRAIN_SECONDS) == (30, 5, 60)
    coordinator, scope, pins = serving
    now = 1000.0
    monkeypatch.setattr("stel.retrieval.query_pins.monotonic", lambda: now)
    with _query(coordinator, scope, pins) as first:
        pass
    now += 10
    with _query(coordinator, scope, pins) as reused:
        assert reused.lease_id == first.lease_id
    now += 29
    pins.sweep()
    assert coordinator.status(scope).query_leases == 1
    now += 1
    pins.sweep()
    assert coordinator.status(scope).query_leases == 0


def test_concurrent_first_admissions_share_one_pin_until_the_last_worker_finishes(
    serving: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#541: serialize pin admission, while allowing both readers to run."""
    coordinator, scope, pins = serving
    coordinator._adapter = SerializedAdapter(coordinator._adapter, Lock())
    acquire = coordinator.acquire_query
    entered, resume, attempted, duplicate = Event(), Event(), Event(), Event()
    finish_first, finish_second = Event(), Event()
    joined = Barrier(3)
    calls: list[Any] = []

    def blocked_acquire(scope: Any, *, legacy_scope: Any) -> Any:
        calls.append(scope)
        if len(calls) > 1:
            duplicate.set()
        entered.set()
        assert resume.wait(timeout=5)
        return acquire(scope, legacy_scope=legacy_scope)

    def worker(finish: Event, *, second: bool) -> str:
        if second:
            attempted.set()
        with _query(coordinator, scope, pins) as lease:
            joined.wait(timeout=5)
            assert finish.wait(timeout=5)
            return lease.lease_id

    monkeypatch.setattr(coordinator, "acquire_query", blocked_acquire)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(worker, finish_first, second=False)
        try:
            assert entered.wait(timeout=5)
            second = executor.submit(worker, finish_second, second=True)
            assert attempted.wait(timeout=5)
            assert not duplicate.wait(timeout=0.1)
            resume.set()
            joined.wait(timeout=5)
            finish_first.set()
            first_id = first.result(timeout=5)
            assert coordinator.status(scope).query_leases == 1
            pins.close()
            assert coordinator.status(scope).query_leases == 1
            finish_second.set()
            assert second.result(timeout=5) == first_id
        finally:
            resume.set()
            finish_first.set()
            finish_second.set()
    assert len(calls) == 1
    assert coordinator.status(scope).query_leases == 0


def test_cutover_retires_the_cache_without_releasing_a_concurrent_holder(serving: Any) -> None:
    """#541: an old holder survives cutover; new queries must see the new pointer."""
    coordinator, scope, pins = serving
    with _query(coordinator, scope, pins) as reader:
        publish = coordinator.acquire_publish(
            scope, expected_code_version="v2", config_fingerprint="cfg2",
            preserves_active_generation=True,
        )
        coordinator.mark_ready(
            publish, active_generation="g2", config_fingerprint="cfg2", counts=(1, 0, 0, 0)
        )
        with pytest.raises(StaleServingLeaseError):
            with _query(coordinator, scope, pins):
                pytest.fail("A new reader joined the old generation")
        coordinator.validate_query(reader)
        assert coordinator.status(scope).query_leases == 1
    with _query(coordinator, scope, pins) as fresh:
        assert fresh.pinned_generation == "g2"
        assert fresh.config_fingerprint == "cfg2"


def test_reused_pin_reports_current_degraded_status_and_is_fenced_by_recovery(serving: Any) -> None:
    """#541: caching a lease cannot cache readiness or hide recovery."""
    coordinator, scope, pins = serving
    with _query(coordinator, scope, pins) as first:
        pass
    publish = coordinator.acquire_publish(
        scope, expected_code_version="v2", config_fingerprint="cfg",
        preserves_active_generation=True,
    )
    coordinator.mark_failed(
        publish, safe_error_code="store_error", counts=(0, 0, 0, 0),
        active_generation="g1", config_fingerprint="cfg",
    )
    with _query(coordinator, scope, pins) as reused:
        assert reused.lease_id == first.lease_id
        assert reused.status == STATUS_DEGRADED
        assert reused.safe_error_code == "store_error"
    coordinator.recover(scope, owner_terminated=True)
    with pytest.raises(StaleServingLeaseError):
        with _query(coordinator, scope, pins):
            pytest.fail("Recovery must fence out a cached pin")
    with _query(coordinator, scope, pins) as fresh:
        assert fresh.lease_id != first.lease_id


def test_failed_delete_stays_registered_until_cleanup_can_retry(
    serving: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#541: a broken connection must not silently forget an unreleased pin."""
    coordinator, scope, pins = serving
    release = ServingCoordinator.release_query

    def fail(self: Any, lease: Any) -> None:
        raise AdapterError("safe category")

    monkeypatch.setattr(ServingCoordinator, "release_query", fail)
    with pytest.raises(AdapterError):
        with _query(coordinator, scope, pins):
            raise ValueError("query failed")
    assert coordinator.status(scope).query_leases == 1
    monkeypatch.setattr(ServingCoordinator, "release_query", release)
    pins.sweep()
    assert coordinator.status(scope).query_leases == 0


def test_a_stale_drain_request_cannot_pause_a_new_generation(
    serving: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#541: a publisher losing its fence cannot gate its successor's readers."""
    coordinator, scope, pins = serving
    reader = coordinator.acquire_query(scope)
    request = coordinator._request_query_drain

    def cutover_before_request(scope: Any, *, fencing_token: int) -> None:
        publish = coordinator.acquire_publish(
            scope, expected_code_version="v2", config_fingerprint="cfg",
            preserves_active_generation=True,
        )
        coordinator.mark_ready(
            publish, active_generation="g2", config_fingerprint="cfg", counts=(1, 0, 0, 0)
        )
        request(scope, fencing_token=fencing_token)

    monkeypatch.setattr(coordinator, "_request_query_drain", cutover_before_request)
    with pytest.raises(ServingBusyError):
        coordinator.acquire_publish(scope, expected_code_version="v2", config_fingerprint="cfg")
    with _query(coordinator, scope, pins) as fresh:
        assert fresh.pinned_generation == "g2"
    coordinator.release_query(reader)


def test_a_drain_between_admission_read_and_insert_refuses_the_new_pin(
    serving: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#541: checking the gate only in Python leaves a stale-admission window."""
    coordinator, scope, pins = serving
    execute = coordinator._adapter.execute

    def request_before_insert(sql: str, params: Any = None) -> Any:
        if "INSERT INTO" in sql and "pinned_generation" in sql:
            coordinator._request_query_drain(scope, fencing_token=1)
        return execute(sql, params)

    monkeypatch.setattr(coordinator._adapter, "execute", request_before_insert)
    with pytest.raises(ServingBusyError):
        with _query(coordinator, scope, pins):
            pytest.fail("A stale admission bypassed the drain gate")
    assert coordinator.status(scope).query_leases == 0


def test_idle_pin_is_released_by_background_maintenance(
    serving: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#541: an idle server cannot retain a healthy pin indefinitely."""
    coordinator, scope, original = serving
    original.close()
    monkeypatch.setattr("stel.retrieval.query_pins.PIN_SWEEP_SECONDS", 0.01)
    monkeypatch.setattr("stel.retrieval.query_pins.IDLE_PIN_SECONDS", 0.0)
    released = Event()
    release = ServingCoordinator.release_query

    def observe(self: Any, lease: Any) -> None:
        release(self, lease)
        released.set()

    monkeypatch.setattr(ServingCoordinator, "release_query", observe)
    pins = SessionQueryPins(lambda: nullcontext(coordinator._adapter))
    try:
        with _query(coordinator, scope, pins):
            pass
        assert released.wait(timeout=5)
        # The event alone precedes the cache removal; close joins the worker.
    finally:
        pins.close()
    assert coordinator.status(scope).query_leases == 0
