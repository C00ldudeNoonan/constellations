"""The serving-ledger protocol, exercised against a real warehouse.

#474 rewrote the ledger SQL that `ServingCoordinator` runs — a `CASE WHEN ...
ELSE ? END` inside an UPDATE, a lease INSERT conditioned on the observed
active generation, a fencing guard on the planning fence — and every test of
it ran on DuckDB. The publish that depends on it runs on BigQuery. One
sequence, run twice: on DuckDB here (always), and on BigQuery when
`STEL_BQ_TEST_PROJECT` is set (`docs/release.md`, step 5). The DuckDB run is
what proves the sequence itself is right; the BigQuery run is what proves the
SQL survives a second dialect and a second parameter binder.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from stel.adapters import (
    StateRecord,
    StateScope,
    WarehouseAdapter,
    create_adapter,
    parse_warehouse_config,
)
from stel.adapters.bigquery import BigQueryAdapter
from stel.retrieval import ServingCoordinator
from stel.retrieval.coordination import STATUS_DEGRADED, STATUS_READY, ServingBusyError

_BQ_PROJECT = os.environ.get("STEL_BQ_TEST_PROJECT")


def _scope(model_name: str) -> StateScope:
    return StateScope.for_target_descriptor(
        model_name,
        stage="retrieval_publish",
        descriptor={"store_type": "lancedb", "collection": model_name},
    )


def exercise_ledger_protocol(coordinator: ServingCoordinator) -> None:
    """Every ledger statement #474 changed, in the order a real publish uses them."""
    scope = _scope("ledger_live")

    # An in-place claim on an empty scope, published and activated.
    first = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        first,
        active_generation="gen1",
        config_fingerprint="cfg1",
        counts=(2, 0, 0, 0),
        active_collection="ledger_live__g1",
    )
    ready = coordinator.status(scope)
    assert ready.status == STATUS_READY
    assert ready.active_generation == "gen1" and ready.config_fingerprint == "cfg1"

    # A reader pins the ready generation: the reworked lease INSERT.
    reader = coordinator.acquire_query(scope)
    assert reader.pinned_generation == "gen1" and reader.config_fingerprint == "cfg1"
    coordinator.validate_query(reader, require_active=True)

    # A private build claims while the reader is pinned: the CASE WHEN keeps
    # the live fingerprint in the ledger, the lease carries the new one, and
    # the plan is fenced on the observed token.
    build = coordinator.acquire_publish(
        scope,
        expected_code_version="v2",
        config_fingerprint="cfg2",
        preserves_active_generation=True,
        expected_fencing_token=ready.fencing_token,
    )
    during = coordinator.status(scope)
    assert during.config_fingerprint == "cfg1", "the live generation keeps its own stamp"
    assert build.config_fingerprint == "cfg2"
    assert during.active_generation == "gen1", "a private claim preserves the pointer"
    coordinator.validate_query(reader)  # the pinned reader survives the claim

    # Activation moves both pointers together; the old-generation pin survives.
    coordinator.mark_ready(
        build,
        active_generation="gen2",
        config_fingerprint="cfg2",
        counts=(2, 0, 0, 0),
        active_collection="ledger_live__g2",
    )
    after = coordinator.status(scope)
    assert after.active_generation == "gen2" and after.config_fingerprint == "cfg2"
    coordinator.validate_query(reader)
    coordinator.release_query(reader)
    fresh = coordinator.acquire_query(scope)
    assert fresh.pinned_generation == "gen2" and fresh.config_fingerprint == "cfg2"
    coordinator.release_query(fresh)

    # Recovery keeps a private build's live generation and its fingerprint.
    stranded = coordinator.acquire_publish(
        scope,
        expected_code_version="v3",
        config_fingerprint="cfg3",
        preserves_active_generation=True,
    )
    assert stranded.config_fingerprint == "cfg3"
    recovered = coordinator.recover(scope, owner_terminated=True)
    assert recovered.status == STATUS_DEGRADED
    assert recovered.active_generation == "gen2" and recovered.config_fingerprint == "cfg2"
    again = coordinator.acquire_query(scope)
    coordinator.release_query(again)

    # The CASE's other branch: a private claim on a scope with nothing live
    # writes the claiming fingerprint, since there is nothing to retain.
    empty = _scope("ledger_live_empty")
    claim = coordinator.acquire_publish(
        empty,
        expected_code_version="v1",
        config_fingerprint="cfgA",
        preserves_active_generation=True,
    )
    assert coordinator.status(empty).config_fingerprint == "cfgA"
    # Nothing is live and a publisher holds the scope: admission says *busy*,
    # not *not ready* -- the publisher is the accurate reason.
    with pytest.raises(ServingBusyError):
        coordinator.acquire_query(empty)
    coordinator.mark_ready(
        claim, active_generation="genA", config_fingerprint="cfgA", counts=(1, 0, 0, 0),
        active_collection="ledger_live_empty__gA",
    )
    assert coordinator.status(empty).status == STATUS_READY


def test_the_ledger_protocol_on_duckdb(tmp_path: Path) -> None:
    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "ledger.duckdb"), "schema": "main"}
    )
    with create_adapter(config) as adapter:
        exercise_ledger_protocol(ServingCoordinator(adapter))


@pytest.mark.skipif(
    not _BQ_PROJECT, reason="set STEL_BQ_TEST_PROJECT to run BigQuery integration"
)
def test_the_ledger_protocol_on_bigquery() -> None:
    """The same sequence on the warehouse the #473 publish actually runs on."""
    config = parse_warehouse_config(
        {
            "type": "bigquery",
            "project": _BQ_PROJECT,
            "dataset": "stel_it_" + os.urandom(3).hex(),
        }
    )
    adapter = create_adapter(config)
    try:
        with adapter:
            exercise_ledger_protocol(ServingCoordinator(adapter))
    finally:
        assert isinstance(adapter, BigQueryAdapter)
        adapter._reset_storage_for_test()


# ─── the state half of a seeded generation (issues #495, #505) ───────────────
#
# `_seed_publication_state` pages the serving scope's state out of stel_state
# and upserts it into the new generation's scope — deliberately not through
# `replace_state_scope`, whose fence checks a ledger entry a generation scope
# does not have. On the 3.6M-row corpus #495 exists for, that is 3.6M state
# rows through paged MERGEs, which #431 showed full-scan an unclustered
# stel_state. A new BigQuery operation needs a live test in this file before
# it ships (docs/release.md); this is that test, run on DuckDB always and on
# BigQuery when STEL_BQ_TEST_PROJECT is set.
#
# What "bounded" means here, stated so nobody expects more of it: the gate
# asserts residency — every page the seed hands to upsert_state is at most the
# page size, and there are exactly as many as the scope needs — which is the
# form the bounded-memory tests use, and the one failure mode a scratch-sized
# scope can detect. It cannot detect an unclustered full scan by cost:
# BigQuery bills a 10MB minimum per table per job, so at this size a clustered
# and an unclustered MERGE bill identically. #431's clustering is pinned where
# it is applied, in the state-table DDL.

_SEED_ROWS = 2_500
_SEED_PAGE = 1_000  # three pages: two full, one short


def exercise_state_seeding(adapter: WarehouseAdapter, coordinator: ServingCoordinator) -> None:
    """Seed a generation scope from a serving scope, exactly as a publish does."""
    from stel.execution.search import _generation_state_scope, _seed_publication_state

    serving = _scope("seed_live")
    source = [StateRecord(f"k{i:06d}", f"fp{i:06d}", "v1") for i in range(_SEED_ROWS)]
    adapter.upsert_state(serving, source)
    lease = coordinator.acquire_publish(
        serving,
        expected_code_version="v2",
        config_fingerprint="cfg2",
        preserves_active_generation=True,
    )
    generation = _generation_state_scope("seed_live", "seed_live__gseed")

    # Observe every page the seed writes, through the adapter it actually
    # writes with. An instance attribute shadows the bound method and is
    # removed afterwards, so the adapter is unchanged for the assertions.
    pages: list[int] = []
    real_upsert = adapter.upsert_state

    def observed(scope: StateScope, records: list[StateRecord]) -> None:
        if scope == generation:
            pages.append(len(records))
        real_upsert(scope, records)

    adapter.upsert_state = observed  # type: ignore[method-assign]
    try:
        copied = _seed_publication_state(
            adapter,
            coordinator,
            serving_scope=serving,
            publish_scope=generation,
            lease=lease,
            page_size=_SEED_PAGE,
            code_version="v2",
        )
    finally:
        del adapter.upsert_state

    assert copied == _SEED_ROWS
    # Residency is bounded by the page, not the scope: a regression to loading
    # the whole scope at once shows up here as one page carrying every row.
    assert pages == [_SEED_PAGE, _SEED_PAGE, _SEED_ROWS - 2 * _SEED_PAGE], pages

    # The seeded scope matches the source exactly, restamped to the new code
    # version, and the source itself is untouched.
    original = adapter.fetch_state(serving)
    seeded = adapter.fetch_state(generation)
    assert len(original) == _SEED_ROWS
    assert set(seeded) == set(original)
    assert all(
        seeded[key].input_fingerprint == original[key].input_fingerprint for key in original
    )
    assert {value.code_version for value in seeded.values()} == {"v2"}
    assert {value.code_version for value in original.values()} == {"v1"}

    coordinator.mark_ready(
        lease,
        active_generation="gseed",
        config_fingerprint="cfg2",
        counts=(0, 0, _SEED_ROWS, 0),
        active_collection="seed_live__gseed",
    )
    assert coordinator.status(serving).status == STATUS_READY


def test_state_seeding_on_duckdb(tmp_path: Path) -> None:
    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "seed.duckdb"), "schema": "main"}
    )
    with create_adapter(config) as adapter:
        exercise_state_seeding(adapter, ServingCoordinator(adapter))


@pytest.mark.skipif(
    not _BQ_PROJECT, reason="set STEL_BQ_TEST_PROJECT to run BigQuery integration"
)
def test_state_seeding_on_bigquery() -> None:
    """The same sequence on the warehouse the #495 publish actually runs on."""
    config = parse_warehouse_config(
        {
            "type": "bigquery",
            "project": _BQ_PROJECT,
            "dataset": "stel_it_" + os.urandom(3).hex(),
        }
    )
    adapter = create_adapter(config)
    try:
        with adapter:
            exercise_state_seeding(adapter, ServingCoordinator(adapter))
    finally:
        assert isinstance(adapter, BigQueryAdapter)
        adapter._reset_storage_for_test()
