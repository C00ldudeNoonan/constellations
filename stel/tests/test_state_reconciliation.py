"""Paged publication-state reconciliation (issue #153).

Covers the adapter contract (ordered paged state reads, bounded subset
lookups, fenced atomic scope replacement) against real DuckDB storage, the
core stream validation, and bounded-residency guarantees with state domains
larger than one page.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

from stel.adapters import (
    AdapterCapabilityError,
    AdapterError,
    StaleStateFenceError,
    StateAbsenceProbe,
    StatePage,
    StatePageReader,
    StatePageRecord,
    StateRecord,
    StateScope,
    StateScopeFence,
    StateValue,
    WarehouseCapability,
    create_adapter,
    parse_warehouse_config,
)
from stel.adapters.duckdb import DuckDBAdapter
from stel.retrieval import ServingCoordinator
from stel.state_reconciliation import (
    BoundedReconciler,
    UpstreamRecord,
    classify_batch,
    iter_validated_state_pages,
)

SCOPE = StateScope("chunks")


def _open_adapter(tmp_path: Path) -> DuckDBAdapter:
    cfg = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "state.duckdb")}
    )
    adapter = create_adapter(cfg, project_dir=tmp_path)
    assert isinstance(adapter, DuckDBAdapter)
    return adapter


def _seed(adapter: DuckDBAdapter, keys: list[str], *, version: str = "v1") -> None:
    adapter.upsert_state(
        SCOPE, [StateRecord(key, f"fp-{key}", version) for key in keys]
    )


def _keys(count: int) -> list[str]:
    return [f"k{index:04d}" for index in range(count)]


# ─── ordered paged iteration ────────────────────────────────────────────────


def test_multi_page_iteration_is_ordered_and_complete(tmp_path: Path) -> None:
    with _open_adapter(tmp_path) as adapter:
        _seed(adapter, _keys(25))
        collected: list[StatePageRecord] = []
        with adapter.state_page_reader(SCOPE, page_size=10) as reader:
            page = reader.fetch_page(None)
            sizes = [len(page.records)]
            collected.extend(page.records)
            while page.next_cursor is not None:
                page = reader.fetch_page(page.next_cursor)
                sizes.append(len(page.records))
                collected.extend(page.records)
        assert sizes == [10, 10, 5]
        assert [record.record_key for record in collected] == _keys(25)
        assert all(record.committed_at is not None for record in collected)
        assert collected[0].input_fingerprint == "fp-k0000"
        assert collected[0].code_version == "v1"


def test_exact_page_multiple_ends_with_empty_final_page(tmp_path: Path) -> None:
    with _open_adapter(tmp_path) as adapter:
        _seed(adapter, _keys(20))
        with adapter.state_page_reader(SCOPE, page_size=10) as reader:
            pages = list(iter_validated_state_pages(reader))
        assert [len(page) for page in pages] == [10, 10]


def test_pages_are_scoped_to_one_snapshot_under_own_deletes(tmp_path: Path) -> None:
    """Deleting rows between pages — the production stale-delete pattern —
    must not skip or repeat records within an open reader."""
    with _open_adapter(tmp_path) as adapter:
        _seed(adapter, _keys(30))
        collected: list[str] = []
        with adapter.state_page_reader(SCOPE, page_size=10) as reader:
            page = reader.fetch_page(None)
            collected.extend(record.record_key for record in page.records)
            adapter.delete_state(SCOPE, [record.record_key for record in page.records])
            adapter.delete_state(SCOPE, _keys(30)[20:25])
            while page.next_cursor is not None:
                page = reader.fetch_page(page.next_cursor)
                collected.extend(record.record_key for record in page.records)
        assert collected == _keys(30)
        assert sorted(adapter.fetch_state(SCOPE)) == _keys(30)[10:20] + _keys(30)[25:]


def test_cursor_retry_returns_identical_page(tmp_path: Path) -> None:
    with _open_adapter(tmp_path) as adapter:
        _seed(adapter, _keys(15))
        with adapter.state_page_reader(SCOPE, page_size=5) as reader:
            first = reader.fetch_page(None)
            assert first.next_cursor is not None
            once = reader.fetch_page(first.next_cursor)
            again = reader.fetch_page(first.next_cursor)
            assert once == again


def test_foreign_and_malformed_cursors_are_rejected(tmp_path: Path) -> None:
    with _open_adapter(tmp_path) as adapter:
        _seed(adapter, _keys(6))
        with adapter.state_page_reader(SCOPE, page_size=2) as reader:
            stolen = reader.fetch_page(None).next_cursor
        assert stolen is not None
        with adapter.state_page_reader(SCOPE, page_size=2) as reader:
            with pytest.raises(AdapterError, match="does not belong"):
                reader.fetch_page(stolen)
            with pytest.raises(AdapterError, match="cursor"):
                reader.fetch_page("not-a-cursor")
            with pytest.raises(AdapterError, match="cursor"):
                reader.fetch_page("x" * 9000)


def test_closed_reader_rejects_further_pages(tmp_path: Path) -> None:
    with _open_adapter(tmp_path) as adapter:
        _seed(adapter, _keys(4))
        with adapter.state_page_reader(SCOPE, page_size=2) as reader:
            reader.fetch_page(None)
        assert reader.closed
        with pytest.raises(AdapterError, match="closed"):
            reader.fetch_page(None)


def test_early_close_is_deterministic_and_reusable(tmp_path: Path) -> None:
    """Abandoning a reader mid-stream releases its transaction so state
    mutation continues normally afterwards."""
    with _open_adapter(tmp_path) as adapter:
        _seed(adapter, _keys(12))
        with adapter.state_page_reader(SCOPE, page_size=5) as reader:
            reader.fetch_page(None)
        adapter.clear_state(SCOPE)
        assert adapter.fetch_state(SCOPE) == {}


# ─── absence probe (stale discovery) ────────────────────────────────────────


def _materialize_upstream(adapter: DuckDBAdapter, keys: list[str]) -> None:
    adapter.materialize_full(
        "chunks_upstream",
        pl.DataFrame({"chunk_id": keys, "text": [f"t-{key}" for key in keys]}),
    )


def test_absence_probe_streams_exactly_the_stale_complement(tmp_path: Path) -> None:
    with _open_adapter(tmp_path) as adapter:
        keys = _keys(40)
        _seed(adapter, keys)
        _materialize_upstream(adapter, keys[10:30])
        reconciler = BoundedReconciler(
            adapter, SCOPE, code_version="v1", page_size=7
        )
        pages = list(
            reconciler.iter_stale_pages(
                upstream_table="chunks_upstream", key_column="chunk_id"
            )
        )
        stale = [record.record_key for page in pages for record in page]
        assert stale == keys[:10] + keys[30:]
        assert all(len(page) <= 7 for page in pages)
        assert pages[0][0].input_fingerprint == "fp-k0000"


def test_empty_upstream_marks_every_state_row_stale(tmp_path: Path) -> None:
    with _open_adapter(tmp_path) as adapter:
        _seed(adapter, _keys(23))
        _materialize_upstream(adapter, [])
        reconciler = BoundedReconciler(
            adapter, SCOPE, code_version="v1", page_size=10
        )
        pages = list(
            reconciler.iter_stale_pages(
                upstream_table="chunks_upstream", key_column="chunk_id"
            )
        )
        assert [record.record_key for page in pages for record in page] == _keys(23)


def test_empty_state_yields_no_stale_pages(tmp_path: Path) -> None:
    with _open_adapter(tmp_path) as adapter:
        _materialize_upstream(adapter, _keys(5))
        reconciler = BoundedReconciler(
            adapter, SCOPE, code_version="v1", page_size=10
        )
        assert (
            list(
                reconciler.iter_stale_pages(
                    upstream_table="chunks_upstream", key_column="chunk_id"
                )
            )
            == []
        )


def test_missing_probe_relation_fails_rather_than_deleting(tmp_path: Path) -> None:
    with _open_adapter(tmp_path) as adapter:
        _seed(adapter, _keys(3))
        with pytest.raises(AdapterError, match="absence probe"):
            with adapter.state_page_reader(
                SCOPE,
                page_size=5,
                absent_from=StateAbsenceProbe(
                    table="missing_table", key_column="chunk_id"
                ),
            ):
                pass


# ─── bounded subset lookups ─────────────────────────────────────────────────


def test_fetch_state_subset_returns_exactly_requested_keys(tmp_path: Path) -> None:
    with _open_adapter(tmp_path) as adapter:
        _seed(adapter, _keys(10))
        subset = adapter.fetch_state_subset(SCOPE, ["k0002", "k0007", "missing"])
        assert subset == {
            "k0002": StateValue("fp-k0002", "v1"),
            "k0007": StateValue("fp-k0007", "v1"),
        }
        assert adapter.fetch_state_subset(SCOPE, []) == {}


def test_fetch_state_subset_rejects_duplicate_and_empty_keys(tmp_path: Path) -> None:
    with _open_adapter(tmp_path) as adapter:
        with pytest.raises(AdapterError, match="duplicate"):
            adapter.fetch_state_subset(SCOPE, ["a", "a"])
        with pytest.raises(AdapterError, match="empty"):
            adapter.fetch_state_subset(SCOPE, [""])


# ─── bounded residency ──────────────────────────────────────────────────────


class _ResidencyProbeAdapter(DuckDBAdapter):
    """Real DuckDB storage plus per-call residency accounting, with the
    full-scope escape hatch forbidden."""

    max_subset = 0
    max_page = 0

    def fetch_state(self, scope: StateScope) -> dict[str, StateValue]:
        raise AssertionError(
            "bounded reconciliation must never fetch the full state scope"
        )

    def _fetch_state_subset(self, scope, record_keys):  # type: ignore[no-untyped-def]
        type(self).max_subset = max(type(self).max_subset, len(record_keys))
        return super()._fetch_state_subset(scope, record_keys)

    def _open_state_page_reader(self, request):  # type: ignore[no-untyped-def]
        inner = super()._open_state_page_reader(request)
        original_fetch = inner.fetch_page

        def fetch_page(cursor: str | None = None) -> StatePage:
            page = original_fetch(cursor)
            type(self).max_page = max(type(self).max_page, len(page.records))
            return page

        cast(Any, inner).fetch_page = fetch_page
        return inner


def test_reconciliation_residency_is_bounded_by_page_size(tmp_path: Path) -> None:
    """State 25x larger than the page: every observation stays page-sized and
    the full-scope read path is never taken."""
    cfg = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "state.duckdb")}
    )
    _ResidencyProbeAdapter.max_subset = 0
    _ResidencyProbeAdapter.max_page = 0
    with _ResidencyProbeAdapter(cfg, project_dir=tmp_path) as adapter:
        total = _keys(500)
        adapter.upsert_state(
            SCOPE, [StateRecord(key, f"fp-{key}", "v1") for key in total]
        )
        _materialize_upstream(adapter, total[:250])
        reconciler = BoundedReconciler(
            adapter, SCOPE, code_version="v1", page_size=20
        )
        for offset in range(0, 250, 20):
            batch = [
                UpstreamRecord(key, f"fp-{key}")
                for key in total[offset : offset + 20]
            ]
            prior = reconciler.prior_state_for(batch)
            outcome = reconciler.classify(batch, prior=prior)
            assert len(outcome.unchanged) == len(batch)
        stale = [
            record.record_key
            for page in reconciler.iter_stale_pages(
                upstream_table="chunks_upstream", key_column="chunk_id"
            )
            for record in page
        ]
        assert stale == total[250:]
    assert _ResidencyProbeAdapter.max_subset <= 20
    assert _ResidencyProbeAdapter.max_page <= 20


# ─── merge classification ───────────────────────────────────────────────────


def test_classification_orders_new_changed_unchanged(tmp_path: Path) -> None:
    prior = {
        "b": StateValue("fp-b", "v1"),
        "c": StateValue("stale-fp", "v1"),
        "d": StateValue("fp-d", "v0"),
    }
    batch = [
        UpstreamRecord("a", "fp-a"),
        UpstreamRecord("b", "fp-b"),
        UpstreamRecord("c", "fp-c"),
        UpstreamRecord("d", "fp-d"),
    ]
    outcome = classify_batch(batch, prior=prior, code_version="v1")
    assert [record.record_key for record in outcome.new] == ["a"]
    assert [record.record_key for record in outcome.changed] == ["c", "d"]
    assert [record.record_key for record in outcome.unchanged] == ["b"]

    forced = classify_batch(batch, prior=prior, code_version="v1", force_publish=True)
    assert [record.record_key for record in forced.new] == ["a"]
    assert [record.record_key for record in forced.changed] == ["b", "c", "d"]
    assert forced.unchanged == ()


# ─── core stream validation ─────────────────────────────────────────────────


def _scripted_reader(pages: list[StatePage], *, page_size: int = 10) -> StatePageReader:
    script = list(pages)

    def fetch(cursor: str | None) -> StatePage:
        return script.pop(0)

    return StatePageReader(page_size=page_size, fetch=fetch, close=lambda: None)


def _record(key: str) -> StatePageRecord:
    return StatePageRecord(key, f"fp-{key}", "v1")


def test_out_of_order_page_is_rejected() -> None:
    reader = _scripted_reader(
        [StatePage(records=(_record("b"), _record("a")), next_cursor=None)]
    )
    with pytest.raises(AdapterError, match="strictly ordered"):
        list(iter_validated_state_pages(reader))


def test_duplicate_key_within_page_is_rejected() -> None:
    reader = _scripted_reader(
        [StatePage(records=(_record("a"), _record("a")), next_cursor=None)]
    )
    with pytest.raises(AdapterError, match="strictly ordered"):
        list(iter_validated_state_pages(reader))


def test_backwards_page_boundary_is_rejected() -> None:
    reader = _scripted_reader(
        [
            StatePage(records=(_record("a"), _record("b")), next_cursor="c1"),
            StatePage(records=(_record("b"),), next_cursor=None),
        ]
    )
    with pytest.raises(AdapterError, match="across page boundaries"):
        list(iter_validated_state_pages(reader))


def test_empty_non_final_page_is_rejected() -> None:
    reader = _scripted_reader(
        [StatePage(records=(), next_cursor="c1")]
    )
    with pytest.raises(AdapterError, match="not final"):
        list(iter_validated_state_pages(reader))


def test_non_advancing_cursor_is_rejected() -> None:
    reader = _scripted_reader(
        [
            StatePage(records=(_record("a"),), next_cursor="c1"),
            StatePage(records=(_record("b"),), next_cursor="c1"),
            StatePage(records=(_record("c"),), next_cursor=None),
        ]
    )
    with pytest.raises(AdapterError, match="did not advance"):
        list(iter_validated_state_pages(reader))


def test_null_record_key_is_rejected_at_construction() -> None:
    with pytest.raises(AdapterError, match="record_key"):
        StatePageRecord("", "fp", "v1")


def test_oversized_page_is_rejected() -> None:
    reader = _scripted_reader(
        [StatePage(records=(_record("a"), _record("b")), next_cursor=None)],
        page_size=1,
    )
    with pytest.raises(AdapterError, match="exceeded"):
        list(iter_validated_state_pages(reader))


# ─── fenced atomic scope replacement ────────────────────────────────────────


def test_replace_state_scope_streams_batches_atomically(tmp_path: Path) -> None:
    with _open_adapter(tmp_path) as adapter:
        _seed(adapter, ["old1", "old2"])
        replacement = _keys(5)
        written = adapter.replace_state_scope(
            SCOPE,
            (
                [StateRecord(key, f"fp-{key}", "v2") for key in replacement[offset : offset + 2]]
                for offset in range(0, 5, 2)
            ),
        )
        assert written == 5
        state = adapter.fetch_state(SCOPE)
        assert sorted(state) == replacement
        assert state["k0000"] == StateValue("fp-k0000", "v2")


def test_replace_state_scope_with_no_batches_clears_the_scope(tmp_path: Path) -> None:
    with _open_adapter(tmp_path) as adapter:
        _seed(adapter, _keys(4))
        assert adapter.replace_state_scope(SCOPE, iter([])) == 0
        assert adapter.fetch_state(SCOPE) == {}


def test_replace_state_scope_rolls_back_cross_batch_duplicates(tmp_path: Path) -> None:
    with _open_adapter(tmp_path) as adapter:
        _seed(adapter, ["survivor"])
        with pytest.raises(AdapterError, match="duplicate record keys"):
            adapter.replace_state_scope(
                SCOPE,
                iter(
                    [
                        [StateRecord("dup", "fp1", "v2")],
                        [StateRecord("dup", "fp2", "v2")],
                    ]
                ),
            )
        assert sorted(adapter.fetch_state(SCOPE)) == ["survivor"]


def test_replace_state_scope_preserves_other_scopes(tmp_path: Path) -> None:
    other = StateScope("other_model")
    with _open_adapter(tmp_path) as adapter:
        _seed(adapter, ["mine"])
        adapter.upsert_state(other, [StateRecord("theirs", "fp", "v1")])
        adapter.replace_state_scope(
            SCOPE, iter([[StateRecord("new", "fp", "v2")]])
        )
        assert sorted(adapter.fetch_state(SCOPE)) == ["new"]
        assert sorted(adapter.fetch_state(other)) == ["theirs"]


def test_fenced_replace_succeeds_only_for_the_live_claim(tmp_path: Path) -> None:
    with _open_adapter(tmp_path) as adapter:
        _seed(adapter, ["old"])
        coordinator = ServingCoordinator(adapter)
        lease = coordinator.acquire_publish(
            SCOPE, expected_code_version="v2", config_fingerprint="cfg"
        )
        fence = StateScopeFence(lease.publication_id, lease.fencing_token)
        adapter.replace_state_scope(
            SCOPE, iter([[StateRecord("new", "fp", "v2")]]), fence=fence
        )
        assert sorted(adapter.fetch_state(SCOPE)) == ["new"]

        # Administrative recovery advances the fence; the old claim must fail
        # without mutating the replaced state.
        coordinator.recover(SCOPE, owner_terminated=True)
        with pytest.raises(StaleStateFenceError):
            adapter.replace_state_scope(
                SCOPE, iter([[StateRecord("zombie", "fp", "v3")]]), fence=fence
            )
        assert sorted(adapter.fetch_state(SCOPE)) == ["new"]


def test_fenced_replace_without_ledger_fails_closed(tmp_path: Path) -> None:
    with _open_adapter(tmp_path) as adapter:
        _seed(adapter, ["old"])
        fence = StateScopeFence("nonexistent", 1)
        with pytest.raises(StaleStateFenceError):
            adapter.replace_state_scope(
                SCOPE, iter([[StateRecord("new", "fp", "v2")]]), fence=fence
            )
        assert sorted(adapter.fetch_state(SCOPE)) == ["old"]


def test_fence_values_are_validated() -> None:
    with pytest.raises(AdapterError, match="publication_id"):
        StateScopeFence("", 1)
    with pytest.raises(AdapterError, match="fencing_token"):
        StateScopeFence("p", 0)


# ─── capability gating ──────────────────────────────────────────────────────


class _EagerOnlyAdapter(DuckDBAdapter):
    @classmethod
    def capabilities(cls) -> frozenset[WarehouseCapability]:
        return super().capabilities() - {
            WarehouseCapability.PAGED_STATE_RECONCILIATION,
            WarehouseCapability.ATOMIC_STATE_SCOPE_REPLACE,
        }


def test_operations_require_their_capabilities(tmp_path: Path) -> None:
    cfg = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "state.duckdb")}
    )
    adapter = _EagerOnlyAdapter(cfg, project_dir=tmp_path)
    with pytest.raises(AdapterCapabilityError, match="paged_state_reconciliation"):
        adapter.fetch_state_subset(SCOPE, ["a"])
    with pytest.raises(AdapterCapabilityError, match="paged_state_reconciliation"):
        with adapter.state_page_reader(SCOPE, page_size=10):
            pass
    with pytest.raises(AdapterCapabilityError, match="atomic_state_scope_replace"):
        adapter.replace_state_scope(SCOPE, iter([]))
