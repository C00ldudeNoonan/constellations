"""Where stel is allowed to run things at the same time (issue #432).

Two properties, both of which were quietly false and neither of which any
existing test could see:

- A streaming stage must not hold the shared adapter lock for its whole run.
  `_SerializedAdapter` serializes warehouse access so models can share one
  connection under `--threads N`, but it special-cased `table_snapshot` and
  held the lock across the entire context. A streaming stage keeps that context
  open for its whole run — provider calls and publishes included — so the first
  model to open a snapshot blocked every other one until it finished. That is
  serialized *execution*, not serialized I/O, and it arrived with the
  bounded-memory work rather than being there all along.

- `embed:` must overlap its provider batches. It was the only provider stage
  with no executor-level concurrency: batches went out one at a time and
  blocked, so the only overlap was whatever a provider arranged inside a single
  call.

Written to be deterministic rather than timing-based. The lock test asks
whether the lock is *acquirable*, not whether two threads happened to
interleave; the concurrency test uses a barrier, so it either observes real
overlap or times out rather than flaking on a slow machine.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from stel.adapters import create_adapter, parse_warehouse_config
from stel.runner import _SerializedAdapter


def _adapter(tmp_path: Path) -> Any:
    return create_adapter(
        parse_warehouse_config(
            {"type": "duckdb", "path": str(tmp_path / "w.duckdb"), "schema": "s"}
        )
    )


# ─── the shared adapter lock ────────────────────────────────────────────────


def test_a_snapshot_does_not_hold_the_adapter_lock_while_streaming(
    tmp_path: Path,
) -> None:
    """The #432 regression, asserted where it is observable.

    Under `--threads N` every model shares one `_SerializedAdapter`. If the
    lock is held for the snapshot's whole context, no other model can make a
    single warehouse call until the streaming stage finishes — which for embed
    or chunk is the entire model, provider latency included.
    """
    with _adapter(tmp_path) as adapter:
        adapter.materialize_full(
            "rows", pl.DataFrame({"id": [f"r{i}" for i in range(20)]})
        )
        lock = threading.Lock()
        guarded = _SerializedAdapter(adapter, lock)

        with guarded.table_snapshot("rows", batch_size=5) as snapshot:
            # Mid-stream: consume one batch, then check the lock is free.
            batches = iter(snapshot)
            first = next(batches)
            assert first.num_rows > 0
            assert lock.acquire(blocking=False), (
                "the adapter lock is held while a snapshot streams, so any "
                "other model on --threads N is blocked for this stage's whole "
                "run (issue #432)"
            )
            lock.release()
            # And the rest of the stream still works after that check.
            assert sum(batch.num_rows for batch in batches) == 15


def test_the_snapshot_open_is_still_guarded(tmp_path: Path) -> None:
    """The narrowing must not remove the guard entirely: opening the snapshot
    creates a cursor from the shared connection, which is the thing the lock
    protects."""
    with _adapter(tmp_path) as adapter:
        adapter.materialize_full("rows", pl.DataFrame({"id": ["a"]}))
        lock = threading.Lock()
        held_during_open: list[bool] = []
        real_snapshot = adapter.table_snapshot

        def spy(*args: Any, **kwargs: Any) -> Any:
            held_during_open.append(lock.locked())
            return real_snapshot(*args, **kwargs)

        adapter.table_snapshot = spy  # type: ignore[method-assign]
        guarded = _SerializedAdapter(adapter, lock)
        with guarded.table_snapshot("rows") as snapshot:
            list(snapshot)

    assert held_during_open == [True], (
        "the snapshot open ran without the lock, but creating its cursor "
        "touches the shared connection the lock protects"
    )


def test_a_failing_snapshot_still_closes_under_the_lock(tmp_path: Path) -> None:
    """The close is the other half of the open. An exception mid-stream must
    not leave the snapshot unclosed just because the lock moved."""
    with _adapter(tmp_path) as adapter:
        adapter.materialize_full(
            "rows", pl.DataFrame({"id": [f"r{i}" for i in range(10)]})
        )
        lock = threading.Lock()
        guarded = _SerializedAdapter(adapter, lock)

        with pytest.raises(RuntimeError, match="deliberate"):
            with guarded.table_snapshot("rows", batch_size=2) as snapshot:
                next(iter(snapshot))
                raise RuntimeError("deliberate")

        # The lock is released, so the run can carry on.
        assert lock.acquire(blocking=False)
        lock.release()


# ─── embed provider concurrency ─────────────────────────────────────────────


def _embed_project(root: Path, *, docs: int, batch_size: int, max_concurrent: int) -> Path:
    import json

    project = root / "proj"
    (project / "models").mkdir(parents=True)
    (project / "sources").mkdir()
    (project / "data").mkdir()
    (project / "stel_project.yml").write_text(
        "name: c\nversion: '0.1.0'\nprofile: c\n", encoding="utf-8"
    )
    (project / "profiles.yml").write_text(
        "c:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n        schema: d\n",
        encoding="utf-8",
    )
    (project / "sources" / "s.yml").write_text(
        "version: 2\nsources:\n  - name: documents\n    path: data\n"
        "    file_pattern: '*.json'\n",
        encoding="utf-8",
    )
    (project / "models" / "m.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: document_registry\n"
        "    source: ref('documents')\n"
        "    extraction:\n      backend: json\n"
        "      options:\n        fields: [title, body]\n"
        "    materialization: incremental\n"
        "  - name: document_chunks\n"
        "    depends_on: [ref('document_registry')]\n"
        "    chunk:\n      text_field: body\n      chunk_size: 1000\n"
        "      chunk_overlap: 0\n"
        "    materialization: incremental\n"
        "  - name: document_embeddings\n"
        "    depends_on: [ref('document_chunks')]\n"
        "    embed:\n      provider: deterministic\n      model: contract-v1\n"
        "      text_field: text\n      id_field: chunk_id\n"
        "      vector_field: embedding\n      dimensions: 4\n"
        f"      batch_size: {batch_size}\n"
        f"      max_concurrent: {max_concurrent}\n"
        "      flush_every: 1000\n"
        "    materialization: incremental\n",
        encoding="utf-8",
    )
    for index in range(docs):
        (project / "data" / f"d{index}.json").write_text(
            json.dumps({"title": f"t{index}", "body": f"body {index}"}),
            encoding="utf-8",
        )
    return project


def _run_embed_recording_overlap(
    project: Path, monkeypatch: pytest.MonkeyPatch, *, expected_batches: int
) -> int:
    """Run the embed model, returning the peak number of concurrent provider calls.

    A barrier rather than a sleep: each call waits for `expected_batches` peers
    before returning, so genuine overlap passes immediately and its absence
    fails on the barrier timeout instead of on a tuned duration.
    """
    from stel.providers.deterministic import DeterministicEmbeddingProvider
    from stel.runner import run_project

    barrier = threading.Barrier(expected_batches, timeout=10)
    active = 0
    peak = 0
    guard = threading.Lock()
    original = DeterministicEmbeddingProvider._embed

    def instrumented(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass  # fewer callers than expected: the assertion below reports it
        try:
            return original(self, *args, **kwargs)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(DeterministicEmbeddingProvider, "_embed", instrumented)
    run_project(project, select="document_registry")
    run_project(project, select="document_chunks")
    run_project(project, select="document_embeddings")
    return peak


def test_embed_issues_provider_batches_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of #432's embed half: four batches must be in flight at once,
    not four one after another."""
    project = _embed_project(tmp_path, docs=4, batch_size=1, max_concurrent=4)
    peak = _run_embed_recording_overlap(project, monkeypatch, expected_batches=4)
    assert peak == 4, (
        f"embed reached only {peak} concurrent provider call(s) with "
        "max_concurrent=4; batches are still going out one at a time (#432)"
    )


def test_embed_max_concurrent_one_stays_sequential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The knob has to be able to turn the pool off — a provider with a strict
    serial rate limit needs that, and the barrier proves nothing overlapped."""
    from stel.providers.deterministic import DeterministicEmbeddingProvider
    from stel.runner import run_project

    project = _embed_project(tmp_path, docs=4, batch_size=1, max_concurrent=1)
    active = 0
    peak = 0
    guard = threading.Lock()
    original = DeterministicEmbeddingProvider._embed

    def instrumented(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        try:
            return original(self, *args, **kwargs)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(DeterministicEmbeddingProvider, "_embed", instrumented)
    run_project(project, select="document_registry")
    run_project(project, select="document_chunks")
    run_project(project, select="document_embeddings")
    assert peak == 1


def test_embed_max_concurrent_does_not_move_code_version(tmp_path: Path) -> None:
    """Throughput tuning must never re-embed a corpus. `max_concurrent` joins
    `batch_size`, `max_retries` and `flush_every` outside the identity."""
    from stel.config.model import EmbedConfig
    from stel.versioning import compute_code_version

    def version(max_concurrent: int) -> str:
        return compute_code_version(
            extraction=None,
            transform=None,
            embed=EmbedConfig(
                provider="deterministic",
                model="contract-v1",
                dimensions=4,
                max_concurrent=max_concurrent,
            ),
            depends_on=["chunks"],
            project_dir=tmp_path,
        )

    assert version(8) == version(64)


def test_a_call_cap_is_not_overrun_by_concurrent_batches() -> None:
    """The #432 review finding: admission and reservation must be one step.

    `ensure_headroom` then charge-on-return is a check-then-act. Sequentially
    that is fine — nothing runs between the two — but with batches in flight
    concurrently every worker passes admission against the same pre-charge
    total, and a `max_api_calls: 1` cap silently buys as many calls as there
    are workers.
    """
    from stel.budget import (
        BudgetExceededError,
        BudgetGuard,
        BudgetLedger,
        LLMBudgetConfig,
    )

    guard = BudgetGuard(
        BudgetLedger(LLMBudgetConfig(max_api_calls=1), scope="model"), None
    )
    admitted = 0
    refused = 0
    lock = threading.Lock()
    start = threading.Barrier(4, timeout=10)

    def worker() -> None:
        nonlocal admitted, refused
        start.wait()
        try:
            guard.reserve_calls(1)
        except BudgetExceededError:
            with lock:
                refused += 1
        else:
            with lock:
                admitted += 1

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert admitted == 1, (
        f"{admitted} workers were admitted against max_api_calls=1; a spending "
        "cap that admits more than it allows is not a cap"
    )
    assert refused == 3


def test_settling_charges_only_what_the_provider_billed_beyond_the_reservation() -> None:
    """A fan-out larger than estimated still lands on the ledger, and an
    over-estimate is left conservative rather than refunded — failing safe."""
    from stel.budget import BudgetGuard, BudgetLedger, LLMBudgetConfig

    ledger = BudgetLedger(LLMBudgetConfig(max_api_calls=100), scope="model")
    guard = BudgetGuard(ledger, None)

    guard.reserve_calls(2)
    guard.settle_calls(reserved=2, actual=5)
    assert ledger.snapshot()["api_calls"] == 5

    guard.reserve_calls(4)
    guard.settle_calls(reserved=4, actual=1)
    assert ledger.snapshot()["api_calls"] == 9, "an over-estimate must not refund"
