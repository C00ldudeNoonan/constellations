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
        held: list[bool] = []

        class _WatchingLock:
            def __init__(self) -> None:
                self._lock = threading.Lock()

            def __enter__(self) -> None:
                self._lock.acquire()
                held.append(True)

            def __exit__(self, *exc: object) -> None:
                self._lock.release()

        guarded = _SerializedAdapter(adapter, _WatchingLock())  # type: ignore[arg-type]
        with guarded.table_snapshot("rows") as snapshot:
            opened_under_lock = len(held)
            list(snapshot)
    assert opened_under_lock >= 1, "the snapshot open was not guarded"


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
