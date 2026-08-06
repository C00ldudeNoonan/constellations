"""Fetch-staging temp directory cleanup (#273).

A killed/crashed process never runs `TemporaryDirectory.__exit__`, leaking
everything an extraction run fetched. Two mitigations: per-document cleanup
bounds a *live* run's peak disk use to in-flight documents, and a startup
sweep self-heals directories a *dead* run left behind.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from dbt_ml.execution import extraction as extraction_module
from dbt_ml.runner import run_project
from dbt_ml.synth import generate_invoices

# ── _cleanup_fetched: per-document staging removal ───────────────────────────


def test_cleanup_fetched_removes_direct_file(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    fetched = work_dir / "doc1.json"
    fetched.write_text("{}")

    extraction_module._cleanup_fetched(fetched, work_dir)

    assert not fetched.exists()


def test_cleanup_fetched_removes_per_document_subdirectory(tmp_path: Path) -> None:
    # sources/local.py nests each fetch under work_dir/<document_id>/<name>.
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    doc_dir = work_dir / "doc1"
    doc_dir.mkdir()
    fetched = doc_dir / "invoice.json"
    fetched.write_text("{}")

    extraction_module._cleanup_fetched(fetched, work_dir)

    assert not doc_dir.exists()


def test_cleanup_fetched_ignores_path_outside_work_dir(tmp_path: Path) -> None:
    # fetch()'s contract guarantees paths live under work_dir; a violation must
    # be a safe no-op, never a deletion outside the staging area.
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")

    extraction_module._cleanup_fetched(outside, work_dir)

    assert outside.exists()


# ── _sweep_stale_fetch_dirs: startup self-healing for killed runs ───────────


def test_sweep_removes_only_stale_prefixed_dirs(tmp_path: Path) -> None:
    stale = tmp_path / "dbt_ml_fetch_dead"
    stale.mkdir()
    (stale / "leftover.bin").write_bytes(b"x" * 100)
    old = time.time() - 999_999
    os.utime(stale, (old, old))

    live = tmp_path / "dbt_ml_fetch_live"
    live.mkdir()  # fresh mtime: looks like an in-progress run

    unrelated = tmp_path / "some_other_dir"
    unrelated.mkdir()
    old_unrelated = time.time() - 999_999
    os.utime(unrelated, (old_unrelated, old_unrelated))

    extraction_module._sweep_stale_fetch_dirs(tmp_path, max_age_seconds=3600)

    assert not stale.exists()
    assert live.exists()
    assert unrelated.exists()  # never touch anything without the dbt-ml prefix


def test_sweep_tolerates_missing_root(tmp_path: Path) -> None:
    extraction_module._sweep_stale_fetch_dirs(
        tmp_path / "does_not_exist", max_age_seconds=3600
    )  # must not raise


def test_sweep_once_runs_a_single_time(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[None] = []
    monkeypatch.setattr(
        extraction_module, "_sweep_stale_fetch_dirs", lambda *a, **k: calls.append(None)
    )
    monkeypatch.setattr(extraction_module, "_swept_stale_fetch_dirs", False)

    extraction_module._sweep_stale_fetch_dirs_once()
    extraction_module._sweep_stale_fetch_dirs_once()

    assert len(calls) == 1


# ── end-to-end: a real incremental run never accumulates staged files ───────


def test_incremental_extraction_bounds_peak_fetch_staging(
    tmp_path: Path, example_project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    shutil.copytree(
        example_project_dir,
        project,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    generate_invoices(5, project / "data" / "invoices", seed=1)

    real_cleanup = extraction_module._cleanup_fetched
    max_entries_seen = 0

    def _spy(path: Path, work_dir: Path) -> None:
        real_cleanup(path, work_dir)
        nonlocal max_entries_seen
        max_entries_seen = max(max_entries_seen, len(list(work_dir.iterdir())))

    monkeypatch.setattr(extraction_module, "_cleanup_fetched", _spy)

    # threads=1 (default): sequential processing, so an empty directory after
    # every cleanup call proves peak usage never exceeds one in-flight document
    # — not the whole 5-document corpus (#273).
    run_project(project)

    assert max_entries_seen == 0
