"""llm map models publish at flush boundaries (issue #401's pattern).

Of every stage in the pipeline this is the one where an all-or-nothing write
costs the most: an llm map model spends one provider call per input, so a
failure or budget exhaustion near the end used to discard every completion
already paid for. The executor even said so in a comment — "this model writes
once at the end, so nothing is published and state is unchanged" — which is
accurate about the mechanism and expensive about the consequence.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import duckdb
import pytest

from stel.execution.contracts import RunError
from stel.runner import run_project

INPUTS = 7


def _project(
    tmp_path: Path,
    *,
    flush_every: int,
    cardinality: str = "one",
    max_concurrent: int = 1,
    max_api_calls: int | None = None,
) -> Path:
    project = tmp_path / "project"
    project.mkdir(parents=True)
    (project / "stel_project.yml").write_text(
        "name: p\nversion: '0.1.0'\nprofile: p\n", encoding="utf-8"
    )
    budget = (
        "      llm:\n"
        "        provider: deterministic\n"
        "        model: m\n"
        "        budget:\n"
        f"          max_api_calls: {max_api_calls}\n"
        if max_api_calls is not None
        else ""
    )
    (project / "profiles.yml").write_text(
        "p:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n"
        "        schema: docs\n"
        f"{budget}",
        encoding="utf-8",
    )
    (project / "sources").mkdir()
    (project / "sources" / "s.yml").write_text(
        "version: 2\nsources:\n  - name: documents\n    path: data\n"
        "    file_pattern: '*.json'\n",
        encoding="utf-8",
    )
    (project / "models").mkdir()
    (project / "models" / "m.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: registry\n    source: ref('documents')\n"
        "    extraction:\n      backend: json\n      options:\n        fields: [body]\n"
        "    materialization: incremental\n"
        "  - name: facts\n    depends_on: [ref('registry')]\n"
        "    llm:\n      input_field: body\n      prompt: p\n"
        "      provider: deterministic\n      model: m\n"
        "      id_field: document_id\n"
        f"      max_concurrent: {max_concurrent}\n"
        f"      flush_every: {flush_every}\n"
        f"      output_cardinality: {cardinality}\n"
        "    fields:\n      - {name: sentiment, type: string}\n"
        "    materialization: incremental\n",
        encoding="utf-8",
    )
    data = project / "data"
    data.mkdir()
    for index in range(INPUTS):
        (data / f"d{index}.json").write_text(
            json.dumps({"body": f"body {index}"}), encoding="utf-8"
        )
    return project


def _query(project: Path, sql: str) -> list[tuple[Any, ...]]:
    connection = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def _fact_count(project: Path) -> int:
    return int(_query(project, 'SELECT COUNT(*) FROM "db".docs.facts')[0][0])


def _fail_after(calls: int, monkeypatch: pytest.MonkeyPatch) -> None:
    import stel.execution.llm as llm_module

    original = llm_module.execute_map_item
    served = 0

    def limited(*args: Any, **kwargs: Any) -> Any:
        nonlocal served
        if served >= calls:
            raise RuntimeError("provider exhausted")
        served += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(llm_module, "execute_map_item", limited)


# ─── paid completions survive a later failure ───────────────────────────────


def test_a_failure_mid_run_keeps_earlier_flushes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, flush_every=2)
    run_project(project, select="registry")
    _fail_after(5, monkeypatch)

    with pytest.raises(RunError):
        run_project(project, select="facts")

    # Two windows of two published; the third died mid-window and is discarded.
    assert _fact_count(project) == 4


def test_the_rerun_only_pays_for_what_was_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, flush_every=2)
    run_project(project, select="registry")
    _fail_after(5, monkeypatch)
    with pytest.raises(RunError):
        run_project(project, select="facts")
    monkeypatch.undo()

    [resumed] = run_project(project, select="facts")

    assert resumed.metrics["provider_calls"] == INPUTS - 4
    assert _fact_count(project) == INPUTS


def test_concurrent_llm_rows_do_not_overrun_the_call_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wire the atomic reservation through the native llm executor itself.

    The barrier makes all four workers finish the old check-before-call before
    any can report usage. Without reservation all four provider calls run
    against a cap of one; with it, exactly one reaches the provider.
    """
    import stel.execution.llm as llm_module

    project = _project(
        tmp_path,
        flush_every=4,
        max_concurrent=4,
        max_api_calls=1,
    )
    run_project(project, select="registry")
    original = llm_module.execute_map_item
    start = threading.Barrier(4, timeout=10)

    def coordinated(
        content: str,
        runtime: Any,
        *,
        budget: Any = None,
    ) -> Any:
        start.wait()
        return original(content, runtime, budget=budget)

    monkeypatch.setattr(llm_module, "execute_map_item", coordinated)

    [result] = run_project(project, select="facts")

    assert result.status == "budget_exceeded"
    assert result.metrics["provider_calls"] == 1
    assert result.metrics["api_calls"] == 1


def test_whole_corpus_completes(tmp_path: Path) -> None:
    project = _project(tmp_path, flush_every=2)

    run_project(project)

    assert _fact_count(project) == INPUTS
    assert all(
        row[0] is not None
        for row in _query(project, 'SELECT sentiment FROM "db".docs.facts')
    )


def test_fan_out_windows_do_not_disturb_earlier_parents(tmp_path: Path) -> None:
    """`output_cardinality: many` publishes through replace_children, which is
    scoped to the window's parents. A window that replaced children globally
    would delete every row an earlier window had just published."""
    project = _project(tmp_path, flush_every=2, cardinality="many")

    run_project(project)

    parents = _query(
        project, 'SELECT COUNT(DISTINCT document_id) FROM "db".docs.facts'
    )
    assert int(parents[0][0]) == INPUTS


# ─── cadence must not change content ────────────────────────────────────────


def test_flush_size_does_not_change_the_output(tmp_path: Path) -> None:
    columns = (
        "SELECT document_id, sentiment FROM \"db\".docs.facts ORDER BY document_id"
    )

    def _run(flush_every: int) -> list[tuple[Any, ...]]:
        project = _project(tmp_path / f"f{flush_every}", flush_every=flush_every)
        run_project(project)
        return _query(project, columns)

    assert _run(1) == _run(2) == _run(1000)


def test_flush_every_does_not_move_code_version(tmp_path: Path) -> None:
    """A moved code_version re-calls the provider for every existing row,
    silently, because the run still succeeds."""
    from stel.config.model import LLMTransformConfig
    from stel.versioning import compute_code_version

    def _version(flush_every: int) -> str:
        return compute_code_version(
            extraction=None,
            transform=None,
            llm=LLMTransformConfig(
                input_field="body",
                prompt="p",
                provider="deterministic",
                model="m",
                flush_every=flush_every,
            ),
            project_dir=tmp_path,
        )

    assert _version(1) == _version(1000) == _version(100_000)
