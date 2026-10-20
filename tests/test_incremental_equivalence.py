"""Incremental, resumed and clean execution converge (constellations#707, ALE-284).

A skip that is not equivalent to a rebuild is a silent wrong-number defect: it
is green, it is fast, and it withholds rows nobody notices missing until a
downstream number disagrees. The mutation matrix pins the identity half --
which changes must invalidate state -- and this harness pins the outcome half:
for deterministic providers, an incremental run and an interrupted-then-resumed
run land on exactly the rows and state a clean build produces.

Every scenario is one project with two warehouse targets: `dev` is built
incrementally and `clean` is built once, from scratch, after the mutation.
Inputs, project files and -- deliberately -- absolute paths are shared, so
the only difference between the two databases is how the rows arrived. The
chain (extract -> chunk -> embed, plus an llm map over the extraction) runs on
DuckDB with the two deterministic providers, so rows are a pure function of
inputs and identity; extract is free, embed and llm are the paid stages whose
calls the harness counts.

Rows are compared without the columns that record where a document came from
(`source_path`/`source_uri`) or when a run happened (`*_at` timestamps); state
is compared without `last_run_at` and `target_identity`. Everything else --
ids, hashes, vectors, values, `code_version` -- must be equal row for row.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pytest

from stel.adapters import AdapterError
from stel.adapters.duckdb import DuckDBAdapter
from stel.runner import RunError, run_project

# Runs whole projects, so the file belongs to the `e2e` tier (issue #518).
pytestmark = pytest.mark.e2e

_MODELS_YML = """version: 2
models:
  - name: document_registry
    source: ref('documents')
    extraction:
      backend: json
      options:
        fields: [title, body]
    materialization: incremental
  - name: document_chunks
    depends_on: [ref('document_registry')]
    chunk:
      text_field: body
      chunk_size: 1000
      chunk_overlap: 0
    materialization: incremental
  - name: document_embeddings
    depends_on: [ref('document_chunks')]
    embed:
      provider: deterministic
      model: contract-v1
      text_field: text
      id_field: chunk_id
      vector_field: embedding
      dimensions: 4
      batch_size: 1
      flush_every: 1
    materialization: incremental
  - name: document_facts
    depends_on: [ref('document_registry')]
    llm:
      input_field: body
      id_field: document_id
      prompt: 'Extract the key fact.'
      provider: deterministic
      model: deterministic-v1
      flush_every: 1
    fields:
      - {name: sentiment, type: string}
      - {name: score, type: integer}
    materialization: incremental
"""

_TABLES = (
    "document_registry",
    "document_chunks",
    "document_embeddings",
    "document_facts",
)


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir(parents=True)
    (project / "stel_project.yml").write_text(
        "name: equivalence\nversion: '0.1.0'\nprofile: equivalence\n"
    )
    (project / "profiles.yml").write_text(
        "equivalence:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n        schema: docs\n"
        "    clean:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/clean/db.duckdb\n        schema: docs\n"
    )
    (project / "sources").mkdir()
    (project / "sources" / "documents.yml").write_text(
        "version: 2\nsources:\n  - name: documents\n    path: data\n    file_pattern: '*.json'\n"
    )
    (project / "models").mkdir()
    (project / "models" / "models.yml").write_text(_MODELS_YML)
    _write_document(project, "a", "employment rose")
    _write_document(project, "b", "inflation cooled")
    return project


def _write_document(project: Path, name: str, body: str) -> None:
    (project / "data").mkdir(exist_ok=True)
    (project / "data" / f"{name}.json").write_text(json.dumps({"title": name, "body": body}))


def _clean_build(project: Path) -> None:
    """Build `clean` once, from scratch, from whatever the inputs are now."""
    run_project(project, target="clean")


# A DuckDB file's catalog is named after its stem, so both databases are named
# `db.duckdb` (the clean one in a subdirectory) and every query can address
# them the same way: `"db".docs.<table>`.
_DATABASES = {"dev": "db.duckdb", "clean": "clean/db.duckdb"}

_FROZEN_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
# The two stages whose row timestamps reach a downstream model's
# `input_fingerprint`: a chunk's fingerprint covers the registry row it came
# from, and an embed's covers the chunk row. Only these two need a frozen
# clock for two independent builds to agree; `embedded_at`/`generated_at`
# never reach a fingerprint because nothing is downstream of them.
_CLOCK_MODULES = ("stel.execution.extraction", "stel.execution.chunk")


def _freeze_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """One clock for every run, so only produced content can differ.

    Deterministic providers make rows a function of inputs and identity; a
    frozen clock extends that to the row timestamps those functions stamp.
    Without it, two independent builds of the same inputs necessarily
    disagree on `extracted_at`, and that disagreement reaches state through
    every downstream `input_fingerprint` -- in production an unchanged row
    keeps its timestamp, so only the comparison across builds needs the
    freeze.
    """

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return _FROZEN_NOW if tz is None else _FROZEN_NOW.astimezone(tz)

    for module in _CLOCK_MODULES:
        monkeypatch.setattr(f"{module}.datetime", _FrozenDatetime)


def _query(project: Path, database: str, sql: str) -> list[tuple[Any, ...]]:
    connection = duckdb.connect(str(project / "target" / _DATABASES[database]), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def _columns(project: Path, database: str, table: str) -> list[str]:
    rows = _query(
        project,
        database,
        "SELECT column_name FROM information_schema.columns "
        f"WHERE table_schema = 'docs' AND table_name = '{table}' "
        "ORDER BY ordinal_position",
    )
    return [row[0] for row in rows]


def _rows(project: Path, database: str, table: str) -> list[tuple[Any, ...]]:
    """Every published row, without the where/when columns (see docstring)."""
    columns = [
        name
        for name in _columns(project, database, table)
        if not name.endswith("_at") and name not in ("source_path", "source_uri")
    ]
    selected = ", ".join(f'"{name}"' for name in columns)
    ordering = ", ".join(f'"{name}"' for name in columns[:2]) or selected
    return _query(
        project,
        database,
        f'SELECT {selected} FROM "db".docs.{table} ORDER BY {ordering}',
    )


def _state(project: Path, database: str) -> list[tuple[Any, ...]]:
    """Published state, without `last_run_at`/`target_identity` (see docstring)."""
    return _query(
        project,
        database,
        "SELECT model_name, state_scope, record_key, input_fingerprint, "
        'code_version FROM "db".docs.stel_state ORDER BY 1, 2, 3',
    )


def _result(results: list[Any], model_name: str) -> Any:
    return next(item for item in results if item.model_name == model_name)


def _provider_calls(results: list[Any], model_name: str) -> int:
    metrics = _result(results, model_name).metrics
    return int(metrics.get("provider_calls", 0))


def _assert_converged(project: Path) -> None:
    """The incremental `dev` database equals the once-built `clean` one."""
    for table in _TABLES:
        dev = _rows(project, "dev", table)
        clean = _rows(project, "clean", table)
        assert dev == clean, table
    assert _state(project, "dev") == _state(project, "clean")


def _add_document(project: Path) -> None:
    _write_document(project, "c", "output grew")


def _change_document(project: Path) -> None:
    _write_document(project, "a", "employment rose sharply")


def _delete_document(project: Path) -> None:
    (project / "data" / "b.json").unlink()


@pytest.mark.parametrize(
    "mutation",
    [_add_document, _change_document, _delete_document],
    ids=["add", "change", "delete"],
)
def test_input_mutations_converge_to_the_clean_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: Callable[[Path], None]
) -> None:
    """A mutated input reaches the same rows and state either way it arrives.

    `dev` runs first, mutates, then runs again; `clean` is built once from the
    mutated inputs. Convergence is the claim the skip logic makes every day:
    nothing about arriving in two runs may leave rows or state different from
    arriving in one.
    """
    project = _project(tmp_path)
    _freeze_clock(monkeypatch)
    run_project(project)
    mutation(project)
    results = run_project(project)
    _clean_build(project)

    _assert_converged(project)
    # Paid work follows the change and nothing else: one new or changed
    # document reaches both paid stages once, and a deletion reaches neither.
    expected = 0 if mutation is _delete_document else 1
    assert _provider_calls(results, "document_embeddings") == expected
    assert _provider_calls(results, "document_facts") == expected


@pytest.mark.parametrize(
    ("model", "old", "new"),
    [
        ("document_embeddings", "      batch_size: 1", "      batch_size: 2"),
        ("document_facts", "      flush_every: 1", "      flush_every: 2"),
    ],
    ids=["embed-batch_size", "llm-flush_every"],
)
def test_execution_only_mutations_cost_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, old: str, new: str
) -> None:
    """Tuning execution never re-bills published rows (#572, #587).

    The identity half is pinned in `test_rebuild_mutation_matrix.py`; this is
    the outcome half, which is the one a bill arrives for: after the knob
    moves, the run must process nothing, call no provider, and leave rows and
    state exactly where the clean build left them.
    """
    project = _project(tmp_path)
    _freeze_clock(monkeypatch)
    run_project(project)
    _replace_models_text(project, old, new)
    results = run_project(project)
    _clean_build(project)

    _assert_converged(project)
    for model_name in ("document_embeddings", "document_facts"):
        assert _provider_calls(results, model_name) == 0
        assert _result(results, model_name).documents_processed == 0


def _replace_models_text(project: Path, old: str, new: str) -> None:
    path = project / "models" / "models.yml"
    path.write_text(path.read_text().replace(old, new))


def test_prompt_change_reprocesses_paid_rows_and_converges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A semantic change pays for what it invalidates, and nothing else.

    The llm model's prompt is semantic, so every published fact reprocesses;
    the embed model's input is untouched, so it must pay for none of it. The
    cascade in the other direction -- a paid model reprocessing because an
    upstream re-keyed it -- is the `change` row above.
    """
    prompt_a = "      prompt: 'Extract the key fact.'"
    prompt_b = "      prompt: 'Extract the two key facts.'"
    project = _project(tmp_path)
    _freeze_clock(monkeypatch)
    run_project(project)
    _replace_models_text(project, prompt_a, prompt_b)
    # The reprocess guard refuses this run by default (issue #530): a paid
    # model whose code_version moved reprocesses at provider prices, and the
    # operator says yes with --accept-reprocess.
    results = run_project(project, accept_reprocess=True)
    _clean_build(project)

    _assert_converged(project)
    assert _provider_calls(results, "document_facts") == 2
    assert _provider_calls(results, "document_embeddings") == 0


def _fail_state_advance_once(
    monkeypatch: pytest.MonkeyPatch, model_name: str, *, on_attempt: int
) -> None:
    """Make one `upsert_state` for `model_name` fail after the write landed.

    This is the failure `FlushPublisher` rule 2 orders around: the row is in
    the target and the state advance is the step that fails, so the next run
    must complete from a target that is ahead of its state.
    """
    real = DuckDBAdapter.upsert_state
    attempts = 0

    def upsert_state(self: DuckDBAdapter, scope: Any, records: Any) -> None:
        nonlocal attempts
        if scope.model_name == model_name:
            attempts += 1
            if attempts == on_attempt:
                raise AdapterError("state advance failed on cue")
        real(self, scope, records)

    monkeypatch.setattr(DuckDBAdapter, "upsert_state", upsert_state)


def test_a_failed_state_advance_resumes_to_the_clean_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run interrupted after a write resumes without paying for it twice.

    The interrupted run writes a flush and then fails its state advance, so
    the target is one row ahead of its state. The resume must reprocess the
    changed row -- the stale state record says it is not done -- while the
    vector the interrupted flush already wrote comes back out of the target
    by input hash rather than from the provider.
    """
    project = _project(tmp_path)
    _freeze_clock(monkeypatch)
    run_project(project)
    _change_document(project)
    _fail_state_advance_once(monkeypatch, "document_embeddings", on_attempt=1)
    with pytest.raises(RunError):
        run_project(project)

    results = run_project(project)
    _clean_build(project)

    _assert_converged(project)
    embed = _result(results, "document_embeddings")
    assert embed.documents_processed == 1
    assert embed.metrics["provider_calls"] == 0
    assert embed.metrics["cache_hits"] == 1


def test_an_interrupted_full_refresh_rebuild_completes_from_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted rebuild completes from state on the next run.

    With `flush_every: 1` the rebuild publishes one row at a time, so failing
    the second flush's state advance leaves the first row published, its state
    advanced, and the rest unrebuilt. That partial table is the deliberate
    trade behind flushing at all -- the alternative is a corpus that cannot be
    rebuilt -- and the next run must finish it, reusing the flush the failed
    run already paid for.
    """
    prompt_a = "      prompt: 'Extract the key fact.'"
    prompt_b = "      prompt: 'Extract the two key facts.'"
    project = _project(tmp_path)
    _freeze_clock(monkeypatch)
    run_project(project)
    _replace_models_text(project, prompt_a, prompt_b)
    _fail_state_advance_once(monkeypatch, "document_embeddings", on_attempt=2)
    with pytest.raises(RunError):
        run_project(project, full_refresh=True)

    results = run_project(project)
    _clean_build(project)

    _assert_converged(project)
    embed = _result(results, "document_embeddings")
    assert embed.documents_processed == 1
    assert embed.metrics["provider_calls"] == 0
    assert embed.metrics["cache_hits"] == 1
