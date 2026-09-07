"""The documented way to experiment on one step (issue #531), end to end.

`docs/reference.md` says: put one `for_each` axis on the step under test, the
same axis on the models below it referencing their own variant, run the
branches with `tag:<base>+`, and when the experiment ends drop the axis value
and let `stel ls --orphans` name what it left behind. This test is that
recipe on the rag example, so the section cannot drift from what works.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from stel.orphans import find_orphans
from stel.plan import plan_project
from stel.runner import run_project

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

CHUNKS = """version: 2

models:
  - name: document_chunks
    depends_on: [ref('document_registry')]
    for_each:
      chunk_size: [200, 400]
    chunk:
      strategy: recursive
      text_field: text
      chunk_size: ${matrix.chunk_size}
      chunk_overlap: 100
    materialization: incremental
"""

EMBEDDINGS = """version: 2

models:
  - name: chunk_embeddings
    for_each:
      chunk_size: [200, 400]
    depends_on: ["ref('document_chunks__chunk_size_${matrix.chunk_size}')"]
    embed:
      provider: deterministic
      model: deterministic-v1
      text_field: text
      id_field: chunk_id
      vector_field: embedding
      dimensions: 8
    materialization: incremental
"""

SEARCH = """version: 2

models:
  - name: chunk_search
    for_each:
      chunk_size: [200, 400]
    depends_on: ["ref('chunk_embeddings__chunk_size_${matrix.chunk_size}')"]
    materialization: incremental
    search:
      access: public
      store: local
      collection: document_chunks_${matrix.chunk_size}
      id_field: chunk_id
      document_id_field: document_id
      chunk_id_field: chunk_id
      text_fields: [text]
      return_text_fields: [text]
      vector:
        field: embedding
        dimensions: 8
        metric: cosine
        search: exact
        embedding: inherit
      full_text:
        fields: [text]
      query:
        modes: [vector, text, filter]
        consistency: strong
"""

WIN, LOSE = "200", "400"
VARIANTS = (WIN, LOSE)
BASES = ("document_chunks", "chunk_embeddings", "chunk_search")


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    pytest.importorskip("lancedb")
    project = tmp_path / "experiment"
    shutil.copytree(
        EXAMPLES / "rag_chunks_pipeline",
        project,
        ignore=shutil.ignore_patterns("target", "__pycache__"),
    )
    models = project / "models"
    # The llm and golden-set models reference the template names, which a
    # for_each expansion removes; the experiment is on the retrieval chain.
    for stale in ("chunk_facts.yml", "chunk_entities.yml", "chunk_search_golden.yml"):
        (models / stale).unlink()
    (models / "document_chunks.yml").write_text(CHUNKS, encoding="utf-8")
    (models / "chunk_embeddings.yml").write_text(EMBEDDINGS, encoding="utf-8")
    (models / "chunk_search.yml").write_text(SEARCH, encoding="utf-8")
    return project


def _variant(base: str, size: str) -> str:
    return f"{base}__chunk_size_{size}"


def test_the_documented_recipe_runs_and_cleans_up(experiment: Path) -> None:
    # The serving pipeline above the experiment already exists, as it would
    # in production; the recipe's selector deliberately leaves it alone.
    run_project(experiment, select="document_registry")

    # Before the variants run, every one is `new`: nothing existing re-keys.
    plans = {m.name: m for m in plan_project(experiment, select="tag:document_chunks+").models}
    expected = {_variant(base, size) for base in BASES for size in VARIANTS}
    assert set(plans) == expected
    assert {m.status for m in plans.values()} == {"new"}

    # Both branches build side by side under the base-name tag.
    results = run_project(experiment, select="tag:document_chunks+")
    assert {r.model_name for r in results} == expected
    assert all(not r.errors for r in results), [r.errors for r in results if r.errors]
    rows_by_name = {r.model_name: r.rows_written for r in results}
    sizes = {size: rows_by_name[_variant("document_chunks", size)] for size in VARIANTS}
    # The experiment did what it says: a smaller chunk size makes more chunks.
    assert sizes[WIN] > sizes[LOSE] > 0
    assert find_orphans(experiment, target=None, profiles_dir=None).is_empty()

    # The experiment ends: the smaller size wins, the other axis value goes away.
    for filename in ("document_chunks.yml", "chunk_embeddings.yml", "chunk_search.yml"):
        path = experiment / "models" / filename
        path.write_text(
            path.read_text(encoding="utf-8").replace(f"[{WIN}, {LOSE}]", f"[{WIN}]"),
            encoding="utf-8",
        )
    report = find_orphans(experiment, target=None, profiles_dir=None)
    # The loser's tables: chunks and embeddings. A search index has no
    # warehouse table, only a serving scope.
    assert [t.name for t in report.tables] == [
        _variant("chunk_embeddings", LOSE),
        _variant("document_chunks", LOSE),
    ]
    assert {s.name for s in report.state_scopes} == {_variant(base, LOSE) for base in BASES}
    search_scope = next(
        s for s in report.state_scopes if s.name == _variant("chunk_search", LOSE)
    )
    assert search_scope.summary.scope.stage == "retrieval_publish"
    # The winner is untouched: still claimed, still unchanged.
    plans = {m.name: m for m in plan_project(experiment).models}
    assert plans[_variant("chunk_embeddings", WIN)].status == "unchanged"
