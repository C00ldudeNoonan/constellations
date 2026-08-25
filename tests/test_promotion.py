"""Promoting candidate judgments into a golden set (#380, #329 phase 3).

The acceptance criterion #380 set is at the bottom of this file: a promoted
golden set runs unchanged through the existing `retrieval_tests:` machinery,
with no eval code changes at all. The rest guards the two ways promotion could
quietly produce a worthless test — a row with no provenance, and a set
promoted in an id space the target index does not key on.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pytest
import yaml

from stel.promotion import GoldenSetFile, PromotionError, load_golden_set
from stel.retrieval_eval import run_retrieval_evaluation
from stel.runner import run_project

# ─── the file contract ──────────────────────────────────────────────────────


def _query(**overrides: Any) -> dict[str, Any]:
    query = {
        "query_id": "q_prices",
        "query_text": "consumer prices inflation",
        "relevant_ids": ["abc"],
        "promoted_by": "alex",
        "promoted_at": "2026-08-25",
        "evidence": {"sessions": ["sess-1"], "harness": "claude-code"},
    }
    query.update(overrides)
    return query


def _document(**overrides: Any) -> dict[str, Any]:
    document = {"version": 1, "id_space": "chunk_id", "queries": [_query()]}
    document.update(overrides)
    return document


def _write(tmp_path: Path, document: Any) -> Path:
    path = tmp_path / "golden.yml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_a_valid_golden_set_round_trips(tmp_path: Path) -> None:
    golden = load_golden_set(_write(tmp_path, _document()))

    assert golden.id_space == "chunk_id"
    assert golden.queries[0].promoted_at == date(2026, 8, 25)
    assert golden.queries[0].evidence.sessions == ("sess-1",)


def test_a_promotion_must_name_its_sessions(tmp_path: Path) -> None:
    """The first question a reviewer asks is where a row came from. A promoted
    golden that cannot answer it is indistinguishable from an invented one."""
    document = _document(queries=[_query(evidence={"sessions": []})])

    with pytest.raises(PromotionError, match="at least one session"):
        load_golden_set(_write(tmp_path, document))


def test_a_query_that_asserts_nothing_is_rejected(tmp_path: Path) -> None:
    document = _document(queries=[_query(relevant_ids=[])])

    with pytest.raises(PromotionError, match="asserts nothing"):
        load_golden_set(_write(tmp_path, document))


def test_duplicate_query_ids_are_rejected(tmp_path: Path) -> None:
    document = _document(queries=[_query(), _query()])

    with pytest.raises(PromotionError, match="duplicate query_id"):
        load_golden_set(_write(tmp_path, document))


def test_query_text_is_required(tmp_path: Path) -> None:
    """`retrieval_tests` replays each query through `search()`, and the corpus
    records only a fingerprint unless text capture was opted into — so the
    reviewer supplying the text is what makes the row re-runnable (#380)."""
    document = _document(queries=[_query(query_text="")])

    with pytest.raises(PromotionError):
        load_golden_set(_write(tmp_path, document))


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(PromotionError):
        load_golden_set(_write(tmp_path, _document(queries=[_query(oops=1)])))


def test_malformed_files_are_named_not_traced(tmp_path: Path) -> None:
    path = tmp_path / "golden.yml"
    path.write_text("just a string", encoding="utf-8")
    with pytest.raises(PromotionError, match="YAML mapping"):
        load_golden_set(path)

    path.write_text("a: [1,\n", encoding="utf-8")
    with pytest.raises(PromotionError, match="not valid YAML"):
        load_golden_set(path)

    with pytest.raises(PromotionError, match="regular file"):
        load_golden_set(tmp_path / "missing.yml")


def test_the_version_is_pinned(tmp_path: Path) -> None:
    with pytest.raises(PromotionError):
        load_golden_set(_write(tmp_path, _document(version=2)))
    assert GoldenSetFile.model_validate(_document()).version == 1


# ─── end to end, against the real eval machinery ────────────────────────────

_DOCS = {
    "inflation.json": {
        "title": "Consumer prices",
        "body": "Inflation moderated as consumer price growth slowed.",
    },
    "labor.json": {
        "title": "Employment report",
        "body": "Payroll employment increased and unemployment remained stable.",
    },
}

_SEARCH_MODELS = """version: 2
models:
  - name: release_documents
    source: ref('releases')
    extraction:
      backend: json
      options:
        fields: [title, body]
    materialization: incremental
  - name: release_chunks
    depends_on: [ref('release_documents')]
    chunk:
      text_field: body
      chunk_size: 1000
      chunk_overlap: 0
    materialization: incremental
  - name: release_embeddings
    depends_on: [ref('release_chunks')]
    embed:
      provider: deterministic
      model: promotion-demo-v1
      text_field: text
      id_field: chunk_id
      vector_field: embedding
      dimensions: 8
    materialization: incremental
  - name: release_search
    depends_on: [ref('release_embeddings')]
    materialization: incremental
    search:
      access: public
      id_field: chunk_id
      document_id_field: document_id
      chunk_id_field: chunk_id
      text_fields: [text]
      return_text_fields: [text]
      full_text:
        fields: [text]
      query:
        modes: [text]
        consistency: strong
    retrieval_tests:
      - name: release_search_quality
        golden_set: ref('promoted_goldens')
        mode: text
        at: [1, 3]
        thresholds:
          recall_at_3: {min: 1.0, severity: error}
"""


def _golden_model(search_model: str = "release_search") -> str:
    return (
        "version: 2\n"
        "models:\n"
        "  - name: promoted_goldens\n"
        "    depends_on: [ref('release_documents')]\n"
        "    transform:\n"
        "      type: python\n"
        "      module: stel.promotion.golden_set\n"
        "      options:\n"
        "        path: golden_sets/release_search.yml\n"
        f"        search_model: {search_model}\n"
        "    materialization: full\n"
    )


def _project(tmp_path: Path, *, search_model: str = "release_search") -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "stel_project.yml").write_text(
        "name: promo_demo\nversion: '0.1.0'\nprofile: promo_demo\n", encoding="utf-8"
    )
    (project / "profiles.yml").write_text(
        "promo_demo:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      warehouse:\n"
        "        type: duckdb\n"
        "        path: target/data.duckdb\n"
        "        schema: analytics\n"
        "      retrieval:\n"
        "        default: local\n"
        "        allow_public_indexes: true\n"
        "        stores:\n"
        "          local:\n"
        "            type: lancedb\n"
        "            path: target/lancedb\n",
        encoding="utf-8",
    )
    (project / "sources").mkdir()
    (project / "sources" / "documents.yml").write_text(
        "version: 2\nsources:\n  - name: releases\n    path: data\n"
        "    file_pattern: '*.json'\n",
        encoding="utf-8",
    )
    (project / "models").mkdir()
    (project / "models" / "search.yml").write_text(_SEARCH_MODELS, encoding="utf-8")
    (project / "models" / "goldens.yml").write_text(
        _golden_model(search_model), encoding="utf-8"
    )
    data = project / "data"
    data.mkdir()
    for name, payload in _DOCS.items():
        (data / name).write_text(json.dumps(payload), encoding="utf-8")
    return project


def _chunk_id(project: Path, stem: str) -> str:
    con = duckdb.connect(str(project / "target" / "data.duckdb"))
    try:
        row = con.execute(
            "select chunk_id from analytics.release_chunks where source_path like ?",
            [f"%{stem}%"],
        ).fetchone()
        assert row is not None
        return str(row[0])
    finally:
        con.close()


def _write_promotion(project: Path, *, id_space: str, queries: list[dict[str, Any]]) -> None:
    directory = project / "golden_sets"
    directory.mkdir(exist_ok=True)
    (directory / "release_search.yml").write_text(
        yaml.safe_dump({"version": 1, "id_space": id_space, "queries": queries}),
        encoding="utf-8",
    )


def _promoted(project: Path, **overrides: Any) -> dict[str, Any]:
    query = {
        "query_id": "q_prices",
        "query_text": "consumer prices inflation",
        "relevant_ids": [_chunk_id(project, "inflation")],
        "promoted_by": "alex",
        "promoted_at": "2026-08-25",
        "evidence": {
            "sessions": ["sess-1"],
            "harness": "claude-code",
            "query_fingerprint": "f" * 32,
        },
    }
    query.update(overrides)
    return query


def test_a_promoted_golden_set_runs_through_retrieval_tests_unchanged(
    tmp_path: Path,
) -> None:
    """#380's acceptance criterion. `retrieval_tests.golden_set` already refs
    an ordinary model, so promotion needed no eval changes at all."""
    project = _project(tmp_path)
    # The search index has to exist before ids can be promoted against it.
    _write_promotion(project, id_space="chunk_id", queries=[])
    run_project(project)

    _write_promotion(project, id_space="chunk_id", queries=[_promoted(project)])
    run_project(project, select="promoted_goldens")

    results = run_retrieval_evaluation(project)

    assert len(results) == 1
    assert results[0].status == "pass"
    assert results[0].aggregate["recall"][3] == pytest.approx(1.0)


def test_a_mislabelled_promotion_fails_the_threshold(tmp_path: Path) -> None:
    """Proves the pass above is not vacuous: the same machinery, with the
    wrong chunk promoted as relevant, has to fail."""
    project = _project(tmp_path)
    _write_promotion(project, id_space="chunk_id", queries=[])
    run_project(project)

    mislabelled = _promoted(project, relevant_ids=[_chunk_id(project, "labor")])
    _write_promotion(project, id_space="chunk_id", queries=[mislabelled])
    run_project(project, select="promoted_goldens")

    results = run_retrieval_evaluation(project)

    assert results[0].status == "fail"
    assert results[0].aggregate["recall"][3] < 1.0


def test_promoting_in_the_wrong_id_space_fails_loudly(tmp_path: Path) -> None:
    """The failure this guard exists for is silent: every id fails to match a
    returned record_id and the eval reports zero recall, as if retrieval were
    broken rather than the golden set mislabelled (#380, constraint 3)."""
    project = _project(tmp_path)
    _write_promotion(project, id_space="chunk_id", queries=[])
    run_project(project)

    # The index keys on chunk_id; promote context_ids instead.
    _write_promotion(project, id_space="context_id", queries=[_promoted(project)])

    with pytest.raises(Exception, match="context_id"):
        run_project(project, select="promoted_goldens")


def test_an_invalid_promotion_fails_before_anything_executes(tmp_path: Path) -> None:
    """The promotion is validated at compile time, not when the transform
    runs. By execution the upstream models have already spent provider calls
    and written to the warehouse, so a malformed file discovered there costs
    all of it (Codex review; AGENTS.md preflight rule)."""
    project = _project(tmp_path)
    # Wrong id space: the index keys on chunk_id.
    _write_promotion(
        project,
        id_space="context_id",
        queries=[
            {
                "query_id": "q",
                "query_text": "t",
                "relevant_ids": ["abc"],
                "promoted_by": "alex",
                "promoted_at": "2026-08-25",
                "evidence": {"sessions": ["s"]},
            }
        ],
    )

    with pytest.raises(Exception, match="context_id"):
        run_project(project)

    # Nothing ran: no warehouse file, so no extraction, embedding, or publish.
    assert not (project / "target" / "data.duckdb").exists()


def test_blank_and_repeated_ids_are_rejected(tmp_path: Path) -> None:
    """`excluded_ids: [""]` is a non-empty tuple, so it satisfies a naive
    "asserts something" check while no search result can ever match it — the
    query would then drop out of the ranking aggregates rather than fail
    (Codex review)."""
    blank = _document(queries=[_query(relevant_ids=[], excluded_ids=[""])])
    with pytest.raises(PromotionError, match="blank entry"):
        load_golden_set(_write(tmp_path, blank))

    repeated = _document(queries=[_query(relevant_ids=["abc", "abc"])])
    with pytest.raises(PromotionError, match="repeats an id"):
        load_golden_set(_write(tmp_path, repeated))


def test_a_symlinked_promotion_artifact_is_refused(tmp_path: Path) -> None:
    """A symlink whose target is still inside the project survives
    `resolve_within_project`, which dereferences it — so containment alone
    does not keep the reviewed artifact a real, reviewable file."""
    project = _project(tmp_path)
    real = project / "elsewhere.yml"
    real.write_text(
        yaml.safe_dump({"version": 1, "id_space": "chunk_id", "queries": []}),
        encoding="utf-8",
    )
    (project / "golden_sets").mkdir(exist_ok=True)
    link = project / "golden_sets" / "release_search.yml"
    try:
        link.symlink_to(real)
    except OSError:  # pragma: no cover - platform-dependent privilege
        pytest.skip("creating symlinks requires privileges on this platform")

    with pytest.raises(Exception, match="symlink"):
        run_project(project)


def test_a_path_outside_the_project_is_refused(tmp_path: Path) -> None:
    project = _project(tmp_path)
    (project / "models" / "goldens.yml").write_text(
        _golden_model().replace(
            "path: golden_sets/release_search.yml",
            "path: ../escape.yml",
        ),
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="outside the project"):
        run_project(project)


def test_an_unknown_search_model_is_refused(tmp_path: Path) -> None:
    project = _project(tmp_path, search_model="no_such_model")
    _write_promotion(project, id_space="chunk_id", queries=[])

    with pytest.raises(Exception, match="not a model in this project"):
        run_project(project)


def test_the_promoted_rows_carry_their_provenance(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _write_promotion(project, id_space="chunk_id", queries=[])
    run_project(project)
    _write_promotion(project, id_space="chunk_id", queries=[_promoted(project)])
    run_project(project, select="promoted_goldens")

    con = duckdb.connect(str(project / "target" / "data.duckdb"))
    try:
        row = con.execute(
            "select query_id, promoted_by, promoted_at, evidence_sessions, "
            "query_fingerprint, id_space from analytics.promoted_goldens"
        ).fetchone()
    finally:
        con.close()

    assert row is not None
    query_id, promoted_by, promoted_at, sessions, fingerprint, id_space = row
    assert query_id == "q_prices"
    assert promoted_by == "alex"
    assert promoted_at == "2026-08-25"
    assert json.loads(sessions) == ["sess-1"]
    assert fingerprint == "f" * 32
    assert id_space == "chunk_id"
