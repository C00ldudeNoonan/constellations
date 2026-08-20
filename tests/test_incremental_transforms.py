"""Incremental Python transforms with child-row deletion semantics (issue #218).

Exercises the generic one-to-many machinery end to end on DuckDB through a
project-local transform, so it needs no optional NLP extras: `word_tokens`
explodes a document's body into one stable child row per word.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pytest

from stel.runner import RunError, run_project

_TRANSFORM_SOURCE = '''
from __future__ import annotations

import polars as pl

from stel.hashing import canonical_fingerprint
from stel.transforms import IncrementalContract, TransformContext

_SCHEMA = {
    "word_id": pl.String(),
    "document_id": pl.String(),
    "position": pl.Int64(),
    "word": pl.String(),
}


def declared_incremental_contract(options):
    return IncrementalContract(
        parent_key="document_id",
        child_key="word_id",
        parent_source_key="document_id",
    )


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    frame = next(iter(deps.values()))
    rows = []
    for record in frame.iter_rows(named=True):
        document_id = str(record["document_id"])
        words = str(record.get("body") or "").split()
        for position, word in enumerate(words):
            rows.append(
                {
                    "word_id": canonical_fingerprint(
                        {"document_id": document_id, "position": position, "word": word},
                        domain="test.word",
                    ),
                    "document_id": document_id,
                    "position": position,
                    "word": word,
                }
            )
    return pl.DataFrame(rows, schema=_SCHEMA)
'''


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "stel_project.yml").write_text(
        "name: docs\nversion: '0.1.0'\nprofile: docs\n"
    )
    (project / "profiles.yml").write_text(
        "docs:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n        schema: docs\n"
    )
    (project / "sources").mkdir()
    (project / "sources" / "src.yml").write_text(
        "version: 2\nsources:\n  - name: raw_docs\n    path: data/docs\n"
        "    file_pattern: '*.json'\n"
    )
    (project / "models").mkdir()
    (project / "models" / "documents.yml").write_text(
        "version: 2\nmodels:\n  - name: documents\n"
        "    source: ref('raw_docs')\n    extraction:\n      backend: json\n"
        "      options:\n        fields: [body]\n"
        "    materialization: incremental\n"
    )
    (project / "models" / "word_tokens.yml").write_text(
        "version: 2\nmodels:\n  - name: word_tokens\n"
        "    depends_on: [ref('documents')]\n"
        "    transform:\n      type: python\n      module: transforms.word_tokens\n"
        "    materialization: incremental\n"
    )
    (project / "transforms").mkdir()
    (project / "transforms" / "word_tokens.py").write_text(_TRANSFORM_SOURCE)
    (project / "data" / "docs").mkdir(parents=True)
    return project


def _write_doc(project: Path, name: str, body: str) -> None:
    (project / "data" / "docs" / name).write_text(json.dumps({"body": body}))


def _tokens(project: Path) -> list[tuple[Any, ...]]:
    con = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        return con.execute(
            'SELECT word_id, document_id, position, word FROM "db".docs.word_tokens '
            "ORDER BY document_id, position"
        ).fetchall()
    finally:
        con.close()


def _result(results: list[Any], name: str):
    return next(r for r in results if r.model_name == name)


def _state_keys(project: Path) -> set[str]:
    con = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        rows = con.execute(
            "SELECT record_key FROM \"db\".docs.stel_state "
            "WHERE model_name = 'word_tokens'"
        ).fetchall()
    finally:
        con.close()
    return {row[0] for row in rows}


def test_first_run_materializes_all_children(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _write_doc(project, "a.json", "alpha beta gamma")
    _write_doc(project, "b.json", "delta")

    results = run_project(project)
    tokens = _result(results, "word_tokens")
    assert tokens.documents_processed == 2
    assert tokens.rows_written == 4  # 3 + 1 words
    assert len(_tokens(project)) == 4
    assert len(_state_keys(project)) == 2


def test_unchanged_corpus_skips_every_parent(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _write_doc(project, "a.json", "alpha beta")
    run_project(project)

    again = _result(run_project(project), "word_tokens")
    assert again.documents_processed == 0
    assert again.documents_skipped == 1


def test_new_parent_only_processes_that_parent(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _write_doc(project, "a.json", "alpha beta")
    run_project(project)

    _write_doc(project, "b.json", "gamma delta epsilon")
    result = _result(run_project(project), "word_tokens")
    assert result.documents_processed == 1
    assert result.documents_skipped == 1
    assert result.documents_deleted == 0
    by_doc = {row[1] for row in _tokens(project)}
    assert len(by_doc) == 2
    assert len(_tokens(project)) == 5


def test_changed_parent_with_fewer_children_replaces_only_its_rows(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    _write_doc(project, "a.json", "alpha beta gamma")
    _write_doc(project, "b.json", "delta epsilon")
    run_project(project)
    b_rows_before = [row for row in _tokens(project) if row[1] != _doc_id(project, "a.json")]

    # Shrink a.json from three words to one; b.json is untouched.
    _write_doc(project, "a.json", "alpha")
    result = _result(run_project(project), "word_tokens")
    assert result.documents_processed == 1
    assert result.documents_skipped == 1

    tokens = _tokens(project)
    a_id = _doc_id(project, "a.json")
    a_rows = [row for row in tokens if row[1] == a_id]
    b_rows_after = [row for row in tokens if row[1] != a_id]
    assert len(a_rows) == 1
    assert a_rows[0][3] == "alpha"
    # b.json's child rows are byte-for-byte unchanged.
    assert b_rows_after == b_rows_before


def test_removed_parent_deletes_its_children_and_state(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _write_doc(project, "a.json", "alpha beta")
    _write_doc(project, "b.json", "gamma")
    run_project(project)
    assert len(_state_keys(project)) == 2

    (project / "data" / "docs" / "b.json").unlink()
    result = _result(run_project(project), "word_tokens")
    assert result.documents_deleted == 1
    assert result.documents_processed == 0
    assert result.documents_skipped == 1
    docs = {row[1] for row in _tokens(project)}
    assert len(docs) == 1
    assert len(_state_keys(project)) == 1


def test_transform_code_change_reprocesses_every_parent(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _write_doc(project, "a.json", "alpha beta")
    _write_doc(project, "b.json", "gamma")
    run_project(project)

    # Appending a comment changes the module hash → code_version → full
    # reprocessing even though inputs are byte-identical.
    module = project / "transforms" / "word_tokens.py"
    module.write_text(module.read_text() + "\n# code change\n")
    result = _result(run_project(project), "word_tokens")
    assert result.documents_processed == 2
    assert result.documents_skipped == 0


def test_parent_producing_zero_children_is_processed_then_skipped(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    _write_doc(project, "a.json", "alpha beta")
    _write_doc(project, "empty.json", "")  # zero words → zero child rows
    first = _result(run_project(project), "word_tokens")
    assert first.documents_processed == 2
    assert len(_tokens(project)) == 2  # only a.json's words

    # The empty document is recorded in state, so it is skipped next run rather
    # than reprocessed as if it were new.
    again = _result(run_project(project), "word_tokens")
    assert again.documents_processed == 0
    assert again.documents_skipped == 2


def test_changed_parent_that_becomes_empty_drops_its_children(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _write_doc(project, "a.json", "alpha beta")
    _write_doc(project, "b.json", "gamma")
    run_project(project)
    a_id = _doc_id(project, "a.json")

    _write_doc(project, "a.json", "")  # now empty
    result = _result(run_project(project), "word_tokens")
    assert result.documents_processed == 1
    assert not [row for row in _tokens(project) if row[1] == a_id]
    assert [row for row in _tokens(project) if row[1] != a_id]  # b.json survives


def test_full_refresh_replaces_and_resets_state(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _write_doc(project, "a.json", "alpha beta")
    run_project(project)

    _write_doc(project, "b.json", "gamma delta")
    result = _result(run_project(project, full_refresh=True), "word_tokens")
    assert result.documents_processed == 2
    assert result.documents_skipped == 0
    assert len(_state_keys(project)) == 2
    assert len(_tokens(project)) == 4


def test_failed_publication_leaves_no_stale_state_and_preserves_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A publication failure while rewriting a changed parent must not advance
    # that parent's state (so a retry reprocesses it) and must not touch an
    # unchanged parent's rows.
    from stel.adapters.base import AdapterError
    from stel.adapters.duckdb import DuckDBAdapter

    # replace_children is atomic: a failure rolls back the whole transaction,
    # leaving both old rows and old state intact (issue #229).
    project = _project(tmp_path)
    _write_doc(project, "a.json", "alpha beta")
    _write_doc(project, "b.json", "gamma")
    run_project(project)
    a_id = _doc_id(project, "a.json")
    b_rows_before = [row for row in _tokens(project) if row[1] != a_id]

    _write_doc(project, "a.json", "alpha beta gamma delta")
    original = DuckDBAdapter.replace_children
    failed = False

    def fail_once(self: DuckDBAdapter, table: str, *args: Any, **kwargs: Any) -> int:
        nonlocal failed
        if table == "word_tokens" and not failed:
            failed = True
            raise AdapterError("simulated publication failure")
        return original(self, table, *args, **kwargs)

    monkeypatch.setattr(DuckDBAdapter, "replace_children", fail_once)
    with pytest.raises(RunError, match="simulated publication failure"):
        run_project(project)

    # Atomic rollback: a.json's old state and rows are both still present.
    assert a_id in _state_keys(project)
    assert [row for row in _tokens(project) if row[1] != a_id] == b_rows_before

    monkeypatch.undo()
    result = _result(run_project(project), "word_tokens")
    assert result.documents_processed == 1  # only the reprocessed a.json
    a_rows = [row for row in _tokens(project) if row[1] == a_id]
    assert [row[3] for row in a_rows] == ["alpha", "beta", "gamma", "delta"]
    assert [row for row in _tokens(project) if row[1] != a_id] == b_rows_before


def _doc_id(project: Path, filename: str) -> str:
    con = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        rows = con.execute(
            'SELECT document_id, source_path FROM "db".docs.documents'
        ).fetchall()
    finally:
        con.close()
    for document_id, source_path in rows:
        if source_path and source_path.endswith(filename):
            return str(document_id)
    raise AssertionError(f"no document for {filename}")


# ─── adapter-neutral orchestration (stands in for BigQuery) ──────────────────
#
# The incremental executor is adapter-agnostic; BigQuery's delete_rows_and_state,
# keyed upsert, and state ops already have contract tests in
# test_bigquery_adapter.py. These tests drive the executor against a recording
# adapter that advertises BigQuery's capability set, verifying it issues the
# parent-keyed delete + child-keyed upsert + state advance in the right order
# without live BigQuery credentials.

_REF_TRANSFORM_SOURCE = '''
from __future__ import annotations

import polars as pl

from stel.hashing import canonical_fingerprint
from stel.transforms import IncrementalContract

_SCHEMA = {"row_id": pl.String(), "document_id": pl.String(), "kept": pl.String()}


def declared_incremental_contract(options):
    return IncrementalContract(
        parent_key="document_id",
        child_key="row_id",
        parent_source="docs",
        parent_source_key="document_id",
        reference_deps=("vocab",),
    )


def run(deps: dict[str, pl.DataFrame], ctx=None) -> pl.DataFrame:
    vocab = set(deps["vocab"]["term"].to_list())
    rows = []
    for record in deps["docs"].iter_rows(named=True):
        document_id = str(record["document_id"])
        for position, word in enumerate(str(record["body"]).split()):
            if word in vocab:
                rows.append(
                    {
                        "row_id": canonical_fingerprint(
                            {"document_id": document_id, "position": position},
                            domain="test.ref",
                        ),
                        "document_id": document_id,
                        "kept": word,
                    }
                )
    return pl.DataFrame(rows, schema=_SCHEMA)
'''


class _RecordingAdapter:
    """Duck-typed WarehouseAdapter subset the incremental executor calls,
    advertising BigQuery's capabilities and recording the operations issued."""

    def __init__(
        self,
        upstream: dict[str, pl.DataFrame],
        *,
        warehouse_options: object | None = None,
        preexisting: dict[str, pl.DataFrame] | None = None,
    ) -> None:
        self._upstream = upstream
        self._tables: dict[str, pl.DataFrame] = dict(preexisting or {})
        self._state: dict[str, dict[str, object]] = {}
        self._warehouse_options = warehouse_options
        self.calls: list[tuple[Any, ...]] = []

    def capabilities(self):
        from stel.adapters.bigquery import BigQueryAdapter

        return BigQueryAdapter.capabilities()

    def adapter_type(self) -> str:
        return "bigquery"

    def parse_warehouse_options(self, options, *, model_name):
        return self._warehouse_options

    def relation_exists(self, name: str) -> bool:
        return name in self._tables

    def read_table(self, name: str):
        return self._upstream[name]

    def fetch_state(self, scope):
        return dict(self._state.get(scope.model_name, {}))

    def delete_rows_and_state(self, model, *, key_col, keys, state_scope):
        keys = list(keys)
        self.calls.append(("delete_rows_and_state", key_col, sorted(keys)))
        state = self._state.setdefault(state_scope.model_name, {})
        for key in keys:
            state.pop(key, None)
        table = self._tables.get(model)
        if table is not None and table.height:
            self._tables[model] = table.filter(~pl.col(key_col).is_in(keys))

    def replace_children(
        self,
        model,
        *,
        parent_key,
        parent_ids,
        child_key,
        new_rows,
        state_scope,
        state_records,
        on_schema_change="fail",
        options=None,
    ):
        from stel.adapters import StateValue

        self.calls.append(("replace_children", child_key, sorted(parent_ids), new_rows.height))
        table = self._tables.get(model)
        if table is not None and table.height and parent_ids:
            self._tables[model] = table.filter(~pl.col(parent_key).is_in(list(parent_ids)))
        if new_rows.height > 0 and new_rows.width > 0:
            existing = self._tables.get(model)
            if existing is None or not existing.height:
                self._tables[model] = new_rows
            else:
                keep = existing.filter(~pl.col(child_key).is_in(new_rows[child_key].to_list()))
                self._tables[model] = pl.concat([keep, new_rows])
        state = self._state.setdefault(state_scope.model_name, {})
        for record in state_records:
            state[record.record_key] = StateValue(record.input_fingerprint, record.code_version)
        return new_rows.height

    def materialize_incremental(self, model, frame, *, key_col, on_schema_change, options):
        self.calls.append(("materialize_incremental", key_col, frame.height))
        existing = self._tables.get(model)
        if existing is None or not existing.height:
            self._tables[model] = frame
        else:
            keep = existing.filter(~pl.col(key_col).is_in(frame[key_col].to_list()))
            self._tables[model] = pl.concat([keep, frame])
        return frame.height

    def materialize_full(self, model, frame, *, options):
        self.calls.append(("materialize_full", frame.height))
        self._tables[model] = frame
        return frame.height

    def upsert_state(self, scope, records):
        from stel.adapters import StateValue

        self.calls.append(("upsert_state", sorted(r.record_key for r in records)))
        state = self._state.setdefault(scope.model_name, {})
        for record in records:
            state[record.record_key] = StateValue(record.input_fingerprint, record.code_version)

    def replace_state(self, scope, records):
        from stel.adapters import StateValue

        self.calls.append(("replace_state", sorted(r.record_key for r in records)))
        self._state[scope.model_name] = {
            record.record_key: StateValue(record.input_fingerprint, record.code_version)
            for record in records
        }


def _incremental_model(name: str, module: str, depends_on: list[str]):
    from stel.config.model import ModelConfig

    return ModelConfig(
        name=name,
        depends_on=depends_on,
        transform={"type": "python", "module": module},
        materialization="incremental",
    )


def _resolved(tmp_path: Path):
    from stel.adapters.duckdb import DuckDBWarehouseConfig
    from stel.profile import ResolvedProfile

    return ResolvedProfile(
        profile_name="p",
        target_name="dev",
        warehouse=DuckDBWarehouseConfig(path=tmp_path / "db.duckdb"),
        llm=None,
        source_paths={},
        profiles_path=tmp_path / "profiles.yml",
    )


def _run_incremental(tmp_path: Path, adapter, model, *, full_refresh: bool = False):
    from stel.config.project import ProjectConfig
    from stel.execution.transform import run_transform_model

    return run_transform_model(
        model=model,
        project=ProjectConfig(name="p"),
        project_dir=tmp_path,
        adapter=adapter,
        resolved=_resolved(tmp_path),
        full_refresh=full_refresh,
    )


def test_bigquery_shaped_incremental_orchestration(tmp_path: Path) -> None:
    (tmp_path / "transforms").mkdir()
    (tmp_path / "transforms" / "word_tokens.py").write_text(_TRANSFORM_SOURCE)
    model = _incremental_model("word_tokens", "transforms.word_tokens", ["ref('documents')"])

    upstream = {
        "documents": pl.DataFrame(
            {"document_id": ["docA", "docB"], "body": ["alpha beta gamma", "delta"]}
        )
    }
    adapter = _RecordingAdapter(upstream)

    first = _run_incremental(tmp_path, adapter, model)
    assert first.documents_processed == 2
    # All-new parents: no parent-keyed delete, one replace_children (no parent_ids).
    assert not [c for c in adapter.calls if c[0] == "delete_rows_and_state"]
    rc_calls = [c for c in adapter.calls if c[0] == "replace_children"]
    assert rc_calls == [("replace_children", "word_id", [], 4)]

    # docA changes (fewer words), docB is removed.
    adapter.calls.clear()
    adapter._upstream["documents"] = pl.DataFrame(
        {"document_id": ["docA"], "body": ["alpha"]}
    )
    second = _run_incremental(tmp_path, adapter, model)
    assert second.documents_processed == 1
    assert second.documents_deleted == 1
    # Removed parent goes through delete_rows_and_state; changed parent through replace_children.
    delete_calls = [c for c in adapter.calls if c[0] == "delete_rows_and_state"]
    assert delete_calls == [("delete_rows_and_state", "document_id", ["docB"])]
    assert [c for c in adapter.calls if c[0] == "replace_children"] == [
        ("replace_children", "word_id", ["docA"], 1)
    ]

    # Nothing changes: no warehouse mutation at all.
    adapter.calls.clear()
    third = _run_incremental(tmp_path, adapter, model)
    assert third.documents_processed == 0
    assert third.documents_skipped == 1
    mutating = {"delete_rows_and_state", "replace_children"}
    assert not [c for c in adapter.calls if c[0] in mutating]


def test_reference_dep_change_reprocesses_every_parent(tmp_path: Path) -> None:
    (tmp_path / "transforms").mkdir()
    (tmp_path / "transforms" / "filtered.py").write_text(_REF_TRANSFORM_SOURCE)
    model = _incremental_model(
        "filtered", "transforms.filtered", ["ref('docs')", "ref('vocab')"]
    )

    adapter = _RecordingAdapter(
        {
            "docs": pl.DataFrame(
                {"document_id": ["docA", "docB"], "body": ["alpha beta", "alpha"]}
            ),
            "vocab": pl.DataFrame({"term": ["alpha"]}),
        }
    )

    first = _run_incremental(tmp_path, adapter, model)
    assert first.documents_processed == 2

    # The documents are byte-identical, but the reference vocab changed — every
    # parent's fingerprint moves, so all parents are reprocessed via replace_children.
    adapter.calls.clear()
    adapter._upstream["vocab"] = pl.DataFrame({"term": ["beta"]})
    second = _run_incremental(tmp_path, adapter, model)
    assert second.documents_processed == 2
    assert second.documents_skipped == 0
    # No removed parents → no delete_rows_and_state; changed parents go to replace_children.
    assert not [c for c in adapter.calls if c[0] == "delete_rows_and_state"]
    rc_calls = [c for c in adapter.calls if c[0] == "replace_children"]
    assert len(rc_calls) == 1
    assert rc_calls[0][1] == "row_id"
    assert rc_calls[0][2] == ["docA", "docB"]


def test_preexisting_target_without_state_is_rebuilt(tmp_path: Path) -> None:
    # A transform switched from `materialization: full` has a populated target
    # but no per-parent state. Incremental must rebuild (full replace) rather
    # than upsert over orphan children (Codex review P1).
    (tmp_path / "transforms").mkdir()
    (tmp_path / "transforms" / "word_tokens.py").write_text(_TRANSFORM_SOURCE)
    model = _incremental_model("word_tokens", "transforms.word_tokens", ["ref('documents')"])

    stale = pl.DataFrame(
        {
            "word_id": ["stale"],
            "document_id": ["docA"],
            "position": [99],
            "word": ["orphan"],
        }
    )
    adapter = _RecordingAdapter(
        {"documents": pl.DataFrame({"document_id": ["docA"], "body": ["alpha beta"]})},
        preexisting={"word_tokens": stale},
    )

    result = _run_incremental(tmp_path, adapter, model)
    assert result.documents_processed == 1
    # Full replace, not an upsert, so the orphan row is gone and state is reset.
    assert [c[0] for c in adapter.calls if c[0].startswith("materialize")] == [
        "materialize_full"
    ]
    assert any(c[0] == "replace_state" for c in adapter.calls)
    words = set(adapter._tables["word_tokens"]["word"].to_list())
    assert words == {"alpha", "beta"}


def test_insert_overwrite_strategy_is_rejected(tmp_path: Path) -> None:
    from types import SimpleNamespace

    (tmp_path / "transforms").mkdir()
    (tmp_path / "transforms" / "word_tokens.py").write_text(_TRANSFORM_SOURCE)
    model = _incremental_model("word_tokens", "transforms.word_tokens", ["ref('documents')"])

    adapter = _RecordingAdapter(
        {"documents": pl.DataFrame({"document_id": ["docA"], "body": ["alpha"]})},
        warehouse_options=SimpleNamespace(incremental_strategy="insert_overwrite"),
    )
    with pytest.raises(RunError, match="insert_overwrite"):
        _run_incremental(tmp_path, adapter, model)
    # Rejected before any warehouse mutation.
    assert not adapter.calls
