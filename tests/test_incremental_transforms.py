"""Incremental Python transforms with child-row deletion semantics (issue #218).

Exercises the generic one-to-many machinery end to end on DuckDB through a
project-local transform, so it needs no optional NLP extras: `word_tokens`
explodes a document's body into one stable child row per word.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, cast

import duckdb
import polars as pl
import pytest

from stel.adapters import StateValue
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


class _FakeSnapshot:
    def __init__(self, table, batch_size: int) -> None:
        self._table = table
        self._batch_size = batch_size
        self.schema = table.schema

    def __iter__(self):
        yield from self._table.to_batches(max_chunksize=self._batch_size)


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
        self.calls.append(("read_table", name))
        return self._upstream[name]

    @contextlib.contextmanager
    def table_snapshot(
        self,
        table: str,
        *,
        columns=None,
        batch_size: int = 10_000,
        predicate=None,
        key_column=None,
    ):
        """The streamed, predicate-pushed read the incremental classifier uses
        (issue #385). Filtering here mirrors what a real adapter pushes into
        SQL, so a test that scopes a read exercises the same path."""
        from stel.adapters.base import ReadPredicate, ReadPredicateOperator

        frame = self._upstream[table]
        predicates = (
            []
            if predicate is None
            else [predicate]
            if isinstance(predicate, ReadPredicate)
            else list(predicate)
        )
        for entry in predicates:
            if entry.operator is not ReadPredicateOperator.IN:
                raise AssertionError(f"unsupported predicate {entry.operator}")
            frame = frame.filter(pl.col(entry.column).is_in(list(entry.value)))
        if columns is not None:
            frame = frame.select(list(columns))
        self.calls.append(("table_snapshot", table, len(predicates)))
        yield _FakeSnapshot(frame.to_arrow(), batch_size)


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


# ─── keyed reference deps (issue #364) ───────────────────────────────────────

_KEYED_REF_TRANSFORM_SOURCE = '''
from __future__ import annotations

import polars as pl

from stel.hashing import canonical_fingerprint
from stel.transforms import IncrementalContract, ReferenceDep

_SCHEMA = {
    "row_id": pl.String(),
    "document_id": pl.String(),
    "word": pl.String(),
    "label": pl.String(),
}


def declared_incremental_contract(options):
    return IncrementalContract(
        parent_key="document_id",
        child_key="row_id",
        parent_source="docs",
        parent_source_key="document_id",
        reference_deps=(ReferenceDep("registry", join_key="document_id"),),
    )


def run(deps, ctx=None):
    labels = {
        str(row["document_id"]): str(row["label"])
        for row in deps["registry"].iter_rows(named=True)
    }
    rows = []
    for record in deps["docs"].iter_rows(named=True):
        document_id = str(record["document_id"])
        for position, word in enumerate(str(record["body"]).split()):
            rows.append(
                {
                    "row_id": canonical_fingerprint(
                        {"document_id": document_id, "position": position},
                        domain="test.keyedref",
                    ),
                    "document_id": document_id,
                    "word": word,
                    "label": labels.get(document_id, "unlabeled"),
                }
            )
    return pl.DataFrame(rows, schema=_SCHEMA)
'''


def _keyed_ref_setup(
    tmp_path: Path, registry: pl.DataFrame
) -> tuple[Any, Any]:
    (tmp_path / "transforms").mkdir()
    (tmp_path / "transforms" / "labeled.py").write_text(
        _KEYED_REF_TRANSFORM_SOURCE, encoding="utf-8"
    )
    model = _incremental_model(
        "labeled", "transforms.labeled", ["ref('docs')", "ref('registry')"]
    )
    adapter = _RecordingAdapter(
        {
            "docs": pl.DataFrame(
                {"document_id": ["docA", "docB"], "body": ["alpha beta", "gamma"]}
            ),
            "registry": registry,
        }
    )
    return model, adapter


def test_keyed_reference_change_reprocesses_only_its_parents(tmp_path: Path) -> None:
    model, adapter = _keyed_ref_setup(
        tmp_path,
        pl.DataFrame({"document_id": ["docA", "docB"], "label": ["x", "y"]}),
    )

    first = _run_incremental(tmp_path, adapter, model)
    assert first.documents_processed == 2

    # Only docB's registry row changes: docA must be skipped, and docB's
    # children replaced with the new label — the corpus-wide reprojection
    # this contract shape previously forced (issue #364).
    adapter.calls.clear()
    adapter._upstream["registry"] = pl.DataFrame(
        {"document_id": ["docA", "docB"], "label": ["x", "y2"]}
    )
    second = _run_incremental(tmp_path, adapter, model)
    assert second.documents_processed == 1
    assert second.documents_skipped == 1
    rc_calls = [c for c in adapter.calls if c[0] == "replace_children"]
    assert len(rc_calls) == 1
    assert rc_calls[0][2] == ["docB"]
    table = adapter._tables["labeled"]
    assert set(table.filter(pl.col("document_id") == "docB")["label"]) == {"y2"}
    assert set(table.filter(pl.col("document_id") == "docA")["label"]) == {"x"}


def test_keyed_reference_gaining_and_losing_rows_moves_only_that_parent(
    tmp_path: Path,
) -> None:
    # docB starts with no registry row: the empty-group fingerprint.
    model, adapter = _keyed_ref_setup(
        tmp_path, pl.DataFrame({"document_id": ["docA"], "label": ["x"]})
    )

    first = _run_incremental(tmp_path, adapter, model)
    assert first.documents_processed == 2
    assert set(adapter._tables["labeled"].filter(pl.col("document_id") == "docB")["label"]) == {
        "unlabeled"
    }

    # docB gains its first registry row: only docB reprocesses.
    adapter.calls.clear()
    adapter._upstream["registry"] = pl.DataFrame(
        {"document_id": ["docA", "docB"], "label": ["x", "y"]}
    )
    second = _run_incremental(tmp_path, adapter, model)
    assert second.documents_processed == 1
    assert second.documents_skipped == 1
    assert next(c for c in adapter.calls if c[0] == "replace_children")[2] == ["docB"]

    # And loses it again: only docB reprocesses back to the empty group.
    adapter.calls.clear()
    adapter._upstream["registry"] = pl.DataFrame(
        {"document_id": ["docA"], "label": ["x"]}
    )
    third = _run_incremental(tmp_path, adapter, model)
    assert third.documents_processed == 1
    assert third.documents_skipped == 1
    assert set(adapter._tables["labeled"].filter(pl.col("document_id") == "docB")["label"]) == {
        "unlabeled"
    }


def test_keyed_reference_row_for_absent_parent_is_inert(tmp_path: Path) -> None:
    model, adapter = _keyed_ref_setup(
        tmp_path,
        pl.DataFrame({"document_id": ["docA", "docB"], "label": ["x", "y"]}),
    )
    first = _run_incremental(tmp_path, adapter, model)
    assert first.documents_processed == 2

    # A registry row keyed to a parent that does not exist cannot contribute
    # to any current parent's children under the declared join semantics.
    adapter.calls.clear()
    adapter._upstream["registry"] = pl.DataFrame(
        {"document_id": ["docA", "docB", "docZ"], "label": ["x", "y", "z"]}
    )
    second = _run_incremental(tmp_path, adapter, model)
    assert second.documents_processed == 0
    assert second.documents_skipped == 2
    assert not [
        c for c in adapter.calls if c[0] in {"replace_children", "delete_rows_and_state"}
    ]


def test_keyed_reference_null_join_key_is_rejected(tmp_path: Path) -> None:
    model, adapter = _keyed_ref_setup(
        tmp_path,
        pl.DataFrame({"document_id": ["docA", None], "label": ["x", "y"]}),
    )
    with pytest.raises(RunError, match="null or empty"):
        _run_incremental(tmp_path, adapter, model)
    mutating = {"delete_rows_and_state", "replace_children", "materialize_full"}
    assert not [c for c in adapter.calls if c[0] in mutating]


def test_keyed_reference_missing_join_key_column_is_rejected(tmp_path: Path) -> None:
    model, adapter = _keyed_ref_setup(
        tmp_path, pl.DataFrame({"doc": ["docA"], "label": ["x"]})
    )
    with pytest.raises(RunError, match="join_key column 'document_id'"):
        _run_incremental(tmp_path, adapter, model)


def test_keyless_reference_dep_fingerprints_match_string_form(tmp_path: Path) -> None:
    """`ReferenceDep(name)` without a join_key must be byte-identical to the
    plain-string form, so normalizing an existing contract re-keys nothing."""
    keyless_source = _REF_TRANSFORM_SOURCE.replace(
        'reference_deps=("vocab",),',
        'reference_deps=(ReferenceDep("vocab"),),',
    ).replace(
        "from stel.transforms import IncrementalContract",
        "from stel.transforms import IncrementalContract, ReferenceDep",
    )
    assert 'ReferenceDep("vocab")' in keyless_source

    upstream = {
        "docs": pl.DataFrame({"document_id": ["docA"], "body": ["alpha beta"]}),
        "vocab": pl.DataFrame({"term": ["alpha"]}),
    }
    fingerprints: list[str] = []
    for label, source in (("str", _REF_TRANSFORM_SOURCE), ("dep", keyless_source)):
        subdir = tmp_path / label
        (subdir / "transforms").mkdir(parents=True)
        (subdir / "transforms" / "filtered.py").write_text(source, encoding="utf-8")
        model = _incremental_model(
            "filtered", "transforms.filtered", ["ref('docs')", "ref('vocab')"]
        )
        adapter = _RecordingAdapter({k: v.clone() for k, v in upstream.items()})
        _run_incremental(subdir, adapter, model)
        state_value = adapter._state["filtered"]["docA"]
        assert isinstance(state_value, StateValue)
        fingerprints.append(state_value.input_fingerprint)

    assert fingerprints[0] == fingerprints[1]


def test_reference_dep_contract_validation() -> None:
    from stel.transforms import IncrementalContract, ReferenceDep

    def contract(*refs: Any) -> IncrementalContract:
        return IncrementalContract(
            parent_key="document_id",
            child_key="row_id",
            parent_source="docs",
            parent_source_key="document_id",
            reference_deps=tuple(refs),
        )

    contract("vocab", ReferenceDep("registry", join_key="document_id")).validate_against(
        ["docs", "vocab", "registry"]
    )

    with pytest.raises(ValueError, match=r"ReferenceDep\.name"):
        contract(ReferenceDep("")).validate_against(["docs"])
    with pytest.raises(ValueError, match=r"ReferenceDep\.join_key"):
        contract(ReferenceDep("vocab", join_key=" ")).validate_against(["docs", "vocab"])
    with pytest.raises(ValueError, match="duplicates"):
        contract("vocab", ReferenceDep("vocab", join_key="k")).validate_against(
            ["docs", "vocab"]
        )
    with pytest.raises(ValueError, match="must not also be a reference_dep"):
        contract(ReferenceDep("docs", join_key="k")).validate_against(["docs"])


# ─── transform memory: streaming parent groups (issue #383) ─────────────────


def test_parent_fingerprint_properties_survive_the_digest_encoding() -> None:
    """The fingerprint decides whether a parent is reprocessed, so a change to
    how rows are combined re-keys every parent in every project at once.
    #385 made that change deliberately (rows -> per-row digests, so the corpus
    need not be resident), and `version=2` plus the goldens in
    `test_frozen_names.py` are what make it loud. What must *not* drift are
    the three properties the encoding has always had.
    """
    from stel.execution.transform import _parent_fingerprint, _row_digest

    rows = [
        {"parent_id": "p1", "n": 2, "s": "b"},
        {"parent_id": "p1", "n": 1, "s": "a"},
    ]
    digests = [_row_digest(row) for row in rows]
    fingerprint = _parent_fingerprint("p1", digests, {"ref": "abc"})

    # 1. Order-insensitive within a parent.
    assert _parent_fingerprint("p1", list(reversed(digests)), {"ref": "abc"}) == fingerprint
    # 2. Multiplicity counts — a repeated row is not absorbed, which an XOR or
    #    sum accumulator would have done silently.
    assert _parent_fingerprint("p1", digests + digests[:1], {"ref": "abc"}) != fingerprint
    # 3. Every column participates: no narrowing of what invalidates a parent.
    for column in ("parent_id", "n", "s"):
        changed = dict(rows[0]) | {column: "different"}
        assert _row_digest(changed) != digests[0]
    # And the parent key and reference fingerprints still bind.
    assert _parent_fingerprint("p2", digests, {"ref": "abc"}) != fingerprint
    assert _parent_fingerprint("p1", digests, {"ref": "xyz"}) != fingerprint


def test_the_parent_read_is_scoped_to_the_changed_parents(tmp_path: Path) -> None:
    """The point of #385. Classification streams, and rows come back only for
    the parents that changed — so an incremental run moves data proportional
    to the change set, not the corpus. Without this pinned, a refactor could
    reinstate the whole-table read and every behavioural test would still
    pass."""
    (tmp_path / "transforms").mkdir()
    (tmp_path / "transforms" / "word_tokens.py").write_text(_TRANSFORM_SOURCE)
    model = _incremental_model("word_tokens", "transforms.word_tokens", ["ref('documents')"])
    adapter = _RecordingAdapter(
        {
            "documents": pl.DataFrame(
                {"document_id": ["docA", "docB"], "body": ["alpha beta", "gamma"]}
            )
        }
    )

    _run_incremental(tmp_path, adapter, model)

    # Only docA changes.
    adapter.calls.clear()
    adapter._upstream["documents"] = pl.DataFrame(
        {"document_id": ["docA", "docB"], "body": ["alpha beta delta", "gamma"]}
    )
    result = _run_incremental(tmp_path, adapter, model)
    assert result.documents_processed == 1
    assert result.documents_skipped == 1

    snapshots = [call for call in adapter.calls if call[0] == "table_snapshot"]
    # One unfiltered scan to classify, then one predicate-scoped read for the
    # single changed parent.
    assert [call[2] for call in snapshots] == [0, 1]
    # And the parent table was never read whole.
    assert ("read_table", "documents") not in adapter.calls


def test_rows_that_changed_since_classification_are_refused(tmp_path: Path) -> None:
    """Classification and the scoped read are separate snapshots (issue #385).
    If the upstream is replaced between them, a parent would publish one
    generation's rows under another generation's fingerprint — state
    describing content the table no longer holds (Codex review)."""
    (tmp_path / "transforms").mkdir()
    (tmp_path / "transforms" / "word_tokens.py").write_text(_TRANSFORM_SOURCE)
    model = _incremental_model("word_tokens", "transforms.word_tokens", ["ref('documents')"])

    class _ShiftingAdapter(_RecordingAdapter):
        """Swaps the parent table's contents after classification has read it."""

        def __init__(self, upstream, replacement):
            super().__init__(upstream)
            self._replacement = replacement
            self._scans = 0

        @contextlib.contextmanager
        def table_snapshot(self, table, **kwargs):
            with super().table_snapshot(table, **kwargs) as snapshot:
                yield snapshot
            self._scans += 1
            if self._scans == 1:
                self._upstream[table] = self._replacement

    adapter = _ShiftingAdapter(
        {"documents": pl.DataFrame({"document_id": ["docA"], "body": ["alpha"]})},
        pl.DataFrame({"document_id": ["docA"], "body": ["totally different"]}),
    )

    with pytest.raises(RunError, match="changed between classification"):
        _run_incremental(tmp_path, adapter, model)


def _digests(frame, key_col: str = "parent_id", model: str = "m"):
    """Classification against a fake adapter serving `frame` (issue #385)."""
    from stel.execution.transform import _stream_parent_digests

    adapter = cast(Any, _RecordingAdapter({"parents": frame}))
    return _stream_parent_digests(adapter, "parents", key_col, model_name=model)


def test_classification_preserves_first_appearance_parent_order() -> None:
    """Processed-parent order is observable — it drives `replace_children`
    batching and the reported counts — so it must not drift with the read."""
    import polars as pl

    frame = pl.DataFrame(
        {"parent_id": ["b", "a", "b", "c"], "n": [1, 2, 3, 4]}
    )

    assert list(_digests(frame)) == ["b", "a", "c"]


def test_classification_keeps_digests_not_rows() -> None:
    """The fix for #385: what survives the scan is a digest per row, not the
    row. A refactor that accumulated rows again would put the corpus back in
    memory while every behavioural test still passed."""
    import polars as pl

    frame = pl.DataFrame({"parent_id": ["a", "a", "b"], "n": [1, 2, 3]})

    digests = _digests(frame)

    assert [len(group) for group in digests.values()] == [2, 1]
    for group in digests.values():
        assert all(isinstance(entry, str) for entry in group)


def test_an_unusable_parent_key_is_still_refused() -> None:
    import polars as pl

    from stel.runner import RunError

    for bad in (None, "", "   "):
        frame = pl.DataFrame(
            {"parent_id": ["a", bad], "n": [1, 2]},
            schema={"parent_id": pl.String, "n": pl.Int64},
        )
        with pytest.raises(RunError, match="null or empty"):
            _digests(frame)


def test_a_missing_parent_key_column_is_refused() -> None:
    import polars as pl

    from stel.runner import RunError

    with pytest.raises(RunError, match="missing the parent key column"):
        _digests(pl.DataFrame({"other": ["a"]}))
    with pytest.raises(RunError, match="must be string-typed"):
        _digests(pl.DataFrame({"parent_id": [1]}))


# ─── batched commits (issue #379) ───────────────────────────────────────────


def _set_commit_every(project: Path, value: int) -> None:
    path = project / "models" / "word_tokens.yml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "      module: transforms.word_tokens\n",
            f"      module: transforms.word_tokens\n      commit_every: {value}\n",
        ),
        encoding="utf-8",
    )


def test_batching_does_not_change_what_gets_published(tmp_path: Path) -> None:
    """Same rows, same state, whether committed in one batch or four."""
    (tmp_path / "one").mkdir()
    (tmp_path / "many").mkdir()
    single = _project(tmp_path / "one")
    batched = _project(tmp_path / "many")
    _set_commit_every(batched, 1)
    for project in (single, batched):
        for index in range(4):
            _write_doc(project, f"d{index}.json", f"word{index} shared")
        run_project(project)

    assert _tokens(single) == _tokens(batched)
    assert _state_keys(single) == _state_keys(batched)


def test_a_failure_mid_run_keeps_the_batches_that_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of #379. Before this, a failure at the last parent re-paid the
    whole corpus; now the parents whose state advanced stay done."""
    from stel.adapters.duckdb import DuckDBAdapter

    project = _project(tmp_path)
    _set_commit_every(project, 1)
    for index in range(4):
        _write_doc(project, f"d{index}.json", f"word{index}")

    real = DuckDBAdapter.replace_children
    calls = {"n": 0}

    def _fail_on_third(self: Any, *args: Any, **kwargs: Any) -> int:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("warehouse blew up mid-run")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(DuckDBAdapter, "replace_children", _fail_on_third)
    with pytest.raises(Exception, match="warehouse blew up"):
        run_project(project)

    # Two batches committed before the failure, and their state advanced.
    survived = _state_keys(project)
    assert len(survived) == 2

    monkeypatch.undo()
    results = run_project(project)

    # The relaunch reclassifies only the parents that never committed.
    assert _result(results, "word_tokens").documents_processed == 2
    assert len(_state_keys(project)) == 4
    assert len(_tokens(project)) == 4


def test_a_run_smaller_than_the_batch_size_commits_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default is high enough that ordinary projects keep one MERGE.

    Batching that split every small run into many warehouse round trips would
    trade a rare failure cost for a constant one.
    """
    from stel.adapters.duckdb import DuckDBAdapter

    project = _project(tmp_path)
    for index in range(3):
        _write_doc(project, f"d{index}.json", f"word{index}")

    real = DuckDBAdapter.replace_children
    calls = {"n": 0}

    def _counting(self: Any, *args: Any, **kwargs: Any) -> int:
        calls["n"] += 1
        return real(self, *args, **kwargs)

    monkeypatch.setattr(DuckDBAdapter, "replace_children", _counting)
    run_project(project)

    assert calls["n"] == 1


@pytest.mark.parametrize("uses_llm", [False, True])
def test_commit_every_stays_out_of_code_version(tmp_path: Path, uses_llm: bool) -> None:
    """Both branches of the code-version payload, and that is the point.

    `uses_llm` transforms build their own `effective_transform`, so an
    exclusion applied only to the fallback branch left the dial inside the
    hash for exactly the models where a needless reprocess costs the most —
    every parent back through inference (Codex review). The non-LLM case alone
    passed while the bug was live.
    """
    from stel.config import load_project
    from stel.versioning import compute_model_code_version

    project = _project(tmp_path)
    _write_doc(project, "d0.json", "word0")
    project_config, _sources, models = load_project(project)
    model = next(item for item in models if item.name == "word_tokens")
    assert model.transform is not None
    if uses_llm:
        model = model.model_copy(
            update={"transform": model.transform.model_copy(update={"uses_llm": True})}
        )
        assert model.transform is not None

    baseline = compute_model_code_version(model, project_config, project)
    tuned = model.model_copy(
        update={"transform": model.transform.model_copy(update={"commit_every": 7})}
    )
    changed = model.model_copy(
        update={"transform": model.transform.model_copy(update={"module": "other"})}
    )

    assert compute_model_code_version(tuned, project_config, project) == baseline
    assert compute_model_code_version(changed, project_config, project) != baseline


def test_a_later_batch_that_adds_a_column_still_reconciles_schema(
    tmp_path: Path,
) -> None:
    """A transform's output schema can be data-dependent, so a later batch may
    emit a column the first never did. Forcing `ignore` after the first batch
    dropped it silently while still advancing state, making the loss
    unrecoverable (Codex review). Under `fail` the drift must surface."""
    project = _project(tmp_path)
    _set_commit_every(project, 1)
    for index in range(3):
        _write_doc(project, f"d{index}.json", f"word{index}")
    # A first run is a full materialization, so it never batches. The state it
    # leaves is what makes the next run incremental.
    run_project(project)

    transform = project / "transforms" / "word_tokens.py"
    widened = chr(10).join(
        [
            "    frame = pl.DataFrame(rows, schema=_SCHEMA)",
            "    # Every parent but the first emits a column the first never did.",
            "    if rows and rows[0]['word'] != 'word0':",
            "        frame = frame.with_columns(extra=pl.lit('x'))",
            "    return frame",
        ]
    )
    transform.write_text(
        transform.read_text(encoding="utf-8").replace(
            "    return pl.DataFrame(rows, schema=_SCHEMA)", widened
        ),
        encoding="utf-8",
    )

    # The edited module changes code_version, so every parent reprocesses —
    # batch 1 (d0) emits the original schema, batch 2 (d1) adds a column.
    with pytest.raises(Exception, match="Schema change"):
        run_project(project)


def test_commit_every_does_not_invalidate_existing_state(tmp_path: Path) -> None:
    """It changes execution cadence, never output content — so it must stay out
    of code_version, exactly like extraction's flush_every. Including it would
    re-run every corpus on the run after an operator tunes it."""
    project = _project(tmp_path)
    for index in range(3):
        _write_doc(project, f"d{index}.json", f"word{index}")
    run_project(project)

    _set_commit_every(project, 1)
    results = run_project(project)

    assert _result(results, "word_tokens").documents_processed == 0
    assert _result(results, "word_tokens").documents_skipped == 3
