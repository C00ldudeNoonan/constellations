"""The bounded-memory contract, and the audit that keeps it honest (issue #414).

One root cause has now produced four incidents on the same corpus: a stage
holds something proportional to the corpus rather than to its flush window.
Transforms (#383/#385), embed output (#401/#402), the embed resume lookup
(#407), embed input (#410), and BigQuery's snapshot key check (#418) were each
fixed individually and each fix was right — but the sequence says the missing
piece is an invariant, not another patch.

**The contract.** For every model kind, peak memory is O(flush window) +
O(per-parent unit) + O(distinct keys), never O(corpus **bytes**).

The key term is deliberate and these tests do not cover it: stages that
reconcile deletions hold every id at once (~108 bytes each, so ~370MB for a
3.6M-row corpus), and a cumulative container like that is invisible to a
per-frame measurement. What is asserted here is the O(corpus bytes) failure the
incidents actually were. `docs/architecture/bounded-memory.md` carries the key
term and its numbers, and issue #428 tracks removing it.

`adapter.read_table()` is the primitive that breaks it: it is `SELECT *` into
one Polars frame, so any call on a corpus-scale relation is corpus-scale
residency. The table below classifies every call site in `src/stel`, and the
scan underneath fails when a new one appears unclassified — the point being
that a new whole-table read has to be an argued decision rather than the
default that four incidents made it.

Not a substitute for measuring: `_read_table_sites` sees only what the source
says. #418 was corpus-sized buffering inside BigQuery, from an unpartitioned
`OVER()` attached to a read that was streamed on stel's side — the same
invariant broken by a different owner, invisible to any call-site audit. The
residency tests at the bottom cover the property; this table covers the shape.

**Why residency and not memory.** The obvious gate — run a stage under a memory
ceiling — measures peak working set, which is an allocator high-water mark
rather than what is held. Measured on the DuckDB read path, a streaming loop
grows +203MB across a 4x corpus purely from allocating and freeing per-batch
frames, against +528MB for a whole-table read: real separation, but a factor of
1.7 on a number that also moves with the platform allocator. That is not a
margin to fail a build on. Counting the rows a stage ever materializes at once
measures the invariant directly and is exact, so that is what these assert.
"""
from __future__ import annotations

import ast
import json
import pathlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from stel.adapters.duckdb import DuckDBAdapter
from stel.execution import transform as transform_module
from stel.execution.embed import _INPUT_BATCH_ROWS
from stel.runner import run_project

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "stel"

# A read whose residency is bounded by something other than the corpus.
BOUNDED = "bounded"
# A whole-table read that is correct for what the stage does, with the reason
# stated. These are decisions, not oversights.
EXCEPTION = "exception"
# A whole-table read that should be bounded and is not, with the issue that
# tracks it. This list may only shrink — see `test_the_unbounded_list_only_shrinks`.
GAP = "gap"

# (module, qualname, call_count, verdict, why)
_READ_TABLE_SITES: tuple[tuple[str, str, int, str, str], ...] = (
    (
        "cli.py",
        "show",
        1,
        BOUNDED,
        "`stel show` passes an explicit row limit; the read is capped by the "
        "operator's `--limit`, not by the relation",
    ),
    (
        "execution/embed.py",
        "run_embed_model",
        1,
        BOUNDED,
        "a zero-row probe for column names and dtypes; the rows stream in "
        "batches to fill each flush window (issue #410)",
    ),
    (
        "execution/embed.py",
        "_EmbeddingReuseReader._load_keys",
        1,
        BOUNDED,
        "a zero-row probe for the reuse-column contract; the id column streams "
        "projected and reuse candidates are fetched a window at a time (#407)",
    ),
    (
        "retrieval_eval.py",
        "_run_one",
        1,
        BOUNDED,
        "a golden set is a hand-curated list of labeled queries, bounded by "
        "what a human wrote rather than by the corpus it evaluates",
    ),
    (
        "execution/chunk.py",
        "run_chunk_model",
        1,
        BOUNDED,
        "a zero-row probe for the column contract; documents stream in batches "
        "and chunks publish every `flush_every` documents (issue #423)",
    ),
    (
        "execution/llm.py",
        "run_llm_model",
        1,
        GAP,
        "issue #424: reads the whole upstream into one frame before the first "
        "provider call — the #410 hole, unfixed for llm",
    ),
    (
        "execution/llm.py",
        "_existing_llm_id_values",
        1,
        GAP,
        "issue #424: reads the entire existing target, generated text columns "
        "included, to build a map of id values — the #407 hole, unfixed for llm",
    ),
    (
        "classic_ml/classifier.py",
        "_run_classifier",
        1,
        EXCEPTION,
        "scikit-learn fits one matrix; training is not incremental here, so the "
        "training set is resident by definition. Bounding this means changing "
        "what the stage is, not how it reads",
    ),
    (
        "classic_ml/text.py",
        "_run_features",
        1,
        EXCEPTION,
        "same as the classifier: feature extraction fits a vectorizer over the "
        "whole training set",
    ),
    (
        "execution/eval.py",
        "run_eval_model",
        3,
        EXCEPTION,
        "predictions and expected are joined by key and scored in memory, and "
        "the third read is the existing metric table, whose cardinality is "
        "metrics x labels x versions rather than corpus rows. Streaming the "
        "join is real work, not a projection fix",
    ),
    (
        "execution/transform.py",
        "run_transform_model",
        1,
        EXCEPTION,
        "non-parent dependencies. #385 bounded the parent source only, and the "
        "python transform contract hands the user whole DataFrames, so a large "
        "secondary dependency is corpus-scale by the interface stel offers",
    ),
    (
        "execution/transform.py",
        "_read_parent_rows",
        2,
        EXCEPTION,
        "the full-refresh branch: rebuilding every parent needs every parent. "
        "The incremental branch beside it is keyed and chunked (#385)",
    ),
    (
        "concept_cloud/export.py",
        "export_concept_cloud",
        6,
        EXCEPTION,
        "an artifact export, not a pipeline stage: it builds one self-contained "
        "file from several models at once and has no flush window to be bounded "
        "by. Reads embeddings whole, so it is corpus-scale on a large project",
    ),
)


class _ReadTableVisitor(ast.NodeVisitor):
    """Counts `.read_table(...)` calls, keyed by the scope that encloses them."""

    def __init__(self, module: str, found: dict[tuple[str, str], int]) -> None:
        self._module = module
        self._found = found
        self._scope: list[str] = []

    def _scoped(self, node: ast.AST) -> None:
        self._scope.append(getattr(node, "name", "?"))
        self.generic_visit(node)
        self._scope.pop()

    visit_FunctionDef = _scoped
    visit_AsyncFunctionDef = _scoped
    visit_ClassDef = _scoped

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "read_table":
            key = (self._module, ".".join(self._scope) or "<module>")
            self._found[key] = self._found.get(key, 0) + 1
        self.generic_visit(node)


@cache
def _read_table_sites() -> dict[tuple[str, str], int]:
    """Every `.read_table(...)` call in `src/stel`, keyed by module and scope.

    Keyed by enclosing qualname rather than line number so ordinary edits above
    a call site do not churn the table. Cached: every test here asks the same
    question, and parsing the package once per assertion is the difference
    between a fast check and one people skip.
    """
    found: dict[tuple[str, str], int] = {}
    for path in sorted(_SRC.rglob("*.py")):
        module = path.relative_to(_SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        _ReadTableVisitor(module, found).visit(tree)
    return found


_DECLARED: frozenset[tuple[str, str]] = frozenset(
    (module, qualname) for module, qualname, *_rest in _READ_TABLE_SITES
)


def test_every_read_table_call_site_is_classified() -> None:
    """A new whole-table read must be argued for, not merely written.

    This is the teeth on the contract. `read_table` is `SELECT *` into one
    frame, so an unreviewed call on a corpus-scale relation is an O(corpus)
    peak — which is how the same bug reached production four times.
    """
    actual = _read_table_sites()
    undeclared = sorted(set(actual) - _DECLARED)
    assert not undeclared, (
        "New `read_table` call site(s) with no entry in _READ_TABLE_SITES: "
        f"{undeclared}. `read_table` reads the whole relation into one frame, "
        "so on a corpus-scale table it breaks the bounded-memory contract this "
        "module states (issue #414). Either read it bounded — a zero-row probe "
        "for schema, `table_snapshot` for rows — or add an entry saying why "
        "whole-table residency is correct here."
    )


def test_no_classified_call_site_has_disappeared() -> None:
    """The table describes the code, so a stale row is a lie about it."""
    actual = _read_table_sites()
    missing = sorted(_DECLARED - set(actual))
    assert not missing, (
        f"_READ_TABLE_SITES describes call site(s) that no longer exist: "
        f"{missing}. Remove the entries — a table that disagrees with the code "
        "stops being an audit."
    )


@pytest.mark.parametrize(
    ("module", "qualname", "count", "verdict", "why"),
    _READ_TABLE_SITES,
    ids=[f"{row[0]}::{row[1]}" for row in _READ_TABLE_SITES],
)
def test_call_site_count_is_pinned(
    module: str, qualname: str, count: int, verdict: str, why: str
) -> None:
    """A second `read_table` added beside a classified one is a new decision.

    Without this, `run_eval_model` could grow a fourth whole-table read under
    cover of the three already argued for.
    """
    actual = _read_table_sites().get((module, qualname), 0)
    assert actual == count, (
        f"{module}::{qualname} now makes {actual} read_table call(s), not "
        f"{count}. The classification on record is `{verdict}`: {why}. Update "
        "the count only if the new call is covered by that same reasoning."
    )


def test_the_unbounded_list_only_shrinks() -> None:
    """The gaps are tracked, and no new one may be added silently.

    Per #414: the list of stages that do not yet hold the contract may only
    shrink. Fixing #423 or #424 means deleting its row, not editing this
    number down after the fact.
    """
    gaps = sorted(
        f"{module}::{qualname}"
        for module, qualname, _count, verdict, _why in _READ_TABLE_SITES
        if verdict == GAP
    )
    assert gaps == [
        "execution/llm.py::_existing_llm_id_values",
        "execution/llm.py::run_llm_model",
    ], (
        "The set of stages known to break the bounded-memory contract changed. "
        "Removing one is the goal — delete its row. Adding one needs an issue "
        "and a line here saying why it shipped unbounded."
    )


def test_every_gap_names_its_tracking_issue() -> None:
    """A gap without a tracker is just a comment nobody will action."""
    for module, qualname, _count, verdict, why in _READ_TABLE_SITES:
        if verdict == GAP:
            assert "issue #" in why, (
                f"{module}::{qualname} is recorded as an unbounded gap but "
                "names no tracking issue."
            )


def test_every_verdict_is_one_of_the_three() -> None:
    for module, qualname, _count, verdict, _why in _READ_TABLE_SITES:
        assert verdict in {BOUNDED, EXCEPTION, GAP}, (
            f"{module}::{qualname} has an unknown verdict {verdict!r}"
        )


# ─── residency: what a stage materializes at once ───────────────────────────


@dataclass
class _Residency:
    """The largest single frame a stage materialized, and where it came from."""

    largest_frame_rows: int = 0
    largest_frame_source: str = ""
    largest_batch_rows: int = 0

    def record_frame(self, table: str, rows: int) -> None:
        if rows > self.largest_frame_rows:
            self.largest_frame_rows = rows
            self.largest_frame_source = f"read_table({table!r})"

    def record_batch(self, rows: int) -> None:
        self.largest_batch_rows = max(self.largest_batch_rows, rows)


class _CountingSnapshot:
    """Passes a snapshot through, recording each batch's size."""

    def __init__(self, inner: Any, seen: _Residency) -> None:
        self._inner = inner
        self._seen = seen

    def iter_batches(self) -> Iterator[Any]:
        """Traverse the snapshot, recording each batch's size.

        Named rather than left in `__iter__`: this drives the warehouse read.
        """
        for batch in self._inner:
            self._seen.record_batch(batch.num_rows)
            yield batch

    def __iter__(self) -> Iterator[Any]:
        # O(1): hands back the generator, does not consume it.
        return self.iter_batches()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@contextmanager
def _measure_residency(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Residency]:
    """Record every frame and batch the DuckDB adapter hands to a stage.

    Exact rather than statistical: the contract is about how many rows are
    alive at once, and that is a number the code states, not one an allocator
    reports.
    """
    seen = _Residency()
    real_read = DuckDBAdapter.read_table
    real_snapshot = DuckDBAdapter.table_snapshot

    def read_table(self: Any, table: str, *, limit: int | None = None) -> Any:
        frame = real_read(self, table, limit=limit)
        seen.record_frame(table, frame.height)
        return frame

    @contextmanager
    def table_snapshot(self: Any, table: str, **kwargs: Any) -> Iterator[Any]:
        with real_snapshot(self, table, **kwargs) as snapshot:
            yield _CountingSnapshot(snapshot, seen)

    monkeypatch.setattr(DuckDBAdapter, "read_table", read_table)
    monkeypatch.setattr(DuckDBAdapter, "table_snapshot", table_snapshot)
    yield seen


_CORPUS_DOCS = 40
_FLUSH_EVERY = 25


def _embed_project(root: Path, *, docs: int) -> Path:
    project = root / "project"
    (project / "models").mkdir(parents=True)
    (project / "sources").mkdir()
    (project / "data").mkdir()
    (project / "stel_project.yml").write_text(
        "name: bounded\nversion: '0.1.0'\nprofile: bounded\n", encoding="utf-8"
    )
    (project / "profiles.yml").write_text(
        "bounded:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n"
        "        schema: docs\n",
        encoding="utf-8",
    )
    (project / "sources" / "documents.yml").write_text(
        "version: 2\nsources:\n  - name: documents\n    path: data\n"
        "    file_pattern: '*.json'\n",
        encoding="utf-8",
    )
    (project / "models" / "models.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: document_registry\n"
        "    source: ref('documents')\n"
        "    extraction:\n      backend: json\n"
        "      options:\n        fields: [title, body]\n"
        f"      flush_every: {_FLUSH_EVERY}\n"
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
        "      vector_field: embedding\n      dimensions: 4\n      batch_size: 8\n"
        f"      flush_every: {_FLUSH_EVERY}\n"
        "    materialization: incremental\n",
        encoding="utf-8",
    )
    for index in range(docs):
        (project / "data" / f"doc{index}.json").write_text(
            json.dumps({"title": f"t{index}", "body": f"body {index}"}),
            encoding="utf-8",
        )
    return project


def test_extraction_never_materializes_more_than_its_flush_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extraction has been flush-bounded since #77; pinned here as a contract
    rather than an implementation detail."""
    project = _embed_project(tmp_path, docs=_CORPUS_DOCS)
    with _measure_residency(monkeypatch) as seen:
        run_project(project, select="document_registry")
    assert seen.largest_frame_rows <= _FLUSH_EVERY, (
        f"extraction materialized {seen.largest_frame_rows} rows at once via "
        f"{seen.largest_frame_source}, above its flush window of "
        f"{_FLUSH_EVERY}. Peak must follow the window, not the corpus (#414)."
    )


def test_chunk_never_materializes_the_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #423. Chunk is the worst place to hold a corpus: it sits upstream
    of embed and its input is the document registry, so a regression here is
    reached before any of embed's bounded-memory work matters. It also
    *amplifies* — one registry row becomes many chunk rows — so the accumulated
    output is larger than the input it came from."""
    project = _embed_project(tmp_path, docs=_CORPUS_DOCS)
    run_project(project, select="document_registry")
    with _measure_residency(monkeypatch) as seen:
        run_project(project, select="document_chunks")

    assert seen.largest_frame_rows == 0, (
        f"chunk materialized a {seen.largest_frame_rows}-row frame via "
        f"{seen.largest_frame_source}. Its only whole-relation read is a "
        "zero-row schema probe (#423); anything else is the corpus."
    )
    assert seen.largest_batch_rows > 0, "the streamed read did not run at all"


def test_embed_never_materializes_the_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The invariant #410 and #407 bought, stated as a property.

    A regression to `read_table(upstream)` shows up here as a frame the size of
    the chunk table, whatever the machine or allocator says about memory.
    """
    project = _embed_project(tmp_path, docs=_CORPUS_DOCS)
    run_project(project, select="document_registry")
    run_project(project, select="document_chunks")
    with _measure_residency(monkeypatch) as seen:
        run_project(project, select="document_embeddings")

    assert seen.largest_frame_rows == 0, (
        f"embed materialized a {seen.largest_frame_rows}-row frame via "
        f"{seen.largest_frame_source}. Its only whole-relation reads are "
        "zero-row schema probes (#410, #407); anything else is the corpus."
    )
    assert seen.largest_batch_rows > 0, "the streamed read did not run at all"


def test_embed_residency_is_capped_by_a_constant_not_the_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever embed holds at once is bounded by a declared constant.

    The companion to the assertion above, and the weaker of the two: below the
    read batch size a batch *is* the whole table, so this cannot separate a
    streamed read from a whole-table one on a fixture small enough for the
    ordinary suite — that separation is what `largest_frame_rows == 0` above
    provides, and it holds at any size. What this adds is the ceiling: the
    number of rows alive at once is `_INPUT_BATCH_ROWS`, a constant in the
    source, rather than anything derived from the input.
    """
    project = _embed_project(tmp_path, docs=_CORPUS_DOCS)
    run_project(project, select="document_registry")
    run_project(project, select="document_chunks")
    with _measure_residency(monkeypatch) as seen:
        run_project(project, select="document_embeddings")

    resident = max(seen.largest_frame_rows, seen.largest_batch_rows)
    assert resident <= _INPUT_BATCH_ROWS, (
        f"embed held {resident} rows at once, above the {_INPUT_BATCH_ROWS}-row "
        "read batch that is supposed to cap it (issue #414)."
    )


# ─── transform residency (issues #383, #385, #379) ──────────────────────────

_TRANSFORM_COMMIT_EVERY = 3

_TOKENS_TRANSFORM = """
from __future__ import annotations

import polars as pl

from stel.hashing import canonical_fingerprint
from stel.transforms import IncrementalContract, TransformContext


def declared_incremental_contract(options):
    return IncrementalContract(
        parent_key="document_id",
        child_key="token_id",
        parent_source_key="document_id",
    )


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    frame = next(iter(deps.values()))
    rows = []
    for record in frame.iter_rows(named=True):
        document_id = str(record["document_id"])
        for position, word in enumerate(str(record.get("body") or "").split()):
            rows.append(
                {
                    "token_id": canonical_fingerprint(
                        {"document_id": document_id, "position": position},
                        domain="test.token",
                    ),
                    "document_id": document_id,
                    "position": position,
                    "word": word,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "token_id": pl.String(),
            "document_id": pl.String(),
            "position": pl.Int64(),
            "word": pl.String(),
        },
    )
"""


def _transform_project(root: Path, *, docs: int) -> Path:
    project = root / "proj"
    (project / "models").mkdir(parents=True)
    (project / "sources").mkdir()
    (project / "transforms").mkdir()
    (project / "data" / "docs").mkdir(parents=True)
    (project / "stel_project.yml").write_text(
        "name: docs\nversion: '0.1.0'\nprofile: docs\n", encoding="utf-8"
    )
    (project / "profiles.yml").write_text(
        "docs:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n"
        "        schema: docs\n",
        encoding="utf-8",
    )
    (project / "sources" / "src.yml").write_text(
        "version: 2\nsources:\n  - name: raw_docs\n    path: data/docs\n"
        "    file_pattern: '*.json'\n",
        encoding="utf-8",
    )
    (project / "models" / "documents.yml").write_text(
        "version: 2\nmodels:\n  - name: documents\n"
        "    source: ref('raw_docs')\n    extraction:\n      backend: json\n"
        "      options:\n        fields: [body]\n"
        "    materialization: incremental\n",
        encoding="utf-8",
    )
    (project / "models" / "doc_lengths.yml").write_text(
        "version: 2\nmodels:\n  - name: doc_lengths\n"
        "    depends_on: [ref('documents')]\n"
        "    transform:\n      type: python\n"
        "      module: transforms.doc_lengths\n"
        f"      commit_every: {_TRANSFORM_COMMIT_EVERY}\n"
        "    materialization: incremental\n",
        encoding="utf-8",
    )
    (project / "transforms" / "doc_lengths.py").write_text(
        _TOKENS_TRANSFORM, encoding="utf-8"
    )
    for index in range(docs):
        (project / "data" / "docs" / f"doc{index}.json").write_text(
            json.dumps({"body": f"body text for document {index}"}),
            encoding="utf-8",
        )
    return project


def test_an_incremental_transform_reads_per_commit_batch_not_per_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #383, and the question its stale last comment left open.

    The reported incident was a re-key: `code_version` moved, so *every* parent
    was classified changed, and the run OOM-killed a 10GiB container. The last
    data point on that issue — still dying at 16GiB — predates #385 by four
    hours, so nobody had measured the shape since.

    This is that measurement. Every parent changes, and the rows still arrive
    one `commit_every` batch at a time (#385 composed with #379's batching)
    rather than as one corpus-sized frame.
    """
    docs = _TRANSFORM_COMMIT_EVERY * 4
    project = _transform_project(tmp_path, docs=docs)
    run_project(project, select="documents")
    run_project(project, select="doc_lengths")

    # Move every parent, the way a re-key does.
    for index in range(docs):
        (project / "data" / "docs" / f"doc{index}.json").write_text(
            json.dumps({"body": f"revised body for document {index}"}),
            encoding="utf-8",
        )
    run_project(project, select="documents")

    # The parent rows the transform is actually handed, per invocation.
    handed: list[int] = []
    real_read = transform_module._read_parent_rows

    def spy(*args: Any, **kwargs: Any) -> Any:
        frame = real_read(*args, **kwargs)
        handed.append(frame.height)
        return frame

    monkeypatch.setattr(transform_module, "_read_parent_rows", spy)
    with _measure_residency(monkeypatch) as seen:
        run_project(project, select="doc_lengths")

    assert handed, "the transform never read its parent"
    assert max(handed) <= _TRANSFORM_COMMIT_EVERY, (
        f"an incremental transform with every parent changed was handed "
        f"{max(handed)} parents at once, above its commit batch of "
        f"{_TRANSFORM_COMMIT_EVERY}. The point of #385 composed with #379 is "
        "that a re-key costs a batch at a time, not a corpus (issue #383)."
    )
    assert len(handed) > 1, (
        "every parent changed, so this must have taken several batches — one "
        "call means the batching did not engage and the assertion above is "
        "passing vacuously"
    )
    # And no whole-table frame anywhere: classification streams the parent and
    # keeps only a ~32-byte digest per row (#385), so `read_table` is not on
    # this path at all.
    assert seen.largest_frame_rows == 0, (
        f"the incremental path materialized {seen.largest_frame_rows} rows via "
        f"{seen.largest_frame_source}"
    )


def test_a_full_refresh_transform_is_the_recorded_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the same story, asserted so it stays a known trade.

    A full refresh reads every parent in one frame, because the python
    transform contract hands the user one DataFrame and a rebuild means all of
    them. That is the `exception` verdict in _READ_TABLE_SITES, and it is worth
    a test so nobody later reads the bounded case above and assumes transform
    is bounded everywhere.
    """
    docs = _TRANSFORM_COMMIT_EVERY * 4
    project = _transform_project(tmp_path, docs=docs)
    run_project(project, select="documents")

    with _measure_residency(monkeypatch) as seen:
        run_project(project, select="doc_lengths", full_refresh=True)

    assert seen.largest_frame_rows == docs, (
        "a full refresh is expected to read every parent at once; if this now "
        "reads less, the exception recorded in _READ_TABLE_SITES is stale."
    )
