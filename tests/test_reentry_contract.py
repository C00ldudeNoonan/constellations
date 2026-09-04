"""The idempotent, re-enterable contract, and the audit that keeps it honest (issue #493).

One incident stated the problem exactly. A `sec_chunk_search` publish wrote
3,613,979 rows across 145 pages over 4.2 hours, correctly and completely, and
failed in its last six seconds at the index step. The rows were fine — every
index was later built by hand on that same generation in 173s — but nothing
could re-enter the publish at the step that failed, so the next run read the
whole corpus again (#492). The expensive work succeeded and was discarded,
because the step that failed was not separable from the steps that had not.

#490, #491 and #492 fixed that incident. This module is the invariant they are
instances of, filed for the same reason #414 was for memory: the sequence says
the missing piece is a contract, not another patch.

**The contract.** For every step stel runs:

1. **Idempotent.** Re-running it converges on the same result and never
   corrupts what is there.
2. **Re-enterable.** A step that failed resumes from where it stopped rather
   than from the beginning, and a step that *succeeded* is not redone because a
   later one failed. The unit of re-entry is the step's own checkpoint: the
   flush window for stages that publish as they go, the page or the generation
   for a search publish, the whole step where the warehouse does the work in
   one atomic replace and there is nothing partial to resume.

`docs/architecture/idempotent-reentry.md` states the contract in full and
answers the design questions #493 left open; ADR-0005 records why the unit of
re-entry is the existing checkpoint and not a new phase ledger.

**What the audit found.** Both parts hold for every step, which is the outcome
#493 said it might: this issue shrank to writing the contract down and gating
it. Three kinds hold part 2 only at whole-step granularity, and each is
recorded below as an `EXCEPTION` with the reason it is the right unit — the
warehouse replaces the table atomically and there is no partial work to lose.
The `GAP` list is empty and pinned empty; adding to it needs an issue.

**The teeth.** Two things are scanned from the source rather than written down
by hand, so a new one has to be classified before it ships: every top-level
`run_*_model` entry in `src/stel/execution/`, and every `timings.phase("...")`
literal — the phase vocabulary #486 built for attributing wall clock is also
the vocabulary of where a publish can be interrupted. Every gate a row cites
must exist as a test function, so a deleted gate fails here rather than
silently unpinning its step.

Not a substitute for the gates themselves. This table says *which* test pins
each property; those tests, spread across the suite, are what actually run the
step, kill it, and rerun it. What this adds is the map, and the guarantee that
the map and the code do not drift apart.
"""

from __future__ import annotations

import ast
import pathlib
import re
from dataclasses import dataclass
from functools import cache

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "stel"
_EXECUTION = _SRC / "execution"
_TESTS = pathlib.Path(__file__).resolve().parent

# The property holds, and the cited gates pin it.
HOLDS = "holds"
# The property holds only at whole-step granularity, for a stated reason. These
# are decisions about the right unit of re-entry, not oversights.
EXCEPTION = "exception"
# The property does not hold, with the issue that tracks it. This list may only
# shrink — see `test_the_gap_list_only_shrinks`.
GAP = "gap"

_VERDICTS = frozenset({HOLDS, EXCEPTION, GAP})
_ENTRY_POINT = re.compile(r"^run_[a-z_]+_model$")


@dataclass(frozen=True)
class Step:
    """One classified step: a model kind's entry point, or a phase inside one."""

    module: str
    name: str
    idempotent: str
    reenterable: str
    unit: str
    why: str
    gates: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str]:
        return (self.module, self.name)

    @property
    def label(self) -> str:
        return f"{self.module}::{self.name}"


# ── Model kinds: every top-level `run_*_model` in src/stel/execution ─────────

_ENTRY_POINTS: tuple[Step, ...] = (
    Step(
        "extraction.py",
        "run_extraction_model",
        HOLDS,
        HOLDS,
        "flush window",
        "Each flush merges by document key and advances state only after the "
        "merge lands (publish-then-state), so an interrupted run never records a "
        "document it did not write and the rerun pays only for the flush that was "
        "lost. A full refresh replaces state after the write, so an interruption "
        "leaves state behind the target, never ahead of it.",
        (
            "test_second_run_is_incremental",
            "test_crash_mid_run_keeps_completed_flushes",
            "test_full_document_failure_preserves_target_and_state",
            "test_full_materialization_replaces_state_snapshot",
        ),
    ),
    Step(
        "chunk.py",
        "run_chunk_model",
        HOLDS,
        HOLDS,
        "flush window",
        "Publishes through FlushPublisher, which owns the three checkpoint rules "
        "once for every windowed stage: a full rebuild clears state before its "
        "first write, state advances only after a write lands, and a publication "
        "failure is reported without the warehouse's own words. Chunk ids are "
        "deterministic, so a rerun reproduces rather than duplicates.",
        (
            "test_chunk_ids_stable_across_reruns",
            "test_full_rebuild_clears_state_before_the_first_write",
            "test_state_advances_after_the_write",
            "test_a_failed_write_never_records_state",
        ),
    ),
    Step(
        "embed.py",
        "run_embed_model",
        HOLDS,
        HOLDS,
        "flush window, and the vector for an unchanged text",
        "FlushPublisher for the window; on top of it, a row whose text is "
        "unchanged reuses its stored vector rather than paying the provider "
        "again, and a resume reads only the rows it needs. A budget stop is an "
        "ordinary interruption: committed windows stay committed.",
        (
            "test_the_rerun_only_pays_for_what_was_lost",
            "test_a_budget_stop_is_graceful_and_resumable",
            "test_resume_never_reads_the_whole_target",
            "test_resume_still_reuses_vectors_for_metadata_only_changes",
        ),
    ),
    Step(
        "llm.py",
        "run_llm_model",
        HOLDS,
        HOLDS,
        "flush window, and the provider batch",
        "FlushPublisher for the window. Provider batch ids are persisted before "
        "the batch is awaited, so an interrupted batch is polled on the next run "
        "rather than resubmitted at full price.",
        (
            "test_the_rerun_only_pays_for_what_was_lost",
            "test_interrupted_batch_resumes_without_resubmission",
        ),
    ),
    Step(
        "transform.py",
        "run_sql_model",
        HOLDS,
        EXCEPTION,
        "the whole step",
        "A full materialization is one atomic replace — DuckDB's CREATE OR "
        "REPLACE TABLE ... AS SELECT, BigQuery's WRITE_TRUNCATE load job — and an "
        "incremental one is a keyed merge. There is no partial state to re-enter "
        "and nothing expensive to lose: the warehouse does the work, and an "
        "interrupted run leaves the target exactly as it was.",
        ("test_incremental_rerun_is_idempotent",),
    ),
    Step(
        "transform.py",
        "run_transform_model",
        HOLDS,
        HOLDS,
        "parent batch under the incremental contract; the whole step without one",
        "Under the incremental contract, changed parents are invoked and "
        "published in batches and each batch's state advances after its "
        "publish, so a partially published run is coherent: the parents whose "
        "state advanced are done and the rest are redone. Without a contract "
        "the output is one atomic replace, as for a SQL model.",
        (
            "test_unchanged_corpus_skips_every_parent",
            "test_changed_parent_with_fewer_children_replaces_only_its_rows",
            "test_removed_parent_deletes_its_children_and_state",
        ),
    ),
    Step(
        "ml.py",
        "run_ml_model",
        HOLDS,
        EXCEPTION,
        "the whole step",
        "Full-only (incremental ML is issue #53): every run retrains, replaces "
        "the primary and secondary relations atomically, and publishes the "
        "fitted artifact from a staged copy or discards it. A failure between "
        "the primary and a secondary replace leaves a mismatched pair until the "
        "rerun heals it. Recorded rather than fixed: training has no per-row "
        "checkpoint to resume from and costs no provider spend, so the step is "
        "the unit.",
        ("test_kmeans_predict_reuses_persisted_artifact",),
    ),
    Step(
        "eval.py",
        "run_eval_model",
        HOLDS,
        EXCEPTION,
        "the whole step",
        "Pure warehouse arithmetic over two materialized relations — no "
        "provider, no credentials. A full run replaces; an incremental run "
        "deletes the metric rows a shrunken label universe left stale, then "
        "merges on metric_id. Cheap enough to run on every change, so the step "
        "is the unit.",
        ("test_incremental_rerun_removes_stale_metric_rows",),
    ),
    Step(
        "search.py",
        "run_search_model",
        HOLDS,
        HOLDS,
        "the page for an in-place publish; the generation for a private build",
        "Every page is a keyed upsert with a durable receipt and its state "
        "advances only after the receipt, so an in-place publish killed between "
        "pages resumes from state and pays for the lost page alone. A private "
        "build that fails after its rows land is adopted by the next run under "
        "the same configuration fingerprint (#492) and its index step is "
        "retried with backoff (#491); an index-only change seeds its generation "
        "from the store instead of the warehouse (#495); a generation nothing "
        "will resume takes its state scope with it when swept (#502). "
        "Activation is fenced, and a failure after the state swap leaves the "
        "previous generation serving (ADR-0001).",
        (
            "test_an_interrupted_in_place_publish_resumes_from_state",
            "test_online_failure_keeps_the_old_index_and_retry_succeeds",
            "test_republishing_converges_rather_than_duplicating",
            "test_a_generation_built_under_this_configuration_is_resumable",
            "test_activation_clears_the_generation_scope",
            "test_sweeping_an_orphaned_generation_clears_its_scope",
        ),
    ),
)

# ── Phases: every `timings.phase("...")` literal in src/stel/execution ───────
#
# The phase vocabulary was built to attribute wall clock (#486). It is also the
# vocabulary of where a step can be interrupted, which is why each phase is
# classified on its own: "the search publish is re-enterable" is a claim about
# every one of these, and a new phase is a new place a failure can land.

_PHASES: tuple[Step, ...] = (
    Step(
        "search.py",
        "read",
        HOLDS,
        HOLDS,
        "page",
        "A cursor-paged snapshot read of the upstream relation. Retrying a "
        "cursor returns the identical page, and a read mutates nothing.",
        ("test_cursor_retry_returns_identical_page",),
    ),
    Step(
        "search.py",
        "store_write",
        HOLDS,
        HOLDS,
        "page",
        "Keyed upsert with a durable receipt: merge_insert on the id, so a "
        "republished page converges rather than duplicates. Append is used only "
        "for a generation that is fresh, unseeded and unresumed, where nothing "
        "can collide.",
        (
            "test_republishing_converges_rather_than_duplicating",
            "test_an_interrupted_in_place_publish_resumes_from_state",
        ),
    ),
    Step(
        "search.py",
        "state",
        HOLDS,
        HOLDS,
        "page",
        "Advances only after the page's receipt, per page. A failure between "
        "the write and the state leaves that page unrecorded, and an unrecorded "
        "page is redone — which is correct, and is the difference between a "
        "safe redo and re-entry.",
        ("test_an_interrupted_in_place_publish_resumes_from_state",),
    ),
    Step(
        "search.py",
        "index_reconcile",
        HOLDS,
        HOLDS,
        "index",
        "create_index with replace=True is idempotent and num_unindexed_rows "
        "drives whether it runs at all; each index kind has its own bounded "
        "retry with backoff (#491). A failure here keeps the generation, and the "
        "next run resumes it rather than re-reading the corpus (#492).",
        (
            "test_online_failure_keeps_the_old_index_and_retry_succeeds",
            "test_each_index_kind_gets_its_own_retry_budget",
        ),
    ),
    Step(
        "embed.py",
        "read",
        HOLDS,
        HOLDS,
        "input batch",
        "Bounded upstream reads; a resume reads only the rows it still needs, "
        "never the whole target.",
        ("test_resume_never_reads_the_whole_target",),
    ),
    Step(
        "embed.py",
        "reuse",
        HOLDS,
        HOLDS,
        "row",
        "A row whose text is unchanged reuses its stored vector, so a "
        "metadata-only change costs no provider call.",
        ("test_resume_still_reuses_vectors_for_metadata_only_changes",),
    ),
    Step(
        "embed.py",
        "provider",
        HOLDS,
        HOLDS,
        "flush window",
        "Provider calls are made per window and never for rows whose state has "
        "advanced. A budget stop between windows is an ordinary interruption: "
        "what was committed stays committed and the rerun continues from it.",
        (
            "test_a_budget_stop_is_graceful_and_resumable",
            "test_a_run_stopped_mid_corpus_keeps_its_committed_work_and_resumes",
        ),
    ),
    Step(
        "embed.py",
        "publish",
        HOLDS,
        HOLDS,
        "flush window",
        "FlushPublisher: state advances only after the write lands, and a full "
        "rebuild clears state before its first write.",
        (
            "test_the_rerun_only_pays_for_what_was_lost",
            "test_state_advances_after_the_write",
        ),
    ),
)

_STEPS: tuple[Step, ...] = _ENTRY_POINTS + _PHASES


# ── Scans ────────────────────────────────────────────────────────────────────


@cache
def _scanned_entry_points() -> frozenset[tuple[str, str]]:
    """Every top-level `run_*_model` in src/stel/execution, keyed by module."""
    found: set[tuple[str, str]] = set()
    for path in sorted(_EXECUTION.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and _ENTRY_POINT.match(node.name):
                found.add((path.name, node.name))
    return frozenset(found)


def _is_timings(node: ast.expr) -> bool:
    """A local `timings` or an attribute such as `self._timings`."""
    if isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, ast.Attribute):
        name = node.attr
    else:
        return False
    return name.lstrip("_").endswith("timings")


@cache
def _scanned_phases() -> frozenset[tuple[str, str]]:
    """Every `timings.phase("<literal>")` in src/stel/execution, keyed by module."""
    found: set[tuple[str, str]] = set()
    for path in sorted(_EXECUTION.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "phase"):
                continue
            if not _is_timings(func.value):
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.add((path.name, first.value))
    return frozenset(found)


@cache
def _test_functions() -> frozenset[str]:
    """Every top-level test function name in this directory."""
    found: set[str] = set()
    for path in sorted(_TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                found.add(node.name)
    return frozenset(found)


# ── The audit is complete, and describes the code ────────────────────────────


def test_every_entry_point_is_classified() -> None:
    """A new model kind must be argued for against both parts of the contract.

    This is the teeth: a kind that ships without a row here ships without
    anyone having said whether an interrupted run of it can be resumed, which
    is how #492 cost 4.7 hours to recover from a six-second failure.
    """
    undeclared = sorted(_scanned_entry_points() - {s.key for s in _ENTRY_POINTS})
    assert not undeclared, (
        f"New run_*_model entry point(s) with no row in _ENTRY_POINTS: {undeclared}. "
        "Classify the kind against the idempotent/re-enterable contract stated in "
        "this module and docs/architecture/idempotent-reentry.md (issue #493)."
    )


def test_every_phase_is_classified() -> None:
    """A new timing phase is a new place a step can be interrupted."""
    undeclared = sorted(_scanned_phases() - {s.key for s in _PHASES})
    assert not undeclared, (
        f"New timings.phase(...) literal(s) with no row in _PHASES: {undeclared}. "
        "A phase is where a publish can fail; say what a failure there costs and "
        "which test pins it (issue #493)."
    )


def test_no_classified_step_has_disappeared() -> None:
    """The table describes the code, so a stale row is a lie about it."""
    scanned = _scanned_entry_points() | _scanned_phases()
    missing = sorted(s.label for s in _STEPS if s.key not in scanned)
    assert not missing, (
        f"Rows describe step(s) that no longer exist: {missing}. Remove them — a "
        "table that disagrees with the code stops being an audit."
    )


def test_no_step_is_classified_twice() -> None:
    keys = [s.key for s in _STEPS]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("step", _STEPS, ids=[s.label for s in _STEPS])
def test_every_cited_gate_exists(step: Step) -> None:
    """Deleting a gate must fail here, not silently unpin its step."""
    assert step.gates, f"{step.label} cites no gate; a verdict with no test is an opinion"
    missing = sorted(set(step.gates) - _test_functions())
    assert not missing, (
        f"{step.label} cites test(s) that do not exist: {missing}. Either the gate "
        "was renamed — update the row — or it was deleted, and the step is no "
        "longer pinned."
    )


@pytest.mark.parametrize("step", _STEPS, ids=[s.label for s in _STEPS])
def test_every_verdict_is_argued(step: Step) -> None:
    assert step.idempotent in _VERDICTS and step.reenterable in _VERDICTS
    assert step.why.strip(), f"{step.label} has no reason"
    if GAP in (step.idempotent, step.reenterable):
        assert re.search(r"#\d+", step.why), (
            f"{step.label} is a GAP with no tracking issue named in its reason"
        )


def test_the_gap_list_only_shrinks() -> None:
    """The set of steps known to break the contract may only shrink.

    Per #493, after #414: a step that does not hold the contract is tracked
    here with its issue, and no new one may be added silently. The audit that
    established this table found none, which is the outcome the issue said it
    might — so the pinned value is empty, and growing it needs an issue and a
    row saying why it shipped that way.
    """
    gaps = sorted(s.label for s in _STEPS if GAP in (s.idempotent, s.reenterable))
    assert gaps == [], (
        "The set of steps known to break the idempotent/re-enterable contract "
        "changed. Removing one is the goal. Adding one needs an issue and a line "
        f"here saying why it shipped that way. Now: {gaps}"
    )


def test_every_kind_has_exactly_one_entry_point_row_per_module_function() -> None:
    """Sanity on the scan itself: the nine kinds stel runs today are all found.

    Pinned so a scan that silently found nothing — a moved directory, a
    renamed convention — cannot make every other test here pass vacuously.
    """
    assert len(_scanned_entry_points()) >= 9
    assert len(_scanned_phases()) >= 8
