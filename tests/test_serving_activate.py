"""`stel serving activate`: serve a complete generation without re-paging (issue #615).

The shape every test here starts from is the one the issue was filed in: a
private generation holding every row sits in the store, the serving ledger
names nothing (a fail-closed in-place failure cleared the pointer), the
publication state for those rows is split between the generation's own scope
and the serving scope, and a release has moved the `code_version` hash without
changing a row. The only path back was a publish that re-read the corpus and
could not finish inside its read session (#614).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from stel.adapters import StateRecord, StateScope, create_adapter
from stel.cli import cli
from stel.execution.activation import UNVERIFIED_INPUT_FINGERPRINT
from stel.execution.search import _generation_state_scope
from stel.retrieval import ServingCoordinator, StoreRole
from stel.retrieval.coordination import STATUS_DEGRADED, STATUS_FAILED, STATUS_READY

# Runs a whole project and opens a retrieval store (issue #518).
pytestmark = pytest.mark.e2e

STALE_HASH = "code-version-before-the-release"


def _write_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "stel_project.yml").write_text(
        "name: activate_demo\nversion: '0.1.0'\nprofile: activate_demo\n"
    )
    (project / "profiles.yml").write_text(
        "activate_demo:\n"
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
        "            path: target/lancedb\n"
    )
    (project / "sources").mkdir()
    (project / "sources" / "documents.yml").write_text(
        "version: 2\n"
        "sources:\n"
        "  - name: releases\n"
        "    path: data\n"
        "    file_pattern: '*.json'\n"
    )
    (project / "models").mkdir()
    (project / "models" / "search.yml").write_text(
        "version: 2\n"
        "models:\n"
        "  - name: release_documents\n"
        "    source: ref('releases')\n"
        "    extraction:\n"
        "      backend: json\n"
        "      options:\n"
        "        fields: [title, body, category]\n"
        "    materialization: incremental\n"
        "  - name: release_chunks\n"
        "    depends_on: [ref('release_documents')]\n"
        "    chunk:\n"
        "      text_field: body\n"
        "      chunk_size: 1000\n"
        "      chunk_overlap: 0\n"
        "    materialization: incremental\n"
        "  - name: release_embeddings\n"
        "    depends_on: [ref('release_chunks')]\n"
        "    embed:\n"
        "      provider: deterministic\n"
        "      model: activate-demo-v1\n"
        "      text_field: text\n"
        "      id_field: chunk_id\n"
        "      vector_field: embedding\n"
        "      dimensions: 8\n"
        "    materialization: incremental\n"
        "  - name: release_search\n"
        "    depends_on: [ref('release_embeddings')]\n"
        "    materialization: incremental\n"
        "    search:\n"
        "      access: public\n"
        "      id_field: chunk_id\n"
        "      document_id_field: document_id\n"
        "      chunk_id_field: chunk_id\n"
        "      text_fields: [text]\n"
        "      return_text_fields: [text]\n"
        "      vector:\n"
        "        field: embedding\n"
        "        dimensions: 8\n"
        "        metric: cosine\n"
        "        search: exact\n"
        "        embedding: inherit\n"
        "      full_text:\n"
        "        fields: [text]\n"
        "      attributes:\n"
        "        - name: category\n"
        "          data_type: string\n"
        "          filter_role: user\n"
        "          returned: true\n"
        "      display_fields: [title]\n"
        "      query:\n"
        "        modes: [vector, text, hybrid, filter]\n"
        "        consistency: strong\n"
    )
    data = project / "data"
    data.mkdir()
    _write_doc(project, "inflation", "Consumer prices", "Inflation moderated.", "prices")
    _write_doc(project, "labor", "Employment report", "Payroll employment increased.", "labor")
    _write_doc(project, "output", "GDP report", "Economic output expanded this quarter.", "growth")
    return project


def _write_doc(project: Path, name: str, title: str, body: str, category: str) -> None:
    (project / "data" / f"{name}.json").write_text(
        json.dumps({"title": title, "body": body, "category": category})
    )


class _Ledger:
    """One warehouse connection's view of the serving scope and its state."""

    def __init__(self, project: Path) -> None:
        from stel.config import load_project
        from stel.profile import resolve_profile
        from stel.retrieval import create_store

        project_config, _sources, models = load_project(project)
        model = next(item for item in models if item.name == "release_search")
        self.resolved = resolve_profile(project_config, project)
        assert self.resolved.retrieval is not None and model.search is not None
        alias = model.search.store or self.resolved.retrieval.default
        self.store = create_store(
            self.resolved.retrieval.stores[alias],
            project_name=project_config.name,
            target_name=self.resolved.target_name,
            alias=alias,
            role=StoreRole.INSPECT,
        )
        self.project = project
        self.scope = StateScope.for_target_descriptor(
            model.name,
            stage="retrieval_publish",
            descriptor=self.store.state_descriptor("release_search").descriptor(),
        )

    def adapter(self) -> Any:
        return create_adapter(self.resolved.warehouse, project_dir=self.project)

    def status(self) -> Any:
        with self.adapter() as adapter:
            return ServingCoordinator(adapter, ensure_schema=True).status(self.scope)

    def state(self, scope: StateScope | None = None) -> dict[str, Any]:
        with self.adapter() as adapter:
            return adapter.fetch_state(scope or self.scope)

    def generation_scope(self, collection: str) -> StateScope:
        return _generation_state_scope("release_search", collection)

    def lose_the_pointer(self) -> None:
        """The fail-closed in-place failure ADR-0001 specified for a store that
        cannot promise a sound collection: pointer cleared at claim, failure
        recorded, nothing retained. The state stays where it was written."""
        with self.adapter() as adapter:
            coordinator = ServingCoordinator(adapter, ensure_schema=True)
            lease = coordinator.acquire_publish(
                self.scope, expected_code_version="any", config_fingerprint="any"
            )
            coordinator.mark_failed(lease, safe_error_code="store_error")

    def split_state_across_a_release(self, collection: str, moved_key: str) -> None:
        """The #615 state: every record re-stamped as if written before a
        hash-only release, and one of them recorded only by the resumed
        generation's own scope, so neither scope alone describes the rows."""
        with self.adapter() as adapter:
            records = adapter.fetch_state(self.scope)
            adapter.upsert_state(
                self.scope,
                [
                    StateRecord(key, value.input_fingerprint, STALE_HASH)
                    for key, value in records.items()
                ],
            )
            adapter.upsert_state(
                self.generation_scope(collection),
                [StateRecord(moved_key, records[moved_key].input_fingerprint, STALE_HASH)],
            )
            adapter.delete_state(self.scope, [moved_key])


def _activate(project: Path, collection: str, *extra: str) -> Any:
    return CliRunner().invoke(
        cli,
        [
            "serving",
            "activate",
            "release_search",
            "--generation",
            collection,
            "--project-dir",
            str(project),
            *extra,
        ],
    )


def _search(project: Path) -> Any:
    from stel.search import SearchMode, SearchRequest, search

    return search(
        project,
        SearchRequest(model="release_search", query="inflation", mode=SearchMode.TEXT),
    )


def test_activate_serves_a_complete_generation_whose_pointer_was_lost(tmp_path: Path) -> None:
    """The incident, end to end: a complete private generation, a ledger that
    names nothing, state split across two scopes at a pre-release hash, and a
    reader refused. Activation needs no corpus read; afterwards the index
    answers, the state is one scope at the current hash, and the next
    incremental run finds nothing to republish -- the re-stamp did its job.
    """
    from stel.runner import run_project
    from stel.search import SearchError

    project = _write_project(tmp_path)
    run_project(project, full_refresh=True)
    ledger = _Ledger(project)
    before = ledger.status()
    assert before.status == STATUS_READY
    generation = before.active_collection
    assert generation is not None and "__g" in generation
    keys = sorted(ledger.state())
    assert len(keys) == 3

    ledger.lose_the_pointer()
    ledger.split_state_across_a_release(generation, moved_key=keys[0])
    lost = ledger.status()
    assert lost.status == STATUS_FAILED and lost.active_generation is None
    assert len(ledger.state()) == 2
    assert len(ledger.state(ledger.generation_scope(generation))) == 1
    with pytest.raises(SearchError, match="no ready publication"):
        _search(project)

    result = _activate(project, generation, "--rows-verified", "--target", "dev")

    assert result.exit_code == 0, result.output
    assert f"Activated '{generation}'" in result.output
    assert "3 row(s)" in result.output
    assert "1 row(s) from the generation's own publication, 2 filled from the serving scope" in (
        result.output
    )
    assert "serving:           generation " in result.output
    after = ledger.status()
    assert after.status == STATUS_READY
    assert after.active_collection == generation
    assert after.active_generation is not None
    assert after.publication_id is None
    assert _search(project)
    # One scope, current hash, nothing left in the generation's.
    state = ledger.state()
    assert sorted(state) == keys
    versions = {value.code_version for value in state.values()}
    assert len(versions) == 1 and STALE_HASH not in versions
    assert ledger.state(ledger.generation_scope(generation)) == {}

    results = run_project(project)
    assert results[-1].serving_resource is not None
    assert results[-1].serving_resource["status"] == "ready"
    assert results[-1].rows_written == 0
    assert ledger.status().active_collection == generation


def test_activate_serves_a_generation_behind_the_upstream_and_the_next_run_catches_up(
    tmp_path: Path,
) -> None:
    """The upstream grows with every embedding run, so a generation is
    "exactly complete" only for the hours between its last write and the next
    filing. A fourth document upstream that the collection never saw is a
    week's staleness, not damage: the index is activated, says by how much it
    is behind, and the next incremental run publishes exactly that row."""
    from stel.runner import run_project

    project = _write_project(tmp_path)
    run_project(project, full_refresh=True)
    ledger = _Ledger(project)
    generation = ledger.status().active_collection
    assert generation is not None
    ledger.lose_the_pointer()
    _write_doc(project, "housing", "Housing starts", "Housing starts rose sharply.", "housing")
    run_project(project, select="+release_embeddings")

    result = _activate(project, generation, "--rows-verified", "--target", "dev")

    assert result.exit_code == 0, result.output
    assert "3 row(s)" in result.output
    assert "pending:           the upstream row count exceeds the collection's by 1" in (
        result.output
    )
    assert ledger.status().status == STATUS_READY
    assert _search(project)

    results = run_project(project)
    assert results[-1].rows_written == 1
    assert ledger.status().active_collection == generation


def test_activate_refuses_a_generation_holding_rows_the_upstream_does_not(
    tmp_path: Path,
) -> None:
    """The other direction is refused: rows the upstream does not have cannot
    be told, by a count, from another relation's collection. Refused before
    the claim, so the ledger is exactly as it was."""
    from stel.runner import run_project

    project = _write_project(tmp_path)
    run_project(project)
    ledger = _Ledger(project)
    before = ledger.status()
    collection = before.active_collection or ledger.store.physical_collection("release_search")
    with ledger.adapter() as adapter:
        key = next(iter(adapter.fetch_state(ledger.scope)))
        adapter.execute(
            f"DELETE FROM {adapter.table_ref('release_embeddings')} WHERE chunk_id = ?",
            [key],
        )

    result = _activate(project, collection, "--rows-verified", "--target", "dev")

    assert result.exit_code != 0
    assert "holds 3 row(s)" in result.output
    assert "upstream relation has only 2" in result.output
    after = ledger.status()
    assert (after.status, after.fencing_token, after.active_generation) == (
        before.status,
        before.fencing_token,
        before.active_generation,
    )


def test_activate_serves_rows_the_state_does_not_describe_and_the_next_run_republishes_them(
    tmp_path: Path,
) -> None:
    """A page whose slices committed before its state advanced leaves rows the
    state does not know. They are in the collection and queryable, so the
    generation is activated and the gap reported -- and each such row is
    recorded under a marker fingerprint no upstream row can match, so the
    next run classifies it as changed and re-upserts it, idempotently, which
    replaces the marker with the row's real state."""
    from stel.runner import run_project

    project = _write_project(tmp_path)
    run_project(project, full_refresh=True)
    ledger = _Ledger(project)
    generation = ledger.status().active_collection
    assert generation is not None
    ledger.lose_the_pointer()
    with ledger.adapter() as adapter:
        key = next(iter(adapter.fetch_state(ledger.scope)))
        adapter.delete_state(ledger.scope, [key])

    result = _activate(project, generation, "--rows-verified", "--target", "dev")

    assert result.exit_code == 0, result.output
    assert "2 filled from the serving scope" in result.output
    assert "1 held row(s) had no state and are marked unverified" in result.output
    assert ledger.status().status == STATUS_READY
    state = ledger.state()
    assert len(state) == 3
    assert state[key].input_fingerprint == UNVERIFIED_INPUT_FINGERPRINT

    results = run_project(project)
    assert results[-1].rows_written == 1
    state = ledger.state()
    assert len(state) == 3
    assert state[key].input_fingerprint != UNVERIFIED_INPUT_FINGERPRINT


def test_a_row_without_state_whose_key_is_deleted_upstream_is_removed_by_the_next_run(
    tmp_path: Path,
) -> None:
    """The hole Codex found in #630: stale discovery enumerates *state* keys
    absent upstream, so a collection row with no state whose upstream key is
    then deleted would never be found again -- served for good, through every
    later run. The marker activation writes is what puts the row where the
    sweep can see it: after the deletion and one incremental run, the row is
    gone from the collection and from the state."""
    from stel.runner import run_project

    project = _write_project(tmp_path)
    run_project(project, full_refresh=True)
    ledger = _Ledger(project)
    generation = ledger.status().active_collection
    assert generation is not None
    ledger.lose_the_pointer()
    with ledger.adapter() as adapter:
        key = next(iter(adapter.fetch_state(ledger.scope)))
        adapter.delete_state(ledger.scope, [key])

    result = _activate(project, generation, "--rows-verified", "--target", "dev")
    assert result.exit_code == 0, result.output
    with ledger.store:
        assert ledger.store.count_present(generation, [key], id_field="chunk_id") == 1

    with ledger.adapter() as adapter:
        adapter.execute(
            f"DELETE FROM {adapter.table_ref('release_embeddings')} WHERE chunk_id = ?",
            [key],
        )
    results = run_project(project, select="release_search")

    assert results[-1].documents_deleted == 1
    assert key not in ledger.state()
    with ledger.store:
        assert ledger.store.count_present(generation, [key], id_field="chunk_id") == 0
    assert ledger.status().active_collection == generation


def test_activate_refuses_state_naming_more_rows_than_the_collection_holds(
    tmp_path: Path,
) -> None:
    """State that vouches for a row the collection lacks would make the
    reconciler skip that row for good. More state rows than collection rows
    proves at least one such row without sampling, so it is refused outright."""
    from stel.runner import run_project

    project = _write_project(tmp_path)
    run_project(project, full_refresh=True)
    ledger = _Ledger(project)
    generation = ledger.status().active_collection
    assert generation is not None
    ledger.lose_the_pointer()
    with ledger.adapter() as adapter:
        value = next(iter(adapter.fetch_state(ledger.scope).values()))
        adapter.upsert_state(
            ledger.scope, [StateRecord("ghost-row", value.input_fingerprint, STALE_HASH)]
        )

    result = _activate(project, generation, "--rows-verified", "--target", "dev")

    assert result.exit_code != 0
    assert "describes 4 row(s)" in result.output
    assert "holds only 3" in result.output
    assert ledger.status().status == STATUS_FAILED


def test_activate_refuses_state_that_names_rows_the_collection_lacks(tmp_path: Path) -> None:
    """Counts can agree by coincidence; membership cannot. State vouching for
    a row the store does not hold would make reconciliation skip that row for
    good, so a ghost key in otherwise complete state is refused -- and the
    refusal, coming after the claim, leaves the ledger no worse than it was."""
    from stel.runner import run_project
    from stel.search import SearchError

    project = _write_project(tmp_path)
    run_project(project, full_refresh=True)
    ledger = _Ledger(project)
    generation = ledger.status().active_collection
    assert generation is not None
    ledger.lose_the_pointer()
    with ledger.adapter() as adapter:
        records = adapter.fetch_state(ledger.scope)
        real_key, value = next(iter(records.items()))
        adapter.delete_state(ledger.scope, [real_key])
        adapter.upsert_state(
            ledger.scope, [StateRecord("ghost-row", value.input_fingerprint, STALE_HASH)]
        )

    result = _activate(project, generation, "--rows-verified", "--target", "dev")

    assert result.exit_code != 0
    assert "does not hold" in result.output
    after = ledger.status()
    assert after.status == STATUS_FAILED
    assert after.publication_id is None
    with pytest.raises(SearchError):
        _search(project)


def test_the_id_walk_refuses_a_ghost_the_sample_missed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ghost state key and a collection row without state cancel in the row
    count, so neither count check sees them, and a 1,000-key sample of 3.6M
    can miss the ghost (review finding on #631). The walk of the collection's
    ids cannot: with the sample blinded, the ghost is still refused, and the
    serving scope is left as it was."""
    from stel.execution import activation
    from stel.runner import run_project

    monkeypatch.setattr(activation, "ACTIVATION_SAMPLE_SIZE", 0)
    project = _write_project(tmp_path)
    run_project(project, full_refresh=True)
    ledger = _Ledger(project)
    generation = ledger.status().active_collection
    assert generation is not None
    ledger.lose_the_pointer()
    with ledger.adapter() as adapter:
        key, value = next(iter(adapter.fetch_state(ledger.scope).items()))
        adapter.delete_state(ledger.scope, [key])
        adapter.upsert_state(
            ledger.scope, [StateRecord("ghost-row", value.input_fingerprint, STALE_HASH)]
        )

    result = _activate(project, generation, "--rows-verified", "--target", "dev")

    assert result.exit_code != 0
    assert "names 1 row(s)" in result.output
    assert "does not hold" in result.output
    assert ledger.status().status == STATUS_FAILED


def test_a_late_refusal_never_un_serves_the_current_generation(tmp_path: Path) -> None:
    """Re-activating the collection being served (to rebuild its indices,
    say) with state that fails the membership check: the activation is
    recorded as a failed publish and the index keeps answering from the
    generation it had, `degraded`, exactly as #617 requires of every path."""
    from stel.runner import run_project

    project = _write_project(tmp_path)
    run_project(project, full_refresh=True)
    ledger = _Ledger(project)
    before = ledger.status()
    generation = before.active_collection
    assert generation is not None
    with ledger.adapter() as adapter:
        records = adapter.fetch_state(ledger.scope)
        real_key, value = next(iter(records.items()))
        adapter.delete_state(ledger.scope, [real_key])
        adapter.upsert_state(
            ledger.scope, [StateRecord("ghost-row", value.input_fingerprint, STALE_HASH)]
        )

    result = _activate(project, generation, "--rows-verified", "--target", "dev")

    assert result.exit_code != 0
    assert "does not hold" in result.output
    after = ledger.status()
    assert after.status == STATUS_DEGRADED
    assert after.active_generation == before.active_generation
    assert after.active_collection == generation
    assert _search(project)


def test_activate_requires_an_explicit_target_and_the_rows_confirmation(
    tmp_path: Path,
) -> None:
    """Both refusals come before anything is read or claimed, and each names
    what to add. The target check comes first: it decides which store the
    rest of the command is about (#511)."""
    from stel.runner import run_project

    project = _write_project(tmp_path)
    run_project(project)
    ledger = _Ledger(project)
    before = ledger.status()
    collection = before.active_collection or ledger.store.physical_collection("release_search")

    untargeted = _activate(project, collection, "--rows-verified")
    assert untargeted.exit_code != 0
    assert "requires an explicit --target" in untargeted.output
    assert "'dev'" in untargeted.output

    unconfirmed = _activate(project, collection, "--target", "dev")
    assert unconfirmed.exit_code != 0
    assert "--rows-verified" in unconfirmed.output

    assert ledger.status() == before


def test_activate_refuses_while_a_publisher_holds_the_scope(tmp_path: Path) -> None:
    from stel.runner import run_project

    project = _write_project(tmp_path)
    run_project(project)
    ledger = _Ledger(project)
    collection = ledger.status().active_collection or ledger.store.physical_collection(
        "release_search"
    )
    with ledger.adapter() as adapter:
        ServingCoordinator(adapter, ensure_schema=True).acquire_publish(
            ledger.scope, expected_code_version="held", config_fingerprint="held"
        )

    result = _activate(project, collection, "--rows-verified", "--target", "dev")

    assert result.exit_code != 0
    assert "Another publisher owns this serving scope" in result.output


# ─── progress is visible from another process (issue #635) ───────────────────


def test_the_restamp_phase_publishes_progress_other_processes_can_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gap that sent an operator to INFORMATION_SCHEMA.

    The re-stamp phase ran for ~2.3 hours in prod while `serving status`
    showed the *previous* publish's counts and `status: publishing`, so
    nothing said whether the command was alive or how far along. The note
    goes on the ledger because that is what a second terminal can read.

    Observed mid-phase, since the completion write clears it.
    """
    from stel.execution import activation
    from stel.runner import run_project

    project = _write_project(tmp_path)
    run_project(project, full_refresh=True)
    ledger = _Ledger(project)
    generation = ledger.status().active_collection
    assert generation is not None
    keys = sorted(ledger.state())
    ledger.lose_the_pointer()
    ledger.split_state_across_a_release(generation, moved_key=keys[0])

    # What another process would see while the phase is running -- including
    # the `stel serving status` CLI output itself, not just the ledger row
    # a test can read directly but an operator cannot (Codex review, #640).
    observed: list[str | None] = []
    cli_outputs: list[str] = []
    original = activation._restamp_and_sample

    def watched(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        observed.append(ledger.status().progress_note)
        cli_outputs.append(
            CliRunner()
            .invoke(
                cli,
                [
                    "serving",
                    "status",
                    "release_search",
                    "--project-dir",
                    str(project),
                ],
            )
            .output
        )
        return result

    monkeypatch.setattr(activation, "_restamp_and_sample", watched)

    result = _activate(project, generation, "--rows-verified", "--target", "dev")
    assert result.exit_code == 0, result.output

    assert observed and observed[0] is not None, (
        "the re-stamp phase published no progress for another process to read"
    )
    assert "re-stamped" in observed[0]
    assert "of" in observed[0]
    assert cli_outputs and f"progress:          {observed[0]}" in cli_outputs[0]
    # And it does not outlive the publication that wrote it: a note surviving
    # completion would read as a phase still running on a scope that is ready,
    # which is worse than no note at all.
    after = ledger.status()
    assert after.status == STATUS_READY
    assert after.progress_note is None




# ─── the re-stamp does not re-query what it is walking (issue #635) ──────────


def test_filling_state_from_serving_issues_no_keyed_lookups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ~75% of #635's bytes, asserted by absence.

    Deciding which of the serving scope's records the generation lacked used
    a per-batch `record_key IN UNNEST(...)` against the generation scope.
    That lookup re-scanned the generation's whole state slice every batch --
    ~514 MB of the ~3.4 GB a batch cost in prod -- because `IN UNNEST` does
    not prune on the clustering #431 added.

    The warehouse now evaluates the absence as part of the walk, so the fill
    phase must issue no keyed lookup at all. Pinned by counting them rather
    than by measuring bytes, which a test cannot see.
    """
    from stel.adapters.duckdb import DuckDBAdapter
    from stel.execution import activation

    project = _write_project(tmp_path)
    run_project_module = __import__("stel.runner", fromlist=["run_project"])
    run_project_module.run_project(project, full_refresh=True)
    ledger = _Ledger(project)
    generation = ledger.status().active_collection
    assert generation is not None

    # Split the state across both scopes so the fill phase has work: the
    # serving scope keeps every record, the generation scope keeps none.
    scope = _generation_state_scope("context_search", generation)
    with create_adapter(ledger.resolved.warehouse, project_dir=project) as adapter:
        adapter.clear_state(scope)

    lookups: list[int] = []
    original = DuckDBAdapter.fetch_state_subset

    def counting(self: DuckDBAdapter, *args: Any, **kwargs: Any) -> Any:
        lookups.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DuckDBAdapter, "fetch_state_subset", counting)

    filled: list[int] = []
    fill = activation._fill_state_from_serving

    def watched(*args: Any, **kwargs: Any) -> int:
        before = len(lookups)
        taken = fill(*args, **kwargs)
        filled.append(len(lookups) - before)
        return taken

    monkeypatch.setattr(activation, "_fill_state_from_serving", watched)

    result = _activate(project, generation, "--rows-verified", "--target", "dev")
    assert result.exit_code == 0, result.output
    assert filled == [0], (
        "the fill phase must resolve absence in the warehouse, not by "
        f"re-querying state ({filled[0] if filled else '?'} lookups)"
    )


def test_the_generations_own_receipts_are_not_overwritten_from_serving(
    tmp_path: Path,
) -> None:
    """The semantic the probe carries, not just the byte count.

    The fill phase may only take keys the generation never recorded. A row
    the interrupted build rewrote carries the fingerprint of what it wrote;
    the serving scope's older record for the same key would make the next
    incremental run republish a row that is already current.

    Written because a mutation that dropped the absence probe -- leaving the
    phase to upsert every serving record over the generation's own -- was
    caught by no existing test, including the one that counts keyed lookups.
    """
    from stel.runner import run_project

    project = _write_project(tmp_path)
    run_project(project, full_refresh=True)
    ledger = _Ledger(project)
    generation = ledger.status().active_collection
    assert generation is not None
    keys = sorted(ledger.state())
    kept = keys[0]

    ledger.lose_the_pointer()
    ledger.split_state_across_a_release(generation, moved_key=kept)
    generation_scope = ledger.generation_scope(generation)
    own_fingerprint = ledger.state(generation_scope)[kept].input_fingerprint

    # The serving scope also holds a record for the key the generation owns,
    # carrying an older fingerprint. Only the generation's may survive.
    with ledger.adapter() as adapter:
        adapter.upsert_state(ledger.scope, [StateRecord(kept, "stale-fp", STALE_HASH)])

    result = _activate(project, generation, "--rows-verified", "--target", "dev")
    assert result.exit_code == 0, result.output

    after = ledger.state()
    assert after[kept].input_fingerprint == own_fingerprint, (
        "the generation's own receipt was overwritten by the serving scope's"
    )
    assert after[kept].input_fingerprint != "stale-fp"
    # And the keys only the serving scope had were still taken.
    assert sorted(after) == keys
