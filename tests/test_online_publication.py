"""Reader-safe online publication, including failures across batch boundaries."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from stel.adapters import create_adapter
from stel.cli_services.serving import resolve_serving_scope
from stel.retrieval import LanceDBStore, RetrievalError, ServingCoordinator, StoreRole
from stel.retrieval.coordination import (
    ServingBusyError,
    ServingNotReadyError,
    StaleServingLeaseError,
)
from stel.retrieval.retention import retire_superseded_generations
from stel.runner import RunError, run_project
from tests.test_retrieval import (
    _materialize_upstream,
    _rows,
    _set_index_change_policy,
    _write_project,
)


def _prepare(project: Path) -> tuple[Any, Any]:
    _write_project(project)
    _set_index_change_policy(project, "online")
    _materialize_upstream(project, _rows())
    run_project(project, select="context_search")
    path = project / "models" / "retrieval.yml"
    path.write_text(
        path.read_text(encoding="utf-8")
        .replace("search: exact", "search: approximate")
        .replace("batch_size: 2", "batch_size: 1"),
        encoding="utf-8",
    )
    return resolve_serving_scope(
        project, profiles_dir=None, target=None, model_name="context_search"
    )


def test_online_switch_appends_privately_and_preserves_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope, resolved = _prepare(tmp_path)
    written: list[str] = []
    real_seed = LanceDBStore.seed_collection

    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        coordinator = ServingCoordinator(adapter)
        before = coordinator.status(scope)
        first_reader = coordinator.acquire_query(scope)
        readers = [first_reader]

        def seed(self: Any, spec: Any, *, source: str) -> int:
            # The write the seeded path performs (issue #495): the private
            # generation is filled from the live collection, not the warehouse.
            collection = spec.physical_name
            assert "__g" in collection
            assert source == self.physical_collection("context")
            entry = coordinator.status(scope)
            assert entry.status == "publishing"
            assert entry.active_generation == before.active_generation
            assert entry.config_fingerprint == before.config_fingerprint
            coordinator.validate_query(first_reader)
            reader = coordinator.acquire_query(scope)
            readers.append(reader)
            original = self.physical_collection("context")
            assert self.inspect_collection(original).physical_generation == before.active_generation
            assert self.text_search(
                original, "inflation", text_field="text", limit=2
            ).num_rows
            written.append(collection)
            return real_seed(self, spec, source=source)

        def no_live_mutation(*args: Any, **kwargs: Any) -> Any:
            # Nothing changed upstream, so reconciliation must skip every
            # seeded row; a merge here would mean the copy was not recognised
            # as current, and the corpus was about to be rewritten after all.
            pytest.fail("An online configuration change must not merge into the live index")

        monkeypatch.setattr(LanceDBStore, "seed_collection", seed)
        monkeypatch.setattr(LanceDBStore, "upsert", no_live_mutation)
        [result] = run_project(tmp_path, select="context_search")
        # Zero: the rows were copied into the generation, not written to it.
        assert result.rows_written == 0
        assert len(written) == 1 and "__g" in written[0]
        after = coordinator.status(scope)
        assert after.status == "ready"
        assert after.active_collection == written[0]
        assert after.config_fingerprint != before.config_fingerprint
        # Both pre-claim and mid-build readers remain valid through cutover.
        for reader in readers:
            coordinator.validate_query(reader)
            coordinator.release_query(reader)
        with pytest.raises(StaleServingLeaseError):
            coordinator.validate_query(first_reader)
        reader = coordinator.acquire_query(scope)
        assert reader.pinned_collection == after.active_collection
        assert reader.config_fingerprint == after.config_fingerprint
        coordinator.release_query(reader)

    # An unchanged rerun must use the newly activated state, not rewrite it.
    [rerun] = run_project(tmp_path, select="context_search")
    assert rerun.rows_written == 0
    assert rerun.documents_skipped == 2


@pytest.mark.parametrize("failure", ["seed", "indexes", "memory", "killed"])
def test_online_failure_keeps_the_old_index_and_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    scope, resolved = _prepare(tmp_path)
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        coordinator = ServingCoordinator(adapter)
        before = coordinator.status(scope)
        original_state = adapter.fetch_state(scope)
        real_seed = LanceDBStore.seed_collection

        def seed(self: Any, spec: Any, *, source: str) -> int:
            # The rows land in the private generation first, so the failure
            # leaves a complete generation behind with nothing pointing at it.
            seeded = real_seed(self, spec, source=source)
            if failure != "indexes":
                if failure == "memory":
                    raise MemoryError("private sentinel input must not escape")
                if failure == "killed":
                    raise SystemExit("simulated termination without exception cleanup")
                raise RetrievalError("simulated store failure")
            return seeded

        def indexes(*args: Any, **kwargs: Any) -> Any:
            raise RetrievalError("simulated index build failure")

        with monkeypatch.context() as patch:
            patch.setattr(LanceDBStore, "seed_collection", seed)
            if failure == "indexes":
                patch.setattr(LanceDBStore, "ensure_indexes", indexes)
            with pytest.raises(SystemExit if failure == "killed" else RunError) as caught:
                run_project(tmp_path, select="context_search")
            assert "private sentinel" not in str(caught.value)

        entry = coordinator.status(scope)
        if failure == "killed":
            assert entry.status == "publishing"
            # Even the stranded publisher cannot take intact serving data down.
            reader = coordinator.acquire_query(scope)
            coordinator.validate_query(reader)
            coordinator.release_query(reader)
            entry = coordinator.recover(scope, owner_terminated=True)
        assert entry.status == "degraded"
        assert entry.active_generation == before.active_generation
        assert entry.config_fingerprint == before.config_fingerprint
        if failure == "memory":
            assert entry.safe_error_code == "memory_exhausted"
        assert adapter.fetch_state(scope) == original_state
        reader = coordinator.acquire_query(scope)
        coordinator.validate_query(reader)
        coordinator.release_query(reader)

    # The retry resumes the generation the failed attempt left behind instead
    # of building a new one, so rows that were already published stay
    # published and only what the failure actually cost is redone (issue
    # #492). `indexes` is the production case: every row landed, the index
    # build failed, and the retry writes nothing at all.
    #
    # Nothing here has to assert the collection came out right — the publish
    # refuses to activate a generation whose row count disagrees with the rows
    # it read, so a resume that duplicated or dropped a row would raise rather
    # than reach `ready`.
    #
    # On the seeded path (issue #495) what a failure costs is decided by
    # ordering: rows are copied first and their state second, so state never
    # vouches for a row that is not there. A failure injected at the store
    # write therefore leaves a generation with every row and no state, and the
    # resume re-upserts both rows — idempotent on the id, so the count below
    # still holds. Only the index-build failure leaves rows and state both
    # landed, and writes nothing.
    expected_written = 0 if failure == "indexes" else 2
    [retry] = run_project(tmp_path, select="context_search")
    assert retry.rows_written == expected_written
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        recovered = ServingCoordinator(adapter).status(scope)
        assert recovered.status == "ready"
        assert recovered.active_collection is not None
    assert resolved.retrieval is not None
    with LanceDBStore(
        resolved.retrieval.stores["primary"],
        project_name="retrieval_demo", target_name="dev", alias="primary",
        role=StoreRole.PUBLISH,
    ) as store:
        served = store.inspect_collection(recovered.active_collection)
        assert served is not None and served.row_count == 2


def test_online_incompatible_change_is_refused_before_claim(tmp_path: Path) -> None:
    scope, resolved = _prepare(tmp_path)
    path = tmp_path / "models" / "retrieval.yml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("metric: cosine", "metric: dot"),
        encoding="utf-8",
    )
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        coordinator = ServingCoordinator(adapter)
        before = coordinator.status(scope)
        with pytest.raises(RunError, match="requires a rebuild"):
            run_project(tmp_path, select="context_search")
        assert coordinator.status(scope) == before


def test_existing_publisher_is_refused_before_planning_store_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope, resolved = _prepare(tmp_path)
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        coordinator = ServingCoordinator(adapter)
        current = coordinator.status(scope)
        assert current.config_fingerprint is not None
        coordinator.acquire_publish(
            scope, expected_code_version="v2", config_fingerprint=current.config_fingerprint,
            preserves_active_generation=True,
        )

        def no_inspection(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("An active publisher must be refused before planning against its old index")

        monkeypatch.setattr(LanceDBStore, "inspect_collection", no_inspection)
        with pytest.raises(RunError, match="Another publisher"):
            run_project(tmp_path, select="context_search")


def test_online_subset_change_cannot_replace_the_complete_index(tmp_path: Path) -> None:
    scope, resolved = _prepare(tmp_path)
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        coordinator = ServingCoordinator(adapter)
        before = coordinator.status(scope)
        with pytest.raises(RunError, match="unfiltered run"):
            run_project(
                tmp_path, select="context_search", read_filter=[("category", "eq", "macro")]
            )
        assert coordinator.status(scope) == before


def test_retirement_waits_for_readers_of_superseded_generations(tmp_path: Path) -> None:
    scope, resolved = _prepare(tmp_path)
    run_project(tmp_path, select="context_search")
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        coordinator = ServingCoordinator(adapter)
        before = coordinator.status(scope)
        reader = coordinator.acquire_query(scope)
        run_project(tmp_path, select="context_search", full_refresh=True)
        current = coordinator.status(scope)
        assert before.active_collection is not None
        assert current.active_collection is not None
        assert current.config_fingerprint is not None
        assert current.active_collection != before.active_collection
        coordinator.validate_query(reader)

        store = LanceDBStore(
            resolved.retrieval.stores["primary"],
            project_name="retrieval_demo", target_name="dev", alias="primary",
            role=StoreRole.PUBLISH,
        )
        claim = coordinator.acquire_publish(
            scope, expected_code_version="test", config_fingerprint=current.config_fingerprint,
            preserves_active_generation=True,
        )
        with store:
            assert store.inspect_collection(before.active_collection) is not None
            assert retire_superseded_generations(
                store, logical_collection="context", active_collection=current.active_collection,
                coordinator=coordinator, lease=claim,
            spare=None,
            ) == []
            coordinator.release_query(reader)
            assert before.active_collection in retire_superseded_generations(
                store, logical_collection="context", active_collection=current.active_collection,
                coordinator=coordinator, lease=claim,
            spare=None,
            )
            assert store.inspect_collection(current.active_collection) is not None


def test_online_change_includes_concurrent_warehouse_row_changes(tmp_path: Path) -> None:
    scope, resolved = _prepare(tmp_path)
    changed = _rows().with_columns(pl.lit("new title").alias("title"))
    _materialize_upstream(tmp_path, changed)
    [result] = run_project(tmp_path, select="context_search")
    assert result.rows_written == 2
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        entry = ServingCoordinator(adapter).status(scope)
    assert resolved.retrieval is not None and entry.active_collection is not None
    with LanceDBStore(
        resolved.retrieval.stores["primary"],
        project_name="retrieval_demo", target_name="dev", alias="primary",
        role=StoreRole.PUBLISH,
    ) as store:
        hits = store.text_search(entry.active_collection, "inflation", text_field="text", limit=2)
    assert hits.column("title").to_pylist() == ["new title"]


def test_online_can_replace_with_an_empty_snapshot(tmp_path: Path) -> None:
    scope, resolved = _prepare(tmp_path)
    _materialize_upstream(tmp_path, _rows().head(0))
    [result] = run_project(tmp_path, select="context_search")
    assert result.rows_written == 0
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        entry = ServingCoordinator(adapter).status(scope)
    assert entry.status == "ready" and entry.active_collection is not None
    assert resolved.retrieval is not None
    with LanceDBStore(
        resolved.retrieval.stores["primary"],
        project_name="retrieval_demo", target_name="dev", alias="primary",
        role=StoreRole.PUBLISH,
    ) as store:
        metadata = store.inspect_collection(entry.active_collection)
        assert metadata is not None and metadata.row_count == 0


@pytest.mark.parametrize("policy", ["online", "rebuild"])
def test_strategy_can_switch_back_to_exact_without_mutating_ann_generation(
    tmp_path: Path, policy: str,
) -> None:
    scope, resolved = _prepare(tmp_path)
    run_project(tmp_path, select="context_search")
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        coordinator = ServingCoordinator(adapter)
        old = coordinator.status(scope)
        reader = coordinator.acquire_query(scope)
        path = tmp_path / "models" / "retrieval.yml"
        path.write_text(
            path.read_text(encoding="utf-8")
            .replace("search: approximate", "search: exact")
            .replace("on_index_change: online", f"on_index_change: {policy}"),
            encoding="utf-8",
        )
        run_project(tmp_path, select="context_search")
        current = coordinator.status(scope)
        assert current.active_collection != old.active_collection
        coordinator.validate_query(reader)
        assert resolved.retrieval is not None
        assert old.active_collection is not None and current.active_collection is not None
        with LanceDBStore(
            resolved.retrieval.stores["primary"],
            project_name="retrieval_demo", target_name="dev", alias="primary",
            role=StoreRole.PUBLISH,
        ) as store:
            old_metadata = store.inspect_collection(old.active_collection)
            new_metadata = store.inspect_collection(current.active_collection)
            assert old_metadata is not None and new_metadata is not None
            assert old_metadata.physical_generation == old.active_generation
            assert old_metadata.config_fingerprint == old.config_fingerprint
            assert new_metadata.config_fingerprint == current.config_fingerprint
        coordinator.release_query(reader)


def test_online_widens_attributes_in_a_private_generation(tmp_path: Path) -> None:
    scope, resolved = _prepare(tmp_path)
    path = tmp_path / "models" / "retrieval.yml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "      attributes:\n",
            "      attributes:\n"
            "        - name: section\n"
            "          data_type: string\n"
            "          filter_role: user\n"
            "          returned: true\n",
        ), encoding="utf-8",
    )
    _materialize_upstream(tmp_path, _rows().with_columns(pl.lit("filing").alias("section")))
    run_project(tmp_path, select="context_search")
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        entry = ServingCoordinator(adapter).status(scope)
    assert resolved.retrieval is not None and entry.active_collection is not None
    with LanceDBStore(
        resolved.retrieval.stores["primary"],
        project_name="retrieval_demo", target_name="dev", alias="primary",
        role=StoreRole.PUBLISH,
    ) as store:
        original = store.inspect_collection(store.physical_collection("context"))
        updated = store.inspect_collection(entry.active_collection)
        assert original is not None and "section" not in original.schema.names
        assert updated is not None and "section" in updated.schema.names
        hits = store.text_search(entry.active_collection, "inflation", text_field="text", limit=2)
        assert hits.column("section").to_pylist() == ["filing"]


def test_in_place_publish_still_refuses_active_readers(tmp_path: Path) -> None:
    scope, resolved = _prepare(tmp_path)
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        coordinator = ServingCoordinator(adapter)
        reader = coordinator.acquire_query(scope)
        with pytest.raises(ServingBusyError):
            coordinator.acquire_publish(
                scope, expected_code_version="test", config_fingerprint="cfg"
            )
        coordinator.release_query(reader)
        with pytest.raises(StaleServingLeaseError):
            coordinator.validate_query(replace(reader, lease_id="not-held"))


def test_online_publish_recovers_a_scope_left_failed_with_no_generation(tmp_path: Path) -> None:
    """The state #473's reporter is actually in: an in-place publish was killed,
    `stel serving recover --owner-terminated` ran, and the ledger is `failed`
    with no active generation. The next `online` run must build a private
    generation from the warehouse and activate it, with nothing to retain."""
    scope, resolved = _prepare(tmp_path)
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        coordinator = ServingCoordinator(adapter)
        before = coordinator.status(scope)
        assert before.config_fingerprint is not None
        # A stranded in-place publisher: the claim clears the pointer, then dies.
        coordinator.acquire_publish(
            scope, expected_code_version="dead", config_fingerprint=before.config_fingerprint
        )
        stranded = coordinator.recover(scope, owner_terminated=True)
        assert stranded.status == "failed"
        assert stranded.active_generation is None
        with pytest.raises(ServingNotReadyError):
            coordinator.acquire_query(scope)

    [result] = run_project(tmp_path, select="context_search")
    assert result.rows_written == 2

    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        coordinator = ServingCoordinator(adapter)
        after = coordinator.status(scope)
        assert after.status == "ready"
        assert after.active_generation is not None
        assert after.active_collection is not None and "__g" in after.active_collection
        assert after.config_fingerprint != before.config_fingerprint
        reader = coordinator.acquire_query(scope)
        coordinator.validate_query(reader)
        coordinator.release_query(reader)


def test_online_index_type_switch_builds_the_declared_type_privately(tmp_path: Path) -> None:
    """A declared index type is a compatible change (issue #476), so switching
    it lands in a fresh private generation carrying the declared structure,
    while the old generation — and its `IvfHnswFlat` — keep serving until
    activation. `ivf_hnsw_sq` here because this fixture has two rows and
    LanceDB needs 256 to train `ivf_pq`; the PQ build, type switch, and the
    too-small refusal are covered at the store level."""
    scope, resolved = _prepare(tmp_path)
    run_project(tmp_path, select="context_search")  # approximate, default type
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        coordinator = ServingCoordinator(adapter)
        old = coordinator.status(scope)
        reader = coordinator.acquire_query(scope)
        path = tmp_path / "models" / "retrieval.yml"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "        search: approximate\n",
                "        search: approximate\n        index: ivf_hnsw_sq\n",
            ),
            encoding="utf-8",
        )
        [result] = run_project(tmp_path, select="context_search")
        # Zero, not two: an index-type switch reaches no row, so the new
        # generation is seeded from the one it replaces instead of re-read
        # from the warehouse (issue #495). The rows are all still there: the
        # post-publication row-count validation refuses the run otherwise.
        assert result.rows_written == 0
        current = coordinator.status(scope)
        assert current.active_collection != old.active_collection
        coordinator.validate_query(reader)
        coordinator.release_query(reader)
        assert resolved.retrieval is not None
        assert old.active_collection is not None and current.active_collection is not None

        def _types(store: Any, name: str) -> list[str]:
            table = store._open_owned_table(name)
            return [i.index_type for i in table.list_indices() if i.columns == ["embedding"]]

        with LanceDBStore(
            resolved.retrieval.stores["primary"],
            project_name="retrieval_demo", target_name="dev", alias="primary",
            role=StoreRole.PUBLISH,
        ) as store:
            assert _types(store, old.active_collection) == ["IvfHnswFlat"]
            assert _types(store, current.active_collection) == ["IvfHnswSq"]

    # Unchanged config afterwards: nothing to republish, nothing to rebuild.
    [rerun] = run_project(tmp_path, select="context_search")
    assert rerun.rows_written == 0


# ─── a generation's publication state does not outlive it (issue #502) ──────


def _generation_scope_rows(adapter: Any, generation: str) -> int:
    from stel.execution.search import _generation_state_scope

    return len(adapter.fetch_state(_generation_state_scope("context_search", generation)))


def test_activation_clears_the_generation_scope(tmp_path: Path) -> None:
    """The success half, asserted by nobody until now. Activation moves a
    generation's state into the serving scope and clears the generation's own;
    the code comment says leaving it would accumulate one dead scope per
    rebuild, and this is what holds it to that."""
    scope, resolved = _prepare(tmp_path)
    run_project(tmp_path, select="context_search")

    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        entry = ServingCoordinator(adapter).status(scope)
        assert entry.status == "ready" and entry.active_collection is not None
        assert adapter.fetch_state(scope), "the serving scope should now hold the state"
        assert _generation_scope_rows(adapter, entry.active_collection) == 0


def test_sweeping_an_orphaned_generation_clears_its_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure half. A failed private build keeps its scope on purpose, so
    a retry can resume it rather than redo it (#492). A generation nothing will
    ever resume — here, one built under a configuration a later run has moved
    past — is swept, and its state has to go with it: #492's incident left
    roughly 2.1M dead rows in `stel_state` this way."""
    scope, resolved = _prepare(tmp_path)

    def indexes(*args: Any, **kwargs: Any) -> Any:
        raise RetrievalError("simulated index build failure")

    with monkeypatch.context() as patch:
        patch.setattr(LanceDBStore, "ensure_indexes", indexes)
        with pytest.raises(RunError):
            run_project(tmp_path, select="context_search")

    assert resolved.retrieval is not None
    store_config = resolved.retrieval.stores["primary"]
    with LanceDBStore(
        store_config, project_name="retrieval_demo", target_name="dev",
        alias="primary", role=StoreRole.PUBLISH,
    ) as store:
        orphans = [name for name in store.list_collections() if "__g" in name]
    assert len(orphans) == 1
    [orphan] = orphans
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        # Kept, deliberately: this is exactly what a retry would resume from.
        assert _generation_scope_rows(adapter, orphan) == 2

    # A rebuild under a different configuration cannot resume that generation,
    # so the sweep takes the collection — and, now, its scope.
    path = tmp_path / "models" / "retrieval.yml"
    text = path.read_text(encoding="utf-8")
    assert "on_index_change: online" in text and "metric: cosine" in text
    path.write_text(
        text.replace("on_index_change: online", "on_index_change: rebuild").replace(
            "metric: cosine", "metric: dot"
        ),
        encoding="utf-8",
    )
    run_project(tmp_path, select="context_search")

    with LanceDBStore(
        store_config, project_name="retrieval_demo", target_name="dev",
        alias="primary", role=StoreRole.PUBLISH,
    ) as store:
        assert orphan not in store.list_collections()
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        assert _generation_scope_rows(adapter, orphan) == 0
        assert ServingCoordinator(adapter).status(scope).status == "ready"
