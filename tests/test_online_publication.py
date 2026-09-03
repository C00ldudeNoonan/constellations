"""Reader-safe online publication, including failures across batch boundaries."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from stel.adapters import create_adapter
from stel.cli_services.serving import resolve_serving_scope
from stel.retrieval import LanceDBStore, RetrievalError, ServingCoordinator
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
    real_append = LanceDBStore.append

    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        coordinator = ServingCoordinator(adapter)
        before = coordinator.status(scope)
        first_reader = coordinator.acquire_query(scope)
        readers = [first_reader]

        def append(self: Any, collection: str, rows: Any, **kwargs: Any) -> Any:
            assert "__g" in collection
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
            return real_append(self, collection, rows, **kwargs)

        def no_live_mutation(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("An online configuration change must not merge or evolve the live index")

        monkeypatch.setattr(LanceDBStore, "append", append)
        monkeypatch.setattr(LanceDBStore, "upsert", no_live_mutation)
        monkeypatch.setattr(LanceDBStore, "evolve_collection", no_live_mutation)
        [result] = run_project(tmp_path, select="context_search")
        assert result.rows_written == 2
        assert len(written) == 2 and len(set(written)) == 1
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


@pytest.mark.parametrize("failure", ["append", "indexes", "memory", "killed"])
def test_online_failure_keeps_the_old_index_and_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    scope, resolved = _prepare(tmp_path)
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        coordinator = ServingCoordinator(adapter)
        before = coordinator.status(scope)
        original_state = adapter.fetch_state(scope)
        real_append = LanceDBStore.append
        writes = 0

        def append(self: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal writes
            receipt = real_append(self, *args, **kwargs)
            writes += 1
            if writes == 2 and failure != "indexes":
                if failure == "memory":
                    raise MemoryError("private sentinel input must not escape")
                if failure == "killed":
                    raise SystemExit("simulated termination without exception cleanup")
                raise RetrievalError("simulated store failure")
            return receipt

        def indexes(*args: Any, **kwargs: Any) -> Any:
            raise RetrievalError("simulated index build failure")

        with monkeypatch.context() as patch:
            patch.setattr(LanceDBStore, "append", append)
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

    [retry] = run_project(tmp_path, select="context_search")
    assert retry.rows_written == 2
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        assert ServingCoordinator(adapter).status(scope).status == "ready"


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
            ) == []
            coordinator.release_query(reader)
            assert before.active_collection in retire_superseded_generations(
                store, logical_collection="context", active_collection=current.active_collection,
                coordinator=coordinator, lease=claim,
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
