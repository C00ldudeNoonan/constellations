"""A mirrored store, end to end: publish, lose the host, restore, carry on (#666).

The point of the mirror is the last step. A fresh host with an empty primary
restores the served generation and then runs as though nothing happened: the
publication state was in the warehouse all along, so the next run reconciles
nothing and embeds nothing.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import lancedb
import pytest
from click.testing import CliRunner

from stel.adapters import create_adapter
from stel.cli import cli
from stel.cli_services.serving import resolve_serving_scope
from stel.config import load_project
from stel.profile import resolve_profile
from stel.retrieval import (
    LanceDBStore,
    RetrievalError,
    ServingBusyError,
    ServingCoordinator,
    StoreRole,
    create_store,
)
from stel.retrieval.base import mirror_fingerprint
from stel.runner import RunError, run_project
from tests.support_retrieval import materialize_upstream, sample_rows, write_project

pytestmark = pytest.mark.e2e

_COLLECTION = "retrieval_demo__dev__context"


def _mirrored_project(root: Path, *, primary: str = "target/lancedb") -> None:
    write_project(root)
    profile = root / "profiles.yml"
    profile.write_text(
        profile.read_text(encoding="utf-8").replace(
            "            path: target/lancedb\n",
            f"            path: {primary}\n"
            "            identity: demo-store\n"
            "            mirror: mirror\n",
        ),
        encoding="utf-8",
    )


def _ledger(root: Path) -> Any:
    scope, resolved = resolve_serving_scope(
        root, profiles_dir=None, target=None, model_name="context_search"
    )
    with create_adapter(resolved.warehouse, project_dir=root) as adapter:
        return ServingCoordinator(adapter, ensure_schema=True).status(scope)


def _store(root: Path) -> LanceDBStore:
    project, _, _ = load_project(root)
    resolved = resolve_profile(project, root)
    assert resolved.retrieval is not None
    store = create_store(
        resolved.retrieval.stores["primary"],
        project_name=project.name,
        target_name=resolved.target_name,
        alias="primary",
        role=StoreRole.INSPECT,
    )
    assert isinstance(store, LanceDBStore)
    return store


def _generation(location: Path, collection: str = _COLLECTION) -> int:
    return lancedb.connect(str(location)).open_table(collection).version


def test_a_publish_leaves_the_mirror_holding_the_served_generation(tmp_path: Path) -> None:
    _mirrored_project(tmp_path)
    materialize_upstream(tmp_path, sample_rows())
    run_project(tmp_path, select="context_search")

    entry = _ledger(tmp_path)
    assert entry.mirror_generation == entry.active_generation
    assert entry.mirror_target == mirror_fingerprint((tmp_path / "mirror").resolve().as_posix())
    assert _generation(tmp_path / "mirror") == _generation(tmp_path / "target" / "lancedb")

    # An incremental change moves the active generation; the mirror follows.
    materialize_upstream(tmp_path, sample_rows(version=2))
    run_project(tmp_path, select="context_search")
    moved = _ledger(tmp_path)
    assert moved.active_generation != entry.active_generation
    assert moved.mirror_generation == moved.active_generation
    assert _generation(tmp_path / "mirror") == _generation(tmp_path / "target" / "lancedb")


def test_a_fresh_host_restores_and_carries_on_without_re_embedding(tmp_path: Path) -> None:
    """The host is lost and a new one has an empty primary at a different
    path. The declared identity finds the serving record, the restore brings
    the generation back, and the next run has nothing to publish."""
    _mirrored_project(tmp_path)
    materialize_upstream(tmp_path, sample_rows())
    run_project(tmp_path, select="context_search")
    served = _ledger(tmp_path).active_generation
    shutil.rmtree(tmp_path / "target" / "lancedb")
    _mirrored_project_profile_moves_to(tmp_path, "target/new-host-lancedb")

    runner = CliRunner()
    refused = runner.invoke(
        cli, ["serving", "restore", "context_search", "--project-dir", str(tmp_path)]
    )
    assert refused.exit_code == 2
    assert "requires an explicit --target" in refused.output

    restored = runner.invoke(
        cli,
        ["serving", "restore", "context_search", "--project-dir", str(tmp_path), "--target", "dev"],
    )
    assert restored.exit_code == 0, restored.output
    assert "Restored 'retrieval_demo__dev__context'" in restored.output
    assert "identity:          demo-store (declared)" in restored.output
    with _store(tmp_path) as store:
        metadata = store.inspect_collection(_COLLECTION)
        assert metadata is not None
        assert metadata.physical_generation == served
        hits = store.vector_search(
            _COLLECTION, [1.0, 0.0], vector_field="embedding", limit=1, columns=["chunk_id"]
        )
        assert hits.column("chunk_id").to_pylist() == ["c1"]

    again = run_project(tmp_path, select="context_search")
    assert again[0].documents_processed == 0
    assert again[0].documents_skipped == 2

    # Idempotent: the primary already holds the generation.
    second = runner.invoke(
        cli,
        ["serving", "restore", "context_search", "--project-dir", str(tmp_path), "--target", "dev"],
    )
    assert second.exit_code == 0, second.output
    assert "Nothing to restore" in second.output


def _mirrored_project_profile_moves_to(root: Path, primary: str) -> None:
    profile = root / "profiles.yml"
    profile.write_text(
        profile.read_text(encoding="utf-8").replace(
            "path: target/lancedb\n", f"path: {primary}\n"
        ),
        encoding="utf-8",
    )


def test_a_store_restored_to_a_new_path_without_an_identity_is_told_why(
    tmp_path: Path,
) -> None:
    _mirrored_project(tmp_path)
    materialize_upstream(tmp_path, sample_rows())
    run_project(tmp_path, select="context_search")
    profile = tmp_path / "profiles.yml"
    profile.write_text(
        profile.read_text(encoding="utf-8")
        .replace("            identity: demo-store\n", "")
        .replace("path: target/lancedb\n", "path: target/elsewhere\n"),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        ["serving", "restore", "context_search", "--project-dir", str(tmp_path), "--target", "dev"],
    )
    assert result.exit_code == 2
    assert "declares the `identity:` it published under" in result.output


def test_a_rebuild_retires_the_superseded_generation_from_the_mirror(tmp_path: Path) -> None:
    _mirrored_project(tmp_path)
    materialize_upstream(tmp_path, sample_rows())
    run_project(tmp_path, select="context_search")
    run_project(tmp_path, select="context_search", full_refresh=True)
    first = _ledger(tmp_path)
    assert first.active_collection is not None
    run_project(tmp_path, select="context_search", full_refresh=True)
    second = _ledger(tmp_path)
    assert second.active_collection not in {None, first.active_collection}

    held = {path.name for path in (tmp_path / "mirror").iterdir()}
    # The unsuffixed base collection is never retired, here or at the primary.
    assert held == {f"{_COLLECTION}.lance", f"{second.active_collection}.lance"}
    assert second.mirror_generation == second.active_generation


def test_a_collection_the_ledger_did_not_activate_is_not_mirrored(tmp_path: Path) -> None:
    """A write the ledger never vouched for -- here a row added behind its
    back, as a failed in-place publish would leave one -- must not reach the
    mirror, or a restore would serve rows the publication state does not
    describe."""
    _mirrored_project(tmp_path)
    materialize_upstream(tmp_path, sample_rows())
    run_project(tmp_path, select="context_search")
    recorded = _ledger(tmp_path).mirror_generation
    table = lancedb.connect(str(tmp_path / "target" / "lancedb")).open_table(_COLLECTION)
    table.delete("chunk_id = 'c2'")

    result = CliRunner().invoke(
        cli, ["serving", "sync", "context_search", "--project-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "mirror_generation_mismatch" in result.output
    assert _ledger(tmp_path).mirror_generation == recorded
    assert sorted(
        lancedb.connect(str(tmp_path / "mirror")).open_table(_COLLECTION).to_arrow()
        .column("chunk_id").to_pylist()
    ) == ["c1", "c2"]


def test_a_failed_sync_fails_the_model_but_not_the_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mirrored_project(tmp_path)
    materialize_upstream(tmp_path, sample_rows())

    def unreachable(self: LanceDBStore, collection: str) -> Any:
        raise RetrievalError("Store mirror could not copy [OSError] (code=mirror_copy_failed)")

    monkeypatch.setattr(LanceDBStore, "sync_to_mirror", unreachable)
    with pytest.raises(RunError, match="stel serving sync context_search") as raised:
        run_project(tmp_path, select="context_search")
    # What the publish did stands in the run log (issue #623).
    assert raised.value.progress["rows_inserted"] == 2
    entry = _ledger(tmp_path)
    assert entry.status == "ready"
    assert entry.mirror_generation is None
    monkeypatch.undo()

    # The command the message names heals it.
    result = CliRunner().invoke(
        cli, ["serving", "sync", "context_search", "--project-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "holds the served generation" in result.output
    assert _ledger(tmp_path).mirror_generation == entry.active_generation


def test_a_sync_holds_off_an_in_place_publisher_while_it_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sync is a reader: its query lease is what stops an in-place
    publish rewriting the collection it is copying."""
    _mirrored_project(tmp_path)
    materialize_upstream(tmp_path, sample_rows())
    run_project(tmp_path, select="context_search")
    scope, resolved = resolve_serving_scope(
        tmp_path, profiles_dir=None, target=None, model_name="context_search"
    )
    real = LanceDBStore.sync_to_mirror
    attempts: list[type[BaseException]] = []

    def contended(self: LanceDBStore, collection: str) -> Any:
        with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
            try:
                ServingCoordinator(adapter, ensure_schema=False).acquire_publish(
                    scope, expected_code_version="cv", config_fingerprint="cf"
                )
            except ServingBusyError as error:
                attempts.append(type(error))
        return real(self, collection)

    monkeypatch.setattr(LanceDBStore, "sync_to_mirror", contended)
    result = CliRunner().invoke(
        cli, ["serving", "sync", "context_search", "--project-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert attempts == [ServingBusyError]
    # And the lease was released, so the index takes publishers again.
    assert _ledger(tmp_path).query_leases == 0


def test_status_names_the_identity_and_whether_the_mirror_is_current(tmp_path: Path) -> None:
    _mirrored_project(tmp_path)
    materialize_upstream(tmp_path, sample_rows())
    run_project(tmp_path, select="context_search")

    result = CliRunner().invoke(
        cli, ["serving", "status", "context_search", "--project-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "identity:          demo-store (declared)" in result.output
    assert "holds the served generation" in result.output


def test_sync_is_not_offered_on_a_store_without_a_mirror(tmp_path: Path) -> None:
    write_project(tmp_path)
    materialize_upstream(tmp_path, sample_rows())
    run_project(tmp_path, select="context_search")
    assert not (tmp_path / "mirror").exists()

    result = CliRunner().invoke(
        cli, ["serving", "sync", "context_search", "--project-dir", str(tmp_path)]
    )
    assert result.exit_code == 2
    assert "has no mirror" in result.output


def test_serving_activate_ends_with_a_sync(tmp_path: Path) -> None:
    """`serving activate` makes a generation the served one without a
    publish, so it has to sync on its own or the mirror falls behind."""
    _mirrored_project(tmp_path)
    materialize_upstream(tmp_path, sample_rows())
    run_project(tmp_path, select="context_search")
    before = _ledger(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "serving", "activate", "context_search",
            "--generation", _COLLECTION,
            "--rows-verified",
            "--project-dir", str(tmp_path),
            "--target", "dev",
        ],
    )
    assert result.exit_code == 0, result.output
    after = _ledger(tmp_path)
    assert after.active_collection == _COLLECTION
    assert after.mirror_generation == after.active_generation
    assert after.mirrored_epoch is not None
    assert before.mirrored_epoch is not None
    assert after.mirrored_epoch >= before.mirrored_epoch

