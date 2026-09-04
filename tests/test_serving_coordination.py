"""Generation-fenced serving coordination tests (issue #152)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stel.adapters import StateScope, create_adapter, parse_warehouse_config
from stel.config.profile import WarehouseConfig
from stel.retrieval import (
    PUBLISHER_FENCING_FEATURES,
    LanceDBStore,
    RetrievalError,
    ServingBusyError,
    ServingCoordinationError,
    ServingCoordinator,
    ServingNotReadyError,
    StaleServingLeaseError,
    StoreRole,
)
from stel.retrieval.coordination import (
    RECOVERY_ERROR_CODE,
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_PUBLISHING,
    STATUS_READY,
    STATUS_UNPUBLISHED,
    validate_safe_error_code,
)


def _wh(path: Path) -> WarehouseConfig:
    return parse_warehouse_config(
        {"type": "duckdb", "path": str(path), "schema": "serving"}
    )


def _scope(model_name: str = "context_search") -> StateScope:
    return StateScope.for_target_descriptor(
        model_name,
        stage="retrieval_publish",
        descriptor={"store_type": "lancedb", "collection": "context"},
    )


@pytest.fixture
def coordinator(tmp_path: Path) -> Any:
    adapter_cm = create_adapter(_wh(tmp_path / "serving.duckdb"))
    with adapter_cm as adapter:
        yield ServingCoordinator(adapter)


# ─── publish claims and fencing ─────────────────────────────────────────────


def test_exclusive_publish_claim_and_fence_progression(coordinator: Any) -> None:
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    assert lease.fencing_token == 1
    assert coordinator.status(scope).status == STATUS_PUBLISHING

    with pytest.raises(ServingBusyError, match="Another publisher"):
        coordinator.acquire_publish(
            scope, expected_code_version="v1", config_fingerprint="cfg1"
        )

    coordinator.mark_ready(
        lease,
        active_generation="gen1",
        config_fingerprint="cfg1",
        counts=(2, 1, 3, 0),
    )
    entry = coordinator.status(scope)
    assert entry.status == STATUS_READY
    assert entry.active_generation == "gen1"
    assert entry.fencing_token == 1
    assert (entry.rows_inserted, entry.rows_updated) == (2, 1)

    second = coordinator.acquire_publish(
        scope, expected_code_version="v2", config_fingerprint="cfg1"
    )
    assert second.fencing_token == 2


def test_scopes_are_independent(coordinator: Any) -> None:
    first = coordinator.acquire_publish(
        _scope("index_a"), expected_code_version="v1", config_fingerprint="cfg"
    )
    second = coordinator.acquire_publish(
        _scope("index_b"), expected_code_version="v1", config_fingerprint="cfg"
    )
    assert first.fencing_token == 1
    assert second.fencing_token == 1


def test_stale_publisher_cannot_advance_after_recovery(coordinator: Any) -> None:
    scope = _scope()
    zombie = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    recovered = coordinator.recover(scope, owner_terminated=True)
    assert recovered.status == STATUS_FAILED
    assert recovered.safe_error_code == RECOVERY_ERROR_CODE
    assert recovered.fencing_token == zombie.fencing_token + 1

    with pytest.raises(StaleServingLeaseError):
        coordinator.verify_publish(zombie)
    with pytest.raises(StaleServingLeaseError):
        coordinator.mark_ready(
            zombie,
            active_generation="gen1",
            config_fingerprint="cfg1",
            counts=(0, 0, 0, 0),
        )
    with pytest.raises(StaleServingLeaseError):
        coordinator.mark_failed(zombie, safe_error_code="store_error")

    fresh = coordinator.acquire_publish(
        scope, expected_code_version="v2", config_fingerprint="cfg2"
    )
    assert fresh.fencing_token == zombie.fencing_token + 2


def test_failed_publication_is_unavailable_and_clears_generation(
    coordinator: Any,
) -> None:
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        lease, active_generation="gen1", config_fingerprint="cfg1", counts=(1, 0, 0, 0)
    )
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_failed(lease, safe_error_code="store_error", counts=(0, 0, 1, 0))

    entry = coordinator.status(scope)
    assert entry.status == STATUS_FAILED
    assert entry.active_generation is None
    assert entry.safe_error_code == "store_error"
    with pytest.raises(ServingNotReadyError, match="no ready publication"):
        coordinator.acquire_query(scope)


def test_ready_activation_requires_claimed_config_and_generation(
    coordinator: Any,
) -> None:
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    with pytest.raises(ServingCoordinationError, match="configuration fingerprint"):
        coordinator.mark_ready(
            lease,
            active_generation="gen1",
            config_fingerprint="cfg-other",
            counts=(0, 0, 0, 0),
        )
    with pytest.raises(ServingCoordinationError, match="physical generation"):
        coordinator.mark_ready(
            lease,
            active_generation="",
            config_fingerprint="cfg1",
            counts=(0, 0, 0, 0),
        )


def _insert_ledger_row(
    coordinator: Any,
    scope: StateScope,
    *,
    row_id: str,
    fencing_token: int = 0,
    status: str = STATUS_UNPUBLISHED,
    publication_id: str | None = None,
) -> None:
    adapter = coordinator._adapter
    ledger = f"{adapter.schema_ref}.{adapter.quote_ident('stel_serving_ledger')}"
    adapter.execute(
        f"""
        INSERT INTO {ledger} (
            model_name, stage, target_identity, row_id, fencing_token, status,
            publication_id, rows_inserted, rows_updated, rows_skipped, rows_deleted
        )
        SELECT ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0 FROM (SELECT 1) AS seed
        """,
        [
            scope.model_name,
            scope.stage,
            scope.target_identity,
            row_id,
            fencing_token,
            status,
            publication_id,
        ],
    )


def test_duplicate_creation_race_self_heals_on_next_claim(coordinator: Any) -> None:
    # Simulate the benign race: two sessions both inserted the initial,
    # unclaimed ledger row because no warehouse-enforced unique key exists.
    scope = _scope()
    _insert_ledger_row(coordinator, scope, row_id="aaa")
    _insert_ledger_row(coordinator, scope, row_id="bbb")

    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    assert lease.fencing_token == 1
    coordinator.mark_ready(
        lease, active_generation="gen1", config_fingerprint="cfg1", counts=(1, 0, 0, 0)
    )
    assert coordinator.status(scope).status == STATUS_READY


def test_claim_refuses_duplicated_scope_and_recover_rebuilds_it(
    coordinator: Any,
) -> None:
    # A corrupted scope with two claimed rows must never elect a publisher;
    # explicit recovery rebuilds exactly one row above every observed fence.
    scope = _scope()
    _insert_ledger_row(
        coordinator, scope, row_id="aaa", fencing_token=3, publication_id="p-aaa",
        status=STATUS_PUBLISHING,
    )
    _insert_ledger_row(
        coordinator, scope, row_id="bbb", fencing_token=4, publication_id="p-bbb",
        status=STATUS_PUBLISHING,
    )

    with pytest.raises(ServingCoordinationError, match="conflicting rows"):
        coordinator.acquire_publish(
            scope, expected_code_version="v1", config_fingerprint="cfg1"
        )
    with pytest.raises(ServingCoordinationError, match="conflicting rows"):
        coordinator.status(scope)

    entry = coordinator.recover(scope, owner_terminated=True)
    assert entry.status == STATUS_FAILED
    assert entry.fencing_token == 5

    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    assert lease.fencing_token == 6


# ─── query leases ───────────────────────────────────────────────────────────


def test_query_lease_requires_ready_scope(coordinator: Any) -> None:
    scope = _scope()
    with pytest.raises(ServingNotReadyError, match="not been published"):
        coordinator.acquire_query(scope)

    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    with pytest.raises(ServingBusyError, match="reconciling"):
        coordinator.acquire_query(scope)
    coordinator.mark_ready(
        lease, active_generation="gen1", config_fingerprint="cfg1", counts=(1, 0, 0, 0)
    )

    query = coordinator.acquire_query(scope)
    assert query.pinned_generation == "gen1"
    assert query.config_fingerprint == "cfg1"
    assert query.fencing_token == 1
    coordinator.validate_query(query)
    coordinator.release_query(query)


@pytest.mark.parametrize("private", [False, True])
def test_a_stale_publication_plan_cannot_claim_or_clear_a_new_generation(
    coordinator: Any, private: bool,
) -> None:
    scope = _scope()
    first = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        first, active_generation="gen1", config_fingerprint="cfg1", counts=(1, 0, 0, 0)
    )
    planned = coordinator.status(scope)
    other = coordinator.acquire_publish(
        scope, expected_code_version="v2", config_fingerprint="cfg2",
        preserves_active_generation=True,
    )
    coordinator.mark_ready(
        other, active_generation="gen2", config_fingerprint="cfg2", counts=(1, 0, 0, 0)
    )
    latest = coordinator.status(scope)
    with pytest.raises(ServingBusyError, match="changed while planning"):
        coordinator.acquire_publish(
            scope, expected_code_version="v3", config_fingerprint="cfg3",
            preserves_active_generation=private,
            expected_fencing_token=planned.fencing_token,
        )
    assert coordinator.status(scope) == latest


def test_admission_that_races_cutover_is_refused(
    coordinator: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = _scope()
    first = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg-old"
    )
    coordinator.mark_ready(
        first, active_generation="gen-old", config_fingerprint="cfg-old",
        counts=(1, 0, 0, 0),
    )
    build = coordinator.acquire_publish(
        scope, expected_code_version="v2", config_fingerprint="cfg-new",
        preserves_active_generation=True,
    )
    execute = coordinator._adapter.execute

    def cutover_after_insert(sql: str, params: Any = None) -> Any:
        result = execute(sql, params)
        if "INSERT INTO" in sql and "pinned_generation" in sql:
            coordinator.mark_ready(
                build, active_generation="gen-new", config_fingerprint="cfg-new",
                counts=(1, 0, 0, 0),
            )
        return result

    monkeypatch.setattr(coordinator._adapter, "execute", cutover_after_insert)
    with pytest.raises(StaleServingLeaseError):
        coordinator.acquire_query(scope)
    assert coordinator.status(scope).query_leases == 0


def test_shared_query_leases_block_exclusive_publisher(coordinator: Any) -> None:
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        lease, active_generation="gen1", config_fingerprint="cfg1", counts=(1, 0, 0, 0)
    )
    first = coordinator.acquire_query(scope)
    second = coordinator.acquire_query(scope)
    assert coordinator.status(scope).query_leases == 2

    with pytest.raises(ServingBusyError, match="query leases"):
        coordinator.acquire_publish(
            scope, expected_code_version="v1", config_fingerprint="cfg1"
        )

    coordinator.release_query(first)
    with pytest.raises(ServingBusyError, match="query leases"):
        coordinator.acquire_publish(
            scope, expected_code_version="v1", config_fingerprint="cfg1"
        )
    coordinator.release_query(second)
    coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )


def test_query_pin_fails_validation_after_authority_change(coordinator: Any) -> None:
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        lease, active_generation="gen1", config_fingerprint="cfg1", counts=(1, 0, 0, 0)
    )
    query = coordinator.acquire_query(scope)
    coordinator.recover(scope, owner_terminated=True)
    with pytest.raises(StaleServingLeaseError, match="no longer active"):
        coordinator.validate_query(query)


def test_recovery_requires_explicit_owner_termination(coordinator: Any) -> None:
    scope = _scope()
    with pytest.raises(ServingCoordinationError, match="terminating the previous owner"):
        coordinator.recover(scope, owner_terminated=False)
    assert coordinator.status(scope).status == STATUS_UNPUBLISHED


def test_recovery_clears_stuck_query_leases(coordinator: Any) -> None:
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        lease, active_generation="gen1", config_fingerprint="cfg1", counts=(1, 0, 0, 0)
    )
    coordinator.acquire_query(scope)
    coordinator.recover(scope, owner_terminated=True)
    assert coordinator.status(scope).query_leases == 0
    coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )


def test_ledger_holds_only_safe_error_codes(coordinator: Any) -> None:
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    with pytest.raises(ServingCoordinationError, match="lowercase identifiers"):
        coordinator.mark_failed(
            lease, safe_error_code="Boom: /etc/secrets leaked", counts=(0, 0, 0, 0)
        )
    assert validate_safe_error_code("store_error") == "store_error"


# ─── store-side publisher fencing ───────────────────────────────────────────


def test_lancedb_advertises_single_host_publisher_lock() -> None:
    features = LanceDBStore.capabilities().features
    assert features & PUBLISHER_FENCING_FEATURES


def test_lancedb_publisher_lock_excludes_second_publisher(tmp_path: Path) -> None:
    from stel.retrieval import parse_store_config

    config = parse_store_config({"type": "lancedb", "path": str(tmp_path / "lance")})
    store = LanceDBStore(
        config, project_name="demo", target_name="dev", alias="primary",
        role=StoreRole.PUBLISH,
    )
    other = LanceDBStore(
        config, project_name="demo", target_name="dev", alias="primary",
        role=StoreRole.PUBLISH,
    )
    with store.publisher_fence("demo__dev__context"):
        with pytest.raises(RetrievalError, match="publisher_lock_held"):
            with other.publisher_fence("demo__dev__context"):
                pass
    # Released on exit: a new publisher may acquire the fence.
    with other.publisher_fence("demo__dev__context"):
        pass


def test_lancedb_publisher_locks_are_per_collection(tmp_path: Path) -> None:
    from stel.retrieval import parse_store_config

    config = parse_store_config({"type": "lancedb", "path": str(tmp_path / "lance")})
    store = LanceDBStore(
        config, project_name="demo", target_name="dev", alias="primary",
        role=StoreRole.PUBLISH,
    )
    with store.publisher_fence("demo__dev__alpha"), store.publisher_fence(
        "demo__dev__beta"
    ):
        pass


# ─── end-to-end publication, query, and recovery ────────────────────────────


def _write_project(tmp_path: Path, *, access: str = "public") -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "stel_project.yml").write_text(
        "name: serving_demo\nversion: '0.1.0'\nprofile: serving_demo\n"
    )
    (project / "profiles.yml").write_text(
        "serving_demo:\n"
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
    policy_attributes = (
        "        - name: tenant\n"
        "          data_type: string\n"
        "          filter_role: policy\n"
        "          returned: true\n"
        if access == "governed"
        else ""
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
        "        fields: [title, body, category, tenant]\n"
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
        "      model: serving-demo-v1\n"
        "      text_field: text\n"
        "      id_field: chunk_id\n"
        "      vector_field: embedding\n"
        "      dimensions: 8\n"
        "    materialization: incremental\n"
        "  - name: release_search\n"
        "    depends_on: [ref('release_embeddings')]\n"
        "    materialization: incremental\n"
        "    search:\n"
        f"      access: {access}\n"
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
        f"{policy_attributes}"
        "      display_fields: [title]\n"
        "      query:\n"
        "        modes: [vector, text, hybrid, filter]\n"
        "        consistency: strong\n"
    )
    data = project / "data"
    data.mkdir()
    for name, payload in {
        "inflation.json": {
            "title": "Consumer prices",
            "body": "Inflation moderated as consumer price growth slowed.",
            "category": "prices",
            "tenant": "research",
        },
        "labor.json": {
            "title": "Employment report",
            "body": "Payroll employment increased and unemployment remained stable.",
            "category": "labor",
            "tenant": "research",
        },
        "output.json": {
            "title": "GDP report",
            "body": "Economic output expanded during the latest quarter.",
            "category": "growth",
            "tenant": "restricted",
        },
    }.items():
        (data / name).write_text(json.dumps(payload))
    return project


def _serving_scope(project: Path, model_name: str = "release_search") -> StateScope:
    from stel.config import load_project
    from stel.profile import resolve_profile
    from stel.retrieval import create_store

    project_config, _sources, models = load_project(project)
    model = next(item for item in models if item.name == model_name)
    resolved = resolve_profile(project_config, project)
    assert resolved.retrieval is not None and model.search is not None
    alias = model.search.store or resolved.retrieval.default
    store = create_store(
        resolved.retrieval.stores[alias],
        project_name=project_config.name,
        target_name=resolved.target_name,
        alias=alias,
        role=StoreRole.PUBLISH,
    )
    logical = model.search.collection or model.name
    return StateScope.for_target_descriptor(
        model.name,
        stage="retrieval_publish",
        descriptor=store.state_descriptor(logical).descriptor(),
    )


def _ledger_status(project: Path, model_name: str = "release_search") -> Any:
    from stel.config import load_project
    from stel.profile import resolve_profile

    project_config, _sources, _models = load_project(project)
    resolved = resolve_profile(project_config, project)
    with create_adapter(resolved.warehouse, project_dir=project) as adapter:
        return ServingCoordinator(adapter).status(_serving_scope(project, model_name))


def test_publication_marks_scope_ready_and_serves_queries(tmp_path: Path) -> None:
    from stel.runner import run_project
    from stel.search import SearchMode, SearchRequest, search

    project = _write_project(tmp_path)
    results = run_project(project)
    search_result = results[-1]
    assert search_result.serving_resource is not None
    assert search_result.serving_resource["status"] == "ready"
    assert search_result.serving_resource["active_generation"]
    assert search_result.serving_resource["fencing_token"] == 1

    entry = _ledger_status(project)
    assert entry.status == STATUS_READY
    assert entry.query_leases == 0

    hits = search(
        project,
        SearchRequest(model="release_search", query="inflation", mode=SearchMode.TEXT),
    )
    assert hits
    assert _ledger_status(project).query_leases == 0


def test_failed_publication_blocks_queries_until_republished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stel.runner import RunError, run_project
    from stel.search import SearchError, SearchMode, SearchRequest, search

    project = _write_project(tmp_path)

    def _fail_upsert(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise RetrievalError("LanceDB operation 'upsert' failed (code=test_outage)")

    monkeypatch.setattr(LanceDBStore, "upsert", _fail_upsert)
    with pytest.raises(RunError):
        run_project(project)
    monkeypatch.undo()

    entry = _ledger_status(project)
    assert entry.status == STATUS_FAILED
    assert entry.safe_error_code == "store_error"
    with pytest.raises(SearchError, match="no ready publication"):
        search(
            project,
            SearchRequest(
                model="release_search", query="inflation", mode=SearchMode.TEXT
            ),
        )

    results = run_project(project)
    assert results[-1].serving_resource is not None
    assert results[-1].serving_resource["status"] == "ready"
    assert _ledger_status(project).status == STATUS_READY
    assert search(
        project,
        SearchRequest(model="release_search", query="inflation", mode=SearchMode.TEXT),
    )


def test_crashed_publisher_requires_explicit_recovery(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from stel.cli import cli
    from stel.config import load_project
    from stel.profile import resolve_profile
    from stel.runner import RunError, run_project

    project = _write_project(tmp_path)
    run_project(project)

    # Simulate a crash: claim the scope and abandon the lease.
    project_config, _sources, _models = load_project(project)
    resolved = resolve_profile(project_config, project)
    scope = _serving_scope(project)
    with create_adapter(resolved.warehouse, project_dir=project) as adapter:
        ServingCoordinator(adapter).acquire_publish(
            scope, expected_code_version="crashed", config_fingerprint="crashed"
        )

    with pytest.raises(RunError, match="Another publisher"):
        run_project(project)

    runner = CliRunner()
    # Recovery is refused twice over, and the target check comes first: it is
    # the one that decides *which* store the rest of the command is about
    # (issue #511).
    untargeted = runner.invoke(
        cli,
        [
            "serving",
            "recover",
            "release_search",
            "--owner-terminated",
            "--project-dir",
            str(project),
        ],
    )
    assert untargeted.exit_code != 0
    assert "requires an explicit --target" in untargeted.output
    # It names the target it would have used, so confirming is one edit.
    assert "'dev'" in untargeted.output

    refused = runner.invoke(
        cli,
        [
            "serving",
            "recover",
            "release_search",
            "--target",
            "dev",
            "--project-dir",
            str(project),
        ],
    )
    assert refused.exit_code != 0
    assert "terminating the previous owner" in refused.output

    recovered = runner.invoke(
        cli,
        [
            "serving",
            "recover",
            "release_search",
            "--owner-terminated",
            "--target",
            "dev",
            "--project-dir",
            str(project),
        ],
    )
    assert recovered.exit_code == 0, recovered.output
    # Failed, not degraded. The abandoned claim was an ordinary in-place
    # publish, which mutates the collection the activation pointer names — its
    # claim cleared the pointer precisely so that a crash cannot leave a
    # half-rewritten collection being served (issue #449). Recovery has no way
    # to prove otherwise about a publisher that is gone, so it fails closed.
    assert "status=failed" in recovered.output

    results = run_project(project)
    assert results[-1].serving_resource is not None
    assert results[-1].serving_resource["status"] == "ready"


def test_serving_status_command_reports_ledger(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from stel.cli import cli
    from stel.runner import run_project

    project = _write_project(tmp_path)
    run_project(project)
    result = CliRunner().invoke(
        cli, ["serving", "status", "release_search", "--project-dir", str(project)]
    )
    assert result.exit_code == 0, result.output
    assert "status:            ready" in result.output
    assert "fencing_token:     1" in result.output


# ─── naming what a serving command acted on (issue #511) ───────────────────


def test_serving_status_names_the_target_warehouse_and_store(
    tmp_path: Path,
) -> None:
    """The ledger alone is ambiguous. A production `status` reported a clean
    `unpublished` for an index that was actively publishing, because it had
    silently resolved a dev target whose store is a local directory rather
    than the GCS store the index lives in. Nothing in the output said so."""
    from click.testing import CliRunner

    from stel.cli import cli
    from stel.runner import run_project

    project = _write_project(tmp_path)
    run_project(project)
    result = CliRunner().invoke(
        cli, ["serving", "status", "release_search", "--project-dir", str(project)]
    )

    assert result.exit_code == 0, result.output
    assert "target:            dev" in result.output
    assert "warehouse:         duckdb" in result.output
    # The store line carries the location, which is the discriminator that
    # makes a wrong target obvious: dev is a directory, prod is a bucket.
    assert "store:             " in result.output
    assert "(lancedb)" in result.output
    assert "lancedb" in result.output


def test_serving_status_says_when_the_target_has_no_row_for_the_index(
    tmp_path: Path,
) -> None:
    """`unpublished` reads as a settled fact about the index. For a target
    that has never heard of it, it is really an empty result -- the reading
    that turned a wrong-target lookup into a confident wrong answer."""
    from click.testing import CliRunner

    from stel.cli import cli

    # Never run, so the ledger has no row for this scope at all.
    project = _write_project(tmp_path)
    result = CliRunner().invoke(
        cli, ["serving", "status", "release_search", "--project-dir", str(project)]
    )

    assert result.exit_code == 0, result.output
    assert "no ledger row for 'release_search'" in result.output
    assert "check --target" in result.output
    assert "status:            unpublished" in result.output


def test_serving_recover_refuses_a_defaulted_target_without_touching_anything(
    tmp_path: Path,
) -> None:
    """Recovery advances the fencing token and marks the scope failed. It
    already demands `--owner-terminated`; inferring which store to apply that
    to undoes the care, and did -- a recovery reported success against a dev
    scope nobody had asked about while prod stayed stranded."""
    from click.testing import CliRunner

    from stel.cli import cli
    from stel.config import load_project
    from stel.profile import resolve_profile
    from stel.runner import run_project

    project = _write_project(tmp_path)
    run_project(project)
    scope = _serving_scope(project)
    project_config, _sources, _models = load_project(project)
    resolved = resolve_profile(project_config, project)
    with create_adapter(resolved.warehouse, project_dir=project) as adapter:
        before = ServingCoordinator(adapter).status(scope)

    result = CliRunner().invoke(
        cli,
        [
            "serving",
            "recover",
            "release_search",
            "--owner-terminated",
            "--project-dir",
            str(project),
        ],
    )

    assert result.exit_code != 0
    assert "requires an explicit --target" in result.output
    # It names the target it would have used and the store behind it, so the
    # operator can confirm rather than guess.
    assert "'dev'" in result.output
    assert "lancedb" in result.output

    with create_adapter(resolved.warehouse, project_dir=project) as adapter:
        after = ServingCoordinator(adapter).status(scope)
    # Refused before anything moved: the resolution it needed to name the
    # default is a read.
    assert after.fencing_token == before.fencing_token
    assert after.status == before.status


def test_serving_recover_output_names_the_target_it_acted_on(
    tmp_path: Path,
) -> None:
    """`Recovered serving scope for 'x'` was equally true of dev and prod, so
    the output carried no signal that the wrong scope had been touched."""
    from click.testing import CliRunner

    from stel.cli import cli
    from stel.runner import run_project

    project = _write_project(tmp_path)
    run_project(project)
    result = CliRunner().invoke(
        cli,
        [
            "serving",
            "recover",
            "release_search",
            "--owner-terminated",
            "--target",
            "dev",
            "--project-dir",
            str(project),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "target:            dev" in result.output
    assert "on target 'dev'" in result.output
    # `degraded`, not `failed`: this index published cleanly, so the
    # generation it activated survives recovery and keeps serving (#449).
    assert "status=degraded" in result.output


# ─── governed publication and queries ───────────────────────────────────────


def test_governed_publication_revokes_before_upsert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stel.runner import run_project

    project = _write_project(tmp_path, access="governed")
    results = run_project(project)
    assert results[-1].serving_resource is not None
    assert results[-1].serving_resource["status"] == "ready"

    # Change an attribute only: the chunk ID stays stable, so the record is a
    # keyed in-place update — the path that must revoke before upserting.
    data = project / "data" / "inflation.json"
    payload = json.loads(data.read_text())
    payload["tenant"] = "restricted"
    data.write_text(json.dumps(payload))

    calls: list[str] = []
    original_upsert = LanceDBStore.upsert
    original_delete = LanceDBStore.delete

    def _recording_upsert(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append("upsert")
        return original_upsert(self, *args, **kwargs)

    def _recording_delete(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append("delete")
        return original_delete(self, *args, **kwargs)

    monkeypatch.setattr(LanceDBStore, "upsert", _recording_upsert)
    monkeypatch.setattr(LanceDBStore, "delete", _recording_delete)
    run_project(project)
    assert "delete" in calls and "upsert" in calls
    assert calls.index("delete") < calls.index("upsert")


def test_governed_queries_fail_closed_without_policy(tmp_path: Path) -> None:
    from stel.runner import run_project
    from stel.search import (
        SearchError,
        SearchFilter,
        SearchFilterOperator,
        SearchMode,
        SearchRequest,
        search,
    )

    project = _write_project(tmp_path, access="governed")
    run_project(project)
    request = SearchRequest(
        model="release_search", query="inflation", mode=SearchMode.TEXT
    )

    with pytest.raises(SearchError, match="fail closed"):
        search(project, request)

    policy = (SearchFilter("tenant", SearchFilterOperator.EQUAL, "research"),)
    hits = search(project, request, policy_filters=policy)
    assert hits
    assert all(hit.metadata.get("tenant") == "research" for hit in hits)

    restricted = search(
        project,
        SearchRequest(model="release_search", query="output", mode=SearchMode.TEXT),
        policy_filters=(
            SearchFilter("tenant", SearchFilterOperator.EQUAL, "research"),
        ),
    )
    assert all(hit.metadata.get("tenant") == "research" for hit in restricted)


def test_policy_filters_validated_and_user_filters_cannot_touch_policy_fields(
    tmp_path: Path,
) -> None:
    from stel.runner import run_project
    from stel.search import (
        SearchError,
        SearchFilter,
        SearchFilterOperator,
        SearchMode,
        SearchRequest,
        search,
    )

    project = _write_project(tmp_path, access="governed")
    run_project(project)
    policy = (SearchFilter("tenant", SearchFilterOperator.EQUAL, "research"),)

    with pytest.raises(SearchError, match="not a policy attribute"):
        search(
            project,
            SearchRequest(model="release_search", query="jobs", mode=SearchMode.TEXT),
            policy_filters=(
                SearchFilter("category", SearchFilterOperator.EQUAL, "labor"),
            ),
        )

    with pytest.raises(SearchError, match="not available for user filtering"):
        search(
            project,
            SearchRequest(
                model="release_search",
                query="jobs",
                mode=SearchMode.TEXT,
                filters=(
                    SearchFilter("tenant", SearchFilterOperator.EQUAL, "research"),
                ),
            ),
            policy_filters=policy,
        )


def test_public_indexes_reject_policy_filters(tmp_path: Path) -> None:
    from stel.runner import run_project
    from stel.search import (
        SearchError,
        SearchFilter,
        SearchFilterOperator,
        SearchMode,
        SearchRequest,
        search,
    )

    project = _write_project(tmp_path)
    run_project(project)
    with pytest.raises(SearchError, match="do not accept policy filters"):
        search(
            project,
            SearchRequest(
                model="release_search", query="inflation", mode=SearchMode.TEXT
            ),
            policy_filters=(
                SearchFilter("category", SearchFilterOperator.EQUAL, "prices"),
            ),
        )


# ─── serving scope re-keying (issue #355) ───────────────────────────────────


def _legacy_scope(model_name: str = "context_search") -> StateScope:
    """A scope standing in for a pre-#355 physical-collection-keyed identity."""
    return StateScope.for_target_descriptor(
        model_name,
        stage="retrieval_publish",
        descriptor={"store_type": "lancedb", "physical_collection": "proj_dev_context"},
    )


def test_a_query_against_an_unmigrated_scope_names_the_rekey(
    coordinator: Any,
) -> None:
    """Issue #413: an index published before #355 keeps its ledger row under the
    old key, and a miss on the new key looked exactly like never having been
    published — so the error sent operators to `stel run`, i.e. to re-embed a
    corpus, when a one-time re-key was all it needed. A live 0.11 index went
    dark on upgrade and was recovered by republishing every collection."""
    legacy, current = _legacy_scope(), _scope()
    lease = coordinator.acquire_publish(
        legacy, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        lease,
        active_generation="gen1",
        config_fingerprint="cfg1",
        counts=(2, 0, 0, 0),
    )

    with pytest.raises(ServingNotReadyError) as caught:
        coordinator.acquire_query(current, legacy_scope=legacy)

    message = str(caught.value)
    assert "stel serving migrate-scope context_search" in message
    # The point of the message: it must not send the operator to re-embed.
    assert "stel run" not in message


def test_a_genuinely_unpublished_scope_still_says_unpublished(
    coordinator: Any,
) -> None:
    """The re-key hint must not swallow the ordinary case: nothing under either
    key means nothing was ever published."""
    with pytest.raises(ServingNotReadyError) as caught:
        coordinator.acquire_query(_scope(), legacy_scope=_legacy_scope())
    assert "has not been published" in str(caught.value)


def test_the_rekey_hint_costs_nothing_once_the_scope_is_migrated(
    coordinator: Any,
) -> None:
    """After `migrate-scope` the row is under the current key, so the query
    resolves normally and never consults the legacy one."""
    legacy, current = _legacy_scope(), _scope()
    lease = coordinator.acquire_publish(
        legacy, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        lease,
        active_generation="gen1",
        config_fingerprint="cfg1",
        counts=(2, 0, 0, 0),
    )
    assert coordinator.rekey_scope(legacy, current) == 1

    query = coordinator.acquire_query(current, legacy_scope=legacy)
    try:
        assert query.pinned_generation == "gen1"
    finally:
        coordinator.release_query(query)


def test_state_retrieval_target_keys_on_logical_not_physical() -> None:
    """The whole point of #355: two physical names, one stable serving identity.

    If the descriptor kept the physical collection, a generation swap would
    change target_identity, and a reader would need the active generation in
    order to compute the scope that names the active generation.
    """
    from stel.retrieval.base import StateRetrievalTarget

    gen_a = StateRetrievalTarget("lancedb", "routing1", "proj_dev_ctx_g1", "ctx")
    gen_b = StateRetrievalTarget("lancedb", "routing1", "proj_dev_ctx_g2", "ctx")

    assert gen_a.descriptor() == gen_b.descriptor()
    assert "physical_collection" not in gen_a.descriptor()
    assert gen_a.descriptor()["logical_collection"] == "ctx"
    # The physical name is still reported for artifacts, just not keyed on.
    assert gen_a.physical_collection != gen_b.physical_collection
    assert gen_a.legacy_descriptor() != gen_b.legacy_descriptor()


def test_rekey_scope_moves_ledger_row(coordinator: Any) -> None:
    legacy, current = _legacy_scope(), _scope()
    lease = coordinator.acquire_publish(
        legacy, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        lease,
        active_generation="gen1",
        config_fingerprint="cfg1",
        counts=(2, 0, 0, 0),
    )
    assert coordinator.status(legacy).status == STATUS_READY
    assert coordinator.status(current).status == STATUS_UNPUBLISHED

    assert coordinator.rekey_scope(legacy, current) == 1

    moved = coordinator.status(current)
    assert moved.status == STATUS_READY
    assert moved.active_generation == "gen1"
    assert moved.fencing_token == lease.fencing_token


def test_rekey_scope_is_idempotent_and_ignores_absent_source(
    coordinator: Any,
) -> None:
    legacy, current = _legacy_scope(), _scope()
    lease = coordinator.acquire_publish(
        legacy, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        lease,
        active_generation="gen1",
        config_fingerprint="cfg1",
        counts=(2, 0, 0, 0),
    )

    assert coordinator.rekey_scope(legacy, current) == 1
    # Second run finds nothing under the old identity and reports zero rather
    # than failing, so the migration command is safe to re-run.
    assert coordinator.rekey_scope(legacy, current) == 0
    assert coordinator.status(current).status == STATUS_READY


def test_rekey_scope_refuses_when_destination_is_occupied(
    coordinator: Any,
) -> None:
    legacy, current = _legacy_scope(), _scope()
    for scope in (legacy, current):
        lease = coordinator.acquire_publish(
            scope, expected_code_version="v1", config_fingerprint="cfg1"
        )
        coordinator.mark_ready(
        lease,
        active_generation="gen1",
        config_fingerprint="cfg1",
        counts=(2, 0, 0, 0),
    )

    with pytest.raises(ServingCoordinationError, match="already has"):
        coordinator.rekey_scope(legacy, current)
    # Neither side is disturbed by the refusal.
    assert coordinator.status(legacy).status == STATUS_READY
    assert coordinator.status(current).status == STATUS_READY


def test_rekey_scope_refuses_while_a_query_lease_is_outstanding(
    coordinator: Any,
) -> None:
    legacy, current = _legacy_scope(), _scope()
    publish = coordinator.acquire_publish(
        legacy, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        publish,
        active_generation="gen1",
        config_fingerprint="cfg1",
        counts=(2, 0, 0, 0),
    )
    query = coordinator.acquire_query(legacy)

    with pytest.raises(ServingBusyError, match="outstanding query leases"):
        coordinator.rekey_scope(legacy, current)

    # The reader is undisturbed and still validates against its own scope.
    coordinator.validate_query(query)
    coordinator.release_query(query)
    assert coordinator.rekey_scope(legacy, current) == 1


def test_rekey_scope_refuses_to_change_model_or_stage(coordinator: Any) -> None:
    legacy = _legacy_scope()
    other_model = _scope("other_search")
    with pytest.raises(ServingCoordinationError, match="only change target_identity"):
        coordinator.rekey_scope(legacy, other_model)


def test_rekey_state_scope_moves_rows_and_refuses_collisions(
    tmp_path: Path,
) -> None:
    from stel.adapters.base import AdapterError, StateRecord

    legacy, current = _legacy_scope(), _scope()
    with create_adapter(_wh(tmp_path / "state.duckdb")) as adapter:
        adapter.upsert_state(
            legacy,
            [
                StateRecord("doc-1", "fp1", "v1"),
                StateRecord("doc-2", "fp2", "v1"),
            ],
        )
        assert len(adapter.fetch_state(legacy)) == 2
        assert adapter.fetch_state(current) == {}

        assert adapter.rekey_state_scope(legacy, current) == 2

        moved = adapter.fetch_state(current)
        assert set(moved) == {"doc-1", "doc-2"}
        assert moved["doc-1"].input_fingerprint == "fp1"
        assert adapter.fetch_state(legacy) == {}
        # Idempotent: nothing left under the old identity.
        assert adapter.rekey_state_scope(legacy, current) == 0

        # A collision must not silently merge or discard either publication.
        adapter.upsert_state(legacy, [StateRecord("doc-3", "fp3", "v1")])
        with pytest.raises(AdapterError, match="already holds"):
            adapter.rekey_state_scope(legacy, current)
        assert len(adapter.fetch_state(legacy)) == 1
        assert len(adapter.fetch_state(current)) == 2


# ─── ledger-resolved activation (issue #355) ────────────────────────────────


def test_a_ledger_predating_generations_gains_the_activation_column(
    tmp_path: Path,
) -> None:
    """`_ensure_tables` is CREATE IF NOT EXISTS, so an older ledger needs ALTER.

    Without the ALTER, every statement naming `active_collection` fails
    against a ledger written by an earlier version.
    """
    from stel.adapters.base import SERVING_LEDGER_TABLE

    warehouse = _wh(tmp_path / "old.duckdb")
    with create_adapter(warehouse) as adapter:
        adapter.execute(f"CREATE SCHEMA IF NOT EXISTS {adapter.schema_ref}")
        adapter.execute(
            f"""
            CREATE TABLE {adapter.schema_ref}.{adapter.quote_ident(SERVING_LEDGER_TABLE)} (
                model_name STRING NOT NULL,
                stage STRING NOT NULL,
                target_identity STRING NOT NULL,
                row_id STRING NOT NULL,
                fencing_token BIGINT NOT NULL,
                status STRING NOT NULL,
                publication_id STRING,
                expected_code_version STRING,
                config_fingerprint STRING,
                active_generation STRING,
                safe_error_code STRING,
                rows_inserted BIGINT NOT NULL,
                rows_updated BIGINT NOT NULL,
                rows_skipped BIGINT NOT NULL,
                rows_deleted BIGINT NOT NULL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP
            )
            """
        )
        assert "active_collection" not in (
            adapter.table_column_names(SERVING_LEDGER_TABLE) or frozenset()
        )

        coordinator = ServingCoordinator(adapter)

        assert "active_collection" in (
            adapter.table_column_names(SERVING_LEDGER_TABLE) or frozenset()
        )
        # And the upgraded ledger is usable end to end.
        scope = _scope()
        lease = coordinator.acquire_publish(
            scope, expected_code_version="v1", config_fingerprint="cfg1"
        )
        coordinator.mark_ready(
            lease,
            active_generation="gen1",
            config_fingerprint="cfg1",
            counts=(1, 0, 0, 0),
        )
        assert coordinator.status(scope).active_collection is None


def test_mark_ready_records_the_activation_pointer(coordinator: Any) -> None:
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        lease,
        active_generation="gen1",
        config_fingerprint="cfg1",
        counts=(2, 0, 0, 0),
        active_collection="proj__dev__ctx__ga1b2",
    )
    assert coordinator.status(scope).active_collection == "proj__dev__ctx__ga1b2"


def test_activation_defaults_to_the_unsuffixed_collection(
    coordinator: Any,
) -> None:
    """An in-place incremental publish leaves the pointer null.

    Null is what every row written before generations existed means, and it
    resolves to the unsuffixed default — so this is the path that keeps
    already-published indexes working untouched.
    """
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        lease,
        active_generation="gen1",
        config_fingerprint="cfg1",
        counts=(2, 0, 0, 0),
    )
    assert coordinator.status(scope).active_collection is None
    assert coordinator.acquire_query(scope).pinned_collection is None


def test_a_query_lease_pins_the_collection_it_resolved(coordinator: Any) -> None:
    """The pin is what keeps a reader coherent across a later activation."""
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        lease,
        active_generation="gen1",
        config_fingerprint="cfg1",
        counts=(2, 0, 0, 0),
        active_collection="proj__dev__ctx__ga1b2",
    )
    query = coordinator.acquire_query(scope)
    assert query.pinned_collection == "proj__dev__ctx__ga1b2"
    assert query.pinned_generation == "gen1"
    coordinator.release_query(query)


def test_an_in_place_failure_clears_both_pointers_and_stops_serving(
    coordinator: Any,
) -> None:
    """An in-place publish writes into the collection the pointer names, so a
    failure may have corrupted what was live. Neither pointer can be trusted
    and nothing may be served — the case `degraded` deliberately excludes."""
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        lease,
        active_generation="gen1",
        config_fingerprint="cfg1",
        counts=(2, 0, 0, 0),
        active_collection="proj__dev__ctx__ga1b2",
    )
    retry = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_failed(retry, safe_error_code="publish_failed")

    entry = coordinator.status(scope)
    assert entry.status == STATUS_FAILED
    assert entry.active_collection is None
    assert entry.active_generation is None
    with pytest.raises(ServingNotReadyError):
        coordinator.acquire_query(scope)


def test_a_rebuild_failure_keeps_serving_the_previous_generation(
    coordinator: Any,
) -> None:
    """The common case #449 is about, and the one that does not need a crash:
    a private generation build writes where nothing reads, so its failure
    leaves the live generation correct. The index keeps answering from it."""
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        lease,
        active_generation="gen1",
        config_fingerprint="cfg1",
        counts=(2, 0, 0, 0),
        active_collection="proj__dev__ctx__ga1b2",
    )
    retry = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    # What search.py passes on a rebuild: both pointers handed back.
    coordinator.mark_failed(
        retry,
        safe_error_code="publish_failed",
        active_collection="proj__dev__ctx__ga1b2",
        active_generation="gen1",
        # The configuration gen1 was published under, not the one this failed
        # publish claimed: the claim already overwrote the ledger's.
        config_fingerprint="cfg1",
    )

    entry = coordinator.status(scope)
    assert entry.status == STATUS_DEGRADED
    assert entry.active_generation == "gen1"
    assert entry.safe_error_code == "publish_failed"

    query = coordinator.acquire_query(scope)
    assert query.pinned_generation == "gen1"
    assert query.pinned_collection == "proj__dev__ctx__ga1b2"
    coordinator.release_query(query)


def test_a_successful_publish_clears_degraded(coordinator: Any) -> None:
    """Degraded is not sticky: the point is that it heals on the next publish
    that works, without an operator having to clear anything."""
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        lease,
        active_generation="gen1",
        config_fingerprint="cfg1",
        counts=(2, 0, 0, 0),
        active_collection="proj__dev__ctx__ga1b2",
    )
    failed = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_failed(
        failed,
        safe_error_code="publish_failed",
        active_collection="proj__dev__ctx__ga1b2",
        active_generation="gen1",
        config_fingerprint="cfg1",
    )
    assert coordinator.status(scope).status == STATUS_DEGRADED

    retry = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg2"
    )
    coordinator.mark_ready(
        retry,
        active_generation="gen2",
        config_fingerprint="cfg2",
        counts=(3, 0, 0, 0),
        active_collection="proj__dev__ctx__gc3d4",
    )

    entry = coordinator.status(scope)
    assert entry.status == STATUS_READY
    assert entry.active_generation == "gen2"
    assert entry.safe_error_code is None


def test_recover_keeps_serving_the_generation_it_recovered(coordinator: Any) -> None:
    """Recovery must not take a healthy index offline.

    It rebuilds the ledger row, so both activation pointers have to be carried
    across (issues #355, #449). A crashed publisher was writing a *new*
    generation; the one that was live is untouched, so the scope is left
    `degraded` — still answering queries from it — rather than `failed`, which
    admits nothing until the next successful publish. On a large corpus that
    is hours of a working index being unavailable for no reason.
    """
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        lease,
        active_generation="gen1",
        config_fingerprint="cfg1",
        counts=(2, 0, 0, 0),
        active_collection="proj__dev__ctx__ga1b2",
    )

    entry = coordinator.recover(scope, owner_terminated=True)

    assert entry.status == STATUS_DEGRADED
    assert entry.active_generation == "gen1"
    assert entry.active_collection == "proj__dev__ctx__ga1b2"
    # The failure stays on the record: degraded must not read as healthy.
    assert entry.safe_error_code == RECOVERY_ERROR_CODE

    # And the index actually answers, pinned to the generation it recovered.
    query = coordinator.acquire_query(scope)
    assert query.pinned_generation == "gen1"
    assert query.pinned_collection == "proj__dev__ctx__ga1b2"
    coordinator.release_query(query)


def test_recover_refuses_to_serve_after_a_crashed_in_place_publish(
    coordinator: Any,
) -> None:
    """The dangerous case, and the reason the claim records its mode.

    A crashed publisher leaves no record of its intent, so recovery cannot ask
    it what it was doing. An in-place publish mutates the collection the
    pointer names, so serving that generation after a crash could serve a
    half-rewritten index. The claim clears the pointer up front, which is what
    makes recovery fail closed here (PR #450 review).
    """
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        lease,
        active_generation="gen1",
        config_fingerprint="cfg1",
        counts=(2, 0, 0, 0),
        active_collection="proj__dev__ctx__ga1b2",
    )
    # An in-place publisher claims the scope, then dies without finishing.
    coordinator.acquire_publish(
        scope,
        expected_code_version="v2",
        config_fingerprint="cfg1",
        preserves_active_generation=False,
    )

    entry = coordinator.recover(scope, owner_terminated=True)

    assert entry.status == STATUS_FAILED
    assert entry.active_generation is None
    with pytest.raises(ServingNotReadyError):
        coordinator.acquire_query(scope)


def test_recover_serves_on_after_a_crashed_rebuild(coordinator: Any) -> None:
    """The mirror image: a rebuild writes to a private generation, so a
    crashed one leaves the live generation untouched and servable."""
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_ready(
        lease,
        active_generation="gen1",
        config_fingerprint="cfg1",
        counts=(2, 0, 0, 0),
        active_collection="proj__dev__ctx__ga1b2",
    )
    coordinator.acquire_publish(
        scope,
        expected_code_version="v2",
        config_fingerprint="cfg1",
        preserves_active_generation=True,
    )

    entry = coordinator.recover(scope, owner_terminated=True)

    assert entry.status == STATUS_DEGRADED
    assert entry.active_generation == "gen1"
    query = coordinator.acquire_query(scope)
    assert query.pinned_generation == "gen1"
    coordinator.release_query(query)


def test_a_retained_generation_keeps_its_own_configuration_fingerprint(
    coordinator: Any,
) -> None:
    """A config change forces the rebuild, so the claim overwrote the ledger's
    fingerprint with the *new* configuration. Retaining the old generation
    under that fingerprint would advertise the old index as answering for a
    configuration it was never built for (PR #450 review) -- the reader's
    first check compares the two, and it must see the generation's own."""
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg-old"
    )
    coordinator.mark_ready(
        lease,
        active_generation="gen1",
        config_fingerprint="cfg-old",
        counts=(2, 0, 0, 0),
        active_collection="proj__dev__ctx__ga1b2",
    )
    retry = coordinator.acquire_publish(
        scope,
        expected_code_version="v1",
        config_fingerprint="cfg-new",
        preserves_active_generation=True,
    )
    assert coordinator.status(scope).config_fingerprint == "cfg-old"
    assert retry.config_fingerprint == "cfg-new"

    coordinator.mark_failed(
        retry,
        safe_error_code="publish_failed",
        active_collection="proj__dev__ctx__ga1b2",
        active_generation="gen1",
        config_fingerprint="cfg-old",
    )

    entry = coordinator.status(scope)
    assert entry.status == STATUS_DEGRADED
    assert entry.config_fingerprint == "cfg-old"
    query = coordinator.acquire_query(scope)
    assert query.config_fingerprint == "cfg-old"
    coordinator.release_query(query)


def test_retaining_a_generation_without_its_fingerprint_is_refused(
    coordinator: Any,
) -> None:
    """Fail loudly rather than publish an incoherent ledger row."""
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    with pytest.raises(ServingCoordinationError, match="configuration"):
        coordinator.mark_failed(
            lease,
            safe_error_code="publish_failed",
            active_generation="gen1",
        )


def test_recover_leaves_a_scope_with_no_generation_failed(coordinator: Any) -> None:
    """The other half: an in-place publish's failure already cleared the
    generation because it may have corrupted what was live. There is nothing
    safe to serve, so recovery leaves the scope failed and admits nothing."""
    scope = _scope()
    lease = coordinator.acquire_publish(
        scope, expected_code_version="v1", config_fingerprint="cfg1"
    )
    coordinator.mark_failed(lease, safe_error_code="store_error")

    entry = coordinator.recover(scope, owner_terminated=True)

    assert entry.status == STATUS_FAILED
    assert entry.active_generation is None
    with pytest.raises(ServingNotReadyError):
        coordinator.acquire_query(scope)


def test_a_search_publish_attributes_its_phases(tmp_path: Path) -> None:
    """The path #452 was actually about. That publish was dominated by
    per-page round trips — ~3.7s of ledger reads and a MERGE per page, 13,794
    pages — and nothing in the run said so, which is why it took a hand count
    to find. `read`, `store_write` and `state` separate the three candidates
    (#432 item 1)."""
    from stel.runner import run_project

    project = _write_project(tmp_path)

    results = run_project(project)

    [search] = [r for r in results if r.kind == "search"]
    for phase in ("seconds_read", "seconds_store_write", "seconds_state"):
        assert phase in search.metrics, search.metrics
        assert search.metrics[phase] >= 0.0


def test_a_search_publish_attributes_its_index_reconciliation(tmp_path: Path) -> None:
    """The ANN build is the largest term in a large publish and was the last
    one of that size left outside every phase.

    #473 was an index build that exhausted the container; v0.16.0 changed how
    one is chosen and where it runs. A reader could see `read`, `store_write`
    and `state` sum to a fraction of `duration_seconds` with nothing saying
    where the rest went — and the operation the run had just been rebuilt
    around was the missing term.
    """
    from stel.runner import run_project

    project = _write_project(tmp_path)

    results = run_project(project)

    [search] = [r for r in results if r.kind == "search"]
    assert "seconds_index_reconcile" in search.metrics, search.metrics
    assert search.metrics["seconds_index_reconcile"] >= 0.0


def test_the_index_phase_does_not_claim_a_build_it_did_not_do(tmp_path: Path) -> None:
    """A rerun that indexes nothing still reports the phase, which is why it
    is not called `index_build`.

    Every `create_index` in both stores is conditional: an unchanged publish
    lists the indexes, counts rows, and builds nothing. A phase named for the
    build would attribute that metadata check to ANN construction — the exact
    reading the docs promise is impossible, since an absent phase is supposed
    to mean the work did not happen (PR #486 review). The name describes the
    block, and the duration separates a check from a build far more finely
    than a flag would.
    """
    from stel.runner import run_project

    project = _write_project(tmp_path)
    run_project(project)

    results = run_project(project)

    [search] = [r for r in results if r.kind == "search"]
    assert "seconds_index_build" not in search.metrics, search.metrics
    assert "seconds_index_reconcile" in search.metrics, search.metrics


def test_search_timings_cover_deletion_and_activation(tmp_path: Path) -> None:
    """A run whose work is deletion must not report only `seconds_read`.

    The stale-removal pass and generation activation do real store and state
    work; leaving them outside every phase meant an incremental run that
    removed rows attributed almost nothing, which defeats the point of having
    attribution at all (PR #460 review).
    """
    import json

    from stel.runner import run_project

    project = _write_project(tmp_path)
    run_project(project)
    # Remove the upstream rows so the next publish is deletion-dominated.
    data = project / "data" / "inflation.json"
    data.unlink()

    results = run_project(project)

    [search] = [r for r in results if r.kind == "search"]
    assert search.documents_deleted > 0, "expected a deletion-dominated publish"
    # The phases exist on a run that did no upserts at all.
    assert "seconds_store_write" in search.metrics
    assert "seconds_state" in search.metrics
    assert json.dumps(search.metrics)  # metrics stay JSON-serializable
