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
)
from stel.retrieval.coordination import (
    RECOVERY_ERROR_CODE,
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
    ledger = f"{adapter.schema_ref}.{adapter.quote_ident('dbt_ml_serving_ledger')}"
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
        config, project_name="demo", target_name="dev", alias="primary"
    )
    other = LanceDBStore(
        config, project_name="demo", target_name="dev", alias="primary"
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
        config, project_name="demo", target_name="dev", alias="primary"
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
    refused = runner.invoke(
        cli,
        ["serving", "recover", "release_search", "--project-dir", str(project)],
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
            "--project-dir",
            str(project),
        ],
    )
    assert recovered.exit_code == 0, recovered.output
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
