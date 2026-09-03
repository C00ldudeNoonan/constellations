from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import lancedb
import polars as pl
import pyarrow as pa
import pytest
from click.testing import CliRunner

from stel.adapters import (
    AdapterError,
    StateScope,
    TableReadSnapshot,
    create_adapter,
    parse_warehouse_config,
)
from stel.adapters.duckdb import DuckDBAdapter
from stel.cli import cli
from stel.cli_services.serving import resolve_serving_scope
from stel.compiler import (
    validate_project_contract,
    validate_retrieval_capabilities,
)
from stel.config import ConfigError, load_project
from stel.dbt_export import build_dbt_sources
from stel.manifest import build_manifest
from stel.profile import resolve_profile
from stel.retrieval import (
    CollectionSpec,
    IndexedRow,
    LanceDBConfig,
    LanceDBStore,
    RetrievalError,
    RetrievalFeature,
    RetrievalPredicate,
    RetrievalPredicateOperator,
    ServingCoordinator,
    StaleServingLeaseError,
    StoreRole,
    collection_config_fingerprint,
    create_store,
)
from stel.retrieval.retention import (
    retire_superseded_generations,
    superseded_generations,
)
from stel.runner import RunError, run_project


def _write_project(tmp_path: Path, *, allow_public: bool = True) -> None:
    (tmp_path / "sources").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "stel_project.yml").write_text(
        "\n".join(
            [
                "name: retrieval_demo",
                "version: '0.1.0'",
                "profile: retrieval_demo",
                "source-paths: [sources]",
                "model-paths: [models]",
            ]
        )
    )
    (tmp_path / "profiles.yml").write_text(
        "\n".join(
            [
                "retrieval_demo:",
                "  target: dev",
                "  outputs:",
                "    dev:",
                "      warehouse:",
                "        type: duckdb",
                "        path: target/demo.duckdb",
                "        schema: analytics",
                "      retrieval:",
                "        default: primary",
                f"        allow_public_indexes: {str(allow_public).lower()}",
                "        stores:",
                "          primary:",
                "            type: lancedb",
                "            path: target/lancedb",
                "            collection_template: '{project}__{target}__{collection}'",
            ]
        )
    )
    (tmp_path / "sources" / "documents.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "sources:",
                "  - name: documents",
                "    path: data",
                "    file_pattern: '*.json'",
            ]
        )
    )
    (tmp_path / "models" / "retrieval.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: embedding_rows",
                "    source: ref('documents')",
                "    extraction:",
                "      backend: json",
                "  - name: context_search",
                "    depends_on: [ref('embedding_rows')]",
                "    materialization: incremental",
                "    tags: [retrieval, economic-data]",
                "    search:",
                "      access: public",
                "      store: primary",
                "      collection: context",
                "      id_field: chunk_id",
                "      document_id_field: document_id",
                "      chunk_id_field: chunk_id",
                "      text_fields: [text]",
                "      return_text_fields: [text]",
                "      vector:",
                "        field: embedding",
                "        dimensions: 2",
                "        metric: cosine",
                "        search: exact",
                "        embedding:",
                "          provider: fixture",
                "          model: deterministic-2d-v1",
                "          provider_contract_version: 2",
                "          provider_implementation: tests:v1",
                "          semantic_config_fingerprint: deterministic-2d-v1",
                "          dimensions: 2",
                "      full_text:",
                "        fields: [text]",
                "      attributes:",
                "        - name: category",
                "          data_type: string",
                "          filter_role: user",
                "          returned: true",
                "      display_fields: [title]",
                "      query:",
                "        modes: [vector, text, filter]",
                "        consistency: strong",
                "      on_index_change: fail",
                "      batch_size: 2",
            ]
        )
    )


def _write_typed_attribute_project(tmp_path: Path) -> None:
    """A project declaring one filterable attribute per `data_type` (#337).

    The bug this guards: a `date` attribute validated, built, and returned
    fine, but every filter on it failed at query time — the declared surface
    accepted a filter the query path could not execute, so the failure landed
    on the querying agent rather than the author.
    """
    _write_project(tmp_path)
    models = tmp_path / "models" / "retrieval.yml"
    models.write_text(
        models.read_text().replace(
            "\n".join(
                [
                    "      attributes:",
                    "        - name: category",
                    "          data_type: string",
                    "          filter_role: user",
                    "          returned: true",
                ]
            ),
            "\n".join(
                [
                    "      attributes:",
                    "        - name: category",
                    "          data_type: string",
                    "          filter_role: user",
                    "          returned: true",
                    "        - name: filing_date_dt",
                    "          data_type: date",
                    "          nullable: true",
                    "          filter_role: user",
                    "          returned: true",
                    "        - name: observed_at",
                    "          data_type: timestamp",
                    "          nullable: true",
                    "          filter_role: user",
                    "          returned: true",
                    "        - name: page_count",
                    "          data_type: integer",
                    "          nullable: true",
                    "          filter_role: user",
                    "          returned: true",
                    "        - name: is_amended",
                    "          data_type: boolean",
                    "          nullable: true",
                    "          filter_role: user",
                    "          returned: true",
                ]
            ),
        )
    )


def _typed_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "chunk_id": ["c1", "c2"],
            "document_id": ["d1", "d2"],
            "text": ["inflation slowed", "employment increased"],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
            "category": ["prices", "labor"],
            "title": ["CPI", "Payrolls"],
            "filing_date_dt": [date(2019, 1, 1), date(2023, 2, 13)],
            "observed_at": [
                datetime(2019, 1, 1, 9, 30, tzinfo=UTC),
                datetime(2023, 2, 13, 14, 0, tzinfo=UTC),
            ],
            "page_count": [10, 20],
            "is_amended": [False, True],
        }
    )


def _rows(version: int = 1) -> pl.DataFrame:
    if version == 1:
        return pl.DataFrame(
            {
                "chunk_id": ["c1", "c2"],
                "document_id": ["d1", "d2"],
                "text": ["inflation slowed", "employment increased"],
                "embedding": [[1.0, 0.0], [0.0, 1.0]],
                "category": ["prices", "labor"],
                "title": ["CPI", "Payrolls"],
            }
        )
    return pl.DataFrame(
        {
            "chunk_id": ["c1", "c3"],
            "document_id": ["d1", "d3"],
            "text": ["inflation declined", "output expanded"],
            "embedding": [[0.9, 0.1], [0.5, 0.5]],
            "category": ["prices", "growth"],
            "title": ["CPI revision", "GDP"],
        }
    )


def _materialize_upstream(tmp_path: Path, rows: pl.DataFrame) -> None:
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        adapter.materialize_full("embedding_rows", rows)


def test_lancedb_incremental_publication_and_queries(tmp_path: Path) -> None:
    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())

    first = run_project(tmp_path, select="context_search")
    assert len(first) == 1
    assert first[0].rows_inserted == 2
    assert first[0].rows_updated == 0
    assert first[0].documents_deleted == 0
    assert first[0].serving_resource is not None

    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert resolved.retrieval is not None
    config = resolved.retrieval.stores["primary"]
    store = create_store(
        config,
        project_name=project.name,
        target_name=resolved.target_name,
        alias="primary",
        role=StoreRole.PUBLISH,
    )
    with store:
        metadata = store.inspect_collection("retrieval_demo__dev__context")
        assert metadata is not None
        assert metadata.row_count == 2
        vector = store.vector_search(
            metadata.physical_name,
            [1.0, 0.0],
            vector_field="embedding",
            limit=1,
            columns=["chunk_id", "text", "_distance"],
            predicates=[RetrievalPredicate("category", RetrievalPredicateOperator.EQUAL, "prices")],
        )
        assert vector.column("chunk_id").to_pylist() == ["c1"]
        assert vector.column_names.count("_distance") == 1
        text = store.text_search(
            metadata.physical_name,
            "employment",
            text_field="text",
            limit=1,
            columns=["chunk_id", "text", "_score"],
        )
        assert text.column("chunk_id").to_pylist() == ["c2"]
        assert text.column_names.count("_score") == 1
        generation = metadata.physical_generation

    second = run_project(tmp_path, select="context_search")
    assert second[0].documents_processed == 0
    assert second[0].documents_skipped == 2
    with store:
        unchanged = store.inspect_collection("retrieval_demo__dev__context")
        assert unchanged is not None
        assert unchanged.physical_generation == generation

    _materialize_upstream(tmp_path, _rows(version=2))
    third = run_project(tmp_path, select="context_search")
    assert third[0].rows_inserted == 1
    assert third[0].rows_updated == 1
    assert third[0].documents_deleted == 1
    with store:
        updated = store.inspect_collection("retrieval_demo__dev__context")
        assert updated is not None
        assert updated.row_count == 2
        rows = store.vector_search(
            updated.physical_name,
            [0.5, 0.5],
            vector_field="embedding",
            limit=10,
            columns=["chunk_id"],
        )
        assert set(rows.column("chunk_id").to_pylist()) == {"c1", "c3"}


def test_search_manifest_v2_and_dbt_export_projection(tmp_path: Path) -> None:
    _write_project(tmp_path)
    manifest = build_manifest(tmp_path)

    assert manifest["manifest_version"] == 2
    search = next(model for model in manifest["models"] if model["name"] == "context_search")
    assert search["resource_type"] == "search_index"
    assert search["output"]["type"] == "serving_resource"
    assert "relation" not in search["output"]
    serialized = json.dumps(manifest)
    assert str(tmp_path / "target" / "lancedb") not in serialized

    payload = build_dbt_sources(tmp_path, select="context_search")
    assert [table["name"] for table in payload["sources"][0]["tables"]] == ["embedding_rows"]
    excluded = build_dbt_sources(
        tmp_path,
        select="context_search",
        exclude="embedding_rows",
    )
    assert excluded["sources"][0]["tables"] == []


def test_public_search_requires_operator_profile_opt_in(tmp_path: Path) -> None:
    _write_project(tmp_path, allow_public=False)
    project, sources, models = load_project(tmp_path)
    validate_project_contract(project, sources, models, tmp_path)
    resolved = resolve_profile(project, tmp_path)

    with pytest.raises(ConfigError, match="allow_public_indexes"):
        validate_retrieval_capabilities(models, project, resolved)


def test_inherited_embedding_identity_requires_an_upstream_embed_resource(
    tmp_path: Path,
) -> None:
    _write_project(tmp_path)
    model_path = tmp_path / "models" / "retrieval.yml"
    identity = "\n".join(
        [
            "        embedding:",
            "          provider: fixture",
            "          model: deterministic-2d-v1",
            "          provider_contract_version: 2",
            "          provider_implementation: tests:v1",
            "          semantic_config_fingerprint: deterministic-2d-v1",
            "          dimensions: 2",
        ]
    )
    model_path.write_text(model_path.read_text().replace(identity, "        embedding: inherit"))
    project, sources, models = load_project(tmp_path)

    with pytest.raises(ConfigError, match="direct upstream embed model"):
        validate_project_contract(project, sources, models, tmp_path)


def test_hybrid_mode_is_validated_against_store_capabilities(tmp_path: Path) -> None:
    _write_project(tmp_path)
    model_path = tmp_path / "models" / "retrieval.yml"
    model_path.write_text(
        model_path.read_text().replace(
            "modes: [vector, text, filter]", "modes: [vector, text, hybrid]"
        )
    )
    project, sources, models = load_project(tmp_path)
    validate_project_contract(project, sources, models, tmp_path)
    resolved = resolve_profile(project, tmp_path)

    validate_retrieval_capabilities(models, project, resolved)


def test_search_predicate_repr_redacts_value() -> None:
    predicate = RetrievalPredicate(
        "category",
        RetrievalPredicateOperator.EQUAL,
        "sensitive-value'; DROP TABLE context; --",
    )
    assert "sensitive-value" not in repr(predicate)
    assert "<redacted>" in repr(predicate)


def test_search_predicate_rejects_nonfinite_values() -> None:
    with pytest.raises(RetrievalError, match="unsupported"):
        RetrievalPredicate("score", RetrievalPredicateOperator.EQUAL, float("nan"))


def test_search_state_is_scoped_to_safe_target(tmp_path: Path) -> None:
    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())
    run_project(tmp_path, select="context_search")

    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert resolved.retrieval is not None
    config = resolved.retrieval.stores["primary"]
    store = create_store(
        config,
        project_name=project.name,
        target_name=resolved.target_name,
        alias="primary",
        role=StoreRole.PUBLISH,
    )
    descriptor = store.state_descriptor("context")
    scope = StateScope.for_target_descriptor(
        "context_search",
        stage="retrieval_publish",
        descriptor=descriptor.descriptor(),
    )
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        assert set(adapter.fetch_state(scope)) == {"c1", "c2"}


def test_failed_store_mutation_does_not_advance_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())

    def fail_upsert(*args: object, **kwargs: object) -> object:
        raise RetrievalError("safe injected failure")

    monkeypatch.setattr(LanceDBStore, "upsert", fail_upsert)
    with pytest.raises(RunError, match="safe injected failure"):
        run_project(tmp_path, select="context_search")

    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert resolved.retrieval is not None
    store = create_store(
        resolved.retrieval.stores["primary"],
        project_name=project.name,
        target_name=resolved.target_name,
        alias="primary",
        role=StoreRole.PUBLISH,
    )
    descriptor = store.state_descriptor("context")
    scope = StateScope.for_target_descriptor(
        "context_search",
        stage="retrieval_publish",
        descriptor=descriptor.descriptor(),
    )
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        assert adapter.fetch_state(scope) == {}
    with store:
        metadata = store.inspect_collection("retrieval_demo__dev__context")
        assert metadata is not None
        assert metadata.row_count == 0

    monkeypatch.undo()
    retry = run_project(tmp_path, select="context_search")
    assert retry[0].rows_inserted == 2
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        assert set(adapter.fetch_state(scope)) == {"c1", "c2"}
    with store:
        metadata = store.inspect_collection("retrieval_demo__dev__context")
        assert metadata is not None
        assert metadata.row_count == 2


def test_failed_index_validation_keeps_receipted_state_and_blocks_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after durable receipts keeps acknowledged state (issue #153):
    per-batch advancement records exactly what the store acknowledged, while
    the serving ledger keeps the failed publication unqueryable."""
    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())

    def fail_indexes(*args: object, **kwargs: object) -> object:
        raise RetrievalError("safe injected index failure")

    monkeypatch.setattr(LanceDBStore, "ensure_indexes", fail_indexes)
    with pytest.raises(RunError, match="safe injected index failure"):
        run_project(tmp_path, select="context_search")

    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert resolved.retrieval is not None
    store = create_store(
        resolved.retrieval.stores["primary"],
        project_name=project.name,
        target_name=resolved.target_name,
        alias="primary",
        role=StoreRole.PUBLISH,
    )
    scope = StateScope.for_target_descriptor(
        "context_search",
        stage="retrieval_publish",
        descriptor=store.state_descriptor("context").descriptor(),
    )
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        assert sorted(adapter.fetch_state(scope)) == ["c1", "c2"]
        assert ServingCoordinator(adapter).status(scope).status == "failed"
    with store:
        metadata = store.inspect_collection("retrieval_demo__dev__context")
        assert metadata is not None
        assert metadata.row_count == 2

    monkeypatch.undo()
    retry = run_project(tmp_path, select="context_search")
    assert retry[0].rows_inserted == 0
    assert retry[0].documents_skipped == 2
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        assert ServingCoordinator(adapter).status(scope).status == "ready"


def test_failed_snapshot_validation_keeps_receipted_state_and_blocks_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Snapshot invalidation after publication keeps receipt-acknowledged
    state (issue #153) but never activates readiness; the retry reconciles
    from recorded state instead of republishing acknowledged rows."""
    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())

    def fail_snapshot_validation(self: TableReadSnapshot) -> None:
        raise AdapterError("safe injected generation failure")

    monkeypatch.setattr(
        TableReadSnapshot,
        "validate_unchanged",
        fail_snapshot_validation,
    )
    with pytest.raises(RunError, match="safe injected generation failure"):
        run_project(tmp_path, select="context_search")

    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert resolved.retrieval is not None
    store = create_store(
        resolved.retrieval.stores["primary"],
        project_name=project.name,
        target_name=resolved.target_name,
        alias="primary",
        role=StoreRole.PUBLISH,
    )
    scope = StateScope.for_target_descriptor(
        "context_search",
        stage="retrieval_publish",
        descriptor=store.state_descriptor("context").descriptor(),
    )
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        assert sorted(adapter.fetch_state(scope)) == ["c1", "c2"]
        assert ServingCoordinator(adapter).status(scope).status == "failed"

    monkeypatch.undo()
    retry = run_project(tmp_path, select="context_search")
    assert retry[0].rows_inserted == 0
    assert retry[0].documents_skipped == 2


def test_search_publication_never_fetches_the_full_state_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance for issue #153: the production publication path reconciles
    through bounded subset lookups and paged stale discovery only."""
    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())
    first = run_project(tmp_path, select="context_search")
    assert first[0].rows_inserted == 2

    def forbid_full_scope(self: object, scope: object) -> dict[str, object]:
        raise AssertionError(
            "search publication must not fetch the full state scope"
        )

    monkeypatch.setattr(DuckDBAdapter, "fetch_state", forbid_full_scope)
    # Re-publication with prior state exercises classification, per-batch
    # advancement, and the stale pass without the eager escape hatch.
    retry = run_project(tmp_path, select="context_search")
    assert retry[0].documents_skipped == 2
    assert retry[0].rows_inserted == 0


def test_unacknowledged_receipt_advances_no_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """State advances only behind exact durable receipts (issue #153): an
    unacknowledged upsert fails the run before any state row is written."""
    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())

    original_upsert = LanceDBStore.upsert

    def unacknowledged(self: Any, *args: Any, **kwargs: Any) -> object:
        receipt = original_upsert(self, *args, **kwargs)
        return replace(receipt, atomic=False)

    monkeypatch.setattr(LanceDBStore, "upsert", unacknowledged)
    with pytest.raises(RunError, match="durable upsert receipt"):
        run_project(tmp_path, select="context_search")

    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert resolved.retrieval is not None
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        scope = StateScope.for_target_descriptor(
            "context_search",
            stage="retrieval_publish",
            descriptor=create_store(
                resolved.retrieval.stores["primary"],
                project_name=project.name,
                target_name=resolved.target_name,
                alias="primary",
                role=StoreRole.PUBLISH,
            ).state_descriptor("context").descriptor(),
        )
        assert adapter.fetch_state(scope) == {}
        assert ServingCoordinator(adapter).status(scope).status == "failed"


def test_invalid_vector_fails_without_content_or_id_in_error(tmp_path: Path) -> None:
    _write_project(tmp_path)
    rows = _rows().with_columns(pl.Series("embedding", [[float("nan"), 0.0], [0.0, 1.0]]))
    _materialize_upstream(tmp_path, rows)

    with pytest.raises(RunError) as raised:
        run_project(tmp_path, select="context_search")
    rendered = str(raised.value)
    assert "finite" in rendered
    assert "c1" not in rendered
    assert "inflation" not in rendered


def test_wrong_vector_dimensions_fail_without_store_mutation(tmp_path: Path) -> None:
    _write_project(tmp_path)
    rows = _rows().with_columns(
        pl.Series("embedding", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )
    _materialize_upstream(tmp_path, rows)

    with pytest.raises(RunError, match="dimensions"):
        run_project(tmp_path, select="context_search")

    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert resolved.retrieval is not None
    store = create_store(
        resolved.retrieval.stores["primary"],
        project_name=project.name,
        target_name=resolved.target_name,
        alias="primary",
        role=StoreRole.PUBLISH,
    )
    with store:
        assert store.inspect_collection("retrieval_demo__dev__context") is None


@pytest.mark.parametrize("record_id", ["bad\x00id", "x" * 8193])
def test_invalid_record_id_fails_before_collection_creation(
    tmp_path: Path, record_id: str
) -> None:
    _write_project(tmp_path)
    _materialize_upstream(
        tmp_path,
        _rows().with_columns(
            pl.when(pl.col("chunk_id") == "c1")
            .then(pl.lit(record_id))
            .otherwise(pl.col("chunk_id"))
            .alias("chunk_id")
        ),
    )

    with pytest.raises(RunError) as raised:
        run_project(tmp_path, select="context_search")
    assert "byte limit" in str(raised.value)

    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert resolved.retrieval is not None
    store = create_store(
        resolved.retrieval.stores["primary"],
        project_name=project.name,
        target_name=resolved.target_name,
        alias="primary",
        role=StoreRole.PUBLISH,
    )
    with store:
        assert store.inspect_collection("retrieval_demo__dev__context") is None


def test_search_attribute_schema_mismatch_fails_before_store_io(tmp_path: Path) -> None:
    _write_project(tmp_path)
    model_path = tmp_path / "models" / "retrieval.yml"
    model_path.write_text(
        model_path.read_text().replace("data_type: string", "data_type: integer")
    )
    _materialize_upstream(tmp_path, _rows())

    with pytest.raises(RunError, match="warehouse type"):
        run_project(tmp_path, select="context_search")
    assert not (tmp_path / "target" / "lancedb").exists()


def test_nonfinite_search_attribute_fails_safely(tmp_path: Path) -> None:
    _write_project(tmp_path)
    model_path = tmp_path / "models" / "retrieval.yml"
    model_path.write_text(
        model_path.read_text().replace("data_type: string", "data_type: float")
    )
    rows = _rows().with_columns(
        pl.Series("category", [float("inf"), 1.0], dtype=pl.Float64)
    )
    _materialize_upstream(tmp_path, rows)

    with pytest.raises(RunError) as raised:
        run_project(tmp_path, select="context_search")
    assert "must be finite" in str(raised.value)
    assert "Infinity" not in str(raised.value)


def test_empty_input_creates_typed_empty_collection(tmp_path: Path) -> None:
    _write_project(tmp_path)
    empty = pl.DataFrame(
        {
            "chunk_id": pl.Series([], dtype=pl.String),
            "document_id": pl.Series([], dtype=pl.String),
            "text": pl.Series([], dtype=pl.String),
            "embedding": pl.Series([], dtype=pl.List(pl.Float64)),
            "category": pl.Series([], dtype=pl.String),
            "title": pl.Series([], dtype=pl.String),
        }
    )
    _materialize_upstream(tmp_path, empty)

    result = run_project(tmp_path, select="context_search")
    assert result[0].rows_written == 0
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert resolved.retrieval is not None
    store = create_store(
        resolved.retrieval.stores["primary"],
        project_name=project.name,
        target_name=resolved.target_name,
        alias="primary",
        role=StoreRole.PUBLISH,
    )
    with store:
        metadata = store.inspect_collection("retrieval_demo__dev__context")
        assert metadata is not None
        assert metadata.row_count == 0
        assert metadata.schema.field("embedding").type.list_size == 2


def _store_for(project_dir: Path) -> Any:
    project, _, _ = load_project(project_dir)
    resolved = resolve_profile(project, project_dir)
    assert resolved.retrieval is not None
    return create_store(
        resolved.retrieval.stores["primary"],
        project_name=project.name,
        target_name=resolved.target_name,
        alias="primary",
        role=StoreRole.PUBLISH,
    )


def test_full_refresh_rebuilds_into_a_private_generation_and_activates(
    tmp_path: Path,
) -> None:
    """The whole point of #355: replace a live index without emptying it.

    After a full refresh the logical name must resolve to a *different*
    physical collection than before, the ledger must point at it, and the
    superseded one must be gone.
    """
    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())
    run_project(tmp_path, select="context_search")

    scope, resolved = resolve_serving_scope(
        tmp_path, profiles_dir=None, target=None, model_name="context_search"
    )
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        before = ServingCoordinator(adapter).status(scope)
    # An in-place publish leaves the pointer null: the unsuffixed default.
    assert before.active_collection is None

    run_project(tmp_path, select="context_search", full_refresh=True)

    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        after = ServingCoordinator(adapter).status(scope)
    assert after.status == "ready"
    assert after.active_collection is not None
    assert "__g" in after.active_collection

    store = _store_for(tmp_path)
    with store:
        collections = store.list_collections()
        # Exactly one generation survives; the sweep took the rest.
        assert sum("__g" in name for name in collections) == 1
        assert after.active_collection in collections
        # The new generation actually holds the data — an activation that
        # pointed at an empty collection would satisfy every check above.
        rebuilt = store.inspect_collection(after.active_collection)
    assert rebuilt is not None
    assert rebuilt.row_count == len(_rows())


def test_index_config_change_leaves_existing_collection_untouched(tmp_path: Path) -> None:
    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())
    run_project(tmp_path, select="context_search")
    model_path = tmp_path / "models" / "retrieval.yml"
    model_path.write_text(model_path.read_text().replace("metric: cosine", "metric: dot"))

    with pytest.raises(RunError, match="configuration changed"):
        run_project(tmp_path, select="context_search")

    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert resolved.retrieval is not None
    store = create_store(
        resolved.retrieval.stores["primary"],
        project_name=project.name,
        target_name=resolved.target_name,
        alias="primary",
        role=StoreRole.PUBLISH,
    )
    with store:
        metadata = store.inspect_collection("retrieval_demo__dev__context")
        assert metadata is not None
        assert metadata.row_count == 2


def test_tuning_batch_size_does_not_invalidate_the_published_index(
    tmp_path: Path,
) -> None:
    """Issue #344's reported bug, end to end. `batch_size` changes how many
    rows a publish sends per call and never what a row contains, but it used
    to sit inside the collection fingerprint — so tuning publish pacing
    demanded a blue/green rebuild and a full re-embed of the corpus."""
    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())
    run_project(tmp_path, select="context_search")

    model_path = tmp_path / "models" / "retrieval.yml"
    model_path.write_text(
        model_path.read_text().replace("batch_size: 2", "batch_size: 1")
    )

    # Must publish, not raise: the index still describes the same rows.
    run_project(tmp_path, select="context_search")

    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert resolved.retrieval is not None
    store = create_store(
        resolved.retrieval.stores["primary"],
        project_name=project.name,
        target_name=resolved.target_name,
        alias="primary",
        role=StoreRole.PUBLISH,
    )
    with store:
        metadata = store.inspect_collection("retrieval_demo__dev__context")
        assert metadata is not None
        assert metadata.row_count == 2


def test_a_rebuild_forcing_change_names_the_field_that_forced_it(
    tmp_path: Path,
) -> None:
    """The old failure said only that the configuration had changed, which left
    the operator to diff the YAML themselves and gave no signal whether the
    change was additive or genuinely invalidating."""
    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())
    run_project(tmp_path, select="context_search")
    model_path = tmp_path / "models" / "retrieval.yml"
    model_path.write_text(
        model_path.read_text().replace("metric: cosine", "metric: dot")
    )

    with pytest.raises(RunError, match="requires a rebuild: vector"):
        run_project(tmp_path, select="context_search")


def test_search_collection_collisions_fail_before_store_io(tmp_path: Path) -> None:
    _write_project(tmp_path)
    model_path = tmp_path / "models" / "retrieval.yml"
    duplicate = "\n".join(
        [
            "  - name: duplicate_search",
            "    depends_on: [ref('embedding_rows')]",
            "    materialization: incremental",
            "    search:",
            "      access: public",
            "      store: primary",
            "      collection: context",
            "      id_field: chunk_id",
            "      text_fields: [text]",
            "      full_text:",
            "        fields: [text]",
            "      query:",
            "        modes: [text]",
        ]
    )
    model_path.write_text(model_path.read_text() + "\n" + duplicate + "\n")
    project, sources, models = load_project(tmp_path)
    validate_project_contract(project, sources, models, tmp_path)
    resolved = resolve_profile(project, tmp_path)

    with pytest.raises(ConfigError, match="same retrieval collection"):
        validate_retrieval_capabilities(models, project, resolved)
    assert not (tmp_path / "target" / "lancedb").exists()


def test_search_collection_collisions_ignore_profile_aliases(tmp_path: Path) -> None:
    _write_project(tmp_path)
    profiles_path = tmp_path / "profiles.yml"
    profiles_path.write_text(
        profiles_path.read_text()
        + "\n"
        + "\n".join(
            [
                "          mirror:",
                "            type: lancedb",
                "            path: target/lancedb",
                "            collection_template: '{project}__{target}__{collection}'",
                "",
            ]
        )
    )
    model_path = tmp_path / "models" / "retrieval.yml"
    duplicate = "\n".join(
        [
            "  - name: duplicate_search",
            "    depends_on: [ref('embedding_rows')]",
            "    materialization: incremental",
            "    search:",
            "      access: public",
            "      store: mirror",
            "      collection: context",
            "      id_field: chunk_id",
            "      text_fields: [text]",
            "      full_text:",
            "        fields: [text]",
            "      query:",
            "        modes: [text]",
        ]
    )
    model_path.write_text(model_path.read_text() + "\n" + duplicate + "\n")
    project, sources, models = load_project(tmp_path)
    validate_project_contract(project, sources, models, tmp_path)
    resolved = resolve_profile(project, tmp_path)

    with pytest.raises(ConfigError, match="same retrieval collection"):
        validate_retrieval_capabilities(models, project, resolved)
    assert not (tmp_path / "target" / "lancedb").exists()


def test_search_cli_lists_resource_and_show_rejects_relation_access(
    tmp_path: Path,
) -> None:
    _write_project(tmp_path)
    runner = CliRunner()

    listed = runner.invoke(
        cli,
        [
            "--project-dir",
            str(tmp_path),
            "ls",
            "--resource-type",
            "search_index",
        ],
    )
    assert listed.exit_code == 0
    assert "context_search" in listed.output
    assert "search_index" in listed.output
    assert "embedding_rows" not in listed.output

    shown = runner.invoke(cli, ["--project-dir", str(tmp_path), "show", "context_search"])
    assert shown.exit_code == 1
    assert "no warehouse relation" in shown.output
    assert "stel search" in shown.output


def test_build_routes_search_without_warehouse_schema_tests(tmp_path: Path) -> None:
    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())

    result = CliRunner().invoke(
        cli,
        ["--project-dir", str(tmp_path), "build", "--select", "context_search"],
    )
    assert result.exit_code == 0, result.output
    assert "context_search" in result.output
    assert "search" in result.output


def test_search_config_fingerprint_is_stable_across_hash_seeds() -> None:
    script = "\n".join(
        [
            "from stel.config import SearchConfig",
            "from stel.retrieval import collection_config_fingerprint",
            "config = SearchConfig.model_validate({",
            "  'id_field': 'chunk_id', 'text_fields': ['text'],",
            "  'vector': {'field': 'embedding', 'dimensions': 2, 'embedding': {",
            "    'provider': 'fixture', 'model': 'deterministic-2d-v1',",
            "    'provider_contract_version': 2, 'provider_implementation': 'tests:v1',",
            "    'semantic_config_fingerprint': 'deterministic-2d-v1',",
            "    'dimensions': 2}},",
            "  'full_text': {'fields': ['text']},",
            "  'query': {'modes': ['vector', 'text', 'filter']},",
            "})",
            "print(collection_config_fingerprint(",
            "  config.model_dump(mode='python'), store_type='lancedb'))",
        ]
    )
    fingerprints: list[str] = []
    for seed in ("1", "7", "101"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        fingerprints.append(
            subprocess.check_output(
                [sys.executable, "-c", script],
                env=environment,
                text=True,
            ).strip()
        )

    assert len(set(fingerprints)) == 1
    assert fingerprints[0] == collection_config_fingerprint(
        {
            "access": "public",
            "store": None,
            "collection": None,
            "id_field": "chunk_id",
            "document_id_field": "document_id",
            "chunk_id_field": None,
            "text_fields": ("text",),
            "return_text_fields": (),
            "vector": {
                "field": "embedding",
                "dimensions": 2,
                "metric": "cosine",
                "search": "exact",
                "embedding": {
                    "provider": "fixture",
                    "model": "deterministic-2d-v1",
                    "provider_contract_version": 2,
                    "provider_implementation": "tests:v1",
                    "semantic_config_fingerprint": "deterministic-2d-v1",
                    "dimensions": 2,
                },
            },
            "full_text": {"fields": ("text",)},
            "attributes": (),
            "display_fields": (),
            "query": {"modes": frozenset({"vector", "text", "filter"}), "consistency": "strong"},
            "on_index_change": "fail",
            "batch_size": 1000,
            "index_options": {},
        },
        store_type="lancedb",
    )


def test_lancedb_query_api_rejects_unowned_collections(tmp_path: Path) -> None:
    _write_project(tmp_path)
    external_path = tmp_path / "target" / "lancedb"
    database = lancedb.connect(external_path)
    database.create_table(
        "external",
        data=pa.table(
            {
                "chunk_id": ["outside"],
                "text": ["sensitive external content"],
                "embedding": [[1.0, 0.0]],
            }
        ),
    )
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert resolved.retrieval is not None
    store = create_store(
        resolved.retrieval.stores["primary"],
        project_name=project.name,
        target_name=resolved.target_name,
        alias="primary",
        role=StoreRole.PUBLISH,
    )

    with store, pytest.raises(RetrievalError) as raised:
        store.text_search(
            "external",
            "sensitive",
            text_field="text",
            limit=1,
        )
    assert "sensitive" not in str(raised.value)
    assert "external content" not in str(raised.value)


def test_every_declared_attribute_type_round_trips_a_filter(tmp_path: Path) -> None:
    """One filtered query per declared `data_type`, against a real store (#337).

    Date filters validated at authoring and exploded at query time, because
    the predicate compiler rendered temporal values as quoted strings — Utf8
    to the query engine, which will not compare them to a date32 or timestamp
    column. Config validation and index build both passed, so the only place
    it surfaced was the querying agent.

    Covers `timestamp` too, which #337's version could not: DuckDB returned
    timestamps in the session timezone, so a UTC value failed the publish-time
    UTC check. Fixed in #339, which pins the session.
    """
    _write_typed_attribute_project(tmp_path)
    _materialize_upstream(tmp_path, _typed_rows())
    run_project(tmp_path, select="context_search")

    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    assert resolved.retrieval is not None
    store = create_store(
        resolved.retrieval.stores["primary"],
        project_name=project.name,
        target_name=resolved.target_name,
        alias="primary",
        role=StoreRole.PUBLISH,
    )

    # (attribute, operator, value, expected chunk_ids) — one per data_type.
    cases: list[tuple[str, RetrievalPredicateOperator, Any, list[str]]] = [
        ("category", RetrievalPredicateOperator.EQUAL, "prices", ["c1"]),
        (
            "filing_date_dt",
            RetrievalPredicateOperator.GREATER_THAN_OR_EQUAL,
            date(2020, 1, 1),
            ["c2"],
        ),
        (
            "filing_date_dt",
            RetrievalPredicateOperator.EQUAL,
            date(2023, 2, 13),
            ["c2"],
        ),
        (
            "observed_at",
            RetrievalPredicateOperator.GREATER_THAN_OR_EQUAL,
            datetime(2020, 1, 1, tzinfo=UTC),
            ["c2"],
        ),
        ("page_count", RetrievalPredicateOperator.LESS_THAN, 15, ["c1"]),
        ("is_amended", RetrievalPredicateOperator.EQUAL, True, ["c2"]),
    ]
    with store:
        metadata = store.inspect_collection("retrieval_demo__dev__context")
        assert metadata is not None
        for field, operator, value, expected in cases:
            result = store.vector_search(
                metadata.physical_name,
                [1.0, 0.0],
                vector_field="embedding",
                limit=10,
                columns=["chunk_id"],
                predicates=[RetrievalPredicate(field, operator, value)],
            )
            assert sorted(result.column("chunk_id").to_pylist()) == expected, (
                f"filter on {field} ({operator}) did not round-trip"
            )


def test_temporal_predicates_execute_against_lancedb(tmp_path: Path) -> None:
    """The compiled predicate must run, not just look right (#337).

    Exercises `_compile_predicates` output against a real LanceDB table with
    real date32/timestamp columns — the layer where the bug lived. A quoted
    string literal is Utf8 to the query engine and is rejected outright, so
    asserting on the SQL text alone would not have caught this.
    """
    from stel.retrieval.lancedb import _compile_predicates

    db = lancedb.connect(str(tmp_path / "raw"))
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("filing_date_dt", pa.date32()),
            ("observed_at", pa.timestamp("us")),
            ("vector", pa.list_(pa.float32(), 2)),
        ]
    )
    table = db.create_table(
        "t",
        pa.table(
            {
                "id": ["a", "b"],
                "filing_date_dt": [date(2019, 1, 1), date(2023, 2, 13)],
                "observed_at": [
                    datetime(2019, 1, 1, 9, 30),
                    datetime(2023, 2, 13, 14, 0),
                ],
                "vector": [[0.1, 0.2], [0.3, 0.4]],
            },
            schema=schema,
        ),
    )

    cases = [
        (
            "filing_date_dt",
            RetrievalPredicateOperator.GREATER_THAN_OR_EQUAL,
            date(2020, 1, 1),
        ),
        ("filing_date_dt", RetrievalPredicateOperator.EQUAL, date(2023, 2, 13)),
        (
            "observed_at",
            RetrievalPredicateOperator.GREATER_THAN_OR_EQUAL,
            datetime(2020, 1, 1),
        ),
        (
            "observed_at",
            RetrievalPredicateOperator.EQUAL,
            datetime(2023, 2, 13, 14, 0, tzinfo=UTC),
        ),
    ]
    for field, operator, value in cases:
        where = _compile_predicates([RetrievalPredicate(field, operator, value)])
        assert where is not None
        rows = (
            table.search([0.1, 0.2])
            .where(where, prefilter=True)
            .limit(10)
            .to_list()
        )
        assert [row["id"] for row in rows] == ["b"], f"{where} matched the wrong rows"


def test_temporal_literals_are_typed_not_quoted_strings() -> None:
    from stel.retrieval.lancedb import _sql_literal

    assert _sql_literal(date(2020, 1, 1)) == "DATE '2020-01-01'"
    assert _sql_literal(datetime(2020, 1, 1, 9, 30)) == (
        "TIMESTAMP '2020-01-01T09:30:00'"
    )
    # `datetime` is a `date` subclass, so ordering matters.
    assert _sql_literal(datetime(2020, 1, 1, tzinfo=UTC)).startswith("TIMESTAMP")
    # A value with a quote still escapes; the type prefix is not an opening.
    assert _sql_literal("o'brien") == "'o''brien'"


def test_duckdb_reads_timestamps_as_stored_not_as_host_local(
    tmp_path: Path,
) -> None:
    """Every DuckDB session stel opens is pinned to UTC (issue #339).

    DuckDB defaults `TimeZone` to the host's local zone and converts
    `TIMESTAMP WITH TIME ZONE` values into it on read, so a genuinely-UTC
    value came back bearing the developer's offset and failed the publish-time
    "search timestamp attributes must be UTC" check. Invisible in CI, whose
    runners are UTC — so this asserts the setting rather than the symptom.
    """
    from stel.adapters import create_adapter, parse_warehouse_config

    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "tz.duckdb"), "schema": "main"}
    )
    stored = datetime(2019, 1, 1, 9, 30, tzinfo=UTC)

    with create_adapter(config) as adapter:
        adapter.materialize_full("moments", pl.DataFrame({"at": [stored]}))

        # The ordinary read path...
        read_back = adapter.read_table("moments").to_dicts()[0]["at"]
        assert read_back.utcoffset() == timedelta(0)
        # ...and the Arrow snapshot path, which uses a cursor. A cursor starts
        # a fresh session rather than inheriting the connection's, so pinning
        # only at connect left this one host-local.
        with adapter.table_snapshot("moments", batch_size=10) as snapshot:
            rows = [row for batch in snapshot for row in batch.to_pylist()]

    # `utcoffset()`, not equality: aware datetimes compare by *instant*, so
    # `4:30-05:00 == 9:30+00:00` is True and an equality assertion cannot see
    # the host-local rendering at all. The publish check tests the offset, so
    # this does too.
    assert [row["at"].utcoffset() for row in rows] == [timedelta(0)]
    assert rows[0]["at"] == stored


def _set_index_change_policy(tmp_path: Path, policy: str) -> None:
    path = tmp_path / "models" / "context_search.yml"
    candidates = [path] if path.exists() else list((tmp_path / "models").glob("*.yml"))
    for candidate in candidates:
        text = candidate.read_text()
        if "on_index_change: fail" in text:
            candidate.write_text(
                text.replace("on_index_change: fail", f"on_index_change: {policy}")
            )
            return
    raise AssertionError("fixture no longer declares on_index_change")


def test_rebuild_policy_now_compiles(tmp_path: Path) -> None:
    """LanceDB advertises private_generation_build, so `rebuild` is a policy it
    can honor — it builds a new generation and activates it (issue #355)."""
    _write_project(tmp_path)
    _set_index_change_policy(tmp_path, "rebuild")
    project, sources, models = load_project(tmp_path)

    validate_project_contract(project, sources, models, tmp_path)


def test_rebuild_policy_replaces_the_index_on_an_incompatible_change(
    tmp_path: Path,
) -> None:
    """The change that `fail` refuses, `rebuild` absorbs — without a window in
    which the collection is empty or half-built."""
    _write_project(tmp_path)
    _set_index_change_policy(tmp_path, "rebuild")
    _materialize_upstream(tmp_path, _rows())
    run_project(tmp_path, select="context_search")

    model_path = tmp_path / "models" / "retrieval.yml"
    model_path.write_text(
        model_path.read_text(encoding="utf-8").replace("metric: cosine", "metric: dot"),
        encoding="utf-8",
    )
    run_project(tmp_path, select="context_search")

    scope, resolved = resolve_serving_scope(
        tmp_path, profiles_dir=None, target=None, model_name="context_search"
    )
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        entry = ServingCoordinator(adapter).status(scope)
    assert entry.status == "ready"
    assert entry.active_collection is not None and "__g" in entry.active_collection


def test_online_policy_compiles_against_a_store_that_builds_private_generations(
    tmp_path: Path,
) -> None:
    """Online updates require an independent generation, not live mutation."""
    _write_project(tmp_path)
    _set_index_change_policy(tmp_path, "online")
    project, sources, models = load_project(tmp_path)
    validate_project_contract(project, sources, models, tmp_path)
    resolved = resolve_profile(project, tmp_path)

    validate_retrieval_capabilities(models, project, resolved)


def test_online_policy_is_refused_when_the_store_cannot_build_private_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject unsupported publication guarantees before store mutation."""
    from stel.retrieval.base import RetrievalFeature
    from stel.retrieval.lancedb import LanceDBStore

    _write_project(tmp_path)
    _set_index_change_policy(tmp_path, "online")
    project, sources, models = load_project(tmp_path)
    validate_project_contract(project, sources, models, tmp_path)
    resolved = resolve_profile(project, tmp_path)

    full = LanceDBStore.capabilities()
    reduced = replace(
        full,
        features=frozenset(
            f for f in full.features if f is not RetrievalFeature.PRIVATE_GENERATION_BUILD
        ),
    )
    monkeypatch.setattr(LanceDBStore, "capabilities", classmethod(lambda cls: reduced))

    with pytest.raises(Exception, match="private_generation_build"):
        validate_retrieval_capabilities(models, project, resolved)


def test_a_first_publish_warns_while_it_is_still_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """On a first publish there is no prior row count, so the advisory has to
    come from the batch loop.

    Taking it only from the post-publish metadata meant a newly created large
    exact collection paid the whole publication first — and got no warning at
    all if the index build failed (Codex review, #461). Rather than publishing
    390k rows, the modelled scan throughput is dropped so that two rows cost
    what millions would on a real store; every threshold in the advisory then
    derives from it exactly as it does in production.
    """
    from stel.retrieval import servability

    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())
    monkeypatch.setattr(servability, "_SCAN_BYTES_PER_SECOND", 0.1)

    with caplog.at_level(logging.WARNING, logger="stel.execution.search"):
        run_project(tmp_path, select="context_search")

    assert "still streaming" in caplog.text
    # Latched, not repeated: one line per run, not one per batch.
    assert caplog.text.count("declares `search: exact`") == 1


def test_a_small_first_publish_says_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The same path at the real threshold. Two rows is not a design problem,
    and a warning on every publish would be the noise that hides the one that
    matters."""
    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())

    with caplog.at_level(logging.WARNING, logger="stel.execution.search"):
        run_project(tmp_path, select="context_search")

    assert "search: exact" not in caplog.text


def test_a_store_config_refusal_stops_the_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Some refusals depend on the resolved store *config*, not the store type,
    so `capabilities()` cannot express them — DuckDB will not build a
    persistent HNSW index without `hnsw_experimental_persistence`. They must
    still stop the compile, because a compatible vector-search change is
    applied to the live collection: discovering the refusal at index time means
    every row has already been republished and the serving pointer cleared
    (Codex review, #461).
    """
    from stel.retrieval.lancedb import LanceDBStore

    _write_project(tmp_path)
    project, sources, models = load_project(tmp_path)
    validate_project_contract(project, sources, models, tmp_path)
    resolved = resolve_profile(project, tmp_path)

    monkeypatch.setattr(
        LanceDBStore,
        "index_config_refusal",
        lambda self, *, vector_search, vector_index: "this store will not build that index",
    )

    with pytest.raises(Exception, match="will not build that index"):
        validate_retrieval_capabilities(models, project, resolved)


def test_a_store_with_nothing_to_refuse_compiles(tmp_path: Path) -> None:
    """The default is permissive: the base implementation returns None, so a
    store with no config-dependent refusals is unaffected by the seam."""
    _write_project(tmp_path)
    project, sources, models = load_project(tmp_path)
    validate_project_contract(project, sources, models, tmp_path)
    resolved = resolve_profile(project, tmp_path)

    validate_retrieval_capabilities(models, project, resolved)


# ─── private generation build (issue #355) ──────────────────────────────────


def _gen_store(tmp_path: Path) -> Any:
    return LanceDBStore(
        LanceDBConfig(type="lancedb", path=str(tmp_path / "lance")),
        project_name="proj",
        target_name="dev",
        alias="default",
        role=StoreRole.PUBLISH,
    )


def test_unsuffixed_physical_name_is_unchanged_by_generations(
    tmp_path: Path,
) -> None:
    """The pre-#355 name must keep addressing already-published collections.

    Every index published before generations existed lives under this exact
    name; changing it would strand the data rather than migrate it.
    """
    store = _gen_store(tmp_path)
    assert store.physical_collection("ctx") == "proj__dev__ctx"
    assert store.physical_collection("ctx", generation=None) == "proj__dev__ctx"


def test_generation_yields_a_distinct_private_collection_name(
    tmp_path: Path,
) -> None:
    store = _gen_store(tmp_path)
    base = store.physical_collection("ctx")
    gen_a = store.physical_collection("ctx", generation="a1b2")
    gen_b = store.physical_collection("ctx", generation="c3d4")

    assert gen_a != base and gen_b != base and gen_a != gen_b
    assert gen_a.startswith(base)


def test_generation_token_charset_is_enforced(tmp_path: Path) -> None:
    store = _gen_store(tmp_path)
    # The token crosses into a physical collection name, so it is restricted
    # rather than escaped.
    for bad in ("has-dash", "UPPER", "with_underscore", "", "x" * 17, "a b"):
        with pytest.raises(RetrievalError, match="generation token"):
            store.physical_collection("ctx", generation=bad)


def test_generation_suffix_cannot_overflow_the_name_limit(
    tmp_path: Path,
) -> None:
    store = _gen_store(tmp_path)
    long_logical = "c" * 120
    # Unsuffixed it already fills the budget; the suffix must be rejected
    # rather than produce a truncated or invalid collection name.
    with pytest.raises(RetrievalError, match="invalid"):
        store.physical_collection(long_logical, generation="abcd1234")


def test_lancedb_advertises_private_generation_build() -> None:
    assert (
        RetrievalFeature.PRIVATE_GENERATION_BUILD
        in LanceDBStore.capabilities().features
    )


def test_drop_collection_removes_only_an_owned_existing_collection(
    tmp_path: Path,
) -> None:
    store = _gen_store(tmp_path)
    with store:
        name = store.physical_collection("ctx", generation="a1b2")
        # Nothing to drop yet: reported, not raised, so a retirement sweep is
        # safe to run against an already-reclaimed generation.
        assert store.drop_collection(name) is False

        store.create_collection(
            CollectionSpec(
                logical_name="ctx",
                physical_name=name,
                id_field="id",
                text_fields=("body",),
                full_text_fields=(),
                attribute_fields=(),
                scalar_index_fields=(),
                display_fields=("body",),
                vector_field=None,
                vector_dimensions=None,
                distance_metric=None,
                vector_search="exact",
                vector_index=None,
                config_fingerprint="cfg1",
                descriptor="{}",
                legacy_config_fingerprint="legacy1",
                arrow_schema=pa.schema(
                    [pa.field("id", pa.string()), pa.field("body", pa.string())]
                ),
            )
        )
        assert store.inspect_collection(name) is not None
        assert store.drop_collection(name) is True
        assert store.inspect_collection(name) is None


def test_drop_collection_refuses_a_collection_stel_does_not_own(
    tmp_path: Path,
) -> None:
    """A mistyped or externally managed name must not be destroyed here."""
    lancedb = pytest.importorskip("lancedb")
    import pyarrow as pa

    store = _gen_store(tmp_path)
    with store:
        foreign = lancedb.connect(str(tmp_path / "lance"))
        foreign.create_table(
            "proj__dev__foreign",
            schema=pa.schema([pa.field("id", pa.string())]),
        )
        with pytest.raises(RetrievalError, match="not owned by stel"):
            store.drop_collection("proj__dev__foreign")
        assert "proj__dev__foreign" in foreign.list_tables().tables


# ─── generation retirement (issue #355) ─────────────────────────────────────


def _make_collection(store: Any, name: str) -> None:
    store.create_collection(
        CollectionSpec(
            logical_name="ctx",
            physical_name=name,
            id_field="id",
            text_fields=("body",),
            full_text_fields=(),
            attribute_fields=(),
            scalar_index_fields=(),
            display_fields=("body",),
            vector_field=None,
            vector_dimensions=None,
            distance_metric=None,
            vector_search="exact",
            vector_index=None,
            config_fingerprint="cfg1",
            descriptor="{}",
            legacy_config_fingerprint="legacy1",
            arrow_schema=pa.schema(
                [pa.field("id", pa.string()), pa.field("body", pa.string())]
            ),
        )
    )


@contextmanager
def _held_publish_lease(tmp_path: Path) -> Iterator[tuple[Any, Any, Any]]:
    """A coordinator with the publish lease held — the state a sweep requires."""
    config = parse_warehouse_config(
        {
            "type": "duckdb",
            "path": str(tmp_path / "serving.duckdb"),
            "schema": "serving",
        }
    )
    with create_adapter(config) as adapter:
        coordinator = ServingCoordinator(adapter)
        scope = StateScope.for_target_descriptor(
            "ctx_search",
            stage="retrieval_publish",
            descriptor={"store_type": "lancedb", "collection": "ctx"},
        )
        lease = coordinator.acquire_publish(
            scope, expected_code_version="v1", config_fingerprint="cfg1"
        )
        yield coordinator, lease, scope


def test_retirement_never_considers_the_unsuffixed_base_collection(
    tmp_path: Path,
) -> None:
    """The base collection is where an in-place published index lives.

    It has no generation marker, so it must never appear as a candidate —
    dropping it would destroy a working index rather than reclaim garbage.
    """
    store = _gen_store(tmp_path)
    with store:
        base = store.physical_collection("ctx")
        _make_collection(store, base)
        _make_collection(store, store.physical_collection("ctx", generation="a1b2"))

        candidates = superseded_generations(
            store, logical_collection="ctx", active_collection=None
        )
        assert base not in candidates
        assert candidates == [store.physical_collection("ctx", generation="a1b2")]


def test_retirement_spares_the_active_generation(tmp_path: Path) -> None:
    store = _gen_store(tmp_path)
    with store, _held_publish_lease(tmp_path) as (coordinator, lease, _scope):
        active = store.physical_collection("ctx", generation="a1b2")
        superseded = store.physical_collection("ctx", generation="c3d4")
        base = store.physical_collection("ctx")
        for name in (base, active, superseded):
            _make_collection(store, name)

        retired = retire_superseded_generations(
            store,
            logical_collection="ctx",
            active_collection=active,
            coordinator=coordinator,
            lease=lease,
        )

        assert retired == [superseded]
        remaining = store.list_collections()
        assert active in remaining and base in remaining
        assert superseded not in remaining


def test_retirement_is_idempotent(tmp_path: Path) -> None:
    store = _gen_store(tmp_path)
    with store, _held_publish_lease(tmp_path) as (coordinator, lease, _scope):
        active = store.physical_collection("ctx", generation="a1b2")
        _make_collection(store, active)
        _make_collection(store, store.physical_collection("ctx", generation="c3d4"))

        first = retire_superseded_generations(
            store,
            logical_collection="ctx",
            active_collection=active,
            coordinator=coordinator,
            lease=lease,
        )
        second = retire_superseded_generations(
            store,
            logical_collection="ctx",
            active_collection=active,
            coordinator=coordinator,
            lease=lease,
        )
        assert len(first) == 1
        assert second == []


def test_retirement_does_not_reach_another_logical_collection(
    tmp_path: Path,
) -> None:
    """The prefix is per logical collection, so a sweep stays in its lane."""
    store = _gen_store(tmp_path)
    with store, _held_publish_lease(tmp_path) as (coordinator, lease, _scope):
        mine = store.physical_collection("ctx", generation="a1b2")
        theirs = store.physical_collection("other", generation="a1b2")
        _make_collection(store, mine)
        _make_collection(store, theirs)

        retired = retire_superseded_generations(
            store,
            logical_collection="ctx",
            active_collection=None,
            coordinator=coordinator,
            lease=lease,
        )
        assert retired == [mine]
        assert theirs in store.list_collections()


def test_generation_shaped_logical_names_are_rejected(tmp_path: Path) -> None:
    """The `__g<token>` suffix shape is reserved for generation collections.

    A logical collection resolving to `proj__dev__ctx__garchive` would be
    indistinguishable from a retired generation of its sibling `ctx` and
    swept with it, so name resolution refuses the shape outright.
    """
    store = _gen_store(tmp_path)
    with pytest.raises(RetrievalError, match="reserved generation suffix"):
        store.physical_collection("ctx__garchive")
    with pytest.raises(RetrievalError, match="reserved generation suffix"):
        store.physical_collection("ctx__garchive", generation="a1b2")
    # The marker alone, or a marker not in suffix position, stays allowed.
    assert store.physical_collection("ctx__g") == "proj__dev__ctx__g"
    assert store.physical_collection("ctx__gv2__x") == "proj__dev__ctx__gv2__x"


def test_retirement_matches_complete_generation_names_only(
    tmp_path: Path,
) -> None:
    """A collection sharing the prefix without the exact token shape survives.

    Only `<base>__g<token>` with one valid 1-16 lowercase-alphanumeric token
    is a generation; a longer or malformed remainder is somebody else's
    collection, never a sweep candidate.
    """
    store = _gen_store(tmp_path)
    with store:
        generation = store.physical_collection("ctx", generation="a1b2")
        _make_collection(store, generation)
        # Pre-existing or externally created names that merely share the
        # prefix: an invalid token (underscore) and an over-long token.
        for bystander in (
            "proj__dev__ctx__gv2__extra",
            "proj__dev__ctx__g" + "x" * 17,
        ):
            _make_collection(store, bystander)

        candidates = superseded_generations(
            store, logical_collection="ctx", active_collection=None
        )
        assert candidates == [generation]


def test_retirement_aborts_when_the_publish_lease_is_stale(
    tmp_path: Path,
) -> None:
    """A sweep whose lease was reassigned must not delete anything.

    After `recover` clears the claim, another publisher may already be
    building a new private generation; the fence check makes the stale
    sweeper abort before any drop instead of deleting that build.
    """
    store = _gen_store(tmp_path)
    with store, _held_publish_lease(tmp_path) as (coordinator, lease, scope):
        superseded = store.physical_collection("ctx", generation="c3d4")
        _make_collection(store, superseded)
        coordinator.recover(scope, owner_terminated=True)

        with pytest.raises(StaleServingLeaseError):
            retire_superseded_generations(
                store,
                logical_collection="ctx",
                active_collection=None,
                coordinator=coordinator,
                lease=lease,
            )
        assert superseded in store.list_collections()


def test_a_rebuild_that_fails_after_the_state_swap_clears_the_stale_state(
    tmp_path: Path,
) -> None:
    """Activation swaps state before marking ready — the fence forces that
    order. If the swap lands and activation does not, the serving scope
    describes a collection the pointer does not name, and a later incremental
    publish would skip rows the old collection never received.

    The failure path clears that state, so the next run republishes in full.
    """
    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())
    run_project(tmp_path, select="context_search")

    scope, resolved = resolve_serving_scope(
        tmp_path, profiles_dir=None, target=None, model_name="context_search"
    )
    real_mark_ready = ServingCoordinator.mark_ready

    def _fail_activation(self: Any, *args: Any, **kwargs: Any) -> None:
        raise RunError("activation interrupted")

    ServingCoordinator.mark_ready = _fail_activation  # type: ignore[method-assign]
    try:
        with pytest.raises(RunError, match="activation interrupted"):
            run_project(tmp_path, select="context_search", full_refresh=True)
    finally:
        ServingCoordinator.mark_ready = real_mark_ready  # type: ignore[method-assign]

    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        # State that no longer describes the served collection is gone, so the
        # next publish cannot skip rows on the strength of it.
        assert adapter.fetch_state(scope) == {}

    results = run_project(tmp_path, select="context_search")
    published = next(r for r in results if r.model_name == "context_search")
    # Every row republished rather than skipped as already-published.
    assert published.rows_written == len(_rows())


def test_a_failure_after_activation_keeps_the_activated_state(
    tmp_path: Path,
) -> None:
    """Once activation succeeds the swapped state describes the live
    generation. Clearing it because a later step failed would leave the ledger
    ready with empty state, and re-embed the whole index on the next run.
    """
    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())
    run_project(tmp_path, select="context_search")
    # A first rebuild has no predecessor to retire, so take two: the second
    # supersedes the first and does reach the post-activation drop.
    run_project(tmp_path, select="context_search", full_refresh=True)

    scope, resolved = resolve_serving_scope(
        tmp_path, profiles_dir=None, target=None, model_name="context_search"
    )
    real_drop = LanceDBStore.drop_collection

    def _fail_after_activation(self: Any, name: str) -> bool:
        raise RetrievalError("retirement failed")

    LanceDBStore.drop_collection = _fail_after_activation  # type: ignore[method-assign]
    try:
        with pytest.raises(RunError, match="retirement failed"):
            run_project(tmp_path, select="context_search", full_refresh=True)
    finally:
        LanceDBStore.drop_collection = real_drop  # type: ignore[method-assign]

    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        # The activated generation's state survives the failed cleanup.
        assert adapter.fetch_state(scope) != {}


def test_a_subset_invocation_defers_stale_reconciliation(tmp_path: Path) -> None:
    """Issue #417: under a subset run, an id absent from this run's view is
    not stale. A partitioned orchestration upstream must not have the search
    index deleting every other partition's records — reconciliation belongs
    to the next unfiltered run, which must still perform it."""
    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())
    first = run_project(tmp_path, select="context_search")
    assert first[0].rows_inserted == 2

    # The upstream shrinks by one record, and the invocation is a subset run
    # (read_filter present). The vanished record must survive.
    _materialize_upstream(tmp_path, _rows(version=2))
    [filtered] = run_project(
        tmp_path,
        select="context_search",
        read_filter=[("category", "eq", "macro")],
    )
    assert filtered.documents_deleted == 0

    [unfiltered] = run_project(tmp_path, select="context_search")
    assert unfiltered.documents_deleted == 1


# ─── vector index reconciliation (issue #461) ───────────────────────────────


def _vector_spec(
    name: str, *, vector_search: str, vector_index: str | None = None
) -> CollectionSpec:
    if vector_index is None and vector_search == "approximate":
        vector_index = "ivf_hnsw_flat"
    return CollectionSpec(
        logical_name="ctx",
        physical_name=name,
        id_field="id",
        text_fields=("body",),
        full_text_fields=(),
        attribute_fields=(),
        scalar_index_fields=(),
        display_fields=("body",),
        vector_field="embedding",
        vector_dimensions=3,
        distance_metric="cosine",
        vector_search=vector_search,
        vector_index=vector_index,
        config_fingerprint=f"cfg-{vector_search}-{vector_index}",
        descriptor=json.dumps({"vector_search": vector_search, "vector_index": vector_index}),
        legacy_config_fingerprint="legacy",
        arrow_schema=pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("body", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), 3)),
            ]
        ),
    )


def _vector_rows() -> list[IndexedRow]:
    return [
        IndexedRow(
            str(index),
            {
                "id": str(index),
                "body": f"row {index}",
                "embedding": [float(index), 1.0, 0.0],
            },
            f"fp-{index}",
        )
        # HNSW needs more than a handful of rows before LanceDB will build it.
        for index in range(256)
    ]


def _vector_index_types(store: Any, name: str) -> list[str]:
    table = store._open_owned_table(name)
    return [
        index.index_type
        for index in table.list_indices()
        if index.columns == ["embedding"]
    ]


# ─── the merge key is indexed (issue #475) ─────────────────────────────────


def _merge_key_spec(name: str) -> CollectionSpec:
    """A spec whose scalar indexes name the merge key, as #475 makes them."""
    return CollectionSpec(
        logical_name="ctx",
        physical_name=name,
        id_field="id",
        text_fields=("body",),
        full_text_fields=(),
        attribute_fields=(),
        scalar_index_fields=("id",),
        display_fields=("body",),
        vector_field=None,
        vector_dimensions=None,
        distance_metric=None,
        vector_search=None,
        vector_index=None,
        config_fingerprint="cfg",
        descriptor="{}",
        legacy_config_fingerprint="legacy",
        arrow_schema=pa.schema(
            [pa.field("id", pa.string()), pa.field("body", pa.string())]
        ),
    )


def _rows_for(ids: range | list[int], *, body: str) -> list[IndexedRow]:
    return [
        IndexedRow(str(i), {"id": str(i), "body": f"{body} {i}"}, f"fp-{i}")
        for i in ids
    ]


def _body_of(store: Any, name: str, record_id: str) -> str | None:
    table = store._open_owned_table(name)
    rows = table.search().where(f"id = '{record_id}'").limit(1).to_list()
    return rows[0]["body"] if rows else None


def test_the_merge_key_gets_a_scalar_index(tmp_path: Path) -> None:
    """Unindexed, `merge_insert`'s join-key predicate is a full column scan and
    the ack `count_rows` scans it again — two O(table) passes per page on a
    daily incremental, not just a reindex (issue #475)."""
    store = _gen_store(tmp_path)
    with store:
        name = store.physical_collection("ctx")
        store.create_collection(_merge_key_spec(name))
        store.upsert(name, _rows_for(range(64), body="v1"), id_field="id", mutation_digest="d1")

        store.ensure_indexes(_merge_key_spec(name))

        table = store._open_owned_table(name)
        assert [
            index.index_type
            for index in table.list_indices()
            if index.columns == ["id"]
        ] == ["BTree"]


def test_merge_updates_still_apply_after_the_id_index_is_rebuilt(
    tmp_path: Path,
) -> None:
    """Pins lancedb#3177 on the pinned LanceDB version.

    On lancedb 0.30 / lance 3.0, a BTree on the merge key plus an index
    rebuild made later `when_matched_update_all` calls silently stop updating
    rows — no error, just stale data. `ensure_indexes` rebuilds with
    `create_index(replace=True)` on every publish that added rows, which is
    every incremental publish, so #475 walks straight into that sequence.

    It is correct on the pinned version. This test is what keeps a LanceDB
    bump from reintroducing it silently, so it drives the exact order:
    index the id, merge-update, append, rebuild, merge-update, verify.
    """
    store = _gen_store(tmp_path)
    with store:
        name = store.physical_collection("ctx")
        store.create_collection(_merge_key_spec(name))
        store.upsert(name, _rows_for(range(64), body="v1"), id_field="id", mutation_digest="d1")
        store.ensure_indexes(_merge_key_spec(name))

        # Update through the index, before any rebuild.
        store.upsert(name, _rows_for([7], body="v2"), id_field="id", mutation_digest="d2")
        assert _body_of(store, name, "7") == "v2 7"

        # Append leaves unindexed rows behind, which is what makes the next
        # ensure_indexes rebuild rather than skip.
        store.upsert(
            name, _rows_for(range(64, 128), body="v1"), id_field="id", mutation_digest="d3"
        )
        store.ensure_indexes(_merge_key_spec(name))

        # The call that silently no-opped under the bug.
        store.upsert(name, _rows_for([7, 70], body="v3"), id_field="id", mutation_digest="d4")

        assert _body_of(store, name, "7") == "v3 7"
        assert _body_of(store, name, "70") == "v3 70"
        # Updates, not duplicate inserts: the merge key must still match.
        assert store._open_owned_table(name).count_rows() == 128


def test_lancedb_builds_the_ann_index_only_when_approximate(tmp_path: Path) -> None:
    """`exact` is implemented by the absence of an index, which is exactly why
    a 3.6M-row exact collection scanned ~11GB per query (issue #461)."""
    store = _gen_store(tmp_path)
    with store:
        name = store.physical_collection("ctx")
        store.create_collection(_vector_spec(name, vector_search="exact"))
        store.upsert(name, _vector_rows(), id_field="id", mutation_digest="d1")

        store.ensure_indexes(_vector_spec(name, vector_search="exact"))
        assert _vector_index_types(store, name) == []

        store.ensure_indexes(_vector_spec(name, vector_search="approximate"))
        assert any("Hnsw" in kind for kind in _vector_index_types(store, name))


def test_lancedb_switching_back_to_exact_takes_the_index_away(
    tmp_path: Path,
) -> None:
    """LanceDB uses an ANN index whenever one exists, so a stale index would
    keep serving approximate results under a config promising exact ones."""
    store = _gen_store(tmp_path)
    with store:
        name = store.physical_collection("ctx")
        store.create_collection(_vector_spec(name, vector_search="exact"))
        store.upsert(name, _vector_rows(), id_field="id", mutation_digest="d1")
        store.ensure_indexes(_vector_spec(name, vector_search="approximate"))
        assert any("Hnsw" in kind for kind in _vector_index_types(store, name))

        store.ensure_indexes(_vector_spec(name, vector_search="exact"))

        assert _vector_index_types(store, name) == []


@pytest.mark.parametrize(
    "vector_index,reported",
    [("ivf_hnsw_flat", "IvfHnswFlat"), ("ivf_hnsw_sq", "IvfHnswSq"), ("ivf_pq", "IvfPq")],
)
def test_lancedb_builds_the_declared_index_type(
    tmp_path: Path, vector_index: str, reported: str
) -> None:
    """Issue #476: `HnswFlat` was hardcoded, and its build peaks at ~3x the
    vectors' bytes — more than a 20 GiB publisher has for an 11 GB corpus.
    The type is now declared, and what LanceDB reports back is what was asked
    for."""
    store = _gen_store(tmp_path)
    with store:
        name = store.physical_collection("ctx")
        store.create_collection(_vector_spec(name, vector_search="exact"))
        store.upsert(name, _vector_rows(), id_field="id", mutation_digest="d1")

        store.ensure_indexes(
            _vector_spec(name, vector_search="approximate", vector_index=vector_index)
        )

        assert _vector_index_types(store, name) == [reported]


def test_lancedb_switching_index_type_replaces_the_structure(tmp_path: Path) -> None:
    """A type switch must rebuild, not leave the old structure answering under
    the new declaration: `ensure_indexes` compares the reported type, not just
    whether *an* index exists."""
    store = _gen_store(tmp_path)
    with store:
        name = store.physical_collection("ctx")
        store.create_collection(_vector_spec(name, vector_search="exact"))
        store.upsert(name, _vector_rows(), id_field="id", mutation_digest="d1")
        store.ensure_indexes(
            _vector_spec(name, vector_search="approximate", vector_index="ivf_hnsw_flat")
        )
        assert _vector_index_types(store, name) == ["IvfHnswFlat"]

        store.ensure_indexes(
            _vector_spec(name, vector_search="approximate", vector_index="ivf_pq")
        )
        assert _vector_index_types(store, name) == ["IvfPq"]

        store.ensure_indexes(_vector_spec(name, vector_search="exact"))
        assert _vector_index_types(store, name) == []


def test_lancedb_refuses_approximate_without_a_declared_type(tmp_path: Path) -> None:
    """A spec built from a validated config always names a type; reaching the
    store without one is a caller bug and must not silently pick a default."""
    from stel.retrieval import RetrievalError

    store = _gen_store(tmp_path)
    with store:
        name = store.physical_collection("ctx")
        store.create_collection(_vector_spec(name, vector_search="exact"))
        store.upsert(name, _vector_rows(), id_field="id", mutation_digest="d1")
        spec = _vector_spec(name, vector_search="approximate")
        undeclared = replace(spec, vector_index=None)

        with pytest.raises(RetrievalError, match="lancedb_index_type_missing"):
            store.ensure_indexes(undeclared)


def test_compiler_points_a_type_refusal_at_the_index_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store that refuses the *type* should name `index`, not `search`: the
    operator chose the type deliberately, and that is the line to change."""
    from stel.retrieval.lancedb import LanceDBStore

    _write_project(tmp_path)
    model_path = tmp_path / "models" / "retrieval.yml"
    model_path.write_text(
        model_path.read_text(encoding="utf-8").replace(
            "        search: exact\n", "        search: approximate\n        index: ivf_pq\n"
        ),
        encoding="utf-8",
    )
    project, sources, models = load_project(tmp_path)
    validate_project_contract(project, sources, models, tmp_path)
    resolved = resolve_profile(project, tmp_path)
    monkeypatch.setattr(
        LanceDBStore,
        "index_config_refusal",
        lambda self, *, vector_search, vector_index: f"no {vector_index} here",
    )

    with pytest.raises(Exception, match=r"no ivf_pq here") as error:
        validate_retrieval_capabilities(models, project, resolved)
    assert "index" in str(error.value)


def test_lancedb_names_the_pq_training_floor_instead_of_a_generic_failure(
    tmp_path: Path,
) -> None:
    """LanceDB refuses `ivf_pq` below 256 rows with a message stel's error
    boundary sanitizes away; the operator would see only `lancedb_index_failed`
    after a full publish. Checked before the build, with the fix in the text."""
    from stel.retrieval import RetrievalError

    store = _gen_store(tmp_path)
    with store:
        name = store.physical_collection("ctx")
        store.create_collection(_vector_spec(name, vector_search="exact"))
        store.upsert(name, _vector_rows()[:8], id_field="id", mutation_digest="d1")

        with pytest.raises(RetrievalError, match="lancedb_pq_corpus_too_small") as error:
            store.ensure_indexes(
                _vector_spec(name, vector_search="approximate", vector_index="ivf_pq")
            )
    assert "ivf_hnsw_sq" in str(error.value)
    with store:
        assert _vector_index_types(store, name) == []


def test_a_pq_collection_that_shrank_below_the_floor_keeps_publishing(tmp_path: Path) -> None:
    """The floor guards a *build*, not a collection: an index trained on 256
    rows stays valid after deletions, and a publish that needs no rebuild must
    not be refused for a training it will not do."""
    store = _gen_store(tmp_path)
    with store:
        name = store.physical_collection("ctx")
        rows = _vector_rows()
        store.create_collection(_vector_spec(name, vector_search="exact"))
        store.upsert(name, rows, id_field="id", mutation_digest="d1")
        pq = _vector_spec(name, vector_search="approximate", vector_index="ivf_pq")
        store.ensure_indexes(pq)
        assert _vector_index_types(store, name) == ["IvfPq"]

        store.delete(
            name, [row.record_id for row in rows[:200]], id_field="id", mutation_digest="d2"
        )
        assert store.inspect_collection(name).row_count == 56

        store.ensure_indexes(pq)  # nothing new to index, nothing to train

        assert _vector_index_types(store, name) == ["IvfPq"]


def test_the_serving_descriptor_versions_the_index_field(tmp_path: Path) -> None:
    """A new key in a documented artifact is a schema change (AGENTS.md), even
    an additive one: a consumer pinning version 1 should fail loudly rather
    than read the wider shape as the narrower. The default is never written,
    so the only difference between the versions is a declared `index`."""
    _write_project(tmp_path)
    resource = next(
        model for model in build_manifest(tmp_path)["models"] if model["name"] == "context_search"
    )["output"]["serving_resource"]
    assert resource["schema_version"] == 2
    assert "index" not in resource["vector"]

    model_path = tmp_path / "models" / "retrieval.yml"
    model_path.write_text(
        model_path.read_text(encoding="utf-8").replace(
            "        search: exact\n", "        search: approximate\n        index: ivf_pq\n"
        ),
        encoding="utf-8",
    )
    resource = next(
        model for model in build_manifest(tmp_path)["models"] if model["name"] == "context_search"
    )["output"]["serving_resource"]
    assert resource["schema_version"] == 2
    assert resource["vector"]["index"] == "ivf_pq"
