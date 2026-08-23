from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import lancedb
import polars as pl
import pyarrow as pa
import pytest
from click.testing import CliRunner

from stel.adapters import AdapterError, StateScope, TableReadSnapshot, create_adapter
from stel.adapters.duckdb import DuckDBAdapter
from stel.cli import cli
from stel.compiler import (
    validate_project_contract,
    validate_retrieval_capabilities,
)
from stel.config import ConfigError, load_project
from stel.dbt_export import build_dbt_sources
from stel.manifest import build_manifest
from stel.profile import resolve_profile
from stel.retrieval import (
    LanceDBStore,
    RetrievalError,
    RetrievalPredicate,
    RetrievalPredicateOperator,
    ServingCoordinator,
    collection_config_fingerprint,
    create_store,
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
    )
    with store:
        metadata = store.inspect_collection("retrieval_demo__dev__context")
        assert metadata is not None
        assert metadata.row_count == 0
        assert metadata.schema.field("embedding").type.list_size == 2


def test_search_full_refresh_fails_closed(tmp_path: Path) -> None:
    _write_project(tmp_path)
    _materialize_upstream(tmp_path, _rows())

    with pytest.raises(RunError, match="atomic store activation"):
        run_project(tmp_path, select="context_search", full_refresh=True)
    assert not (tmp_path / "target" / "lancedb").exists()


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


def test_rebuild_policy_is_refused_while_no_store_can_activate_atomically(
    tmp_path: Path,
) -> None:
    """`rebuild` needs atomic generation activation, which no store advertises.
    Accepting it would compile a policy that cannot be honored (issue #344)."""
    _write_project(tmp_path)
    _set_index_change_policy(tmp_path, "rebuild")
    project, sources, models = load_project(tmp_path)

    with pytest.raises(ConfigError, match="atomic generation activation"):
        validate_project_contract(project, sources, models, tmp_path)


def test_online_policy_compiles_against_a_store_that_advertises_evolution(
    tmp_path: Path,
) -> None:
    """LanceDB can widen a live collection in place, so `online` is a policy it
    can actually honor — unlike `rebuild`."""
    _write_project(tmp_path)
    _set_index_change_policy(tmp_path, "online")
    project, sources, models = load_project(tmp_path)
    validate_project_contract(project, sources, models, tmp_path)
    resolved = resolve_profile(project, tmp_path)

    validate_retrieval_capabilities(models, project, resolved)


def test_online_policy_is_refused_when_the_store_cannot_evolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The capability is the gate, not the mode name. A store that cannot widen
    in place must not be handed an `online` policy to honor."""
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
            f for f in full.features if f is not RetrievalFeature.ONLINE_SCHEMA_EVOLUTION
        ),
    )
    monkeypatch.setattr(LanceDBStore, "capabilities", classmethod(lambda cls: reduced))

    with pytest.raises(Exception, match="online_schema_evolution"):
        validate_retrieval_capabilities(models, project, resolved)


# ─── private generation build (issue #355) ──────────────────────────────────


def _gen_store(tmp_path: Path) -> Any:
    from stel.retrieval import LanceDBConfig, LanceDBStore

    return LanceDBStore(
        LanceDBConfig(type="lancedb", path=str(tmp_path / "lance")),
        project_name="proj",
        target_name="dev",
        alias="default",
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
    from stel.retrieval import RetrievalError

    store = _gen_store(tmp_path)
    # The token crosses into a physical collection name, so it is restricted
    # rather than escaped.
    for bad in ("has-dash", "UPPER", "with_underscore", "", "x" * 17, "a b"):
        with pytest.raises(RetrievalError, match="generation token"):
            store.physical_collection("ctx", generation=bad)


def test_generation_suffix_cannot_overflow_the_name_limit(
    tmp_path: Path,
) -> None:
    from stel.retrieval import RetrievalError

    store = _gen_store(tmp_path)
    long_logical = "c" * 120
    # Unsuffixed it already fills the budget; the suffix must be rejected
    # rather than produce a truncated or invalid collection name.
    with pytest.raises(RetrievalError, match="invalid"):
        store.physical_collection(long_logical, generation="abcd1234")


def test_lancedb_advertises_private_generation_build() -> None:
    from stel.retrieval import LanceDBStore, RetrievalFeature

    assert (
        RetrievalFeature.PRIVATE_GENERATION_BUILD
        in LanceDBStore.capabilities().features
    )


def test_drop_collection_removes_only_an_owned_existing_collection(
    tmp_path: Path,
) -> None:
    pytest.importorskip("lancedb")
    import pyarrow as pa

    from stel.retrieval import CollectionSpec

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

    from stel.retrieval import RetrievalError

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
