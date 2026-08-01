from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import lancedb
import polars as pl
import pyarrow as pa
import pytest
from click.testing import CliRunner

from dbt_ml.adapters import AdapterError, StateScope, TableReadSnapshot, create_adapter
from dbt_ml.adapters.duckdb import DuckDBAdapter
from dbt_ml.cli import cli
from dbt_ml.compiler import (
    validate_project_contract,
    validate_retrieval_capabilities,
)
from dbt_ml.config import ConfigError, load_project
from dbt_ml.dbt_export import build_dbt_sources
from dbt_ml.manifest import build_manifest
from dbt_ml.profile import resolve_profile
from dbt_ml.retrieval import (
    LanceDBStore,
    RetrievalError,
    RetrievalPredicate,
    RetrievalPredicateOperator,
    ServingCoordinator,
    collection_config_fingerprint,
    create_store,
)
from dbt_ml.runner import RunError, run_project


def _write_project(tmp_path: Path, *, allow_public: bool = True) -> None:
    (tmp_path / "sources").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "dbt_ml_project.yml").write_text(
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
    assert "dbt-ml search" in shown.output


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
            "from dbt_ml.config import SearchConfig",
            "from dbt_ml.retrieval import collection_config_fingerprint",
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
