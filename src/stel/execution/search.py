"""Search publication executor (issue #190).

Owns the search-only lifecycle: retrieval-store selection, collection spec and
arrow-schema contract, publication lease/fence and serving coordination,
bounded reconciliation with per-batch state advancement, durable-receipt
verification, and the serving_resource descriptor. runner.py keeps selection,
DAG scheduling, and result aggregation, and re-exports run_search_model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import date, datetime, timedelta
from math import isfinite
from pathlib import Path
from typing import Any, cast

import pyarrow as pa

from ..adapters import (
    AdapterError,
    StateRecord,
    StateScope,
    WarehouseAdapter,
    WarehouseCapability,
)
from ..config.model import ModelConfig
from ..config.project import ProjectConfig
from ..dag import parse_ref
from ..embedding import effective_search_config, resolve_search_embedding_options
from ..hashing import canonical_fingerprint
from ..profile import ResolvedProfile
from ..retrieval import (
    ChangeKind,
    CollectionMetadata,
    CollectionSpec,
    IndexedRow,
    PublishLease,
    RetrievalError,
    ServingCoordinationError,
    ServingCoordinator,
    classify_descriptor_changes,
    collection_config_fingerprint,
    collection_descriptor,
    create_store,
    descriptor_json,
    legacy_collection_config_fingerprint,
)
from ..state_reconciliation import BoundedReconciler, UpstreamRecord
from ..versioning import compute_model_code_version
from .contracts import ModelRunResult, RunError


def run_search_model(
    *,
    model: ModelConfig,
    models_by_name: Mapping[str, ModelConfig],
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
) -> ModelRunResult:
    search = model.search
    assert search is not None
    if resolved.retrieval is None:
        raise RunError("Search publication requires a configured retrieval target")
    alias = search.store or resolved.retrieval.default
    store_config = resolved.retrieval.stores.get(alias)
    if store_config is None:
        raise RunError("Search publication selected an unknown retrieval target")
    upstream = parse_ref((model.depends_on or [""])[0])
    logical_collection = search.collection or model.name
    code_version = compute_model_code_version(
        model,
        project,
        project_dir,
        resolved=resolved,
    )
    store = create_store(
        store_config,
        project_name=project.name,
        target_name=resolved.target_name,
        alias=alias,
    )
    projected = search.projected_fields()
    inserted = 0
    updated = 0
    skipped = 0
    deleted = 0
    rows_written = 0
    rows_seen = 0
    spec: CollectionSpec | None = None
    state_scope: StateScope | None = None
    coordinator = ServingCoordinator(adapter)
    publish_lease: PublishLease | None = None
    active_generation: str | None = None

    try:
        # Publication-state residency is bounded (issue #153): reconciliation
        # never loads the full scope, so the adapter must provide ordered
        # paged state access before any store I/O begins.
        adapter.require_capability(
            WarehouseCapability.PAGED_STATE_RECONCILIATION,
            operation="bounded publication-state reconciliation",
        )
        with adapter.table_snapshot(
            upstream,
            columns=projected,
            batch_size=search.batch_size,
            key_column=search.id_field,
        ) as snapshot:
            physical = store.physical_collection(logical_collection)
            spec = _search_collection_spec(
                model=model,
                models_by_name=models_by_name,
                physical_collection=physical,
                upstream_schema=snapshot.schema,
                store_type=store_config.type,
                resolved=resolved,
            )
            state_target = store.state_descriptor(logical_collection)
            state_scope = StateScope.for_target_descriptor(
                model.name,
                stage="retrieval_publish",
                descriptor=state_target.descriptor(),
            )
            reconciler = BoundedReconciler(
                adapter,
                state_scope,
                code_version=code_version,
                page_size=search.batch_size,
            )

            publish_lease = coordinator.acquire_publish(
                state_scope,
                expected_code_version=code_version,
                config_fingerprint=spec.config_fingerprint,
            )
            with store.publisher_fence(physical), store:
                existing = store.inspect_collection(physical)
                force_publish = existing is None
                collection_exists = existing is not None
                if existing is not None:
                    evolved = _verify_collection_config(
                        store, existing, spec, policy=search.on_index_change
                    )
                    if evolved:
                        # The collection just changed shape; compare against
                        # what it now is, not the snapshot taken before.
                        existing = store.inspect_collection(physical) or existing
                    _validate_collection_schema(existing.schema, spec)

                for ordinal, batch in enumerate(snapshot):
                    indexed = _indexed_rows(
                        batch,
                        model,
                        spec.config_fingerprint,
                        max_id_bytes=store.capabilities().max_id_bytes,
                    )
                    rows_seen += len(indexed)
                    upstream_records = [
                        UpstreamRecord(row.record_id, row.input_fingerprint)
                        for row in indexed
                    ]
                    prior_state = (
                        reconciler.prior_state_for(upstream_records)
                        if upstream_records
                        else {}
                    )
                    outcome = reconciler.classify(
                        upstream_records,
                        prior=prior_state,
                        force_publish=force_publish,
                    )
                    skipped += len(outcome.unchanged)
                    pending_keys = {
                        record.record_key
                        for record in (*outcome.new, *outcome.changed)
                    }
                    pending: list[IndexedRow] = [
                        row for row in indexed if row.record_id in pending_keys
                    ]
                    pending_inserted = len(outcome.new)
                    pending_updated = len(outcome.changed)
                    if not pending:
                        continue
                    coordinator.verify_publish(publish_lease)
                    if not collection_exists:
                        store.create_collection(spec)
                        collection_exists = True
                    if search.access == "governed":
                        # Revoke-before-upsert: a changed governed record is
                        # removed first, so a failed publish leaves the old,
                        # possibly more permissive row absent, not queryable.
                        changed_ids = [
                            row.record_id
                            for row in pending
                            if row.record_id in prior_state
                        ]
                        if changed_ids:
                            revoke_digest = canonical_fingerprint(
                                {
                                    "snapshot": snapshot.fingerprint,
                                    "config": spec.config_fingerprint,
                                    "ordinal": ordinal,
                                    "revoked": changed_ids,
                                },
                                domain="dbt-ml-search-governed-revoke-batch",
                            )
                            receipt = store.delete(
                                physical,
                                changed_ids,
                                id_field=search.id_field,
                                mutation_digest=revoke_digest,
                            )
                            if not receipt.acknowledged or len(receipt.outcomes) != len(
                                changed_ids
                            ):
                                raise RunError(
                                    "Retrieval store did not return an exact durable "
                                    "governed-revoke receipt"
                                )
                    digest = canonical_fingerprint(
                        {
                            "snapshot": snapshot.fingerprint,
                            "config": spec.config_fingerprint,
                            "ordinal": ordinal,
                            "rows": [row.input_fingerprint for row in pending],
                        },
                        domain="dbt-ml-search-upsert-batch",
                    )
                    receipt = store.upsert(
                        physical,
                        pending,
                        id_field=search.id_field,
                        mutation_digest=digest,
                    )
                    if not receipt.acknowledged or len(receipt.outcomes) != len(pending):
                        raise RunError(
                            "Retrieval store did not return an exact durable upsert receipt"
                        )
                    # Advance state per batch, only for receipt-acknowledged
                    # rows: a later failure leaves those rows durably
                    # published and correctly recorded, while readiness stays
                    # gated by the serving ledger.
                    coordinator.verify_publish(publish_lease)
                    adapter.upsert_state(
                        state_scope,
                        [
                            StateRecord(
                                row.record_id, row.input_fingerprint, code_version
                            )
                            for row in pending
                        ],
                    )
                    inserted += pending_inserted
                    updated += pending_updated
                    rows_written += len(pending)

                # Stale discovery streams state pages whose keys no longer
                # exist upstream, in ascending key order — complete even for
                # an empty upstream, with residency bounded by one page.
                stale_pages = reconciler.iter_stale_pages(
                    upstream_table=upstream,
                    key_column=search.id_field,
                )
                try:
                    for ordinal, stale_page in enumerate(stale_pages):
                        record_ids = [record.record_key for record in stale_page]
                        coordinator.verify_publish(publish_lease)
                        digest = canonical_fingerprint(
                            {
                                "snapshot": snapshot.fingerprint,
                                "config": spec.config_fingerprint,
                                "ordinal": ordinal,
                                "state": [
                                    record.input_fingerprint
                                    for record in stale_page
                                ],
                            },
                            domain="dbt-ml-search-delete-batch",
                        )
                        receipt = store.delete(
                            physical,
                            record_ids,
                            id_field=search.id_field,
                            mutation_digest=digest,
                        )
                        if not receipt.acknowledged or len(receipt.outcomes) != len(
                            record_ids
                        ):
                            raise RunError(
                                "Retrieval store did not return an exact durable "
                                "delete receipt"
                            )
                        adapter.delete_state(state_scope, record_ids)
                        deleted += len(record_ids)
                finally:
                    stale_pages.close()

                if not collection_exists:
                    coordinator.verify_publish(publish_lease)
                    store.create_collection(spec)
                metadata = store.ensure_indexes(spec)
                if metadata.config_fingerprint != spec.config_fingerprint:
                    raise RunError(
                        "Retrieval collection failed post-publication configuration validation"
                    )
                if metadata.row_count != rows_seen:
                    raise RunError(
                        "Retrieval collection failed post-publication row-count validation"
                    )
                active_generation = metadata.physical_generation

        assert state_scope is not None
        coordinator.verify_publish(publish_lease)
        assert active_generation is not None
        assert spec is not None
        coordinator.mark_ready(
            publish_lease,
            active_generation=active_generation,
            config_fingerprint=spec.config_fingerprint,
            counts=(inserted, updated, skipped, deleted),
        )
    except (AdapterError, RetrievalError, RunError) as error:
        if publish_lease is not None:
            _mark_search_publication_failed(
                coordinator,
                publish_lease,
                error,
                counts=(inserted, updated, skipped, deleted),
            )
        if isinstance(error, RunError):
            raise
        raise RunError(str(error)) from None

    assert spec is not None
    safe_target = store.safe_descriptor()
    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="search",
        backend=store_config.type,
        documents_processed=inserted + updated,
        documents_skipped=skipped,
        documents_deleted=deleted,
        rows_written=rows_written,
        rows_inserted=inserted,
        rows_updated=updated,
        serving_resource={
            "type": "retrieval_index",
            "store_type": store_config.type,
            "safe_target_identity": safe_target.safe_target_identity,
            "logical_collection": logical_collection,
            "physical_collection": store.physical_collection(logical_collection),
            "config_fingerprint": spec.config_fingerprint,
            "status": "ready",
            "active_generation": active_generation,
            "fencing_token": publish_lease.fencing_token if publish_lease else None,
        },
    )


def _mark_search_publication_failed(
    coordinator: ServingCoordinator,
    lease: PublishLease,
    error: Exception,
    *,
    counts: tuple[int, int, int, int],
) -> None:
    """Record the failure under a safe code; a stale fence has nothing to record."""
    if isinstance(error, ServingCoordinationError):
        code = "coordination_error"
    elif isinstance(error, RetrievalError):
        code = "store_error"
    elif isinstance(error, AdapterError):
        code = "warehouse_error"
    else:
        code = "publication_failed"
    with suppress(ServingCoordinationError):
        coordinator.mark_failed(lease, safe_error_code=code, counts=counts)


def _search_collection_spec(
    *,
    model: ModelConfig,
    models_by_name: Mapping[str, ModelConfig],
    physical_collection: str,
    upstream_schema: pa.Schema,
    store_type: str,
    resolved: ResolvedProfile,
) -> CollectionSpec:
    search = model.search
    assert search is not None
    projected = search.projected_fields()
    missing = [field for field in projected if field not in upstream_schema.names]
    if missing:
        raise RunError(
            f"Search resource '{model.name}' upstream schema is missing declared fields: "
            f"{', '.join(missing)}"
        )
    fields: list[pa.Field] = []
    for name in projected:
        upstream = upstream_schema.field(name)
        if search.vector is not None and name == search.vector.field:
            if not _is_numeric_list_type(upstream.type):
                raise RunError("Search vector field must use a numeric list warehouse type")
            if (
                pa.types.is_fixed_size_list(upstream.type)
                and upstream.type.list_size != search.vector.dimensions
            ):
                raise RunError(
                    "Search vector warehouse type does not match configured dimensions"
                )
            fields.append(pa.field(name, pa.list_(pa.float32(), search.vector.dimensions)))
        else:
            fields.append(pa.field(name, upstream.type, nullable=upstream.nullable))
    schema = pa.schema(fields)
    _validate_search_schema(model, schema=schema)
    embedding_runtime = resolve_search_embedding_options(
        model,
        models_by_name,
        resolved,
    )
    search_payload = effective_search_config(
        model,
        models_by_name,
        profile_options=embedding_runtime.provider_options
        if embedding_runtime is not None
        else None,
    )
    config_fingerprint = collection_config_fingerprint(
        search_payload, store_type=store_type
    )
    descriptor = descriptor_json(
        collection_descriptor(search_payload, store_type=store_type)
    )
    legacy_fingerprint = legacy_collection_config_fingerprint(
        search_payload, store_type=store_type
    )
    return CollectionSpec(
        logical_name=search.collection or model.name,
        physical_name=physical_collection,
        id_field=search.id_field,
        text_fields=search.text_fields,
        full_text_fields=search.full_text.fields if search.full_text else (),
        attribute_fields=tuple(attribute.name for attribute in search.attributes),
        scalar_index_fields=tuple(
            attribute.name
            for attribute in search.attributes
            if attribute.filter_role != "none" or attribute.sortable
        ),
        display_fields=search.display_fields,
        vector_field=search.vector.field if search.vector else None,
        vector_dimensions=search.vector.dimensions if search.vector else None,
        distance_metric=search.vector.metric if search.vector else None,
        vector_search=search.vector.search if search.vector else None,
        config_fingerprint=config_fingerprint,
        descriptor=descriptor,
        legacy_config_fingerprint=legacy_fingerprint,
        arrow_schema=schema,
    )


def _schema_matches_by_name(existing: pa.Schema, declared: pa.Schema) -> bool:
    """Compare two schemas by column name and type, ignoring order.

    This is the contract stel actually relies on: every read, write, and
    predicate addresses columns by name, so ordering carries no meaning, and a
    store column's nullability is not a guarantee stel makes — rows are
    validated on the way in. Types are still compared exactly.

    An ordered comparison was over-strict, and once collections could be
    widened it was wrong. `add_columns` can only append, while the declared
    schema orders columns by projection, so a newly added attribute lands ahead
    of the display fields in one and behind them in the other; added columns
    are also nullable until the republish fills them. Comparing strictly
    rejected a collection that had just been widened correctly, on every run
    after the widening (Codex review, #344).
    """
    return {field.name: field.type for field in existing} == {
        field.name: field.type for field in declared
    }


def _validate_collection_schema(existing: pa.Schema, spec: CollectionSpec) -> None:
    """Refuse a collection whose physical shape is not the declared one.

    One comparison for every path. A collection that was widened in place is
    reached again on the next run, when nothing has changed and no evolution
    happens, so a stricter check here than the one applied during the widening
    would reject the collection it had just accepted.
    """
    if not _schema_matches_by_name(existing, spec.arrow_schema):
        raise RunError(
            "Search index schema does not match the declared collection contract"
        )


def _verify_collection_config(
    store: Any,
    existing: CollectionMetadata,
    spec: CollectionSpec,
    *,
    policy: str = "fail",
) -> bool:
    """Refuse a publish whose configuration no longer describes the collection.

    Returns whether the collection was widened in place. Under the default
    `fail` policy this only ever returns False or raises, but it says *which*
    field moved and distinguishes a change the existing collection could serve
    from one that invalidates the rows already written. Under `online` a
    compatible change is applied rather than refused (issue #344).
    """
    if existing.descriptor is None:
        # Published before the descriptor existed: it carries only the legacy
        # digest. Recomputing that digest is the one way to prove the
        # configuration is unchanged, in which case the stamp is merely stale
        # in format and is rewritten in place — no rebuild, no re-embed.
        if existing.config_fingerprint == spec.legacy_config_fingerprint:
            store.restamp_collection(spec)
            return False
        raise RunError(
            "Search index configuration changed. This collection predates "
            "configuration classification, so the changed fields cannot be "
            "named; publish under a new collection name, validate it, and cut "
            "consumers over."
        )

    if existing.descriptor == spec.descriptor:
        return False

    changes = classify_descriptor_changes(
        json.loads(existing.descriptor), json.loads(spec.descriptor)
    )
    blocking = [c for c in changes if c.kind is ChangeKind.REBUILD_REQUIRED]
    detail = "; ".join(change.describe() for change in changes) or "store contract"
    if blocking:
        raise RunError(
            f"Search index configuration changed and requires a rebuild: {detail}. "
            "Rows already written were indexed under the previous definition, so "
            "publish under a new collection name, validate it, and cut consumers "
            "over."
        )
    if policy == "online":
        # Compatible-only, and the store advertised that it can widen a live
        # collection. Columns arrive null; every row's fingerprint includes the
        # config digest, so the republish that follows refills the collection
        # from the warehouse — an index rewrite, but no provider calls.
        added = [
            name
            for name in spec.arrow_schema.names
            if name not in set(existing.schema.names)
        ]
        store.evolve_collection(spec, added)
        return True

    raise RunError(
        f"Search index configuration changed: {detail}. The change is additive "
        "and the existing collection could serve it — set `on_index_change: "
        "online` to widen it in place, or publish under a new collection name."
    )


def _indexed_rows(
    batch: pa.RecordBatch,
    model: ModelConfig,
    config_fingerprint: str,
    *,
    max_id_bytes: int | None,
) -> list[IndexedRow]:
    search = model.search
    assert search is not None
    rows: list[IndexedRow] = []
    for raw in batch.to_pylist():
        record_id = raw.get(search.id_field)
        if (
            not isinstance(record_id, str)
            or not record_id
            or "\x00" in record_id
            or (max_id_bytes is not None and len(record_id.encode()) > max_id_bytes)
        ):
            raise RunError(
                "Search input IDs must be non-empty, NUL-free strings within the "
                "retrieval store byte limit"
            )
        values = dict(raw)
        for optional_id_field in (search.document_id_field, search.chunk_id_field):
            if optional_id_field is None:
                continue
            optional_id = values.get(optional_id_field)
            if optional_id is not None and (
                not isinstance(optional_id, str)
                or not optional_id
                or "\x00" in optional_id
                or (max_id_bytes is not None and len(optional_id.encode()) > max_id_bytes)
            ):
                raise RunError("Search document and chunk IDs must be valid strings")
        for text_field in search.text_fields:
            if not isinstance(values.get(text_field), str):
                raise RunError("Search text fields must contain non-null strings")
        if not any(values[text_field] for text_field in search.text_fields):
            raise RunError("Search input must contain at least one non-empty text field")
        if search.vector is not None:
            vector = values.get(search.vector.field)
            if not isinstance(vector, Sequence) or isinstance(vector, str | bytes):
                raise RunError("Search vectors must be finite numeric sequences")
            if any(isinstance(item, bool) for item in vector):
                raise RunError("Search vectors must be finite numeric sequences")
            try:
                normalized = [float(item) for item in vector]
            except (TypeError, ValueError):
                raise RunError("Search vectors must be finite numeric sequences") from None
            if len(normalized) != search.vector.dimensions or any(
                not isfinite(item) for item in normalized
            ):
                raise RunError(
                    "Search vectors must match the configured dimensions and be finite"
                )
            values[search.vector.field] = normalized
        for attribute in search.attributes:
            _validate_search_attribute(values.get(attribute.name), attribute)
        for display_field in search.display_fields:
            if not _is_json_safe(values.get(display_field)):
                raise RunError("Search display fields must contain JSON-safe values")
        input_fingerprint = canonical_fingerprint(
            {"config": config_fingerprint, "row": values},
            domain="dbt-ml-search-indexed-row",
        )
        rows.append(IndexedRow(record_id, values, input_fingerprint))
    return rows


def _validate_search_attribute(value: Any, attribute: Any) -> None:
    if value is None:
        if attribute.nullable:
            return
        raise RunError("Search attributes declared non-nullable must not contain NULL")
    expected: dict[str, type[Any] | tuple[type[Any], ...]] = {
        "string": str,
        "integer": int,
        "float": (int, float),
        "boolean": bool,
        "date": (date, str),
        "timestamp": (datetime, str),
    }
    if attribute.data_type == "array[string]":
        if not isinstance(value, list | tuple) or any(
            not isinstance(item, str) for item in value
        ):
            raise RunError("Search array attributes must contain only strings")
        return
    required = expected[attribute.data_type]
    if attribute.data_type == "integer" and isinstance(value, bool):
        raise RunError("Search attribute type does not match its declaration")
    if attribute.data_type == "float" and isinstance(value, bool):
        raise RunError("Search attribute type does not match its declaration")
    if not isinstance(value, required):
        raise RunError("Search attribute type does not match its declaration")
    if attribute.data_type == "integer" and not -(2**63) <= value < 2**63:
        raise RunError("Search integer attributes must fit signed 64-bit values")
    if attribute.data_type == "float" and not isfinite(float(value)):
        raise RunError("Search float attributes must be finite")
    if attribute.data_type == "date" and isinstance(value, str):
        try:
            date.fromisoformat(value)
        except ValueError:
            raise RunError("Search date attributes must be ISO-8601 dates") from None
    if attribute.data_type == "timestamp":
        parsed = value
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                raise RunError(
                    "Search timestamp attributes must be ISO-8601 timestamps"
                ) from None
        if parsed.utcoffset() != timedelta(0):
            raise RunError("Search timestamp attributes must be UTC")


def _validate_search_schema(model: ModelConfig, *, schema: pa.Schema) -> None:
    search = model.search
    assert search is not None
    for field_name in (search.id_field, search.document_id_field, search.chunk_id_field):
        if field_name is not None and not _is_string_type(schema.field(field_name).type):
            raise RunError("Search ID fields must use a string warehouse type")
    for field_name in search.text_fields:
        if not _is_string_type(schema.field(field_name).type):
            raise RunError("Search text fields must use a string warehouse type")
    for attribute in search.attributes:
        if not _attribute_type_matches(
            attribute.data_type, schema.field(attribute.name).type
        ):
            raise RunError("Search attribute warehouse type does not match its declaration")


def _is_string_type(value: pa.DataType) -> bool:
    return cast(bool, pa.types.is_string(value) or pa.types.is_large_string(value))


def _is_numeric_list_type(value: pa.DataType) -> bool:
    if not (
        pa.types.is_list(value)
        or pa.types.is_large_list(value)
        or pa.types.is_fixed_size_list(value)
    ):
        return False
    return cast(
        bool,
        pa.types.is_integer(value.value_type) or pa.types.is_floating(value.value_type),
    )


def _attribute_type_matches(data_type: str, value: pa.DataType) -> bool:
    if data_type == "string":
        return _is_string_type(value)
    if data_type == "integer":
        return cast(bool, pa.types.is_integer(value))
    if data_type == "float":
        return cast(bool, pa.types.is_floating(value))
    if data_type == "boolean":
        return cast(bool, pa.types.is_boolean(value))
    if data_type == "date":
        return cast(bool, pa.types.is_date(value))
    if data_type == "timestamp":
        return cast(bool, pa.types.is_timestamp(value))
    if not (
        pa.types.is_list(value)
        or pa.types.is_large_list(value)
        or pa.types.is_fixed_size_list(value)
    ):
        return False
    return _is_string_type(value.value_type)


def _is_json_safe(value: Any) -> bool:
    if value is None or isinstance(value, str | bool | int):
        return True
    if isinstance(value, float):
        return isfinite(value)
    if isinstance(value, list | tuple):
        return all(_is_json_safe(item) for item in value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _is_json_safe(item) for key, item in value.items())
    return False
