"""Embed model executor (issue #190).

Owns the embed-only lifecycle: upstream validation, embedding identity and
incremental state, vector reuse from the existing target, provider batching,
row shaping, and materialization. runner.py keeps selection, DAG scheduling,
threading, and result aggregation, and re-exports the public names below for
compatibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

import polars as pl

from ..adapters import (
    AdapterError,
    StateRecord,
    StateScope,
    StateValue,
    WarehouseAdapter,
)
from ..config.model import EMBED_METADATA_FIELDS, ModelConfig
from ..config.project import ProjectConfig
from ..dag import parse_ref
from ..embedding import EmbeddingIdentity, embed_texts
from ..hashing import canonical_fingerprint
from ..profile import ResolvedProfile, resolve_embedding_options
from ..versioning import compute_model_code_version
from .contracts import ModelRunResult, RunError
from .errors import artifact_error_text
from .usage import add_provider_usage
from .values import scalarize
from .warehouse import warehouse_options


@dataclass
class _EmbedWork:
    record_id: str
    record: dict[str, Any]
    input_fingerprint: str
    text_hash: str
    text: str
    vector: tuple[float, ...] | None = None
    embedded_at: str | None = None


def run_embed_model(
    *,
    model: ModelConfig,
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
    full_refresh: bool,
) -> ModelRunResult:
    assert model.embed is not None
    config = model.embed
    if not model.depends_on or len(model.depends_on) != 1:
        raise RunError(
            f"Embed model '{model.name}' must declare exactly one upstream in "
            "`depends_on:`"
        )
    upstream = parse_ref(model.depends_on[0])
    source = adapter.read_table(upstream)
    missing = sorted({config.id_field, config.text_field} - set(source.columns))
    if missing:
        raise RunError(
            f"Embed model '{model.name}': upstream '{upstream}' is missing "
            f"required column(s): {', '.join(missing)}. Available: "
            f"{sorted(source.columns)}"
        )
    generated = set(EMBED_METADATA_FIELDS) | {config.vector_field}
    generated_names = {name.casefold() for name in generated}
    collisions = sorted(
        column for column in source.columns if column.casefold() in generated_names
    )
    if collisions:
        raise RunError(
            f"Embed model '{model.name}': upstream '{upstream}' already contains "
            f"generated embedding column(s): {', '.join(collisions)}"
        )

    record_ids = _embed_record_ids(source, config.id_field, model.name)
    embedding_options = resolve_embedding_options(config.provider, resolved)
    identity = EmbeddingIdentity.from_config(
        config,
        profile_options=embedding_options.provider_options,
    )
    code_version = compute_model_code_version(
        model,
        project,
        project_dir,
        resolved=resolved,
    )
    warehouse_opts = warehouse_options(adapter, model)
    state_scope = StateScope(model.name)
    existing_tables = set(adapter.list_tables())
    is_incremental = model.materialization == "incremental" and not full_refresh
    rebuild_target = is_incremental and model.name not in existing_tables
    processed_state = (
        adapter.fetch_state(state_scope)
        if is_incremental and not rebuild_target
        else {}
    )
    existing_rows = (
        _existing_embedding_rows(
            adapter,
            model.name,
            id_field=config.id_field,
        )
        if is_incremental and not rebuild_target
        else {}
    )

    current_ids = set(record_ids)
    removed = sorted(set(processed_state) - current_ids)
    removed_target_keys = [
        existing_rows[record_id][config.id_field]
        for record_id in removed
        if record_id in existing_rows
    ]
    work: list[_EmbedWork] = []
    skipped = 0
    cache_hits = 0
    for record_id, record in zip(
        record_ids,
        source.iter_rows(named=True),
        strict=True,
    ):
        text_value = record[config.text_field]
        text = "" if text_value is None else str(text_value)
        input_fingerprint = canonical_fingerprint(
            record,
            domain="embedding-input-row",
            version=1,
        )
        if processed_state.get(record_id) == StateValue(
            input_fingerprint,
            code_version,
        ):
            skipped += 1
            continue
        text_hash = canonical_fingerprint(
            {"text": text},
            domain="embedding-input-text",
            version=1,
        )
        item = _EmbedWork(
            record_id=record_id,
            record=record,
            input_fingerprint=input_fingerprint,
            text_hash=text_hash,
            text=text,
        )
        existing = existing_rows.get(record_id)
        if (
            existing is not None
            and existing.get("embedding_input_hash") == text_hash
            and existing.get("embedding_config_hash") == identity.config_hash
        ):
            vector = _coerce_embedding_vector(
                existing.get(config.vector_field),
                dimensions=config.dimensions,
            )
            if vector is not None:
                item.vector = vector
                embedded_at = existing.get("embedded_at")
                item.embedded_at = (
                    str(embedded_at) if embedded_at is not None else None
                )
                cache_hits += 1
        work.append(item)

    pending = [item for item in work if item.vector is None]
    usage_totals: dict[str, int | float] = {}
    provider_batches = 0
    provider_calls = 0
    try:
        for offset in range(0, len(pending), config.batch_size):
            batch = pending[offset : offset + config.batch_size]
            embedded = embed_texts(
                [item.text for item in batch],
                identity,
                input_ids=[item.record_id for item in batch],
                credential_env=embedding_options.api_key_env,
                profile_options=embedding_options.provider_options,
                max_retries=config.max_retries,
                timeout_seconds=embedding_options.timeout_seconds,
            )
            provider_batches += 1
            provider_calls += embedded.provider_requests
            add_provider_usage(usage_totals, embedded.usage.to_metrics())
            for item, vector in zip(batch, embedded.vectors, strict=True):
                item.vector = vector
    except Exception as e:
        raise RunError(
            f"Embed model '{model.name}' provider execution failed: "
            f"{artifact_error_text(e)}"
        ) from e

    now = datetime.now(UTC).isoformat()
    rows = [
        _embedding_row(
            item,
            identity=identity,
            vector_field=config.vector_field,
            embedded_at=item.embedded_at or now,
        )
        for item in work
    ]
    state_records = [
        StateRecord(item.record_id, item.input_fingerprint, code_version)
        for item in work
    ]
    output = (
        pl.DataFrame(rows)
        if rows
        else _empty_embedding_frame(
            source,
            vector_field=config.vector_field,
        )
    )

    rows_written = 0
    use_full = model.materialization == "full" or full_refresh or rebuild_target
    try:
        if use_full:
            rows_written = adapter.materialize_full(
                model.name,
                output,
                options=warehouse_opts,
            )
            adapter.replace_state(state_scope, state_records)
        else:
            if rows:
                rows_written = adapter.materialize_incremental(
                    model.name,
                    output,
                    key_col=config.id_field,
                    on_schema_change=model.on_schema_change,
                    options=warehouse_opts,
                    update_when_changed=model.update_when_changed,
                )
            if removed:
                adapter.delete_rows_and_state(
                    model.name,
                    key_col=config.id_field,
                    keys=removed_target_keys,
                    state_scope=state_scope,
                    state_record_keys=removed,
                )
            if state_records:
                adapter.upsert_state(state_scope, state_records)
    except AdapterError as e:
        raise RunError(str(e)) from e

    metrics: dict[str, Any] = {
        "provider_calls": provider_calls,
        "batches": provider_batches,
        "cache_hits": cache_hits,
        "cache_misses": len(pending),
        "rows_embedded": len(pending),
        "metadata_updates": cache_hits,
        **usage_totals,
    }
    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="embed",
        provider=identity.provider,
        provider_model=identity.model,
        provider_implementation=identity.implementation,
        documents_processed=len(work),
        documents_skipped=skipped,
        documents_deleted=len(removed),
        rows_written=rows_written,
        metrics=metrics,
        artifact_metadata={"embedding": identity.to_dict()},
    )


def _embed_record_ids(
    frame: pl.DataFrame,
    id_field: str,
    model_name: str,
) -> list[str]:
    values = frame[id_field].to_list()
    null_count = sum(value is None for value in values)
    if null_count:
        raise RunError(
            f"Embed model '{model_name}': upstream `{id_field}` contains "
            f"{null_count} NULL value(s)"
        )
    record_ids = [str(value) for value in values]
    empty_count = sum(not value for value in record_ids)
    if empty_count:
        raise RunError(
            f"Embed model '{model_name}': upstream `{id_field}` contains "
            f"{empty_count} empty value(s)"
        )
    duplicate_count = len(record_ids) - len(set(record_ids))
    if duplicate_count:
        raise RunError(
            f"Embed model '{model_name}': upstream `{id_field}` contains "
            f"{duplicate_count} duplicate value(s)"
        )
    return record_ids


def _existing_embedding_rows(
    adapter: WarehouseAdapter,
    table: str,
    *,
    id_field: str,
) -> dict[str, dict[str, Any]]:
    existing = adapter.read_table(table)
    if id_field not in existing.columns:
        return {}
    return {
        str(row[id_field]): row
        for row in existing.iter_rows(named=True)
        if row[id_field] is not None
    }


def _coerce_embedding_vector(
    value: Any,
    *,
    dimensions: int,
) -> tuple[float, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return None
    if len(value) != dimensions:
        return None
    if any(
        isinstance(item, bool)
        or not isinstance(item, int | float)
        or not isfinite(item)
        for item in value
    ):
        return None
    return tuple(float(item) for item in value)


def _embedding_row(
    item: _EmbedWork,
    *,
    identity: EmbeddingIdentity,
    vector_field: str,
    embedded_at: str,
) -> dict[str, Any]:
    assert item.vector is not None
    row = {key: scalarize(value) for key, value in item.record.items()}
    row[vector_field] = list(item.vector)
    row.update(
        {
            "embedding_provider": identity.provider,
            "embedding_model": identity.model,
            "embedding_dimensions": identity.dimensions,
            "embedding_provider_implementation": identity.implementation,
            "embedding_input_hash": item.text_hash,
            "embedding_config_hash": identity.config_hash,
            "embedded_at": embedded_at,
        }
    )
    return row


def _empty_embedding_frame(
    source: pl.DataFrame,
    *,
    vector_field: str,
) -> pl.DataFrame:
    schema: dict[str, Any] = dict(source.schema)
    schema.update(
        {
            vector_field: pl.List(pl.Float64),
            "embedding_provider": pl.String,
            "embedding_model": pl.String,
            "embedding_dimensions": pl.Int64,
            "embedding_provider_implementation": pl.String,
            "embedding_input_hash": pl.String,
            "embedding_config_hash": pl.String,
            "embedded_at": pl.String,
        }
    )
    return pl.DataFrame(schema=schema)
