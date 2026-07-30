from __future__ import annotations

from datetime import UTC, datetime
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
from ..chunking import chunk_id, split_text
from ..config.model import ModelConfig
from ..dag import parse_ref
from ..hashing import canonical_fingerprint
from ..versioning import compute_code_version
from .contracts import ModelRunResult, RunError
from .values import scalarize
from .warehouse import warehouse_options

_CHUNK_GENERATED_FIELDS = frozenset(
    {
        "chunk_id",
        "document_id",
        "chunk_index",
        "chunk_count",
        "text",
        "chunk_strategy",
        "code_version",
        "chunked_at",
    }
)
_CHUNK_INPUT_EXCLUDED_FIELDS = _CHUNK_GENERATED_FIELDS


def run_chunk_model(
    *,
    model: ModelConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    full_refresh: bool,
) -> ModelRunResult:
    assert model.chunk is not None
    chunk_config = model.chunk
    if not model.depends_on or len(model.depends_on) != 1:
        raise RunError(
            f"Chunk model '{model.name}' must declare exactly one upstream in "
            "`depends_on:` (the extraction model to chunk)"
        )
    upstream = parse_ref(model.depends_on[0])
    frame = adapter.read_table(upstream)
    if chunk_config.text_field not in frame.columns:
        raise RunError(
            f"Chunk model '{model.name}': upstream '{upstream}' has no column "
            f"'{chunk_config.text_field}'. Available: {sorted(frame.columns)}"
        )
    if "document_id" not in frame.columns:
        raise RunError(
            f"Chunk model '{model.name}': upstream '{upstream}' has no "
            "`document_id`; chunk models read extraction outputs."
        )
    document_ids = chunk_document_ids(frame, model.name)

    code_version = compute_code_version(
        extraction=None,
        transform=None,
        chunk=chunk_config,
        depends_on=[upstream],
        project_dir=project_dir,
    )
    parsed_warehouse_options = warehouse_options(adapter, model)
    state_scope = StateScope(model.name)
    is_incremental = model.materialization == "incremental" and not full_refresh
    processed_state = adapter.fetch_state(state_scope) if is_incremental else {}

    # Generated values are replaced on every chunk; all other upstream metadata
    # participates in invalidation and remains available for filtering/lineage.
    carry_columns = [
        column
        for column in frame.columns
        if column != chunk_config.text_field and column not in _CHUNK_GENERATED_FIELDS
    ]
    chunked_at = datetime.now(UTC).isoformat()

    rows: list[dict[str, Any]] = []
    state_records: list[StateRecord] = []
    processed = 0
    skipped = 0
    current_ids: set[str] = set()
    changed_ids: list[str] = []

    for document_id, record in zip(
        document_ids, frame.iter_rows(named=True), strict=True
    ):
        current_ids.add(document_id)
        raw_text = record[chunk_config.text_field]
        text = "" if raw_text is None else str(raw_text)
        document_hash = chunk_input_hash(record, text_field=chunk_config.text_field)
        if is_incremental:
            prior = processed_state.get(document_id)
            if prior == StateValue(document_hash, code_version):
                skipped += 1
                continue
            if prior is not None:
                changed_ids.append(document_id)
        processed += 1
        pieces = split_text(text, chunk_config)
        carried = {column: record[column] for column in carry_columns}
        for piece in pieces:
            rows.append(
                chunk_row(
                    carried=carried,
                    document_id=document_id,
                    piece_index=piece.index,
                    chunk_count=len(pieces),
                    text=piece.text,
                    strategy=chunk_config.strategy,
                    code_version=code_version,
                    chunked_at=chunked_at,
                )
            )
        state_records.append(StateRecord(document_id, document_hash, code_version))

    deleted = 0
    if is_incremental:
        removed = [
            document_id
            for document_id in processed_state
            if document_id not in current_ids
        ]
        if removed:
            adapter.delete_rows_and_state(
                model.name,
                key_col="document_id",
                keys=removed,
                state_scope=state_scope,
            )
            deleted = len(removed)

    rows_written = 0
    chunk_frame = pl.DataFrame(rows) if rows else pl.DataFrame()
    if model.materialization == "full" or full_refresh:
        rows_written = adapter.materialize_full(
            model.name,
            chunk_frame,
            options=parsed_warehouse_options,
        )
        adapter.replace_state(state_scope, state_records)
    else:
        try:
            rows_written = adapter.replace_children(
                model.name,
                parent_key="document_id",
                parent_ids=changed_ids,
                child_key="chunk_id",
                new_rows=chunk_frame,
                state_scope=state_scope,
                state_records=state_records,
                on_schema_change=model.on_schema_change,
                options=parsed_warehouse_options,
            )
        except AdapterError as error:
            raise RunError(str(error)) from error

    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="chunk",
        documents_processed=processed,
        documents_skipped=skipped,
        documents_deleted=deleted,
        rows_written=rows_written,
    )


def chunk_document_ids(frame: pl.DataFrame, model_name: str) -> list[str]:
    raw_ids = frame["document_id"].to_list()
    null_count = sum(value is None for value in raw_ids)
    if null_count:
        raise RunError(
            f"Chunk model '{model_name}': upstream `document_id` contains "
            f"{null_count} NULL value(s)"
        )
    document_ids = [str(value) for value in raw_ids]
    empty_count = sum(not value for value in document_ids)
    if empty_count:
        raise RunError(
            f"Chunk model '{model_name}': upstream `document_id` contains "
            f"{empty_count} empty value(s)"
        )
    duplicate_count = len(document_ids) - len(set(document_ids))
    if duplicate_count:
        raise RunError(
            f"Chunk model '{model_name}': upstream `document_id` contains "
            f"{duplicate_count} duplicate value(s)"
        )
    return document_ids


def chunk_input_hash(record: dict[str, Any], *, text_field: str) -> str:
    raw_text = record[text_field]
    effective_input = {
        "document_id": str(record["document_id"]),
        "text": "" if raw_text is None else str(raw_text),
        "carried": {
            key: value
            for key, value in record.items()
            if key != text_field and key not in _CHUNK_INPUT_EXCLUDED_FIELDS
        },
    }
    return canonical_fingerprint(effective_input, domain="chunk-input", version=2)


def chunk_row(
    *,
    carried: dict[str, Any],
    document_id: str,
    piece_index: int,
    chunk_count: int,
    text: str,
    strategy: str,
    code_version: str,
    chunked_at: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        column: scalarize(value) for column, value in carried.items()
    }
    row.update(
        {
            "chunk_id": chunk_id(document_id, piece_index, text),
            "document_id": document_id,
            "chunk_index": piece_index,
            "chunk_count": chunk_count,
            "text": text,
            "chunk_strategy": strategy,
            "code_version": code_version,
            "chunked_at": chunked_at,
        }
    )
    return row
