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
from ..chunking import (
    ChunkingError,
    chunk_id,
    measure,
    render_metadata_block,
    split_text,
)
from ..config.model import CHUNK_GENERATED_FIELDS, ModelConfig
from ..dag import parse_ref
from ..hashing import canonical_fingerprint
from ..progress import get_reporter
from ..versioning import compute_code_version
from .checkpoint import FlushPublisher
from .contracts import ModelRunResult, RunError
from .values import scalarize
from .warehouse import warehouse_options

_CHUNK_GENERATED_FIELDS = CHUNK_GENERATED_FIELDS
_CHUNK_INPUT_EXCLUDED_FIELDS = _CHUNK_GENERATED_FIELDS

# Read batch sizes for the two input passes (issue #423), matching embed's.
# The id pass is projected to one narrow column so it can afford a wide batch;
# the row pass carries the document text and stays an order of magnitude
# smaller. Neither is derived from `flush_every`, which is a publication
# cadence with no upper bound while these bound residency.
_ID_BATCH_ROWS = 100_000
_INPUT_BATCH_ROWS = 1_000


def run_chunk_model(
    *,
    model: ModelConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    full_refresh: bool,
    subset_run: bool = False,
) -> ModelRunResult:
    assert model.chunk is not None
    chunk_config = model.chunk
    if not model.depends_on or len(model.depends_on) != 1:
        raise RunError(
            f"Chunk model '{model.name}' must declare exactly one upstream in "
            "`depends_on:` (the extraction model to chunk)"
        )
    upstream = parse_ref(model.depends_on[0])
    # A zero-row read for the contract, not the corpus (issue #423). Chunk used
    # to pull the whole upstream registry into one frame before splitting
    # anything — the #410 hole one stage earlier, and worse placed, since chunk
    # feeds embed and its input is the document registry.
    schema_probe = adapter.read_table(upstream, limit=0)
    frame = schema_probe
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
    if chunk_config.headings is not None:
        column = chunk_config.headings.column
        if column in frame.columns:
            raise RunError(
                f"Chunk model '{model.name}': `chunk.headings.column` is "
                f"'{column}', which upstream '{upstream}' already has. The "
                "attribution would overwrite it; name the heading column "
                "something else."
            )
    missing_metadata = [
        column
        for column in chunk_config.in_text_metadata
        if column not in frame.columns
    ]
    if missing_metadata:
        raise RunError(
            f"Chunk model '{model.name}': `chunk.in_text_metadata` names "
            f"{missing_metadata}, which upstream '{upstream}' does not have. "
            f"Available: {sorted(frame.columns)}"
        )
    # One projected pass over the id column, before anything is written.
    current_ids, upstream_rows = _stream_document_ids(adapter, upstream, model.name)

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

    state_records: list[StateRecord] = []
    processed = 0
    skipped = 0
    flush_every = chunk_config.flush_every
    use_full = model.materialization == "full" or full_refresh
    # An explicit dtype for the section column: a first window whose pattern
    # matched no headings supplies only nulls, which polars infers as `Null`
    # and DuckDB materializes as an integer column — so the next window that
    # does find a heading fails converting a string into it (Codex review,
    # #343, and the same failure mode as the append-only logs in #333). Fixed
    # for the whole run rather than inferred per window, which is exactly the
    # hazard windowed publication introduces.
    section_schema = (
        {chunk_config.headings.column: pl.String}
        if chunk_config.headings is not None
        else None
    )
    publisher = FlushPublisher(
        adapter,
        model_name=model.name,
        state_scope=state_scope,
        use_full=use_full,
    )

    def _publish_window(
        window_rows: list[dict[str, Any]],
        window_state: list[StateRecord],
        window_changed: list[str],
    ) -> None:
        chunk_frame = (
            pl.DataFrame(window_rows, schema_overrides=section_schema)
            if window_rows
            else pl.DataFrame()
        )
        publisher.publish(
            write_full=lambda: adapter.materialize_full(
                model.name,
                chunk_frame,
                options=parsed_warehouse_options,
            ),
            write_incremental=lambda: adapter.replace_children(
                model.name,
                parent_key="document_id",
                parent_ids=window_changed,
                child_key="chunk_id",
                new_rows=chunk_frame,
                state_scope=state_scope,
                state_records=window_state,
                on_schema_change=model.on_schema_change,
                options=parsed_warehouse_options,
            ),
            state_records=window_state,
            # replace_children applies the state records in the same
            # transaction as the rows; the full-replace branch does not.
            advances_state_itself=not (use_full and publisher.first_publication),
        )

    window_rows: list[dict[str, Any]] = []
    window_state: list[StateRecord] = []
    window_changed: list[str] = []
    window_documents = 0

    with get_reporter().model_task(model.name, "chunk", upstream_rows) as task:
        # Streamed, not read whole (issue #423). The id comes off each record
        # rather than from a parallel list, so this needs no correspondence
        # with the id pass above — which matters, because `table_snapshot`
        # promises no ordering and the two passes are separate snapshots.
        with adapter.table_snapshot(
            upstream, batch_size=_INPUT_BATCH_ROWS
        ) as snapshot:
            for batch in snapshot:
                batch_frame = pl.from_arrow(batch)
                assert isinstance(batch_frame, pl.DataFrame)
                for record in batch_frame.iter_rows(named=True):
                    task.advance(1)
                    document_id = str(record["document_id"])
                    raw_text = record[chunk_config.text_field]
                    text = "" if raw_text is None else str(raw_text)
                    document_hash = chunk_input_hash(
                        record, text_field=chunk_config.text_field
                    )
                    if is_incremental:
                        prior = processed_state.get(document_id)
                        if prior == StateValue(document_hash, code_version):
                            skipped += 1
                            continue
                        if prior is not None:
                            window_changed.append(document_id)
                    processed += 1
                    # Rendered per document, because the values are the
                    # document's. The block is charged against chunk_size
                    # before splitting and prepended after, so every emitted
                    # chunk — block included — stays within the size the
                    # embedder was configured for (issue #308).
                    block = render_metadata_block(
                        record, chunk_config.in_text_metadata
                    )
                    try:
                        pieces = split_text(
                            text,
                            chunk_config,
                            reserved=measure(block, chunk_config) if block else 0,
                        )
                    except ChunkingError as error:
                        raise RunError(
                            f"Chunk model '{model.name}': {error}"
                        ) from error
                    carried = {column: record[column] for column in carry_columns}
                    for piece in pieces:
                        window_rows.append(
                            chunk_row(
                                carried=carried,
                                document_id=document_id,
                                piece_index=piece.index,
                                chunk_count=len(pieces),
                                text=block + piece.text,
                                section_column=(
                                    chunk_config.headings.column
                                    if chunk_config.headings is not None
                                    else None
                                ),
                                section=piece.section,
                                strategy=chunk_config.strategy,
                                code_version=code_version,
                                chunked_at=chunked_at,
                            )
                        )
                    record_state = StateRecord(
                        document_id, document_hash, code_version
                    )
                    window_state.append(record_state)
                    state_records.append(record_state)
                    window_documents += 1
                    if window_documents >= flush_every:
                        try:
                            _publish_window(
                                window_rows, window_state, window_changed
                            )
                        except AdapterError as error:
                            raise RunError(str(error)) from error
                        # Drop the window's rows before the next one is built;
                        # holding them is the O(corpus) the windows exist to
                        # avoid, and chunking amplifies, so the output is
                        # larger than the input it came from.
                        window_rows = []
                        window_state = []
                        window_changed = []
                        window_documents = 0

    if window_documents or not publisher.published_any:
        # The trailing partial window, and the empty-run case: a rebuild still
        # owes the target its (possibly empty) table.
        try:
            _publish_window(window_rows, window_state, window_changed)
        except AdapterError as error:
            raise RunError(str(error)) from error
        window_rows = []
        window_state = []
        window_changed = []

    deleted = 0
    # A subset invocation deliberately narrows what the run sees, so absence
    # is not removal: skipping the delete pass is the same promise
    # --source-filter already makes for extraction (#266), extended one model
    # kind downstream (issue #417). Reconciliation is the job of the next
    # unfiltered run.
    if is_incremental and not subset_run:
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

    rows_written = publisher.rows_written

    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="chunk",
        documents_processed=processed,
        documents_skipped=skipped,
        documents_deleted=deleted,
        rows_written=rows_written,
    )


def _reject_bad_document_ids(
    model_name: str,
    *,
    total: int,
    distinct: int,
    null_count: int,
    empty_count: int,
) -> None:
    """The `document_id` contract, from counts rather than from the values.

    Shared so the frame-based and streamed passes cannot drift into reporting
    the same violation differently.
    """
    if null_count:
        raise RunError(
            f"Chunk model '{model_name}': upstream `document_id` contains "
            f"{null_count} NULL value(s)"
        )
    if empty_count:
        raise RunError(
            f"Chunk model '{model_name}': upstream `document_id` contains "
            f"{empty_count} empty value(s)"
        )
    duplicate_count = total - distinct
    if duplicate_count:
        raise RunError(
            f"Chunk model '{model_name}': upstream `document_id` contains "
            f"{duplicate_count} duplicate value(s)"
        )


def chunk_document_ids(frame: pl.DataFrame, model_name: str) -> list[str]:
    raw_ids = frame["document_id"].to_list()
    document_ids = [str(value) for value in raw_ids if value is not None]
    _reject_bad_document_ids(
        model_name,
        total=len(raw_ids),
        distinct=len({value for value in document_ids if value}),
        null_count=sum(value is None for value in raw_ids),
        empty_count=sum(not value for value in document_ids),
    )
    return document_ids


def _stream_document_ids(
    adapter: WarehouseAdapter, table: str, model_name: str
) -> tuple[set[str], int]:
    """Validate the upstream `document_id` column and return it, streamed.

    One projected pass (issue #423). Residency is proportional to the *key
    count* rather than the row width, which for a document registry is the
    difference between a set of ids and several gigabytes of text.

    Eager rather than folded into the chunk loop: a NULL, empty, or duplicate
    id is a contract violation the run should die on before it has written
    anything, not partway through publishing windows.
    """
    ids: set[str] = set()
    total = 0
    null_count = 0
    empty_count = 0
    with adapter.table_snapshot(
        table, columns=["document_id"], batch_size=_ID_BATCH_ROWS
    ) as snapshot:
        for batch in snapshot:
            frame = pl.from_arrow(batch)
            assert isinstance(frame, pl.DataFrame)
            for value in frame["document_id"].to_list():
                total += 1
                if value is None:
                    null_count += 1
                    continue
                document_id = str(value)
                if not document_id:
                    empty_count += 1
                    continue
                ids.add(document_id)
    _reject_bad_document_ids(
        model_name,
        total=total,
        distinct=len(ids),
        null_count=null_count,
        empty_count=empty_count,
    )
    return ids, total


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
    section_column: str | None = None,
    section: str | None = None,
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
    if section_column is not None:
        row[section_column] = section
    return row
