"""Embed model executor (issue #190).

Owns the embed-only lifecycle: upstream validation, embedding identity and
incremental state, vector reuse from the existing target, provider batching,
row shaping, and materialization. runner.py keeps selection, DAG scheduling,
threading, and result aggregation, and re-exports the public names below for
compatibility.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from math import isfinite
from pathlib import Path
from typing import Any

import polars as pl

from ..adapters import (
    AdapterError,
    ReadPredicate,
    ReadPredicateOperator,
    StateRecord,
    StateScope,
    StateValue,
    WarehouseAdapter,
)
from ..budget import BudgetExceededError, BudgetGuard, BudgetLedger
from ..config.model import EMBED_METADATA_FIELDS, ModelConfig
from ..config.project import ProjectConfig
from ..dag import parse_ref
from ..embedding import EmbeddingIdentity, embed_texts, estimate_embed_requests
from ..hashing import canonical_fingerprint
from ..profile import ResolvedProfile, resolve_embedding_options
from ..progress import get_reporter
from ..versioning import compute_model_code_version
from .checkpoint import FlushPublisher
from .contracts import ModelRunResult, RunError
from .errors import artifact_error_text
from .usage import add_provider_usage
from .values import scalarize
from .warehouse import warehouse_options

log = logging.getLogger(__name__)


# Read batch sizes for the two input passes (issue #410). Deliberately not
# derived from `flush_every`: that is a publication cadence with no upper bound,
# while these bound *residency*. The id pass is projected to one narrow column
# so it can afford a wide batch; the row pass carries the full record, text
# included, so it stays an order of magnitude smaller.
_ID_BATCH_ROWS = 100_000
_INPUT_BATCH_ROWS = 10_000


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
    run_budget: BudgetLedger | None = None,
) -> ModelRunResult:
    assert model.embed is not None
    config = model.embed
    if not model.depends_on or len(model.depends_on) != 1:
        raise RunError(
            f"Embed model '{model.name}' must declare exactly one upstream in "
            "`depends_on:`"
        )
    upstream = parse_ref(model.depends_on[0])
    # A zero-row read for the contract, not the corpus (issue #410). The
    # previous `read_table(upstream)` pulled the whole upstream into one frame
    # before anything else happened, so a fresh run's peak was O(corpus) no
    # matter what #401 did downstream of it: a 7.3GB chunk table climbed for
    # six minutes with no flush committed and no provider call made. Column
    # names and dtypes are all this needs, and both survive a limit-0 read.
    schema_probe = adapter.read_table(upstream, limit=0)
    missing = sorted({config.id_field, config.text_field} - set(schema_probe.columns))
    if missing:
        raise RunError(
            f"Embed model '{model.name}': upstream '{upstream}' is missing "
            f"required column(s): {', '.join(missing)}. Available: "
            f"{sorted(schema_probe.columns)}"
        )
    generated = set(EMBED_METADATA_FIELDS) | {config.vector_field}
    generated_names = {name.casefold() for name in generated}
    collisions = sorted(
        column
        for column in schema_probe.columns
        if column.casefold() in generated_names
    )
    if collisions:
        raise RunError(
            f"Embed model '{model.name}': upstream '{upstream}' already contains "
            f"generated embedding column(s): {', '.join(collisions)}"
        )

    # One projected pass over the id column, before any spend. Kept ahead of
    # the embedding loop rather than folded into it so a NULL, empty, or
    # duplicate id still fails the run before the first provider call instead
    # of after the corpus has been paid for.
    current_ids, upstream_rows = _stream_upstream_ids(
        adapter, upstream, config.id_field, model.name
    )
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
    # The target is never loaded whole. The previous shape read every row --
    # vectors included -- into Python dicts before any embedding began, which
    # is ~25KB per 768-dim row: at 3.6M chunks that is ~90GB spent before the
    # first provider call, and only on the *resume* path, where the run being
    # resumed has already proven the corpus is large (issue #401 follow-up).
    reuse_reader = (
        _EmbeddingReuseReader(adapter, model.name, config=config)
        if is_incremental and not rebuild_target
        else None
    )
    # Embeds were the only provider-spending stage the run budget could not
    # see: --max-cost and friends gated extraction and llm calls while a
    # Vertex embed run spent freely. No cost estimator yet -- embedding
    # pricing is not modeled -- so max_api_calls, max_input_tokens, and
    # max_documents are the enforceable dimensions, and with per-flush
    # publication a budget stop is graceful: published windows stay, state is
    # advanced for exactly them, and the run reports budget_exceeded instead
    # of pretending to fail.
    budget_guard = BudgetGuard(None, run_budget) if run_budget is not None else None

    removed = sorted(set(processed_state) - current_ids)
    removed_target_keys: list[Any] = (
        [
            key
            for record_id in removed
            if (key := reuse_reader.target_key(record_id)) is not None
        ]
        if reuse_reader is not None
        else []
    )
    flush_every = config.flush_every
    skipped = 0
    cache_hits = 0
    embedded_count = 0
    processed = 0
    usage_totals: dict[str, int | float] = {}
    provider_batches = 0
    provider_calls = 0
    use_full = model.materialization == "full" or full_refresh or rebuild_target
    now = datetime.now(UTC).isoformat()
    output_schema = _embedding_output_schema(
        schema_probe, vector_field=config.vector_field
    )
    publisher = FlushPublisher(
        adapter,
        model_name=model.name,
        state_scope=state_scope,
        use_full=use_full,
    )

    def _window(record_id: str, record: dict[str, Any]) -> _EmbedWork | None:
        """Decide this record's fate without holding anything corpus-sized."""
        nonlocal skipped, cache_hits
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
            return None
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
        return item

    def _apply_reuse(window: list[_EmbedWork]) -> None:
        """Reclaim vectors the target already holds, one window at a time.

        One keyed, projected read per window rather than one whole-target
        load per run: residency is bounded by `flush_every` rows of reuse
        columns, and a corpus that never changes text pays warehouse reads,
        not provider calls.
        """
        nonlocal cache_hits
        if reuse_reader is None:
            return
        window_reuse = reuse_reader.rows_for(
            [item.record_id for item in window]
        )
        for item in window:
            existing = window_reuse.get(item.record_id)
            if (
                existing is not None
                and existing.get("embedding_input_hash") == item.text_hash
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

    def _iter_windows() -> Iterator[list[_EmbedWork]]:
        """Yield at most `flush_every` items at a time.

        The whole point of the rewrite (issue #401): the previous shape built
        one list of every row, embedded every vector into it, built a second
        list of output rows, then a DataFrame over that -- three corpus-sized
        structures alive at once, with 768 floats per row held as Python
        objects. A 3.6M-chunk corpus could not finish inside 115GB of virtual
        memory, and because nothing published until the end, 28 hours of paid
        provider calls were lost with it.
        """
        window: list[_EmbedWork] = []
        # Streamed, not zipped against a pre-read frame (issue #410). The id
        # comes off each record rather than from a parallel list, so this needs
        # no correspondence with the id pass above -- which matters, because
        # `table_snapshot` does not promise an ordering and the two passes are
        # separate snapshots.
        with adapter.table_snapshot(
            upstream, batch_size=_INPUT_BATCH_ROWS
        ) as snapshot:
            for batch in snapshot:
                frame = pl.from_arrow(batch)
                assert isinstance(frame, pl.DataFrame)
                for record in frame.iter_rows(named=True):
                    item = _window(str(record[config.id_field]), record)
                    if item is None:
                        continue
                    window.append(item)
                    if len(window) >= flush_every:
                        yield window
                        window = []
        if window:
            yield window

    def _embed_window(window: list[_EmbedWork]) -> None:
        nonlocal provider_batches, provider_calls, embedded_count
        pending = [item for item in window if item.vector is None]
        embedded_count += len(pending)
        if pending and budget_guard is not None:
            budget_guard.charge_documents(len(pending))
        for offset in range(0, len(pending), config.batch_size):
            batch = pending[offset : offset + config.batch_size]
            if budget_guard is not None:
                # One logical batch may fan into many billed requests --
                # Vertex issues one per text for gemini-embedding models --
                # so reserve what the provider says it will bill, not 1.
                budget_guard.ensure_headroom(
                    next_calls=estimate_embed_requests(
                        [item.text for item in batch],
                        identity,
                        profile_options=embedding_options.provider_options,
                    )
                )
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
            if budget_guard is not None:
                # Embedding usage carries no api_calls key -- the request
                # count is provider_requests -- so fold it in explicitly or
                # max_api_calls silently never trips for embeds.
                budget_guard.charge_metrics(
                    {
                        **embedded.usage.to_metrics(),
                        "api_calls": embedded.provider_requests,
                    }
                )
            for item, vector in zip(batch, embedded.vectors, strict=True):
                item.vector = vector

    def _publish(window: list[_EmbedWork]) -> None:
        """Publish one flush, then advance state for exactly its rows.

        Ordering is `FlushPublisher`'s, not this module's: a full rebuild
        clears state before its first write, state advances only after a write
        lands, and a warehouse failure is reported without the warehouse's own
        text. Those rules are identical for every flushing stage, and the one
        time each stage implemented them separately they diverged.
        """
        frame = pl.DataFrame(
            [
                _embedding_row(
                    item,
                    identity=identity,
                    vector_field=config.vector_field,
                    embedded_at=item.embedded_at or now,
                )
                for item in window
            ],
            schema=output_schema,
        )
        publisher.publish(
            write_full=lambda: adapter.materialize_full(
                model.name,
                frame,
                options=warehouse_opts,
            ),
            write_incremental=lambda: adapter.materialize_incremental(
                model.name,
                frame,
                key_col=config.id_field,
                on_schema_change=(
                    model.on_schema_change
                    if publisher.first_publication
                    else "append_new_columns"
                ),
                options=warehouse_opts,
                update_when_changed=model.update_when_changed,
            ),
            state_records=[
                StateRecord(item.record_id, item.input_fingerprint, code_version)
                for item in window
            ],
        )

    run_status: str | None = None
    errors: list[str] = []
    # Counted in source rows and advanced per flushed window: the operator
    # cares how much of the corpus is through, not how many provider requests
    # it took. The id pass already counted the upstream, so the total is known
    # even though both the windows and the read itself stream (issues #401, #410).
    with get_reporter().model_task(model.name, "embed", upstream_rows) as task:
        advanced = 0
        for window in _iter_windows():
            # Provider failures and publication failures are kept apart on
            # purpose. `artifact_error_text` exists to sanitize *provider* text;
            # routing a warehouse error through it hands the raw message to its
            # fallback, and that message can quote the offending row and the
            # statement that touched it (issue #401 review).
            _apply_reuse(window)
            try:
                _embed_window(window)
            except BudgetExceededError as e:
                # Exhaustion fires before the next provider call. Windows
                # already published stay with their state advanced; this
                # window's partial vectors are discarded and re-embedded next
                # run, exactly like a crash at the same point -- which is the
                # entire design.
                run_status = "budget_exceeded"
                errors.append(f"BudgetExceededError: {e}")
                break
            except Exception as e:
                raise RunError(
                    f"Embed model '{model.name}' provider execution failed: "
                    f"{artifact_error_text(e)}"
                ) from e
            processed += len(window)
            _publish(window)
            # Advance by *source rows consumed*, not window size: rows the
            # generator dropped as unchanged never reach a window, so a
            # mostly-incremental run would otherwise leave the bar near zero
            # while the model was in fact nearly done.
            consumed = processed + skipped
            task.advance(consumed - advanced)
            advanced = consumed
            # Drop the window's vectors before the next one is built. Without
            # this the generator's reference keeps one extra flush alive.
            window.clear()
        if run_status is None:
            # Rows skipped after the final window are consumed but never
            # advanced above, so close the bar out on the total it was opened
            # with. A budget stop leaves the bar where it stopped: filling it
            # would report a corpus as done that the cap just cut short.
            task.advance(upstream_rows - advanced)

    deleted_count = 0
    try:
        if run_status is not None:
            pass  # a budget stop mutates nothing further
        elif not publisher.published_any and use_full:
            # Nothing to write, but a rebuild still owes the target its table.
            adapter.replace_state(state_scope, [])
            publisher.rows_written += adapter.materialize_full(
                model.name,
                _empty_embedding_frame(
                    schema_probe, vector_field=config.vector_field
                ),
                options=warehouse_opts,
            )
        if removed and not use_full and run_status is None:
            adapter.delete_rows_and_state(
                model.name,
                key_col=config.id_field,
                keys=removed_target_keys,
                state_scope=state_scope,
                state_record_keys=removed,
            )
            deleted_count = len(removed)
    except AdapterError as e:
        raise RunError(str(e)) from e

    metrics: dict[str, Any] = {
        "provider_calls": provider_calls,
        "batches": provider_batches,
        "cache_hits": cache_hits,
        "cache_misses": embedded_count,
        "rows_embedded": embedded_count,
        "metadata_updates": cache_hits,
        **usage_totals,
    }
    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="embed",
        status=run_status,
        errors=errors,
        provider=identity.provider,
        provider_model=identity.model,
        provider_implementation=identity.implementation,
        documents_processed=processed,
        documents_skipped=skipped,
        # What was executed, not what was planned: a budget stop skips the
        # deletion pass, and a manifest claiming mutations that did not occur
        # is the kind of lie an operator acts on (issue #407 review).
        documents_deleted=deleted_count,
        rows_written=publisher.rows_written,
        metrics=metrics,
        artifact_metadata={"embedding": identity.to_dict()},
    )


def _stream_upstream_ids(
    adapter: WarehouseAdapter,
    table: str,
    id_field: str,
    model_name: str,
) -> tuple[set[str], int]:
    """Validate the upstream id column and return its distinct ids and row count.

    One projected, streamed pass (issue #410). Residency is proportional to the
    *key count*, not the row width -- the same trade `_EmbeddingReuseReader`
    already makes for the target -- so a corpus whose rows carry long text
    costs a set of ids here rather than a frame of everything.

    Deliberately eager rather than folded into the embedding loop: NULL, empty
    and duplicate ids are contract violations the run should die on before it
    has paid a provider for a single call.
    """
    ids: set[str] = set()
    total = 0
    null_count = 0
    empty_count = 0
    with adapter.table_snapshot(
        table, columns=[id_field], batch_size=_ID_BATCH_ROWS
    ) as snapshot:
        for batch in snapshot:
            frame = pl.from_arrow(batch)
            assert isinstance(frame, pl.DataFrame)
            for value in frame[id_field].to_list():
                total += 1
                if value is None:
                    null_count += 1
                    continue
                record_id = str(value)
                if not record_id:
                    empty_count += 1
                    continue
                ids.add(record_id)
    if null_count:
        raise RunError(
            f"Embed model '{model_name}': upstream `{id_field}` contains "
            f"{null_count} NULL value(s)"
        )
    if empty_count:
        raise RunError(
            f"Embed model '{model_name}': upstream `{id_field}` contains "
            f"{empty_count} empty value(s)"
        )
    duplicate_count = total - len(ids)
    if duplicate_count:
        raise RunError(
            f"Embed model '{model_name}': upstream `{id_field}` contains "
            f"{duplicate_count} duplicate value(s)"
        )
    return ids, total


class _EmbeddingReuseReader:
    """Bounded reads of the existing embed target on resume (issue #401).

    The resume path used to read the whole target into Python dicts, vectors
    included, before any embedding began -- ~25KB per 768-dim row, ~90GB at
    3.6M chunks -- and it is reachable only on resume, when the corpus has
    already proven itself too large for exactly that. What the run actually
    needs is far smaller:

    - **The id column, once.** State keys are stringified ids, but the target
      column may be typed (a numeric id is a tested contract), so deleting
      removed rows and pushing keyed predicates both need the *typed* value.
      One streamed, projected pass builds that map: no vectors, no metadata,
      residency proportional to key count rather than row width.
    - **Reuse columns, one window at a time.** A keyed IN predicate over the
      window's typed ids, projected to the hash/vector columns, so residency
      is bounded by `flush_every` whatever the corpus size.
    """

    def __init__(
        self,
        adapter: WarehouseAdapter,
        table: str,
        *,
        config: Any,
    ) -> None:
        self._adapter = adapter
        self._table = table
        self._id_field = config.id_field
        self._columns = (
            config.id_field,
            "embedding_input_hash",
            "embedding_config_hash",
            config.vector_field,
            "embedded_at",
        )
        self._usable = False
        self._target_keys: dict[str, Any] = {}
        self._load_keys()

    def _load_keys(self) -> None:
        # Schema probe via a zero-row read, not a snapshot: the DuckDB
        # snapshot's close() rolls back its transaction but does not close an
        # unexhausted Arrow reader, and that reader pins the database file.
        # Every other snapshot consumer iterates to exhaustion, which is why
        # the leak stayed invisible until a probe that never iterates.
        names = set(self._adapter.read_table(self._table, limit=0).columns)
        if not set(self._columns).issubset(names):
            # A pre-embed table (or an older contract) has nothing to reuse;
            # the old whole-table loader answered {} here too.
            return
        with self._adapter.table_snapshot(
            self._table,
            columns=[self._id_field],
            batch_size=100_000,
        ) as snapshot:
            for batch in snapshot:
                frame = pl.from_arrow(batch)
                assert isinstance(frame, pl.DataFrame)
                for value in frame[self._id_field].to_list():
                    if value is not None:
                        self._target_keys[str(value)] = value
        self._usable = True

    def target_key(self, record_id: str) -> Any | None:
        """The typed id column value for a stringified record id, if present."""
        return self._target_keys.get(record_id)

    # Bounded independently of flush_every: the window size is a publication
    # cadence with no upper limit, while TableReadRequest caps batch_size at
    # 100,000 and an IN list is SQL parameters -- copying the window length
    # into either turns a legal `flush_every: 200000` into a resume that
    # fails before embedding begins.
    _LOOKUP_KEYS_PER_READ = 10_000

    def rows_for(self, record_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not self._usable:
            return {}
        typed = [
            key
            for record_id in record_ids
            if (key := self._target_keys.get(record_id)) is not None
            # A key the predicate contract cannot carry (DuckDB DECIMAL,
            # BigQuery NUMERIC) skips reuse rather than failing the resume:
            # the row re-embeds -- correct output, paid call -- which is the
            # same price the row would pay if its text had changed. The old
            # whole-target dict handled these ids, so this is the one
            # narrowing the bounded path makes, and it is a cost, not a
            # correctness change.
            and isinstance(key, str | int | float | bool | date | datetime)
        ]
        if len(typed) < len(record_ids):
            log.debug(
                "embed reuse lookup skipped %d id(s) with non-scalar key types",
                len(record_ids) - len(typed),
            )
        found: dict[str, dict[str, Any]] = {}
        for offset in range(0, len(typed), self._LOOKUP_KEYS_PER_READ):
            chunk = typed[offset : offset + self._LOOKUP_KEYS_PER_READ]
            predicate = ReadPredicate(
                self._id_field,
                ReadPredicateOperator.IN,
                tuple(chunk),
            )
            with self._adapter.table_snapshot(
                self._table,
                columns=list(self._columns),
                predicate=predicate,
                batch_size=len(chunk),
            ) as snapshot:
                for batch in snapshot:
                    frame = pl.from_arrow(batch)
                    assert isinstance(frame, pl.DataFrame)
                    for row in frame.iter_rows(named=True):
                        key = row.get(self._id_field)
                        if key is not None:
                            found[str(key)] = row
        return found


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


def _embedding_output_schema(
    source: pl.DataFrame,
    *,
    vector_field: str,
) -> dict[str, Any]:
    """The output schema, fixed for the whole run.

    Every flush frame is built with this rather than inferred from its own
    rows (issue #401 review). A passthrough column that happens to be all-NULL
    in the first flush would otherwise infer as Null, the target column would
    be created from that, and a later flush carrying real values would fail on
    conversion -- after its provider calls had been paid for.
    """
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
    return schema


def _empty_embedding_frame(
    source: pl.DataFrame,
    *,
    vector_field: str,
) -> pl.DataFrame:
    return pl.DataFrame(
        schema=_embedding_output_schema(source, vector_field=vector_field)
    )
