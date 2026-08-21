"""Warehouse-table document source (issue #322).

A `warehouse://<relation>` source treats each **row** of a relation as a
document, so text that arrived in the warehouse — Fivetran/Airbyte/dlt loads,
upstream dbt models — enters an `extraction:` pipeline the same way files do.
The relation is read through the *active adapter*, so warehouse dialect stays
behind `adapters/` and per-target `source_paths` overrides point each target
at its own copy.

The `DocumentSource` contract maps onto rows directly:

- `document_id` derives from the source-relative path, whose final segment is
  always the `key_column` value (optionally prefixed by `path_columns` values),
  so `--source-filter 'subreddit/*'` scopes rows exactly as it scopes object
  prefixes.
- `content_hash` is a canonical fingerprint of the whole row, so a changed row
  re-extracts, an unchanged row skips, and a deleted row prunes — the existing
  incremental machinery, unchanged.
- `fetch()` writes the row as canonical JSON into the per-run scratch
  directory, where the ordinary `json` backend extracts declared fields.

Discovery reads the relation once and serves fetches from that snapshot: the
same memory class as `read_table` elsewhere in stel, and it guarantees a run
extracts the rows it discovered, not whatever the table held moments later —
the row-grain analogue of the object sources' verified-snapshot rule.
"""

from __future__ import annotations

import base64
import datetime
import decimal
import json
import logging
from pathlib import Path
from typing import Any

from ..adapters import AdapterError, create_adapter
from ..config.profile import WarehouseConfig
from ..config.source import WAREHOUSE_SOURCE_SCHEME, SourceConfig
from ..hashing import canonical_fingerprint
from ..versioning import compute_document_id
from .base import DocumentRef, DocumentSource, SourceError, SourceScan

log = logging.getLogger(__name__)

# Fingerprint domain for a source row's content hash. Pinned in
# tests/test_frozen_names.py: a drift here reports every row as new and
# re-extracts the whole relation.
ROW_HASH_DOMAIN = "warehouse-source-row"


def _relation(source: SourceConfig) -> str:
    return source.path[len(WAREHOUSE_SOURCE_SCHEME) :]


def _json_value(value: Any) -> Any:
    """Warehouse scalar types the json backend should see as plain JSON.

    Not `hashing.canonical_json` — that is the *typed* serialization for
    fingerprints, and a backend reading it would see tagged tuples instead of
    a document. Timestamps render as ISO strings, decimals as strings (JSON
    floats would silently round them), binary as base64.
    """
    if isinstance(value, datetime.datetime | datetime.date | datetime.time):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    raise TypeError(f"unsupported warehouse value type {type(value).__name__}")


def _row_json(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, default=_json_value)


class WarehouseDocumentSource(DocumentSource):
    """Rows of a warehouse relation, as documents."""

    def __init__(self, warehouse: WarehouseConfig, project_dir: Path) -> None:
        self._warehouse = warehouse
        self._project_dir = project_dir
        # document_id -> canonical JSON of the discovered row, held between
        # discover() and fetch() so extraction consumes the discovered
        # snapshot rather than re-querying a table that may have moved.
        self._rows: dict[str, str] = {}

    def discover(
        self, source: SourceConfig, project_dir: Path
    ) -> list[DocumentRef]:
        relation = _relation(source)
        key = source.key_column
        assert key is not None  # enforced by SourceConfig validation
        try:
            with create_adapter(
                self._warehouse, project_dir=self._project_dir
            ) as adapter:
                frame = adapter.read_relation(relation)
        except AdapterError as error:
            raise SourceError(
                f"Source '{source.name}': cannot read '{relation}': {error}"
            ) from error

        if frame.height > source.max_objects:
            raise SourceError(
                f"Source '{source.name}': more than {source.max_objects} rows "
                f"matched in '{relation}' ({frame.height}). Narrow the "
                "relation (a view is fine) or raise `max_objects` on the "
                "source."
            )
        missing = [
            column
            for column in (key, *source.path_columns)
            if column not in frame.columns
        ]
        if missing:
            raise SourceError(
                f"Source '{source.name}': '{relation}' has no column(s) "
                f"{missing}. Available: {sorted(frame.columns)}"
            )

        refs: list[DocumentRef] = []
        by_path: dict[str, str] = {}
        null_keys = 0
        duplicates: set[str] = set()
        self._rows = {}
        for row in frame.iter_rows(named=True):
            key_value = row[key]
            if key_value is None:
                null_keys += 1
                continue
            segments = [str(row[column]) for column in source.path_columns]
            if any(row[column] is None for column in source.path_columns):
                raise SourceError(
                    f"Source '{source.name}': row {key}={key_value!r} has a "
                    "null path column; `path_columns:` values become the "
                    "document path and cannot be null"
                )
            relative_path = "/".join([*segments, str(key_value)])
            if relative_path in by_path:
                duplicates.add(relative_path)
                continue
            by_path[relative_path] = str(key_value)
            document_id = compute_document_id(source.name, relative_path)
            self._rows[document_id] = _row_json(row)
            refs.append(
                DocumentRef(
                    source_name=source.name,
                    relative_path=relative_path,
                    document_id=document_id,
                    content_hash=canonical_fingerprint(
                        row, domain=ROW_HASH_DOMAIN
                    ),
                    source_uri=f"{source.path}#{key}={key_value}",
                )
            )
        if duplicates:
            sample = ", ".join(sorted(duplicates)[:5])
            raise SourceError(
                f"Source '{source.name}': {len(duplicates)} duplicate document "
                f"path(s) in '{relation}' (e.g. {sample}). `{key}` must be "
                "unique — which duplicate row became the document would depend "
                "on warehouse row order."
            )
        if null_keys:
            raise SourceError(
                f"Source '{source.name}': {null_keys} row(s) in '{relation}' "
                f"have a null `{key}`. A row without a key has no identity "
                "across runs; filter them out (a view is fine) or fix the "
                "upstream load."
            )
        log.info("Source '%s': discovered %d row(s)", source.name, len(refs))
        return refs

    def fetch(self, ref: DocumentRef, work_dir: Path) -> Path:
        payload = self._rows.get(ref.document_id)
        if payload is None:
            raise SourceError(
                f"Source '{ref.source_name}': row for document "
                f"'{ref.relative_path}' was not part of this run's discovery; "
                "re-run discovery instead of fetching a stale reference"
            )
        destination_dir = work_dir / ref.document_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / "row.json"
        destination.write_text(payload, encoding="utf-8")
        return destination

    def scan(self, source: SourceConfig, project_dir: Path) -> SourceScan:
        relation = _relation(source)
        try:
            with create_adapter(
                self._warehouse, project_dir=self._project_dir
            ) as adapter:
                count = adapter.relation_row_count(relation)
        except AdapterError as error:
            raise SourceError(
                f"Source '{source.name}': cannot scan '{relation}': {error}"
            ) from error
        return SourceScan(
            exists=True,
            file_count=count,
            # Rows carry no modification time a listing could read; freshness
            # for warehouse sources needs a declared watermark column, which
            # is future work — reported honestly rather than guessed.
            newest_epoch=None,
            newest_name=None,
            message="warehouse sources report no modification times",
        )
