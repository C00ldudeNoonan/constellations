"""Load a `--format graphiti` graph export into a Graphiti store (ALE-603, ADR-0065).

`stel graph-export --format graphiti` writes Graphiti's own records; this reads
them back through the manifest and saves each with `Model.save(driver)`, so the
export loads with no hand transformation. graphiti-core is imported when a
load starts, never at import time, and without the `graphiti` extra the command
fails naming the install command.

The loader keeps the export's contracts:

- **Memory is one shard.** The manifest names the shards, and each document's
  two files are read, checked and saved before the next is touched.
- **The manifest is the list of what to load.** The directory is never walked,
  so a stray file is not loaded; a shard the manifest names but the directory
  lost, or whose bytes no longer match its fingerprint, fails the load before
  that document is saved rather than loading half an export.
- **Nothing is followed through a symlink.** The shard trees, their fan-out
  directories and the shard files are refused if they are links.
- **A load is idempotent.** Graphiti's `save` merges on `uuid`, and a uuid is a
  function of the entity or relation id, so loading the same export twice, or
  an entity that several documents share, leaves one node.

The loader stamps what the file leaves out (`created_at`, `expired_at`, an
episode with no observed time); see `graph_graphiti.py` for the table.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .graph_export import (
    EDGES_DIRNAME,
    GRAPH_EXPORT_SCHEMA,
    GRAPH_MANIFEST_FILENAME,
    NODES_DIRNAME,
    _fingerprint,
    _shard_path,
)
from .graph_graphiti import read_records, require_graphiti, to_model
from .optional_dependencies import OptionalDependencyError


class GraphLoadError(Exception):
    pass


@dataclass(frozen=True)
class GraphLoadResult:
    documents: int
    entities: int
    episodes: int
    mentions: int
    edges: int


def check_loadable() -> None:
    """Refuse before any file is read when the extra is missing."""
    try:
        require_graphiti("`stel graph-load`")
    except OptionalDependencyError as error:
        raise GraphLoadError(str(error)) from error


def _manifest(export_dir: Path) -> dict[str, Any]:
    path = export_dir / GRAPH_MANIFEST_FILENAME
    if path.is_symlink() or not path.is_file():
        raise GraphLoadError(
            f"{export_dir} has no {GRAPH_MANIFEST_FILENAME}; run "
            "`stel graph-export --format graphiti` first"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GraphLoadError(f"{path} is not readable JSON") from error
    if not isinstance(data, dict) or data.get("schema") != GRAPH_EXPORT_SCHEMA:
        raise GraphLoadError(f"{path} is not a {GRAPH_EXPORT_SCHEMA} export")
    if data.get("format") != "graphiti" or not isinstance(data.get("documents"), dict):
        raise GraphLoadError(
            f"{path} holds a {data.get('format', 'jsonl')!r} export; only `--format "
            "graphiti` loads into Graphiti"
        )
    return data


def _read_shard(export_dir: Path, kind: str, entry: dict[str, Any]) -> bytes:
    if not entry[kind]:
        return b""
    path = _shard_path(export_dir, kind, entry["shard"], "graphiti")
    for part in (export_dir / kind, path.parent, path):
        if part.is_symlink():
            raise GraphLoadError(f"{part} is a symlink; the load never follows symlinks")
    if not path.is_file():
        raise GraphLoadError(
            f"{path} is named by the manifest but missing; export again before loading"
        )
    return path.read_bytes()


def _shards(export_dir: Path, manifest: dict[str, Any]) -> Iterator[tuple[bytes, bytes]]:
    for document_id, entry in manifest["documents"].items():
        nodes = _read_shard(export_dir, NODES_DIRNAME, entry)
        edges = _read_shard(export_dir, EDGES_DIRNAME, entry)
        if _fingerprint(nodes, edges) != entry["sha256"]:
            raise GraphLoadError(
                f"the shard for document {document_id!r} no longer matches the manifest's "
                "fingerprint; export again before loading"
            )
        yield nodes, edges


async def load_graphiti_export(
    export_dir: Path,
    driver: Any,
    *,
    loaded_at: datetime | None = None,
) -> GraphLoadResult:
    """Save every record of a `graphiti` export through `driver`.

    `driver` is any graphiti-core `GraphDriver`. Per document the nodes shard is
    saved before the edges shard, because an edge's two ends must exist.
    """
    check_loadable()
    manifest = _manifest(export_dir)
    stamp = loaded_at or datetime.now(UTC)
    await driver.build_indices_and_constraints()
    counts = {"entity_node": 0, "episodic_node": 0, "mentions": 0, "entity_edge": 0}
    documents = 0
    for nodes, edges in _shards(export_dir, manifest):
        for payload in (nodes, edges):
            for record in read_records(payload):
                await to_model(record, stamp).save(driver)
                counts[record["record"]] += 1
        documents += 1
    return GraphLoadResult(
        documents=documents,
        entities=counts["entity_node"],
        episodes=counts["episodic_node"],
        mentions=counts["mentions"],
        edges=counts["entity_edge"],
    )


def load_into_neo4j(
    export_dir: Path, *, uri: str, user: str, password: str, database: str | None
) -> GraphLoadResult:
    """Load into a Neo4j-backed Graphiti store.

    The password is held only to build the driver. A driver's own error text can
    carry the address or the credentials it was given, so a failure is reported
    by type alone.
    """
    check_loadable()
    from graphiti_core.driver.neo4j_driver import Neo4jDriver

    async def run() -> GraphLoadResult:
        driver = Neo4jDriver(uri=uri, user=user, password=password, database=database or "neo4j")
        try:
            return await load_graphiti_export(export_dir, driver)
        finally:
            await driver.close()

    try:
        return asyncio.run(run())
    except GraphLoadError:
        raise
    except Exception as error:
        raise GraphLoadError(
            f"the load into Neo4j failed ({type(error).__name__}); check the URI, the "
            "credentials and that the server is reachable"
        ) from None
