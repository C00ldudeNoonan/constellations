"""The durable embedding response cache (issue #665, ask 2).

The embed-side twin of the `llm:` `cache_path`. Every vector a provider
returns is written here as soon as its provider batch returns -- before the
window it belongs to publishes -- keyed on what decides the vector: the text
(`embedding_input_hash`) and the embedding configuration
(`embedding_config_hash`). That closes the three ways a corpus paid twice for
text it had already embedded, none of which the target-table reuse of #671
can see:

1. **A crash before a flush.** #401 made embed publish at flush boundaries,
   but a vector embedded after the last flush lived only in memory, so the
   retry paid for it again. astrolabe's 08-27 backfill was relaunched through
   repeated ~1.3h failures exactly that way.
2. **Duplicate text within a run.** The target index is read once, before the
   first window, so it cannot know about a vector the same run produced a
   window ago. SEC risk-factor and accounting-policy boilerplate repeats
   across filings and years.
3. **Duplicate text across runs** once the target row that held the vector is
   gone -- the row was deleted, or the model was rebuilt with
   `--full-refresh`, which turns target reuse off.

Operator-owned and outside `code_version`, like the llm cache: the path lives
in profiles.yml, and deleting the file costs provider calls, never
correctness. It holds hashes, vectors and timestamps -- no text, no ids, no
credentials -- and is kept owner-only for the same reason the llm cache is: a
vector is derived from the text it embeds.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl

from ..cache_files import harden_cache_files, private_cache_umask

log = logging.getLogger(__name__)

_LABEL = "Embedding cache"

# One lock for every cache file in the process. Concurrent provider batches
# write from worker threads, and two embed models in one run may share a
# path; DuckDB refuses a second read-write connection to a file another
# connection in the process holds with a different configuration, so every
# open is serialized here, as the llm cache's are.
_LOCK = threading.Lock()

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    embedding_input_hash VARCHAR NOT NULL,
    embedding_config_hash VARCHAR NOT NULL,
    vector DOUBLE[] NOT NULL,
    embedded_at VARCHAR NOT NULL,
    PRIMARY KEY (embedding_input_hash, embedding_config_hash)
)
"""


@dataclass(frozen=True, slots=True)
class CachedVector:
    vector: tuple[float, ...]
    #: When the provider produced it, so a row restored from the cache reports
    #: the original embedding time rather than the run that found it.
    embedded_at: str


class EmbeddingResponseCache:
    """One embed model's view of a cache file: its configuration only.

    Opened eagerly, so a path that cannot hold a cache -- a symlink, a
    directory, an unwritable location -- fails the run before the first
    provider call rather than after the corpus has been paid for.

    Lookups and writes after that are best-effort. A write that fails
    mid-run (a full disk, say) disables the cache for the rest of the run with
    one warning rather than failing it: the vectors in hand are already paid
    for and the window is about to publish them, so raising would discard
    exactly the spend the cache exists to protect.
    """

    def __init__(self, path: Path, *, config_hash: str, dimensions: int) -> None:
        self._path = path
        self._config_hash = config_hash
        self._dimensions = dimensions
        self._enabled = True
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            self._harden(require_main=False)
            with private_cache_umask():
                connection = duckdb.connect(str(path))
                try:
                    connection.execute(_TABLE_DDL)
                finally:
                    connection.close()
            self._harden(require_main=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_many(self, text_hashes: Sequence[str]) -> dict[str, CachedVector]:
        """Vectors this configuration already paid for, keyed by text hash."""
        wanted = list(dict.fromkeys(text_hashes))
        if not wanted or not self._enabled:
            return {}
        with _LOCK:
            if not self._enabled:
                return {}
            try:
                rows = self._read(wanted)
            except (duckdb.Error, ValueError) as error:
                self._disable("read", error)
                return {}
        found: dict[str, CachedVector] = {}
        for text_hash, vector, embedded_at in rows:
            # A vector of the wrong width is a corrupt or foreign entry, not a
            # usable one; the row re-embeds rather than publishing it.
            if vector is None or len(vector) != self._dimensions:
                continue
            found[text_hash] = CachedVector(
                tuple(float(value) for value in vector), str(embedded_at)
            )
        return found

    def put_many(
        self, entries: Sequence[tuple[str, tuple[float, ...]]], *, embedded_at: str
    ) -> None:
        """Record freshly paid vectors. The first vector for a key wins."""
        if not entries or not self._enabled:
            return
        frame = pl.DataFrame(
            {
                "embedding_input_hash": [text_hash for text_hash, _ in entries],
                "vector": [list(vector) for _, vector in entries],
            },
            schema={"embedding_input_hash": pl.String, "vector": pl.List(pl.Float64)},
        )
        with _LOCK:
            if not self._enabled:
                return
            try:
                self._write(frame, embedded_at=embedded_at)
            except (duckdb.Error, ValueError) as error:
                self._disable("write", error)

    def _read(self, wanted: list[str]) -> list[tuple[str, list[float], str]]:
        if os.name != "nt" and not self._harden(require_main=True):
            return []
        connection = duckdb.connect(str(self._path))
        try:
            connection.register(
                "stel_wanted", pl.DataFrame({"h": wanted}, schema={"h": pl.String})
            )
            return connection.execute(
                """
                SELECT c.embedding_input_hash, c.vector, c.embedded_at
                FROM embedding_cache AS c
                JOIN stel_wanted AS w ON c.embedding_input_hash = w.h
                WHERE c.embedding_config_hash = ?
                """,
                [self._config_hash],
            ).fetchall()
        finally:
            connection.close()

    def _write(self, frame: pl.DataFrame, *, embedded_at: str) -> None:
        if os.name != "nt":
            self._harden(require_main=False)
        with private_cache_umask():
            connection = duckdb.connect(str(self._path))
            try:
                connection.register("stel_incoming", frame)
                connection.execute(
                    """
                    INSERT INTO embedding_cache
                        (embedding_input_hash, embedding_config_hash, vector, embedded_at)
                    SELECT DISTINCT ON (embedding_input_hash)
                        embedding_input_hash, ?, vector, ?
                    FROM stel_incoming
                    ON CONFLICT DO NOTHING
                    """,
                    [self._config_hash, embedded_at],
                )
            finally:
                connection.close()
        if os.name != "nt":
            self._harden(require_main=True)

    def _harden(self, *, require_main: bool) -> bool:
        if os.name == "nt":
            return self._path.exists()
        return harden_cache_files(self._path, require_main=require_main, label=_LABEL)

    def _disable(self, operation: str, error: Exception) -> None:
        # The class name locates the failure; DuckDB's own message can quote
        # the path and the statement, and neither belongs in a log line that
        # outlives the run.
        self._enabled = False
        log.warning(
            "Embedding cache %s failed [%s]; continuing without the cache for "
            "the rest of this model's run. Vectors already published are "
            "unaffected.",
            operation,
            type(error).__name__,
        )
