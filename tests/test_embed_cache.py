"""The embedding response cache's own file boundaries (issue #665).

What the cache saves a run is pinned end to end in test_embed_cost.py; these
are the properties of the file itself, which hold whatever runs on top.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from stel.execution.embed_cache import EmbeddingResponseCache


def test_an_entry_of_the_wrong_width_is_not_served(tmp_path: Path) -> None:
    """A corrupt or foreign entry re-embeds rather than publishing a vector
    the target's dimension contract would reject -- or worse, accept."""
    cache = EmbeddingResponseCache(
        tmp_path / "cache.duckdb", config_hash="config", dimensions=4
    )
    cache.put_many(
        [("short", (1.0, 2.0, 3.0)), ("right", (1.0, 2.0, 3.0, 4.0))],
        embedded_at="2026-10-15T00:00:00+00:00",
    )

    found = cache.get_many(["short", "right"])

    assert set(found) == {"right"}
    assert found["right"].vector == (1.0, 2.0, 3.0, 4.0)
    assert found["right"].embedded_at == "2026-10-15T00:00:00+00:00"


def test_the_first_vector_for_a_text_wins(tmp_path: Path) -> None:
    """A cached vector is what a rebuild publishes; a second write for the
    same text must not move it under rows already published from the first."""
    cache = EmbeddingResponseCache(
        tmp_path / "cache.duckdb", config_hash="config", dimensions=2
    )
    cache.put_many([("text", (1.0, 1.0))], embedded_at="first")
    cache.put_many([("text", (2.0, 2.0))], embedded_at="second")

    hit = cache.get_many(["text"])["text"]
    assert (hit.vector, hit.embedded_at) == ((1.0, 1.0), "first")


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_cache_file_is_private_and_rejects_symlinks(tmp_path: Path) -> None:
    """Same treatment as the llm cache: a vector is derived from the text it
    embeds, so the file is owner-only, and a symlinked path is refused before
    a run spends anything."""
    path = tmp_path / "cache.duckdb"
    original_umask = os.umask(0o022)
    try:
        cache = EmbeddingResponseCache(path, config_hash="config", dimensions=2)
        cache.put_many([("text", (1.0, 1.0))], embedded_at="now")
    finally:
        os.umask(original_umask)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    link = tmp_path / "cache-link.duckdb"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="Embedding cache path must be a regular"):
        EmbeddingResponseCache(link, config_hash="config", dimensions=2)
