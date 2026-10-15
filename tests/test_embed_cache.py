"""The embedding response cache's own file boundaries (issue #665).

What the cache saves a run is pinned end to end in test_embed_cost.py; these
are the properties of the file itself, which hold whatever runs on top.
"""
from __future__ import annotations

import os
import stat
import threading
from pathlib import Path

import pytest

from stel.cache_files import private_cache_umask
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


@pytest.mark.skipif(os.name == "nt", reason="POSIX umask")
def test_overlapping_private_umasks_restore_the_process_mask() -> None:
    """The llm and embedding caches each lock their own writers, but the
    umask is process-wide. Two overlapping contexts without a shared lock
    leave the process's mask at 0o077 for good (Codex review on #690)."""
    first_inside = threading.Event()
    second_inside = threading.Event()
    first_done = threading.Event()

    # The interleaving that breaks an unserialized mask: the second context
    # enters while the first holds 0o077 (so it saves 0o077 as "previous"),
    # and the first exits before the second does. With the shared lock the
    # second cannot enter, the first's wait times out, and they run in turn.
    def first() -> None:
        with private_cache_umask():
            first_inside.set()
            second_inside.wait(timeout=0.5)
        first_done.set()

    def second() -> None:
        first_inside.wait()
        with private_cache_umask():
            second_inside.set()
            first_done.wait(timeout=5)

    original = os.umask(0o022)
    try:
        threads = [threading.Thread(target=first), threading.Thread(target=second)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert os.umask(0o022) == 0o022
    finally:
        os.umask(original)
