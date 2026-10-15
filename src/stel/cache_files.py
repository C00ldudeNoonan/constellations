"""Owner-only, regular-file storage for stel's local response caches.

Shared by the `llm:` response cache and the embedding response cache (issue
#665). Both are DuckDB files at an operator-chosen path holding values the
operator paid a provider for, and both must refuse a symlinked path and stay
readable by their owner only -- a cache can hold sensitive values even when it
holds no source text.
"""

from __future__ import annotations

import os
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# The umask is process-wide, so the lock guarding it must be too. Each cache
# serializes its own writers with its own lock, but an llm-cache write and an
# embedding-cache write can overlap: without this, the first to finish
# restores the ordinary mask while the second is still creating files, and the
# second then "restores" 0o077 as the process's mask for good (Codex review on
# #690). Always taken innermost, after a cache's own lock, so it cannot
# deadlock against one.
_UMASK_LOCK = threading.Lock()


@contextmanager
def private_cache_umask() -> Iterator[None]:
    if os.name == "nt":
        yield
        return
    # DuckDB creates its write-ahead log beside the database, so the mask
    # covers the whole connection, not only the main file's creation.
    with _UMASK_LOCK:
        previous = os.umask(0o077)
        try:
            yield
        finally:
            os.umask(previous)


def harden_cache_files(path: Path, *, require_main: bool, label: str) -> bool:
    """Owner-only permissions on the cache and its WAL; False if absent.

    `label` names the cache in the refusal ("LLM cache", "Embedding cache"),
    so an operator with both configured can tell which path is wrong.
    """
    main_exists = _harden_optional_cache_file(path, label=label)
    _harden_optional_cache_file(Path(f"{path}.wal"), label=label)
    if require_main and not main_exists:
        return False
    return main_exists


def _harden_optional_cache_file(path: Path, *, label: str) -> bool:
    try:
        _harden_existing_cache_file(path, label=label)
    except FileNotFoundError:
        return False
    return True


def _harden_existing_cache_file(path: Path, *, label: str) -> None:
    refusal = f"{label} path must be a regular, non-symlink file"
    if path.is_symlink():
        raise ValueError(refusal)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as error:
        if os.path.lexists(path):
            raise ValueError(refusal) from error
        raise
    except OSError as error:
        raise ValueError(refusal) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(refusal)
        fchmod = getattr(os, "fchmod", None)
        if fchmod is not None:
            fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
    finally:
        os.close(descriptor)
