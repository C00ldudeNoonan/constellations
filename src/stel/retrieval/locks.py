"""Single-host publisher locking, shared by every local-file retrieval store.

Extracted from the LanceDB store when DuckDB became the second store to need
it (issue #371). A second copy of a lock implementation is a defect waiting to
happen: the two would drift, and the drift would show up as two publishers that
believe they hold the same lock. It also keeps the DuckDB store from importing
a module whose third-party dependency is an optional extra.

The guarantee and its boundary are unchanged from #152: an OS file lock
excludes concurrent publishers **on one host**. It cannot fence a publisher on
another machine, whether the two share a network filesystem or an object-store
prefix. Cross-host exclusion needs provider-enforced fencing, which no local
store advertises.
"""
from __future__ import annotations

import os
import sys
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any

from .base import RetrievalError


def default_host_lock_base(store_type: str) -> Path:
    """Fixed per-machine base for locks that cannot live beside their data.

    Deliberately NOT `tempfile.gettempdir()`: TMPDIR is per-process and
    per-container, so two publishers on one host could resolve to different
    directories and their locks would never contend — silently voiding the
    single-host guarantee while appearing to hold it. Publishers in isolated
    mount namespaces still need an explicit lock directory on a shared volume.
    """
    if sys.platform == "win32":
        base = os.environ.get("PROGRAMDATA") or r"C:\ProgramData"
        return Path(base) / "stel" / f"{store_type}-locks"
    return Path(f"/var/tmp/stel-{store_type}-locks")


class PublisherLock(AbstractContextManager[None]):
    """Non-blocking OS file lock excluding concurrent publisher processes."""

    def __init__(self, path: Path, *, store_type: str) -> None:
        self._path = path
        self._store_type = store_type
        self._handle: Any | None = None

    def __enter__(self) -> None:
        handle = self._path.open("a+b")
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise RetrievalError(
                f"Another publisher holds the {self._store_type} collection lock "
                f"(code={self._store_type}_publisher_lock_held); terminate it "
                "before recovering the serving scope"
            ) from None
        self._handle = handle
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
