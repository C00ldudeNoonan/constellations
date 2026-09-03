"""What memory this process may use, as the container sees it.

Written for issue #412, where DuckDB sized itself from host RAM inside a
container and grew past the ceiling the kernel actually kills at. It lived in
the DuckDB adapter while that was its only caller. Issue #476 added a second —
the search publish estimates an ANN index build against the same ceiling
before spending hours on the rows — and the per-batch memory log line wants
the limit beside the RSS it reports, so the reader moved here.

Reads the standard cgroup mount rather than resolving this process's own
cgroup path: container runtimes mount the container's own cgroup there, and
the delegated-subtree cases where that is wrong are ones where an operator can
set an explicit limit instead. Everything here is advisory and never raises.
"""

from __future__ import annotations

import os
from pathlib import Path

_CGROUP_V2_MAX = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V1_MAX = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")


def container_memory_limit_bytes() -> int | None:
    """The container memory ceiling, or None when not running under one."""
    for path in (_CGROUP_V2_MAX, _CGROUP_V1_MAX):
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            # An unreadable cgroup file is not worth failing a run over; it
            # just means no detection.
            continue
        # v2 spells unlimited as "max"; v1 as a sentinel near 2**63. Either
        # way, and for any ceiling at or above physical RAM, there is no
        # container constraint to respect.
        if raw == "max" or not raw.isdigit():
            return None
        value = int(raw)
        physical = physical_memory_bytes()
        if value <= 0 or physical is None or value >= physical:
            return None
        return value
    return None


def physical_memory_bytes() -> int | None:
    """Host RAM, or None where the platform cannot say.

    `os.sysconf` does not exist on Windows. Only reached when a cgroup file was
    found, so in practice this is Linux — the guard keeps the module importable
    and type-checkable everywhere.
    """
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return None
    try:
        return int(sysconf("SC_PAGE_SIZE")) * int(sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError):
        return None
