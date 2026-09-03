"""Small, value-free memory samples for diagnosing an abruptly killed publish.

RSS includes native store allocations that Arrow's allocator does not track.
Linux exposes it without an optional dependency; other platforms still report
Arrow allocation. These are observations, not a guarantee against SIGKILL.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa

_PROCESS_STATUS = Path("/proc/self/status")


def resident_bytes(status_path: Path) -> int | None:
    """Read Linux RSS when available; diagnostics must not fail publication."""
    try:
        if not status_path.is_file():
            return None
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split()
        if len(fields) == 3 and fields[0] == "VmRSS:" and fields[1].isdigit():
            return int(fields[1]) * 1024 if fields[2] == "kB" else None
    return None


def log_publication_memory(
    logger: logging.Logger, model_name: str, *, phase: str, batch: int
) -> None:
    if not logger.isEnabledFor(logging.INFO):
        return
    logger.info(
        "%s: publication memory phase=%s batch=%d rss_bytes=%s arrow_bytes=%d",
        model_name, phase, batch, resident_bytes(_PROCESS_STATUS),
        pa.total_allocated_bytes(),
    )
