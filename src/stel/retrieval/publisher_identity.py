"""Who holds a serving-scope publish claim, and whether they are still running.

The publish claim used to record when it was acquired and nothing about by
whom (issue #621). `stel serving status` printed `publisher: active` for a
process that had been dead for forty minutes, and deciding whether to run
`recover --owner-terminated` took a five-step procedure outside stel: a
command-line-aware process scan on this host, another inside every container
that could run a build, the orchestrator's run history, asking every human who
might be publishing from another machine, and a second `status` to see whether
the row counts moved.

This module is what the claim records instead: the publisher's host, PID,
process start time and an operator-supplied label, plus a heartbeat the page
loop touches. None of it changes the recovery rule -- there is still no
timeout-based stealing, because a publisher inside a ninety-minute index build
has no heartbeat to give and is not dead -- but it makes the operator's
confirmation an informed one, and in the one case that is provable from here
(the recorded owner is this host and no process with that PID and start time
exists) it lets `recover` proceed without the confirmation.

Process start times come from `/proc` and are therefore Linux-only. Elsewhere
the identity records host and PID, and liveness is unknown: the confirmation
stays required there. A dependency on `psutil` for the other platforms was
considered and declined; the core installation stays lean, and the provable
case is the orchestrator-launched build on the same Linux host as the operator.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

# Operator-controlled, so an orchestrator can tag the claim with its run id. It
# is written to the serving ledger and printed by `stel serving status`, so it
# must never carry a credential; a label is a name, not a secret.
PUBLISHER_LABEL_ENV = "STEL_PUBLISHER_LABEL"
_LABEL_MAX_CHARS = 128

# Two processes whose recorded start differs by less than this are the same
# process: the start marker is derived from kernel clock ticks since boot and
# rounded to whole seconds on the way in.
_START_TOLERANCE_SECONDS = 2


class Liveness(StrEnum):
    """What this host can say about a recorded publisher."""

    ALIVE = "alive"
    DEAD = "dead"
    # Another host, or a platform without a process start marker: nothing
    # here can tell, and the operator's confirmation stays required.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PublisherIdentity:
    """What a publish claim records about the process that made it."""

    host: str
    pid: int
    # Unix seconds, from the kernel's start marker; None where the platform
    # cannot say. Stored as an integer rather than a TIMESTAMP so the value
    # means the same thing on every warehouse and in every session time zone.
    started_epoch: int | None
    label: str | None

    @classmethod
    def current(cls) -> PublisherIdentity:
        pid = os.getpid()
        return cls(
            host=socket.gethostname(),
            pid=pid,
            started_epoch=process_started_epoch(pid),
            label=publisher_label(os.environ.get(PUBLISHER_LABEL_ENV)),
        )


def publisher_label(raw: str | None) -> str | None:
    """Normalize an operator-supplied label: one line, bounded, or nothing."""
    if raw is None:
        return None
    label = " ".join(raw.split())
    if not label:
        return None
    return label[:_LABEL_MAX_CHARS]


def process_started_epoch(pid: int) -> int | None:
    """When `pid` started, in Unix seconds, or None.

    None means either that this platform has no `/proc`, or that no process
    with that PID exists; `local_liveness` tells those apart by asking about
    the current process first.
    """
    stat = Path(f"/proc/{pid}/stat")
    boot = Path("/proc/stat")
    if not stat.exists() or not boot.exists():
        return None
    try:
        fields = stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        boot_lines = boot.read_text(encoding="utf-8").splitlines()
    except OSError:
        # The process exited between the existence check and the read.
        return None
    btime = next((int(line.split()[1]) for line in boot_lines if line.startswith("btime ")), None)
    if btime is None or len(fields) < 20:
        return None
    # Field 22 of /proc/<pid>/stat is the start time in clock ticks since
    # boot; after splitting off "pid (comm)" it is index 19.
    ticks = int(fields[19])
    return btime + ticks // os.sysconf("SC_CLK_TCK")


def local_liveness(identity: PublisherIdentity) -> Liveness:
    """Whether the recorded publisher is provably alive or dead on this host.

    Provable only for a publisher that recorded a start marker on *this* host:
    the PID is looked up the same way it recorded itself, and a different
    start time means the PID has been reused by another process. A publisher
    on another host, or one that could not record a start marker, is unknown
    -- a container's hostname is the container's, so a build inside one is
    unknown to the host even when both share a kernel.
    """
    if identity.host != socket.gethostname() or identity.started_epoch is None:
        return Liveness.UNKNOWN
    if process_started_epoch(os.getpid()) is None:
        return Liveness.UNKNOWN
    started = process_started_epoch(identity.pid)
    if started is None:
        return Liveness.DEAD
    if abs(started - identity.started_epoch) <= _START_TOLERANCE_SECONDS:
        return Liveness.ALIVE
    return Liveness.DEAD


def describe_publisher(
    identity: PublisherIdentity | None,
    *,
    heartbeat_epoch: int | None,
    now_epoch: int,
) -> str:
    """One line for `stel serving status`: who holds the claim, and how long
    since they were last heard from. "-" when nobody does."""
    if identity is None:
        return "-"
    parts = [f"active: host={identity.host} pid={identity.pid}"]
    if identity.started_epoch is not None:
        parts.append(f"started={_iso(identity.started_epoch)}")
    if identity.label:
        parts.append(f"label={identity.label}")
    if heartbeat_epoch is None:
        parts.append("no heartbeat recorded")
    else:
        age = max(0, now_epoch - heartbeat_epoch)
        parts.append(f"last heartbeat {_age(age)} ago ({_iso(heartbeat_epoch)})")
    return parts[0] + (", " + ", ".join(parts[1:]) if len(parts) > 1 else "")


def _iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _age(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
