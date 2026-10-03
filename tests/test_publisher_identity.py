"""What a publish claim records about its holder, and what this host can prove
about them (issue #621).

Pure process-level checks with no warehouse: the identity of the current
process, the label normalization, and the three liveness verdicts. The ledger
side -- recording, heartbeat, and recovery acting on the verdict -- is in
`test_serving_coordination.py`.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest

from stel.retrieval.publisher_identity import (
    PUBLISHER_LABEL_ENV,
    Liveness,
    PublisherIdentity,
    describe_publisher,
    local_liveness,
    process_namespace,
    process_started_epoch,
    publisher_label,
)

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="process start markers come from /proc"
)


def test_the_current_identity_is_this_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PUBLISHER_LABEL_ENV, "  dagster run  abc123 ")
    identity = PublisherIdentity.current()
    assert identity.host == socket.gethostname()
    assert identity.pid == os.getpid()
    assert identity.label == "dagster run abc123"
    assert identity.namespace == process_namespace()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("run-1", "run-1"),
        ("two\nlines\there", "two lines here"),
        ("x" * 300, "x" * 128),
    ],
)
def test_a_label_is_one_bounded_line_or_nothing(raw: str | None, expected: str | None) -> None:
    """The label is written to the ledger and printed by `serving status`, so
    it is one line, bounded, and absent rather than blank."""
    assert publisher_label(raw) == expected


@linux_only
def test_this_process_has_a_start_marker_and_a_reaped_one_does_not() -> None:
    assert process_started_epoch(os.getpid()) is not None
    pid = subprocess.Popen([sys.executable, "-c", "pass"])
    pid.wait()
    assert process_started_epoch(pid.pid) is None


@linux_only
def test_liveness_is_alive_for_this_process_and_dead_for_a_reaped_one() -> None:
    """The provable cases. A reaped child is the shape of a crashed build on
    the operator's own host: the PID is gone, so recovery may proceed."""
    assert local_liveness(PublisherIdentity.current()) is Liveness.ALIVE
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    dead = PublisherIdentity(
        host=socket.gethostname(),
        pid=child.pid,
        started_epoch=process_started_epoch(os.getpid()),
        label=None,
        namespace=process_namespace(),
    )
    assert local_liveness(dead) is Liveness.DEAD


@linux_only
def test_a_reused_pid_with_another_start_time_is_dead() -> None:
    """PID reuse is why the start time is recorded at all: the same number
    with a start an hour earlier is a different process, so the one that
    claimed the scope is gone."""
    mine = PublisherIdentity.current()
    assert mine.started_epoch is not None
    earlier = PublisherIdentity(
        host=mine.host,
        pid=mine.pid,
        started_epoch=mine.started_epoch - 3600,
        label=None,
        namespace=mine.namespace,
    )
    assert local_liveness(earlier) is Liveness.DEAD


@linux_only
def test_the_same_hostname_in_another_pid_namespace_is_unknown() -> None:
    """A hostname proves nothing about locality: two containers can be
    configured with the same one, and the PID that is gone here may be alive
    there. Only a matching boot id and PID namespace make the lookup valid;
    anything else keeps the confirmation required (review finding on #637)."""
    mine = PublisherIdentity.current()
    elsewhere = PublisherIdentity(
        host=mine.host,
        pid=mine.pid,
        started_epoch=mine.started_epoch,
        label=None,
        namespace="other-boot/pid:[4026531836]",
    )
    assert local_liveness(elsewhere) is Liveness.UNKNOWN
    unrecorded = PublisherIdentity(
        host=mine.host, pid=mine.pid, started_epoch=mine.started_epoch, label=None, namespace=None
    )
    assert local_liveness(unrecorded) is Liveness.UNKNOWN


def test_liveness_is_unknown_for_another_host_or_without_a_start_marker() -> None:
    """A container's hostname is the container's, so a build inside one is
    unknown to the host even when both share a kernel; and without a start
    marker a PID match proves nothing. Both keep the confirmation required."""
    elsewhere = PublisherIdentity(
        host="some-other-host",
        pid=os.getpid(),
        started_epoch=1,
        label=None,
        namespace="other-boot/pid:[1]",
    )
    assert local_liveness(elsewhere) is Liveness.UNKNOWN
    unmarked = PublisherIdentity(
        host=socket.gethostname(),
        pid=os.getpid(),
        started_epoch=None,
        label=None,
        namespace=process_namespace(),
    )
    assert local_liveness(unmarked) is Liveness.UNKNOWN


def test_no_publisher_reads_as_a_dash() -> None:
    assert describe_publisher(None, heartbeat_epoch=None, now_epoch=1_700_000_000) == "-"


def test_an_active_publisher_names_host_pid_label_and_heartbeat_age() -> None:
    """The line an operator reads before deciding whether to recover: who,
    where, and how long since the last page. The 2026-10-01 row would have
    read `last heartbeat 40m00s ago` instead of `active`."""
    identity = PublisherIdentity(
        host="dagster-user-code",
        pid=4242,
        started_epoch=1_700_000_000,
        label="run-abc",
        namespace="boot/pid:[1]",
    )
    line = describe_publisher(
        identity, heartbeat_epoch=1_700_003_600, now_epoch=1_700_003_600 + 40 * 60
    )
    assert line == (
        "active: host=dagster-user-code pid=4242, started=2023-11-14T22:13:20Z, "
        "label=run-abc, last heartbeat 40m00s ago (2023-11-14T23:13:20Z)"
    )


def test_a_publisher_without_a_start_marker_or_label_says_so_only_by_omission() -> None:
    identity = PublisherIdentity(
        host="laptop", pid=7, started_epoch=None, label=None, namespace=None
    )
    line = describe_publisher(identity, heartbeat_epoch=None, now_epoch=10)
    assert line == "active: host=laptop pid=7, no heartbeat recorded"


@pytest.mark.parametrize(
    ("age", "rendered"),
    [(0, "0s"), (59, "59s"), (60, "1m00s"), (3599, "59m59s"), (3600, "1h00m"), (7_260, "2h01m")],
)
def test_heartbeat_age_reads_in_the_unit_that_matters(age: int, rendered: str) -> None:
    identity = PublisherIdentity(host="h", pid=1, started_epoch=None, label=None, namespace=None)
    line = describe_publisher(identity, heartbeat_epoch=1000, now_epoch=1000 + age)
    assert f"last heartbeat {rendered} ago" in line
