"""`stel serving status` says what a reader gets, in words (issue #617).

`active_generation: -` next to `status: failed` meant three weeks of refused
queries read as an idle index. The `serving:` line is the sentence an operator
would otherwise have to derive from the admission rule. Pure functions over a
ledger entry, so no store or project is opened here.
"""

from __future__ import annotations

import pytest

from stel.cli_services.serving import describe_publisher_claim, describe_serving
from stel.retrieval.coordination import (
    RECOVERY_ERROR_CODE,
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_PUBLISHING,
    STATUS_PUBLISHING_IN_PLACE,
    STATUS_READY,
    STATUS_UNPUBLISHED,
    ServingLedgerEntry,
)
from stel.retrieval.publisher_identity import PublisherIdentity


def _entry(
    status: str,
    *,
    active_generation: str | None = None,
    active_collection: str | None = None,
    safe_error_code: str | None = None,
    publication_id: str | None = None,
    publisher: PublisherIdentity | None = None,
    publisher_heartbeat_epoch: int | None = None,
) -> ServingLedgerEntry:
    return ServingLedgerEntry(
        status=status,
        fencing_token=18,
        publication_id=publication_id,
        expected_code_version="e8c8c918",
        config_fingerprint="d0fa5d3c",
        active_generation=active_generation,
        active_collection=active_collection,
        safe_error_code=safe_error_code,
        rows_inserted=0,
        rows_updated=0,
        rows_skipped=0,
        rows_deleted=0,
        query_leases=0,
        publisher=publisher,
        publisher_heartbeat_epoch=publisher_heartbeat_epoch,
    )


def test_a_ready_index_names_its_generation_and_collection() -> None:
    line = describe_serving(
        _entry(STATUS_READY, active_generation="g17b61b4a", active_collection="ctx__g12fbc")
    )
    assert line == "generation g17b61b4a from ctx__g12fbc"


def test_a_ready_index_on_the_default_collection_says_so() -> None:
    line = describe_serving(_entry(STATUS_READY, active_generation="g17b61b4a"))
    assert line == "generation g17b61b4a from the default collection"


def test_a_degraded_index_says_readers_get_the_previous_generation() -> None:
    """The #617 shape after the fix: the failure is on the row, the
    generation is still served, and the line says both."""
    line = describe_serving(
        _entry(
            STATUS_DEGRADED,
            active_generation="g17b61b4a",
            active_collection="ctx__g12fbc",
            safe_error_code="warehouse_error",
        )
    )
    assert line.startswith("generation g17b61b4a from ctx__g12fbc, degraded")
    assert "(warehouse_error)" in line
    assert "readers get the generation published before it" in line


def test_recovery_reads_as_a_degraded_publish_with_its_own_code() -> None:
    line = describe_serving(
        _entry(
            STATUS_DEGRADED, active_generation="g1", safe_error_code=RECOVERY_ERROR_CODE
        )
    )
    assert f"({RECOVERY_ERROR_CODE})" in line


@pytest.mark.parametrize(
    "entry",
    [
        # The row #617 was filed against: failed, both pointers gone.
        _entry(STATUS_FAILED, safe_error_code="warehouse_error"),
        # A servable status with no generation is still nothing; the
        # admission rule requires both.
        _entry(STATUS_READY),
        _entry(STATUS_DEGRADED, safe_error_code="store_error"),
    ],
)
def test_nothing_is_said_in_words_not_dashes(entry: ServingLedgerEntry) -> None:
    line = describe_serving(entry)
    assert line.startswith("nothing;")
    assert "refused until a publish succeeds" in line
    assert "-" not in line.split(";")[0]


def test_an_unpublished_index_says_it_was_never_published() -> None:
    assert describe_serving(_entry(STATUS_UNPUBLISHED)) == (
        "nothing; the index has never been published"
    )


def test_an_in_place_publish_says_readers_retry_after_it() -> None:
    """The one state where "nothing" is temporary and the operator should
    wait rather than republish: an in-place publisher holds the index and
    kept the pointer, so the fields alone would read as servable."""
    line = describe_serving(
        _entry(
            STATUS_PUBLISHING_IN_PLACE,
            active_generation="g17b61b4a",
            active_collection="ctx__g12fbc",
            publication_id="p1",
        )
    )
    assert line.startswith("nothing while an in-place publish holds the index")
    assert "retry after it completes" in line


def test_a_private_build_keeps_serving_the_live_generation() -> None:
    """`publishing` is the private-generation claim: readers of the live
    generation run alongside it, so the line names what they get."""
    line = describe_serving(
        _entry(STATUS_PUBLISHING, active_generation="g17b61b4a", publication_id="p1")
    )
    assert line == "generation g17b61b4a from the default collection"


def test_the_publisher_line_names_who_holds_the_claim_and_their_last_heartbeat() -> None:
    """`publisher: active` was all the row could say about a process dead for
    forty minutes (issue #621). The line now says which process, where, and
    how long since its last page."""
    entry = _entry(
        STATUS_PUBLISHING_IN_PLACE,
        publication_id="p1",
        publisher=PublisherIdentity(
            host="dagster-user-code",
            pid=4242,
            started_epoch=1_700_000_000,
            label="run-abc",
            namespace="boot/pid:[1]",
        ),
        publisher_heartbeat_epoch=1_700_000_600,
    )
    line = describe_publisher_claim(entry, now_epoch=1_700_003_000)
    assert line.startswith("active: host=dagster-user-code pid=4242")
    assert "label=run-abc" in line
    assert "last heartbeat 40m00s ago" in line


def test_no_claim_is_a_dash_on_the_publisher_line() -> None:
    assert describe_publisher_claim(_entry(STATUS_READY), now_epoch=0) == "-"
