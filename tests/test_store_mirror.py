"""A LanceDB store's mirror is a byte copy that is readable at every instant (#666).

astrolabe read its prod store from `gs://` on a host outside GCP and paid 1.43 TB
of egress for it. #676 let a store declare an identity so the bytes can move to
local disk; this is the other half -- a copy kept in the bucket, so moving off it
does not mean giving up the copy a fresh host is restored from.

The copy is files, not rows: rewriting rows into the mirror would rebuild every
index there. These pin what makes a file copy safe to rely on -- the destination
is a readable table whenever the copy stops, a rerun finishes what an interrupted
one started, what the primary prunes leaves the mirror too, and a restore never
overwrites the primary it writes into.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
import pytest
from lancedb.index import BTree

from stel.cli_services.serving import describe_identity, describe_mirror
from stel.execution.mirror import mirror_restore_refusal
from stel.retrieval import LanceDBStore, RetrievalError, StoreRole, lance_mirror, parse_store_config
from stel.retrieval.base import RetrievalConfigError, mirror_fingerprint
from stel.retrieval.coordination import STATUS_READY, ServingLedgerEntry
from stel.retrieval.lance_mirror import PARTIAL_SUFFIX, copy_table, local_tree
from stel.retrieval.lancedb import LanceDBConfig

_NAME = "econ__prod__chunks"


def _lance(path: str, **extra: object) -> LanceDBConfig:
    parsed = parse_store_config({"type": "lancedb", "path": path, **extra})
    assert isinstance(parsed, LanceDBConfig)
    return parsed


def _table_with_history(root: Path) -> Any:
    """A table with several versions, an index, and a deletion: every kind of
    file a Lance table directory holds."""
    table = lancedb.connect(str(root)).create_table(
        _NAME, pa.table({"id": ["a", "b"], "v": [1, 2]})
    )
    table.add(pa.table({"id": ["c", "d"], "v": [3, 4]}))
    table.delete("id = 'b'")
    table.create_index("id", config=BTree())
    return table


def _files(root: Path) -> dict[str, int]:
    table_dir = root / f"{_NAME}.lance"
    return {
        path.relative_to(table_dir).as_posix(): path.stat().st_size
        for path in table_dir.rglob("*")
        if path.is_file()
    }


def _rows(root: Path) -> list[str]:
    table = lancedb.connect(str(root)).open_table(_NAME)
    return sorted(table.to_arrow().column("id").to_pylist())


# --- configuration --------------------------------------------------------


def test_a_mirror_does_not_change_the_store_identity() -> None:
    """Adding, moving or removing a mirror must not re-key the store: every
    state scope and ledger row is keyed on its identity, and a change there is
    every published index reading as unpublished."""
    plain = _lance("/srv/lancedb")
    mirrored = _lance("/srv/lancedb", mirror="gs://bucket/lancedb")

    def identity(config: LanceDBConfig) -> str:
        store = LanceDBStore(
            config, project_name="econ", target_name="prod", alias="primary",
            role=StoreRole.INSPECT,
        )
        return store.safe_descriptor().safe_target_identity

    assert identity(mirrored) == identity(plain)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        # The copy reads the primary as plain files, which a bucket's
        # `storage_options_env` credentials never reach.
        ({"path": "gs://bucket/primary", "mirror": "s3://bucket/m"}, "local primary"),
        ({"path": "/srv/lancedb", "mirror": "az://container/m"}, "az:// is not supported"),
        ({"path": "/srv/lancedb", "mirror": "gs://"}, "must include a bucket"),
        ({"path": "/srv/lancedb", "mirror": "/srv/lancedb"}, "must not be the store's own path"),
    ],
)
def test_a_mirror_the_copy_cannot_serve_is_refused_at_the_profile(
    extra: dict[str, str], message: str
) -> None:
    with pytest.raises(RetrievalConfigError, match=message):
        parse_store_config({"type": "lancedb", **extra})


def test_a_relative_local_mirror_resolves_against_the_project(tmp_path: Path) -> None:
    config = _lance("target/lancedb", mirror="target/mirror").absolutize(tmp_path)
    assert config.mirror_location() == (tmp_path / "target" / "mirror").resolve().as_posix()
    # A cloud mirror is canonicalized like a cloud path, so `gcs://` and
    # `gs://` are one mirror to the ledger rather than two.
    assert _lance("/srv/l", mirror="GCS://bucket/m/").mirror_location() == "gs://bucket/m"


# --- the copy ---------------------------------------------------------------


@pytest.fixture
def primary(tmp_path: Path) -> Path:
    root = tmp_path / "primary"
    _table_with_history(root)
    return root


def test_a_copy_is_faithful_and_carries_the_index(primary: Path, tmp_path: Path) -> None:
    mirror = tmp_path / "mirror"
    copied = copy_table(local_tree(str(primary)), local_tree(str(mirror)), _NAME, mode="mirror")

    assert _files(mirror) == _files(primary)
    assert copied.files_copied == len(_files(primary))
    assert _rows(mirror) == ["a", "c", "d"]
    table = lancedb.connect(str(mirror)).open_table(_NAME)
    assert table.version == lancedb.connect(str(primary)).open_table(_NAME).version
    # The index came across as files; nothing rebuilt it.
    assert [index.name for index in table.list_indices()] == ["id_idx"]


@pytest.fixture
def interrupt_after(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Make the copy die after `n` files have landed."""
    real = lance_mirror._copy_file
    state = {"remaining": 0}
    lock = threading.Lock()

    def copy_file(*args: Any) -> None:
        with lock:
            if state["remaining"] <= 0:
                raise OSError("simulated interruption")
            state["remaining"] -= 1
        real(*args)

    def arm(n: int) -> None:
        state["remaining"] = n

    monkeypatch.setattr(lance_mirror, "_copy_file", copy_file)
    yield arm


def test_every_instant_of_a_copy_is_a_readable_table(
    primary: Path, tmp_path: Path, interrupt_after: Any
) -> None:
    """Stop the copy after every possible number of files. Whatever has
    landed must open as the table at some version, or not be a table yet --
    never a manifest naming a file that is not there."""
    total = len(_files(primary))
    for landed in range(total):
        mirror = tmp_path / f"mirror_{landed}"
        interrupt_after(landed)
        with pytest.raises(RetrievalError, match="mirror_copy_failed"):
            copy_table(local_tree(str(primary)), local_tree(str(mirror)), _NAME, mode="mirror")
        manifests = [
            name for name in _files(mirror) if name.startswith("_versions/")
        ] if (mirror / f"{_NAME}.lance").exists() else []
        if any(name.endswith(".manifest") for name in manifests):
            assert _rows(mirror) == ["a", "c", "d"], landed


def test_an_interrupted_copy_is_finished_by_the_next(
    primary: Path, tmp_path: Path, interrupt_after: Any
) -> None:
    mirror = tmp_path / "mirror"
    interrupt_after(3)
    with pytest.raises(RetrievalError):
        copy_table(local_tree(str(primary)), local_tree(str(mirror)), _NAME, mode="mirror")
    interrupt_after(10_000)

    resumed = copy_table(local_tree(str(primary)), local_tree(str(mirror)), _NAME, mode="mirror")
    assert resumed.files_copied == len(_files(primary)) - 3
    assert _files(mirror) == _files(primary)
    # And converged: a third run has only the version pointer to refresh.
    again = copy_table(local_tree(str(primary)), local_tree(str(mirror)), _NAME, mode="mirror")
    assert again.files_copied == 1
    assert again.files_deleted == 0


def test_a_torn_file_in_the_mirror_is_copied_again(primary: Path, tmp_path: Path) -> None:
    mirror = tmp_path / "mirror"
    copy_table(local_tree(str(primary)), local_tree(str(mirror)), _NAME, mode="mirror")
    data = next((mirror / f"{_NAME}.lance" / "data").iterdir())
    data.write_bytes(data.read_bytes()[:10])
    (data.parent / f"leftover{PARTIAL_SUFFIX}").write_bytes(b"x")

    copy_table(local_tree(str(primary)), local_tree(str(mirror)), _NAME, mode="mirror")
    assert _files(mirror) == _files(primary)


def test_what_the_primary_prunes_leaves_the_mirror(primary: Path, tmp_path: Path) -> None:
    mirror = tmp_path / "mirror"
    copy_table(local_tree(str(primary)), local_tree(str(mirror)), _NAME, mode="mirror")
    before = set(_files(mirror))
    table = lancedb.connect(str(primary)).open_table(_NAME)
    table.optimize(cleanup_older_than=timedelta(0))
    assert set(_files(primary)) != before

    synced = copy_table(local_tree(str(primary)), local_tree(str(mirror)), _NAME, mode="mirror")
    assert synced.files_deleted == len(before - set(_files(primary)))
    assert _files(mirror) == _files(primary)
    assert _rows(mirror) == ["a", "c", "d"]


def test_a_restore_never_overwrites_a_primary_that_diverged(
    primary: Path, tmp_path: Path
) -> None:
    mirror = tmp_path / "mirror"
    copy_table(local_tree(str(primary)), local_tree(str(mirror)), _NAME, mode="mirror")
    diverged = next((primary / f"{_NAME}.lance" / "data").iterdir())
    original = diverged.read_bytes()
    diverged.write_bytes(original + b"local change")

    with pytest.raises(RetrievalError, match="mirror_restore_conflict"):
        copy_table(local_tree(str(mirror)), local_tree(str(primary)), _NAME, mode="restore")
    assert diverged.read_bytes() == original + b"local change"


def test_a_restore_fills_only_what_the_primary_lacks(primary: Path, tmp_path: Path) -> None:
    mirror = tmp_path / "mirror"
    copy_table(local_tree(str(primary)), local_tree(str(mirror)), _NAME, mode="mirror")
    fresh = tmp_path / "fresh"
    first = copy_table(local_tree(str(mirror)), local_tree(str(fresh)), _NAME, mode="restore")
    assert first.files_copied == len(_files(primary))
    assert _rows(fresh) == ["a", "c", "d"]

    second = copy_table(local_tree(str(mirror)), local_tree(str(fresh)), _NAME, mode="restore")
    assert (second.files_copied, second.files_deleted) == (0, 0)


def test_a_native_error_leaves_as_its_type_only(primary: Path, tmp_path: Path) -> None:
    """Filesystem errors quote bucket paths and response bodies; neither may
    reach a log line or run_results.json, through the message or the cause."""
    # A file where the mirror's directory has to go: the native error that
    # follows names the path.
    blocked = tmp_path / "sensitive-bucket-name"
    blocked.write_text("not a directory", encoding="utf-8")
    with pytest.raises(RetrievalError) as raised:
        copy_table(local_tree(str(primary)), local_tree(str(blocked)), _NAME, mode="mirror")
    assert "mirror_copy_failed" in str(raised.value)
    for text in (str(raised.value), str(raised.value.__cause__)):
        assert "sensitive-bucket-name" not in text
        assert os.fspath(tmp_path) not in text
    assert raised.value.__context__ is None


# --- what the ledger says ---------------------------------------------------


def _entry(
    *,
    active_generation: str | None = "g2",
    mirror_generation: str | None = None,
    mirror_target: str | None = None,
    mirrored_epoch: int | None = None,
) -> ServingLedgerEntry:
    return ServingLedgerEntry(
        status=STATUS_READY,
        fencing_token=3,
        publication_id=None,
        expected_code_version="cv",
        config_fingerprint="cf",
        active_generation=active_generation,
        active_collection=None,
        safe_error_code=None,
        progress_note=None,
        rows_inserted=0,
        rows_updated=0,
        rows_skipped=0,
        rows_deleted=0,
        query_leases=0,
        publisher=None,
        publisher_heartbeat_epoch=None,
        mirror_generation=mirror_generation,
        mirror_target=mirror_target,
        mirrored_epoch=mirrored_epoch,
    )


_MIRROR = "gs://bucket/lancedb"
_TARGET = mirror_fingerprint(_MIRROR)


@pytest.mark.parametrize(
    ("entry", "refusal"),
    [
        (_entry(active_generation=None), "serves no generation"),
        (_entry(), "no record of this mirror"),
        # Synced, but to a mirror the profile no longer names.
        (
            _entry(mirror_generation="g2", mirror_target=mirror_fingerprint("gs://old/l")),
            "no record of this mirror",
        ),
        # Restoring an older generation would serve rows the publication
        # state, which describes g2, does not describe.
        (_entry(mirror_generation="g1", mirror_target=_TARGET), "older generation"),
        (_entry(mirror_generation="g2", mirror_target=_TARGET), None),
    ],
)
def test_a_restore_is_refused_unless_the_mirror_holds_the_served_generation(
    entry: ServingLedgerEntry, refusal: str | None
) -> None:
    found = mirror_restore_refusal(entry, mirror_target=_TARGET)
    if refusal is None:
        assert found is None
    else:
        assert found is not None and refusal in found


@pytest.mark.parametrize(
    ("entry", "says"),
    [
        (_entry(), "never synced"),
        (
            _entry(mirror_generation="g2", mirror_target=mirror_fingerprint("gs://old/l")),
            "never synced",
        ),
        (
            _entry(mirror_generation="g2", mirror_target=_TARGET, mirrored_epoch=1_760_000_000),
            "holds the served generation, synced 2025-10-09T08:53:20Z",
        ),
        (_entry(mirror_generation="g1", mirror_target=_TARGET), "behind: holds generation g1"),
    ],
)
def test_the_mirror_line_says_whether_it_can_restore_what_is_served(
    entry: ServingLedgerEntry, says: str
) -> None:
    assert says in describe_mirror(entry, mirror=_MIRROR)


def test_no_mirror_is_a_dash_and_the_identity_line_says_where_it_came_from() -> None:
    assert describe_mirror(_entry(), mirror=None) == "-"
    assert describe_identity(None) == "derived from the store location"
    assert describe_identity("econ-prod") == "econ-prod (declared)"
