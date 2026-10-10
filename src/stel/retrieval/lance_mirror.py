"""A byte copy of a Lance table directory between two filesystems (issue #666).

A LanceDB store's mirror is not a second LanceDB store that rows are written
into. Writing rows would rebuild every index at the mirror -- the ANN build is
the most expensive step a publish has -- and would make the mirror a different
table that merely holds the same rows. A Lance table is a directory of files
that are written once and never modified, plus manifests that name them, so
the faithful copy is a file copy: the mirror then holds the same versions, the
same indices and the same generation the ledger vouched for.

**Every instant of a copy is a readable table.** Files go in three passes:
data, index, deletion and transaction files first; then the version manifests
that reference them; then the one mutable pointer Lance keeps as a hint to the
newest version. A copy interrupted anywhere leaves the destination at a version
whose files are all present, because no manifest arrives before what it names.
Pruning runs the other way -- manifests that no longer exist at the source are
removed before the files only they referenced.

**The source is listed once.** Everything copied comes from that one listing,
so a version committed while the copy runs is simply not part of it: its
manifest was not listed, and every manifest that was listed names only files
that existed when it was written, which is before the listing.

**Re-running converges.** A file already present at the destination with the
source's size is skipped; one with a different size is a copy an earlier run
did not finish, and is copied again. On a local destination each file is
written under a temporary name and renamed into place, because a local write
can be torn where an object-store upload cannot.

Restoring is the same copy in the other direction with one difference: it
never replaces or removes anything at the destination. A size that differs
there is not a torn copy of this mirror but a table that has diverged from it,
and the restore refuses rather than overwrite the operator's primary.

Native filesystem errors quote bucket paths and response bodies, so they leave
this module as a type name only, like every other store error.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import pyarrow.fs as pafs

from .base import MirrorCopy, RetrievalError, sanitized_retrieval_cause

log = logging.getLogger(__name__)

# Copies are network-bound against an object store, so a handful in flight is
# what makes a multi-gigabyte first sync finish in minutes rather than hours.
# Bounded so a table of thousands of small files does not open thousands of
# connections at once.
_COPY_WORKERS = 8
# Suffix of a file being written to a local destination. Never listed as part
# of a table: a source listing skips it, and a mirror prune removes it.
PARTIAL_SUFFIX = ".stel-partial"
# The mutable files in a Lance table directory: a hint naming the newest
# version (v2 manifest naming) and the v1 equivalent. Every other file is
# written once and never changed.
_POINTER_NAMES = frozenset({"latest_version_hint.json", "_latest.manifest"})
_TABLE_SUFFIX = ".lance"

CopyMode = Literal["mirror", "restore"]


@dataclass(frozen=True)
class TreeLocation:
    """A directory of Lance tables on one filesystem."""

    filesystem: pafs.FileSystem
    root: str

    def table_dir(self, collection: str) -> str:
        return f"{self.root.rstrip('/')}/{collection}{_TABLE_SUFFIX}"

    @property
    def is_local(self) -> bool:
        return isinstance(self.filesystem, pafs.LocalFileSystem)


def local_tree(path: str) -> TreeLocation:
    """A local directory.

    `os.path.abspath` rather than `Path.resolve()`: the filesystem layer needs
    an absolute path, and normalizing lexically is enough to get one without
    following a symlink the operator put in the profile on purpose.
    """
    return TreeLocation(pafs.LocalFileSystem(), Path(os.path.abspath(path)).as_posix())


def cloud_tree(uri: str) -> TreeLocation:
    """An object-store prefix, reached with the environment's default credentials.

    Application Default Credentials for gs://, the AWS default chain for s3://.
    No credential is read or held here: the filesystem resolves its own at
    first use, exactly as `gcloud storage` or the AWS CLI would.
    """
    filesystem: pafs.FileSystem | None = None
    root = ""
    failure: RetrievalError | None = None
    try:
        filesystem, root = pafs.FileSystem.from_uri(uri)
    except Exception as error:
        failure = _failed("open the mirror filesystem", "mirror_filesystem_failed", error)
    if failure is not None:
        raise failure
    assert filesystem is not None
    return TreeLocation(filesystem, root)


def copy_table(
    source: TreeLocation,
    destination: TreeLocation,
    collection: str,
    *,
    mode: CopyMode,
) -> MirrorCopy:
    """Copy one table directory, in the order that keeps the destination readable.

    `mode` is required rather than defaulted: a mirror sync replaces torn files
    and removes what the source pruned, and a restore must do neither to the
    primary it writes into. Which one a caller wants is never obvious from the
    call, and the wrong one either corrupts a primary or leaves a mirror to
    grow without bound.
    """
    source_dir = source.table_dir(collection)
    destination_dir = destination.table_dir(collection)
    source_files = {
        relative: size
        for relative, size in _list_files(source, source_dir).items()
        if not relative.endswith(PARTIAL_SUFFIX)
    }
    if not source_files:
        raise RetrievalError(
            f"Collection '{collection}' has no files to copy at the source "
            "(code=mirror_source_missing)"
        )
    destination_files = _list_files(destination, destination_dir)
    pending: list[list[str]] = [[], [], []]
    for relative, size in sorted(source_files.items()):
        present = destination_files.get(relative)
        phase = _phase(relative)
        if present is None:
            pending[phase].append(relative)
        elif phase == 2:
            # The pointer is the one file a copy is expected to replace; a
            # restore leaves the primary's own alone, since Lance treats it
            # as a hint and finds the newest manifest without it.
            if mode == "mirror":
                pending[phase].append(relative)
        elif present != size:
            if mode == "restore":
                raise RetrievalError(
                    f"Collection '{collection}' at the destination has a file the "
                    "mirror holds at a different size; the two have diverged, and "
                    "a restore never overwrites the primary "
                    "(code=mirror_restore_conflict)"
                )
            pending[phase].append(relative)

    def copy(relative: str) -> int:
        _copy_file(source, destination, f"{source_dir}/{relative}", f"{destination_dir}/{relative}")
        return source_files[relative]

    copied = 0
    copied_bytes = 0
    for phase_files in pending:
        sizes = _run_all(copy, phase_files, operation="copy a table file")
        copied += len(sizes)
        copied_bytes += sum(sizes)
    deleted = 0
    if mode == "mirror":
        stale = [relative for relative in destination_files if relative not in source_files]
        # Manifests and pointers first, so no version the destination still
        # names ever loses a file it references.
        stale.sort(key=lambda relative: -_phase(relative))
        for relative in stale:
            _guarded(
                lambda target=f"{destination_dir}/{relative}": destination.filesystem.delete_file(
                    target
                ),
                operation="remove a pruned file from the mirror",
                code="mirror_prune_failed",
            )
            deleted += 1
    return MirrorCopy(files_copied=copied, bytes_copied=copied_bytes, files_deleted=deleted)


def table_names(location: TreeLocation) -> tuple[str, ...]:
    """Every table directory directly under `location`, by collection name."""
    infos = _guarded(
        lambda: location.filesystem.get_file_info(
            pafs.FileSelector(location.root, recursive=False, allow_not_found=True)
        ),
        operation="list the mirror",
        code="mirror_list_failed",
    )
    names = (
        PurePosixPath(info.path).name
        for info in infos
        if info.type == pafs.FileType.Directory
    )
    return tuple(
        sorted(name[: -len(_TABLE_SUFFIX)] for name in names if name.endswith(_TABLE_SUFFIX))
    )


def drop_table(location: TreeLocation, collection: str) -> None:
    """Remove one table directory and everything in it."""
    _guarded(
        lambda: location.filesystem.delete_dir(location.table_dir(collection)),
        operation="remove a retired generation from the mirror",
        code="mirror_retire_failed",
    )


def _phase(relative: str) -> int:
    """0 for files nothing points at yet, 1 for manifests, 2 for the pointer."""
    path = PurePosixPath(relative)
    if path.name in _POINTER_NAMES:
        return 2
    if path.parts[0] == "_versions" and path.suffix == ".manifest":
        return 1
    return 0


def _list_files(location: TreeLocation, directory: str) -> dict[str, int]:
    infos = _guarded(
        lambda: location.filesystem.get_file_info(
            pafs.FileSelector(directory, recursive=True, allow_not_found=True)
        ),
        operation="list a table directory",
        code="mirror_list_failed",
    )
    prefix = f"{directory}/"
    return {
        info.path[len(prefix) :]: int(info.size)
        for info in infos
        if info.type == pafs.FileType.File and info.path.startswith(prefix)
    }


def _copy_file(
    source: TreeLocation, destination: TreeLocation, source_path: str, destination_path: str
) -> None:
    if destination.is_local:
        # A torn local write would leave a short file under the final name,
        # and a short manifest is a corrupt newest version. Rename is atomic.
        destination.filesystem.create_dir(
            str(PurePosixPath(destination_path).parent), recursive=True
        )
        partial = f"{destination_path}{PARTIAL_SUFFIX}"
        pafs.copy_files(
            source_path,
            partial,
            source_filesystem=source.filesystem,
            destination_filesystem=destination.filesystem,
        )
        destination.filesystem.move(partial, destination_path)
        return
    # An object-store upload is visible whole or not at all.
    pafs.copy_files(
        source_path,
        destination_path,
        source_filesystem=source.filesystem,
        destination_filesystem=destination.filesystem,
    )


def _run_all(
    work: Callable[[str], int], items: Iterable[str], *, operation: str
) -> list[int]:
    """Run `work` over `items` concurrently, failing on the first error."""
    items = list(items)
    if not items:
        return []
    failure: RetrievalError | None = None
    try:
        with ThreadPoolExecutor(max_workers=min(_COPY_WORKERS, len(items))) as pool:
            return list(pool.map(work, items))
    except Exception as error:
        failure = _failed(operation, "mirror_copy_failed", error)
    raise failure


def _guarded[T](call: Callable[[], T], *, operation: str, code: str) -> T:
    failure: RetrievalError | None = None
    try:
        return call()
    except Exception as error:
        failure = _failed(operation, code, error)
    raise failure


def _failed(operation: str, code: str, error: Exception) -> RetrievalError:
    """The artifact-safe failure for a native filesystem error.

    Raised by the caller outside its except block, so the native exception --
    which quotes bucket paths and, for an object store, response bodies -- is
    not retained as `__context__` either. The full exception goes to the
    DEBUG log only, as for the store itself (issue #590).
    """
    log.debug("Mirror operation %r failed", operation, exc_info=error)
    failure = RetrievalError(
        f"Store mirror could not {operation} [{type(error).__name__}] (code={code})"
    )
    failure.__cause__ = sanitized_retrieval_cause(error)
    return failure

