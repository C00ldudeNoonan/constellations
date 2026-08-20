from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..config.loader import ConfigError
from ..config.source import SourceConfig, validate_file_pattern
from ..hashing import HASH_DIGEST_SIZE
from ..paths import resolve_within_project
from ..versioning import compute_document_id
from .base import DocumentRef, DocumentSource, SourceError, SourceScan

log = logging.getLogger(__name__)

_HASH_CHUNK_SIZE = 1024 * 1024
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_BINARY = getattr(os, "O_BINARY", 0)
_HAS_DESCRIPTOR_WALK = (
    bool(_NOFOLLOW)
    and os.open in os.supports_dir_fd
    and os.scandir in os.supports_fd
)


@dataclass(frozen=True)
class _MatchedFile:
    relative_path: str
    path: Path
    stat_result: os.stat_result
    content_hash: str | None


def _source_dir(source: SourceConfig, project_dir: Path) -> Path:
    return resolve_within_project(
        source.path,
        project_dir,
        surface=f"Source '{source.name}' path",
        external=source.external,
        hint="Set `external: true` on the source to allow it.",
    )


def _effective_pattern(source: SourceConfig) -> tuple[str, ...]:
    try:
        file_pattern = validate_file_pattern(source.file_pattern)
    except ValueError as e:
        raise ConfigError(
            f"Source '{source.name}' has invalid configuration: {e}"
        ) from e
    normalized = file_pattern.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if not parts:
        raise ConfigError(
            f"Source '{source.name}' has invalid configuration: "
            "file_pattern must not be empty"
        )
    return ("**", *parts) if source.recursive else parts


def _pattern_states(
    path_parts: tuple[str, ...], pattern_parts: tuple[str, ...]
) -> set[int]:
    def _closure(states: set[int]) -> set[int]:
        out = set(states)
        pending = list(states)
        while pending:
            index = pending.pop()
            if index < len(pattern_parts) and pattern_parts[index] == "**":
                next_index = index + 1
                if next_index not in out:
                    out.add(next_index)
                    pending.append(next_index)
        return out

    states = _closure({0})
    for part in path_parts:
        next_states: set[int] = set()
        for index in states:
            if index == len(pattern_parts):
                continue
            pattern = pattern_parts[index]
            if pattern == "**":
                next_states.add(index)
            elif fnmatch.fnmatch(part, pattern):
                next_states.add(index + 1)
        states = _closure(next_states)
        if not states:
            break
    return states


def _matches(path_parts: tuple[str, ...], pattern_parts: tuple[str, ...]) -> bool:
    return len(pattern_parts) in _pattern_states(path_parts, pattern_parts)


def _could_match_descendant(
    path_parts: tuple[str, ...], pattern_parts: tuple[str, ...]
) -> bool:
    states = _pattern_states(path_parts, pattern_parts)
    return any(index < len(pattern_parts) for index in states)


def _symlink_error(
    source: SourceConfig, relative_path: str, *, would_traverse: bool
) -> ConfigError:
    if would_traverse:
        return ConfigError(
            f"Source '{source.name}' file_pattern would traverse symlink "
            f"'{relative_path}', which may resolve outside the source root. "
            "stel does not traverse symlinked source directories."
        )
    return ConfigError(
        f"Source '{source.name}' file_pattern matched symlink "
        f"'{relative_path}'. stel does not read symlinked source files."
    )


def _hash_fd(fd: int) -> str:
    digest = hashlib.blake2b(digest_size=HASH_DIGEST_SIZE)
    while chunk := os.read(fd, _HASH_CHUNK_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


def _open_at(name: str, flags: int, *, dir_fd: int) -> int:
    return os.open(name, flags | _NOFOLLOW, dir_fd=dir_fd)


def _walk_fd(
    source: SourceConfig,
    source_dir: Path,
    dir_fd: int,
    relative_parts: tuple[str, ...],
    pattern_parts: tuple[str, ...],
    *,
    include_hash: bool,
) -> Iterator[_MatchedFile]:
    with os.scandir(dir_fd) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            child_parts = (*relative_parts, entry.name)
            relative_path = PurePosixPath(*child_parts).as_posix()
            matches = _matches(child_parts, pattern_parts)
            descend = _could_match_descendant(child_parts, pattern_parts)

            if entry.is_symlink():
                if matches or descend:
                    raise _symlink_error(
                        source,
                        relative_path,
                        would_traverse=descend and not matches,
                    )
                continue

            if entry.is_dir(follow_symlinks=False):
                if not descend:
                    continue
                try:
                    child_fd = _open_at(
                        entry.name, _DIRECTORY_FLAGS, dir_fd=dir_fd
                    )
                except OSError as e:
                    raise ConfigError(
                        f"Source '{source.name}' directory '{relative_path}' "
                        "changed or became a symlink during discovery"
                    ) from e
                try:
                    yield from _walk_fd(
                        source,
                        source_dir,
                        child_fd,
                        child_parts,
                        pattern_parts,
                        include_hash=include_hash,
                    )
                finally:
                    os.close(child_fd)
                continue

            if not matches or not entry.is_file(follow_symlinks=False):
                continue
            try:
                file_fd = _open_at(
                    entry.name, os.O_RDONLY | _BINARY, dir_fd=dir_fd
                )
            except OSError as e:
                raise ConfigError(
                    f"Source '{source.name}' file '{relative_path}' changed "
                    "or became a symlink during discovery"
                ) from e
            try:
                file_stat = os.fstat(file_fd)
                if not stat.S_ISREG(file_stat.st_mode):
                    continue
                content_hash = _hash_fd(file_fd) if include_hash else None
            finally:
                os.close(file_fd)
            yield _MatchedFile(
                relative_path=relative_path,
                path=source_dir.joinpath(*child_parts),
                stat_result=file_stat,
                content_hash=content_hash,
            )


def _open_path_checked(path: Path) -> tuple[int, os.stat_result]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode):
        raise OSError(f"refusing to follow symlink: {path}")
    if not stat.S_ISREG(before.st_mode):
        raise OSError(f"refusing to open non-regular file: {path}")
    fd = os.open(path, os.O_RDONLY | _BINARY | _NOFOLLOW)
    after = os.fstat(fd)
    identity_before = (before.st_dev, before.st_ino)
    identity_after = (after.st_dev, after.st_ino)
    if identity_before != identity_after or not stat.S_ISREG(after.st_mode):
        os.close(fd)
        raise OSError(f"path changed while opening: {path}")
    return fd, after


def _walk_path(
    source: SourceConfig,
    source_dir: Path,
    directory: Path,
    relative_parts: tuple[str, ...],
    pattern_parts: tuple[str, ...],
    *,
    include_hash: bool,
) -> Iterator[_MatchedFile]:
    before = directory.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise ConfigError(
            f"Source '{source.name}' directory changed or became a symlink "
            f"during discovery: {directory}"
        )
    with os.scandir(directory) as entries:
        after = directory.lstat()
        if (
            stat.S_ISLNK(after.st_mode)
            or not stat.S_ISDIR(after.st_mode)
            or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise ConfigError(
                f"Source '{source.name}' directory changed or became a symlink "
                f"during discovery: {directory}"
            )
        for entry in sorted(entries, key=lambda item: item.name):
            child_parts = (*relative_parts, entry.name)
            relative_path = PurePosixPath(*child_parts).as_posix()
            matches = _matches(child_parts, pattern_parts)
            descend = _could_match_descendant(child_parts, pattern_parts)

            if entry.is_symlink():
                if matches or descend:
                    raise _symlink_error(
                        source,
                        relative_path,
                        would_traverse=descend and not matches,
                    )
                continue

            path = directory / entry.name
            if entry.is_dir(follow_symlinks=False):
                if descend:
                    yield from _walk_path(
                        source,
                        source_dir,
                        path,
                        child_parts,
                        pattern_parts,
                        include_hash=include_hash,
                    )
                continue
            if not matches or not entry.is_file(follow_symlinks=False):
                continue
            try:
                file_fd, file_stat = _open_path_checked(path)
            except OSError as e:
                raise ConfigError(
                    f"Source '{source.name}' file '{relative_path}' changed "
                    "or became a symlink during discovery"
                ) from e
            try:
                content_hash = _hash_fd(file_fd) if include_hash else None
            finally:
                os.close(file_fd)
            yield _MatchedFile(
                relative_path=relative_path,
                path=source_dir.joinpath(*child_parts),
                stat_result=file_stat,
                content_hash=content_hash,
            )


def _matched_files(
    source: SourceConfig, source_dir: Path, *, include_hash: bool
) -> list[_MatchedFile]:
    pattern_parts = _effective_pattern(source)
    if _HAS_DESCRIPTOR_WALK:
        try:
            root_fd = os.open(source_dir, _DIRECTORY_FLAGS | _NOFOLLOW)
        except OSError as e:
            raise ConfigError(
                f"Source '{source.name}' root changed or became a symlink "
                "during discovery"
            ) from e
        try:
            matches = list(
                _walk_fd(
                    source,
                    source_dir,
                    root_fd,
                    (),
                    pattern_parts,
                    include_hash=include_hash,
                )
            )
        finally:
            os.close(root_fd)
    else:
        matches = list(
            _walk_path(
                source,
                source_dir,
                source_dir,
                (),
                pattern_parts,
                include_hash=include_hash,
            )
        )
    return sorted(matches, key=lambda match: match.relative_path)


def _ref_parts(ref: DocumentRef) -> tuple[Path, tuple[str, ...]]:
    if ref.path is None:
        raise SourceError(f"Local source reference '{ref.relative_path}' has no path")
    relative_parts = PurePosixPath(ref.relative_path).parts
    if not relative_parts or any(part in {"", ".", ".."} for part in relative_parts):
        raise SourceError(
            f"Local source reference has invalid relative path: {ref.relative_path!r}"
        )
    source_root = ref.path
    for _ in relative_parts:
        source_root = source_root.parent
    expected = source_root.joinpath(*relative_parts)
    if expected != ref.path:
        raise SourceError(
            f"Local source reference path does not match '{ref.relative_path}'"
        )
    return source_root, relative_parts


def _open_ref_fd(ref: DocumentRef) -> int:
    source_root, relative_parts = _ref_parts(ref)
    if _HAS_DESCRIPTOR_WALK:
        try:
            dir_fd = os.open(source_root, _DIRECTORY_FLAGS | _NOFOLLOW)
        except OSError as e:
            raise SourceError(
                f"Local source root changed or became a symlink before fetching "
                f"'{ref.relative_path}'"
            ) from e
        try:
            for part in relative_parts[:-1]:
                try:
                    next_fd = _open_at(part, _DIRECTORY_FLAGS, dir_fd=dir_fd)
                except OSError as e:
                    raise SourceError(
                        f"Local source path '{ref.relative_path}' changed or crossed "
                        "a symlink before fetch"
                    ) from e
                os.close(dir_fd)
                dir_fd = next_fd
            try:
                file_fd = _open_at(
                    relative_parts[-1],
                    os.O_RDONLY | _BINARY,
                    dir_fd=dir_fd,
                )
            except OSError as e:
                raise SourceError(
                    f"Local source file '{ref.relative_path}' changed or became "
                    "a symlink before fetch"
                ) from e
            file_stat = os.fstat(file_fd)
            if not stat.S_ISREG(file_stat.st_mode):
                os.close(file_fd)
                raise SourceError(
                    f"Local source file '{ref.relative_path}' is no longer a regular file"
                )
            return file_fd
        finally:
            os.close(dir_fd)

    path = source_root.joinpath(*relative_parts)
    current = source_root
    try:
        for part in relative_parts[:-1]:
            current /= part
            current_stat = current.lstat()
            if stat.S_ISLNK(current_stat.st_mode) or not stat.S_ISDIR(
                current_stat.st_mode
            ):
                raise SourceError(
                    f"Local source path '{ref.relative_path}' changed or crossed "
                    "a symlink before fetch"
                )
        file_fd, _ = _open_path_checked(path)
        return file_fd
    except OSError as e:
        raise SourceError(
            f"Local source file '{ref.relative_path}' changed or became a symlink "
            "before fetch"
        ) from e


class LocalDocumentSource(DocumentSource):
    """Files on disk under `<project_dir>/<source.path>`."""

    def discover(self, source: SourceConfig, project_dir: Path) -> list[DocumentRef]:
        source_dir = _source_dir(source, project_dir)
        if not source_dir.exists():
            log.info("Source '%s': path does not exist: %s", source.name, source_dir)
            return []
        log.info("Source '%s': scanning %s", source.name, source_dir)
        refs: list[DocumentRef] = []
        for match in _matched_files(source, source_dir, include_hash=True):
            assert match.content_hash is not None
            refs.append(
                DocumentRef(
                    source_name=source.name,
                    relative_path=match.relative_path,
                    document_id=compute_document_id(
                        source.name, match.relative_path
                    ),
                    content_hash=match.content_hash,
                    path=match.path,
                    source_uri=match.path.as_uri(),
                )
            )
        log.info("Source '%s': discovered %d document(s)", source.name, len(refs))
        return refs

    def fetch(self, ref: DocumentRef, work_dir: Path) -> Path:
        file_fd = _open_ref_fd(ref)
        destination_dir = work_dir / ref.document_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / Path(ref.relative_path).name
        digest = hashlib.blake2b(digest_size=HASH_DIGEST_SIZE)
        try:
            with destination.open("wb") as output:
                while chunk := os.read(file_fd, _HASH_CHUNK_SIZE):
                    digest.update(chunk)
                    output.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            destination_dir.rmdir()
            raise
        finally:
            os.close(file_fd)
        if digest.hexdigest() != ref.content_hash:
            destination.unlink(missing_ok=True)
            destination_dir.rmdir()
            raise SourceError(
                f"Local source file '{ref.relative_path}' changed after discovery; "
                "run again to refresh source state"
            )
        return destination

    def scan(self, source: SourceConfig, project_dir: Path) -> SourceScan:
        source_dir = _source_dir(source, project_dir)
        if not source_dir.exists():
            return SourceScan(
                exists=False,
                file_count=0,
                newest_epoch=None,
                newest_name=None,
                message=f"source path does not exist: {source_dir}",
            )
        files = _matched_files(source, source_dir, include_hash=False)
        if not files:
            return SourceScan(
                exists=True,
                file_count=0,
                newest_epoch=None,
                newest_name=None,
                message="no matching files",
            )
        newest = max(files, key=lambda match: match.stat_result.st_mtime)
        return SourceScan(
            exists=True,
            file_count=len(files),
            newest_epoch=newest.stat_result.st_mtime,
            newest_name=newest.relative_path,
        )
