"""The first-party polling loop behind `stel run --watch` (ALE-628, #798).

It replaced `watchfiles`, so these pin what that dependency used to promise:
created, modified and deleted files are noticed, a burst yields once, and a
symlink inside a watched tree is not followed.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from stel.cli_services.watch import snapshot_paths, watch_changes


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")


def _writer(path: Path, text: str) -> Callable[[], None]:
    def write() -> None:
        path.write_text(text, encoding="utf-8")

    return write


def _drive(root: Path, mutations: list[Callable[[], None] | None]) -> tuple[int, list[int]]:
    """Run the loop; each sleep applies the next mutation, then Ctrl-C after the last.

    Returns the number of yields and the sleep count at which each occurred.
    """
    calls = 0
    yields: list[int] = []

    def sleep(_: float) -> None:
        nonlocal calls
        if calls >= len(mutations):
            raise KeyboardInterrupt
        step = mutations[calls]
        calls += 1
        if step is not None:
            step()

    try:
        for _ in watch_changes([root], interval=0.0, sleep=sleep):
            yields.append(calls)
    except KeyboardInterrupt:
        pass
    return len(yields), yields


def test_snapshot_records_files_recursively(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("bb", encoding="utf-8")
    snap = snapshot_paths([tmp_path])
    assert set(snap) == {str(tmp_path / "a.txt"), str(tmp_path / "sub" / "b.txt")}
    assert snap[str(tmp_path / "sub" / "b.txt")][1] == 2


def test_snapshot_accepts_a_file_path(tmp_path: Path) -> None:
    f = tmp_path / "one.txt"
    f.write_text("x", encoding="utf-8")
    assert set(snapshot_paths([f])) == {str(f)}


def test_snapshot_of_a_missing_path_is_empty(tmp_path: Path) -> None:
    assert snapshot_paths([tmp_path / "gone"]) == {}


def test_created_file_is_detected(tmp_path: Path) -> None:
    count, _ = _drive(tmp_path, [None, _writer(tmp_path / "new.txt", "n"), None, None])
    assert count == 1


def test_modified_file_is_detected(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text("one", encoding="utf-8")
    count, _ = _drive(tmp_path, [None, _writer(f, "three"), None, None])
    assert count == 1


def test_mtime_only_change_is_detected(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text("same", encoding="utf-8")
    before = f.stat().st_mtime_ns

    def touch() -> None:
        os.utime(f, ns=(before + 5_000_000_000, before + 5_000_000_000))

    count, _ = _drive(tmp_path, [None, touch, None, None])
    assert count == 1


def test_deleted_file_is_detected(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text("x", encoding="utf-8")
    count, _ = _drive(tmp_path, [None, f.unlink, None, None])
    assert count == 1


def test_no_change_never_yields(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("x", encoding="utf-8")
    count, _ = _drive(tmp_path, [None, None, None])
    assert count == 0


def test_burst_of_writes_yields_once_after_the_tree_settles(tmp_path: Path) -> None:
    count, yields = _drive(
        tmp_path,
        [
            None,
            _writer(tmp_path / "a", "a"),
            _writer(tmp_path / "b", "bb"),
            _writer(tmp_path / "c", "ccc"),
            None,
            None,
        ],
    )
    assert count == 1
    # Yielded only once the final write had been quiet for a full interval.
    assert yields == [5]


def test_two_separate_changes_yield_twice(tmp_path: Path) -> None:
    count, _ = _drive(
        tmp_path,
        [_writer(tmp_path / "a", "a"), None, None, _writer(tmp_path / "b", "bb"), None, None],
    )
    assert count == 2


def test_keyboard_interrupt_propagates_out_of_the_loop(tmp_path: Path) -> None:
    def sleep(_: float) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        next(watch_changes([tmp_path], interval=0.0, sleep=sleep))


def test_symlinked_directory_is_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s", encoding="utf-8")
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "real.txt").write_text("r", encoding="utf-8")
    _symlink_or_skip(watched / "link", outside)
    assert set(snapshot_paths([watched])) == {str(watched / "real.txt")}

    count, _ = _drive(watched, [None, _writer(outside / "later.txt", "l"), None, None])
    assert count == 0


def test_symlinked_file_is_not_followed(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("t", encoding="utf-8")
    watched = tmp_path / "watched"
    watched.mkdir()
    _symlink_or_skip(watched / "link.txt", target)
    assert snapshot_paths([watched]) == {}
