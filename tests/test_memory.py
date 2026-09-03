"""The container memory reader, shared since issue #476.

Behavior unchanged from where it lived in the DuckDB adapter (issue #412);
the DuckDB tests still exercise the 75% budget on top of it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from stel import memory


def _mount(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, v2: str | None, v1: str | None,
    physical: int | None = 64 * 1024**3,
) -> None:
    absent = tmp_path / "absent"
    mounts = (("memory.max", v2, "_CGROUP_V2_MAX"), ("limit", v1, "_CGROUP_V1_MAX"))
    for name, contents, attr in mounts:
        if contents is None:
            monkeypatch.setattr(memory, attr, absent)
        else:
            path = tmp_path / name
            path.write_text(contents, encoding="utf-8")
            monkeypatch.setattr(memory, attr, path)
    monkeypatch.setattr(memory, "physical_memory_bytes", lambda: physical)


def test_a_v2_ceiling_below_physical_ram_is_the_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mount(monkeypatch, tmp_path, v2=str(20 * 1024**3), v1=None)
    assert memory.container_memory_limit_bytes() == 20 * 1024**3


def test_v1_is_read_when_v2_is_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _mount(monkeypatch, tmp_path, v2=None, v1=str(8 * 1024**3))
    assert memory.container_memory_limit_bytes() == 8 * 1024**3


@pytest.mark.parametrize(
    "contents,physical,why",
    [
        ("max", 64 * 1024**3, "v2 spells unlimited as max"),
        (str(2**63 - 1), 64 * 1024**3, "v1's unlimited sentinel"),
        (str(128 * 1024**3), 64 * 1024**3, "a ceiling above physical RAM is not a constraint"),
        ("0", 64 * 1024**3, "a zero ceiling is not a real one"),
        (str(4 * 1024**3), None, "without physical RAM there is no way to tell"),
        ("garbage", 64 * 1024**3, "an unparseable file is no detection"),
    ],
)
def test_no_constraint_is_reported_as_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, contents: str, physical: int | None, why: str
) -> None:
    _mount(monkeypatch, tmp_path, v2=contents, v1=None, physical=physical)
    assert memory.container_memory_limit_bytes() is None, why


def test_no_cgroup_files_means_no_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mount(monkeypatch, tmp_path, v2=None, v1=None)
    assert memory.container_memory_limit_bytes() is None
