from __future__ import annotations

import logging
from pathlib import Path

import pytest

from stel.retrieval import publication_memory


def test_rss_sample_is_numeric_and_excludes_other_process_fields(tmp_path: Path) -> None:
    status = tmp_path / "status"
    status.write_text("Name:\tprivate-sentinel\nVmRSS:\t1234 kB\n", encoding="utf-8")
    assert publication_memory.resident_bytes(status) == 1234 * 1024


@pytest.mark.parametrize("contents", ["", "VmRSS: invalid kB", "VmRSS: 3 unknown"])
def test_malformed_rss_is_unavailable(tmp_path: Path, contents: str) -> None:
    status = tmp_path / "status"
    status.write_text(contents, encoding="utf-8")
    assert publication_memory.resident_bytes(status) is None
    assert publication_memory.resident_bytes(tmp_path / "absent") is None


def test_memory_log_contains_only_safe_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    status = tmp_path / "status"
    status.write_text("Name: private-sentinel\nVmRSS: 10 kB\n", encoding="utf-8")
    monkeypatch.setattr(publication_memory, "_PROCESS_STATUS", status)
    with caplog.at_level(logging.INFO):
        publication_memory.log_publication_memory(
            logging.getLogger("test"), "search_model", phase="batch", batch=4
        )
    assert "rss_bytes=10240" in caplog.text
    assert "arrow_bytes=" in caplog.text
    assert "private-sentinel" not in caplog.text


def test_memory_diagnostics_do_not_fail_on_unreadable_proc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = tmp_path / "status"
    status.write_text("VmRSS: 10 kB", encoding="utf-8")

    def denied(*args: object, **kwargs: object) -> str:
        raise PermissionError("private diagnostic sentinel")

    monkeypatch.setattr(Path, "read_text", denied)
    assert publication_memory.resident_bytes(status) is None
