"""`--diagnostics-file` end to end through `stel run` (issue #590).

An orchestrated run is a `stel build` subprocess, so the flag and the
`STEL_DIAGNOSTICS_FILE` variable are the only hatches such an operator has.

The vehicle is a transform that raises. Transform code is trusted, so its own
message is shown on the `ERROR:` line by design; what no other channel carries
is the traceback, which is what these tests look for in the file.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from stel.cli import cli
from stel.synth import generate_invoices

pytestmark = pytest.mark.e2e

SENTINEL = "distinctive-native-secret"
POINTER = "Native error detail was written to"
TRACEBACK = "Traceback (most recent call last)"


def _broken_project(tmp_path: Path, example_project_dir: Path) -> Path:
    """The example project with a transform that raises a URI-shaped error."""
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    generate_invoices(3, dst / "data" / "invoices", seed=1)
    (dst / "transforms" / "summarize.py").write_text(
        "def run(deps):\n"
        f"    raise RuntimeError('gs://bucket/prefix?token={SENTINEL}')\n",
        encoding="utf-8",
    )
    return dst


@pytest.fixture(autouse=True)
def _no_ambient_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STEL_DIAGNOSTICS_FILE", raising=False)
    monkeypatch.delenv("STEL_VERBOSE", raising=False)


def test_run_writes_the_native_detail_to_the_named_file_and_nowhere_else(
    tmp_path: Path, example_project_dir: Path
) -> None:
    dst = _broken_project(tmp_path, example_project_dir)
    path = tmp_path / "diagnostics.log"

    result = CliRunner().invoke(
        cli, ["--project-dir", str(dst), "run", "--diagnostics-file", str(path)]
    )

    assert result.exit_code == 1, (result.stdout, result.stderr)
    assert TRACEBACK not in result.stdout
    assert TRACEBACK not in result.stderr
    assert f"{POINTER} {path}" in result.stderr

    # A transform failure is raised as a RunError, so the pointer follows the
    # sanitized message the operator would otherwise have stopped at.
    assert "Error: Transform model 'invoice_summary' failed" in result.stderr

    text = path.read_text(encoding="utf-8")
    assert SENTINEL in text
    assert TRACEBACK in text
    assert "transform failed for invoice_summary" in text


def test_the_default_writes_nothing_and_the_env_var_names_the_file(
    tmp_path: Path, example_project_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dst = _broken_project(tmp_path, example_project_dir)
    path = tmp_path / "diagnostics.log"

    result = CliRunner().invoke(cli, ["--project-dir", str(dst), "run"])
    assert result.exit_code == 1, (result.stdout, result.stderr)
    assert POINTER not in result.stderr
    assert not path.exists()
    assert TRACEBACK not in result.stderr

    monkeypatch.setenv("STEL_DIAGNOSTICS_FILE", str(path))
    result = CliRunner().invoke(cli, ["--project-dir", str(dst), "run"])
    assert result.exit_code == 1, (result.stdout, result.stderr)
    assert f"{POINTER} {path}" in result.stderr
    assert SENTINEL in path.read_text(encoding="utf-8")
