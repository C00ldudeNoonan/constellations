"""The operator-chosen diagnostics sink (issue #590).

These guard a disclosure boundary, so several of them assert what must NOT be
in the file as firmly as what must. The native error text in the fixtures below
is written to look like the thing that made sanitization necessary in the first
place -- a quoted object-store URI with a credential in it -- so a regression
that starts copying native text through fails here loudly rather than shipping
a run_results.json with a token in it.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from click.testing import CliRunner

from stel.cli import cli
from stel.diagnostics import (
    configure_diagnostics,
    diagnostics_destination,
    record_failure,
)
from stel.env import DIAGNOSTICS_FILE_ENV
from stel.retrieval.lancedb import _operation_failed

# What a native LanceDB error can look like, and precisely what must never be
# copied into a file or an artifact.
_NATIVE_TEXT = "PUT https://acct.blob.core.windows.net/c?sig=SECRET-TOKEN failed"


def _native_error() -> RuntimeError:
    """A raised-and-caught error, so it carries a real traceback to walk."""
    try:
        raise RuntimeError(_NATIVE_TEXT)
    except RuntimeError as error:
        return error


def test_no_destination_configured_writes_nothing_and_does_not_raise() -> None:
    """The default. A sink nobody asked for must be inert, not merely quiet."""
    assert diagnostics_destination() is None
    record_failure(_native_error(), summary="operation='upsert'")


def test_environment_variable_supplies_the_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The orchestrated case: a subprocess caller sets the variable, since it
    is the one caller that cannot attach a DEBUG handler."""
    destination = tmp_path / "diagnostics.log"
    monkeypatch.setenv(DIAGNOSTICS_FILE_ENV, str(destination))

    record_failure(_native_error(), summary="operation='upsert' (code=x)")

    assert "operation='upsert' (code=x)" in destination.read_text(encoding="utf-8")


def test_configured_path_overrides_the_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--diagnostics-file` beats an inherited variable the caller may not
    know is set, rather than writing to both or to the wrong one."""
    monkeypatch.setenv(DIAGNOSTICS_FILE_ENV, str(tmp_path / "from-env.log"))
    chosen = tmp_path / "from-flag.log"
    configure_diagnostics(chosen)

    record_failure(_native_error(), summary="operation='upsert'")

    assert chosen.exists()
    assert not (tmp_path / "from-env.log").exists()


def test_the_record_carries_exception_types_but_never_native_text(
    tmp_path: Path,
) -> None:
    """The disclosure boundary itself.

    A type name and a stel source location are what make a failure
    diagnosable; the native message is what can carry a credential or a
    response body. The file gets the first and never the second.
    """
    destination = tmp_path / "diagnostics.log"
    configure_diagnostics(destination)

    record_failure(_native_error(), summary="operation='index creation'")

    written = destination.read_text(encoding="utf-8")
    assert "builtins.RuntimeError" in written
    assert _NATIVE_TEXT not in written
    assert "SECRET-TOKEN" not in written


def test_a_failure_recorded_through_the_lancedb_path_reaches_the_file(
    tmp_path: Path,
) -> None:
    """The seam that matters: the store's own sanitizing error builder feeds
    the sink, so this works for a real failure and not only a direct call."""
    destination = tmp_path / "diagnostics.log"
    configure_diagnostics(destination)

    failure = _operation_failed(
        "index creation",
        "lancedb_index_failed",
        _native_error(),
        step="BTree index for 'context_id'",
    )

    written = destination.read_text(encoding="utf-8")
    assert "lancedb_index_failed" in written
    assert "BTree index for 'context_id'" in written
    assert _NATIVE_TEXT not in written
    # The raised error is unchanged by the sink -- the file is additional, not
    # a replacement, and must not have leaked native text into the message.
    assert _NATIVE_TEXT not in str(failure)


def test_successive_failures_append_rather_than_overwrite(tmp_path: Path) -> None:
    """A build fails more than once -- three index retries in the case this
    was written for. Truncating would keep only the last and hide the shape."""
    destination = tmp_path / "diagnostics.log"
    configure_diagnostics(destination)

    record_failure(_native_error(), summary="operation='upsert' attempt=1")
    record_failure(_native_error(), summary="operation='upsert' attempt=2")

    written = destination.read_text(encoding="utf-8")
    assert "attempt=1" in written
    assert "attempt=2" in written


@pytest.mark.skipif(
    os.name != "posix", reason="POSIX mode bits; os.open ignores mode on Windows"
)
def test_the_file_is_created_owner_only(tmp_path: Path) -> None:
    """Defense in depth: the content is redacted, but a diagnostics file gets
    left behind in containers and CI workspaces.

    Creation only -- `O_CREAT` ignores the mode when the path already exists,
    which the module documents rather than works around.
    """
    destination = tmp_path / "diagnostics.log"
    configure_diagnostics(destination)

    record_failure(_native_error(), summary="operation='upsert'")

    mode = stat.S_IMODE(destination.stat().st_mode)
    assert mode == 0o600, f"expected owner-only, got {mode:o}"


def test_an_unwritable_destination_does_not_disturb_the_reported_failure(
    tmp_path: Path,
) -> None:
    """The sink absorbs its own failures.

    A diagnostics file that cannot be written is a worse outcome than the
    blindness it fixes if it replaces the error the caller was reporting --
    the operator would then debug the sink instead of the build.

    Unwritable here means a parent that is a regular file, which fails for
    root too. A chmod-based version of this test silently SKIPS under the
    container builds that run the suite as root, so it would certify this
    guard exactly where it never ran.
    """
    not_a_directory = tmp_path / "occupied"
    not_a_directory.write_text("", encoding="utf-8")
    configure_diagnostics(not_a_directory / "diagnostics.log")

    failure = _operation_failed("upsert", "lancedb_upsert_failed", _native_error())

    assert "lancedb_upsert_failed" in str(failure)


def test_the_cli_flag_points_the_sink_at_the_given_path(
    tmp_path: Path, example_project_dir: Path
) -> None:
    """The wiring, which the unit tests above cannot see.

    `--diagnostics-file` is declared with `expose_value=False`, so no command
    body ever reads it -- the Click callback is the whole mechanism. A typo in
    the callback name or a decorator applied to the wrong helper would leave
    every test above passing and the flag inert, which is the failure mode
    worth a test of its own.
    """
    destination = tmp_path / "from-cli.log"

    CliRunner().invoke(
        cli,
        [
            "compile",
            "--project-dir",
            str(example_project_dir),
            "--diagnostics-file",
            str(destination),
        ],
    )

    assert diagnostics_destination() == destination.resolve()
