"""An operator-chosen destination for the failure detail artifacts must drop.

stel sanitizes store and provider failures hard, and for good reason: the
native text quotes object-store URIs, SQL, and response bodies, and the
sanitized message reaches `run_results.json` and the CLI. `redacted_exception_
text` already settles what is safe to disclose -- exception *types*, stel
source locations, and a count of external frames, never native text, because
"exact-value replacement cannot safely redact repr-, JSON-, or URL-encoded
request data" (issue #490).

What was missing is a destination. That detail goes only to `log.debug`, and
`logging_setup` caps `-v` at INFO deliberately, pointing callers who need more
at attaching their own handler. A caller running `stel` as a subprocess -- an
orchestrator, which is how a long build actually runs -- cannot attach
anything. So in the one situation where the detail matters, nothing can
receive it, and the operator is left with a failure that names only its own
opacity:

    LanceDB operation 'index creation' failed on BTree index for 'context_id'
    after 3 attempts [RuntimeError] (code=lancedb_index_failed)

That is astrolabe#705: the same model failed twice eight days apart and
neither failure could be root-caused, because nothing available to the
operator could say what the native RuntimeError was.

This module supplies the destination and nothing else. Off unless a path is
given, and what it writes is `redacted_exception_text`'s output -- the same
allowlist the provider path already ships, not raw text. A log fans out to
whatever is capturing it; a path is a place the operator picked, so this is
where a deliberate disclosure belongs even though the content is already safe.

Written owner-only, and never allowed to fail: a diagnostics sink that breaks
a build would be worse than the blindness it fixes.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from .env import DIAGNOSTICS_FILE_ENV, read_env

# Lives in `providers` because that is where the redaction problem was first
# solved, but nothing about it is provider-specific: it walks a cause chain and
# emits allowlisted type labels plus stel frame locations. Imported rather than
# duplicated so the two disclosure paths can never drift into disagreeing about
# what is safe to write. The direction (retrieval-adjacent code importing from
# providers) is a layering wart worth fixing by moving the helper to a leaf
# module; deferred so this change stays reviewable as one idea.
from .providers.base import redacted_exception_text

log = logging.getLogger(__name__)

# Owner-only. The content is redacted, so this is defense in depth rather than
# the thing standing between a secret and a reader -- but a diagnostics file
# invites being left behind in a container or a CI workspace, and 0600 is the
# posture the cache storage already takes.
#
# This binds at CREATION only: `O_CREAT` ignores the mode for a path that
# already exists, so appending to a file the operator made themselves keeps
# whatever mode they gave it. Deliberate -- silently chmod-ing a path someone
# named would be the more surprising behavior -- but it means the guarantee is
# "stel does not create a readable one", not "this file is always 0600".
_FILE_MODE = 0o600

# `None` means "not configured here" and defers to the environment, which is
# what an orchestrated subprocess sets. A configured path wins so an explicit
# `--diagnostics-file` beats an inherited variable the caller may not know is
# set.
_override: Path | None = None


def configure_diagnostics(path: Path | None) -> None:
    """Point the sink at `path`, or clear it with None.

    Process-global, like logging configuration, so tests restore it in an
    autouse fixture rather than each test remembering to.
    """
    global _override
    _override = path


def diagnostics_destination() -> Path | None:
    """The path failures should be recorded to, or None when disabled."""
    if _override is not None:
        return _override
    raw = read_env(DIAGNOSTICS_FILE_ENV, default="").strip()
    return Path(raw) if raw else None


def record_failure(error: BaseException, *, summary: str) -> None:
    """Append one failure's redacted diagnostics to the operator's file.

    `summary` is the artifact-safe description the caller already built -- the
    operation, step, and code its raised error carries -- so the file can be
    read against the run's own error output without a correlation step.

    Silent when no destination is configured, which is the default.
    """
    destination = diagnostics_destination()
    if destination is None:
        return
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = f"{stamp} {summary}\n{redacted_exception_text(error)}\n\n"
    # LBYL does not apply: the parent may exist and the open still fail (a
    # read-only mount, a full disk, a path that is a directory). This is the
    # failure-reporting path, so it absorbs its own failures -- an unwritable
    # diagnostics file must not replace the error the caller is reporting. The
    # debug log notes it for anyone who does have a handler attached.
    try:
        # Unconditional: a bare filename's parent is `.`, so there is no case
        # where this is skipped, and a guard that never fires reads as one
        # that might.
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            _FILE_MODE,
        )
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(record)
    except OSError:
        log.debug("Could not write diagnostics to the configured path", exc_info=True)
