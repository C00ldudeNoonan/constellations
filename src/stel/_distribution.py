"""The installed distribution's name and version.

`DISTRIBUTION_NAME` must equal `[project] name` in pyproject.toml. Asking
`importlib.metadata` for a name that is not installed does not raise here — it
degrades to `"unknown"`, because a source checkout has no distribution metadata
and that must not break `--help`. The same tolerance means a stale name is
invisible: every lookup keeps working and just reports `"unknown"`.

That matters because the value is durable. It is written into run_results
metadata, into every extracted row's parser identity, and into the `runtime`
block of every fitted classic-ML artifact — where it is validated on load. Four
call sites each performed this lookup independently; one of them drifting would
silently change what those artifacts record. Hence one definition, and
`tests/test_frozen_names.py` asserting the resolved version is not `"unknown"`.

Not memoized: the lookup keeps `importlib.metadata`'s exact semantics so this
module changes no behavior.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

DISTRIBUTION_NAME = "stel"

UNKNOWN_VERSION = "unknown"


def distribution_version() -> str:
    """The installed stel version, or `"unknown"` outside an installation."""
    try:
        return version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return UNKNOWN_VERSION
