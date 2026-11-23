"""Import budget for the CLI entry points (issue #818, ALE-790; theme ALE-789).

A command pays only for what it uses: `stel --version`, `stel --help` and shell
completion must not load the dataframe, warehouse or retrieval stack, which cost
seconds of every invocation. This is a table of entry point against the module
sets that must stay unloaded, asserted over `sys.modules` in a subprocess.

It asserts module sets, never timings: a timing assertion is flaky on a loaded
machine, while a module that is loaded is loaded on every machine. Later issues
under the theme add rows as they lift more commands under budget.

A failure names the module and the entry point. The fix is almost always to move
a module-level import in `cli.py` into the body of the command that uses it (see
the docstring of `stel/cli.py`), not to remove a row.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Heavy third-party libraries: each is a second or more of import time.
_DATAFRAME_AND_WAREHOUSE = (
    "polars",
    "pyarrow",
    "duckdb",
    "numpy",
    "jinja2",
    "lancedb",
    "spacy",
)

# stel's own layers that pull the libraries above in. A prefix covers its
# submodules: "stel.adapters" also forbids "stel.adapters.bigquery".
_HEAVY_STEL_LAYERS = (
    "stel.runner",
    "stel.compiler",
    "stel.adapters",
    "stel.retrieval",
    "stel.text",
    "stel.concept_cloud",
    "stel.docs",
    "stel.graph_export",
    "stel.search",
    "stel.plan",
    "stel.pending",
    "stel.checks",
    "stel.adopt",
)

# What building the group and parsing arguments may not load: not even the
# configuration layer, which is loaded by a command that reads a project.
_CONFIG_LAYER = ("stel.config", "pydantic", "yaml")

_NOTHING_HEAVY = _DATAFRAME_AND_WAREHOUSE + _HEAVY_STEL_LAYERS + _CONFIG_LAYER

# Reading a project is configuration work. The configuration layer still loads
# polars (ALE-791), so the rows for such commands forbid only what is already
# out of reach: the runner and everything the run path alone needs.
_READS_A_PROJECT = (
    "lancedb",
    "spacy",
    "stel.runner",
    "stel.concept_cloud",
    "stel.docs",
    "stel.graph_export",
)

# (id, argv, modules that must not be in sys.modules afterwards). argv may
# hold "{project}", which is replaced by the example project's directory.
ENTRY_POINTS: list[tuple[str, list[str], tuple[str, ...]]] = [
    ("version", ["--version"], _NOTHING_HEAVY),
    ("help", ["--help"], _NOTHING_HEAVY),
    ("ls", ["--project-dir", "{project}", "ls"], _READS_A_PROJECT),
]

# Runs the CLI the way the console script does, then reports what it loaded.
_PROBE = """
import json, sys
from stel.cli import cli

forbidden = json.loads(sys.argv[1])
try:
    cli.main(args=json.loads(sys.argv[2]), prog_name="stel", standalone_mode=False)
except SystemExit:
    pass
assert "click" in sys.modules  # the probe saw a real import graph
loaded = sorted(
    name
    for name in forbidden
    if any(m == name or m.startswith(name + ".") for m in sys.modules)
)
print(json.dumps(loaded))
"""


@pytest.mark.parametrize(
    ("argv", "forbidden"),
    [pytest.param(argv, forbidden, id=entry_id) for entry_id, argv, forbidden in ENTRY_POINTS],
)
def test_entry_point_does_not_load_what_it_does_not_use(
    argv: list[str], forbidden: tuple[str, ...], example_project_dir: Path
) -> None:
    resolved = [arg.replace("{project}", str(example_project_dir)) for arg in argv]
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(list(forbidden)), json.dumps(resolved)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    loaded = json.loads(completed.stdout.strip().splitlines()[-1])
    assert loaded == [], (
        f"`stel {' '.join(argv)}` loaded {loaded}; move the module-level import in "
        "stel/cli.py that reaches it into the command body (issue #818)"
    )


# `--help` on any command, group or subcommand parses arguments and prints usage
# without running a body, so it is under the same budget as `stel --help`. One
# process walks the whole command tree, so a new command is covered the day it
# is added, with no row to remember.
_HELP_WALK = """
import contextlib, io, json, sys
from stel.cli import cli

forbidden = json.loads(sys.argv[1])


def paths(command, prefix):
    yield prefix
    for name, child in getattr(command, "commands", {}).items():
        yield from paths(child, [*prefix, name])


count = 0
for path in paths(cli, []):
    # The help text is discarded: only what the parse loaded matters, and a
    # Windows console codepage cannot encode every character in it.
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            cli.main(args=[*path, "--help"], prog_name="stel", standalone_mode=False)
        except SystemExit:
            pass
    count += 1
loaded = sorted(
    name
    for name in forbidden
    if any(m == name or m.startswith(name + ".") for m in sys.modules)
)
print(json.dumps({"commands": count, "loaded": loaded}))
"""


def test_help_of_every_command_stays_under_budget() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _HELP_WALK, json.dumps(list(_NOTHING_HEAVY))],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["commands"] > 30  # the walk found the command tree
    assert report["loaded"] == [], (
        f"`--help` on some command loaded {report['loaded']}; move the module-level "
        "import in stel/cli.py that reaches it into the command body (issue #818)"
    )
