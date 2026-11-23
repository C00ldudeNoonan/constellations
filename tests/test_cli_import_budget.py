"""Import budget for the CLI entry points (issues #818 and #823; theme ALE-789).

A command pays only for what it uses: `stel --version`, `stel --help` and shell
completion must not load the dataframe, warehouse or retrieval stack, which cost
seconds of every invocation. This is a table of entry point against the module
sets that must stay unloaded, asserted over `sys.modules` in a subprocess.

It asserts module sets, never timings: a timing assertion is flaky on a loaded
machine, while a module that is loaded is loaded on every machine. Later issues
under the theme add rows as they lift more commands under budget.

A failure names the module and the entry point. For `--version`, `--help` and
shell completion the fix is almost always to move a module-level import in
`cli.py` into the body of the command that uses it (see the docstring of
`stel/cli.py`). For a command that reads a project it is usually one layer down:
a module under `config/`, the compiler, or the adapter and retrieval registries,
which resolve the active implementation on first use. Do not remove a row.
"""

from __future__ import annotations

import json
import shutil
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

# Reading a project is configuration work (issue #823, ALE-791): no dataframe
# library, no warehouse adapter and no retrieval store is needed to load and
# validate it, so none is imported until a command does the work that uses it.
_RUN_ONLY_STEL = (
    "stel.runner",
    "stel.concept_cloud",
    "stel.docs",
    "stel.graph_export",
    "stel.text",
)
_STORES = ("stel.retrieval.duckdb", "stel.retrieval.lancedb")
_DATAFRAME_STACK = ("polars", "pyarrow", "numpy", "lancedb", "spacy")
_WAREHOUSE_ADAPTERS = ("stel.adapters.duckdb", "stel.adapters.bigquery")

_READS_A_PROJECT = _DATAFRAME_STACK + _WAREHOUSE_ADAPTERS + _STORES + _RUN_ONLY_STEL

# Validating a project needs the active warehouse's capabilities, so that one
# adapter is imported, and only that one. The dataframe stack arrives with it.
_COMPILES_DUCKDB = ("stel.adapters.bigquery", *_STORES, *_RUN_ONLY_STEL)
_COMPILES_BIGQUERY = ("stel.adapters.duckdb", *_STORES, *_RUN_ONLY_STEL)

# (id, argv, modules that must not be in sys.modules afterwards). argv may hold
# "{project}", the example project's directory, which only a read-only command
# may be pointed at; "{duckdb_project}" and "{bigquery_project}", throwaway
# copies of it that a command may write its target/ into.
ENTRY_POINTS: list[tuple[str, list[str], tuple[str, ...]]] = [
    ("version", ["--version"], _NOTHING_HEAVY),
    ("help", ["--help"], _NOTHING_HEAVY),
    ("ls", ["--project-dir", "{project}", "ls"], _READS_A_PROJECT),
    ("compile-duckdb", ["--project-dir", "{duckdb_project}", "compile"], _COMPILES_DUCKDB),
    (
        "compile-bigquery",
        ["--project-dir", "{bigquery_project}", "compile"],
        _COMPILES_BIGQUERY,
    ),
]

_BIGQUERY_PROFILE = """\
invoice_pipeline:
  target: dev
  outputs:
    dev:
      warehouse:
        type: bigquery
        project: econ
        dataset: documents_dev
"""


def _project_copy(root: Path, example: Path, *, warehouse: str) -> Path:
    destination = root / warehouse
    shutil.copytree(example, destination)
    if warehouse == "bigquery":
        (destination / "profiles.yml").write_text(_BIGQUERY_PROFILE, encoding="utf-8")
    return destination

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
    argv: list[str], forbidden: tuple[str, ...], example_project_dir: Path, tmp_path: Path
) -> None:
    placeholders = {
        "{project}": example_project_dir,
        "{duckdb_project}": _project_copy(tmp_path, example_project_dir, warehouse="duckdb"),
        "{bigquery_project}": _project_copy(tmp_path, example_project_dir, warehouse="bigquery"),
    }
    resolved = list(argv)
    for placeholder, directory in placeholders.items():
        resolved = [arg.replace(placeholder, str(directory)) for arg in resolved]
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(list(forbidden)), json.dumps(resolved)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    loaded = json.loads(completed.stdout.strip().splitlines()[-1])
    assert loaded == [], (
        f"`stel {' '.join(argv)}` loaded {loaded}; move the module-level import that "
        "reaches it into the code that uses it (issues #818, #823)"
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


# The layers a command reads a project through, imported on their own: a command
# is only as cheap as the lowest layer it touches, so the budget is also asserted
# where the cost starts rather than only where a command ends (issue #823).
IMPORTS: list[tuple[str, tuple[str, ...]]] = [
    ("stel.config", ("polars", "pyarrow", "duckdb", "numpy")),
    (
        "stel.compiler",
        (
            "polars",
            "pyarrow",
            "numpy",
            "stel.text",
            "stel.retrieval.duckdb",
            "stel.retrieval.lancedb",
            *_WAREHOUSE_ADAPTERS,
        ),
    ),
]

_IMPORT_PROBE = """
import importlib, json, sys

importlib.import_module(sys.argv[1])
forbidden = json.loads(sys.argv[2])
assert "pydantic" in sys.modules  # the probe saw a real import graph
print(json.dumps(sorted(
    name
    for name in forbidden
    if any(m == name or m.startswith(name + ".") for m in sys.modules)
)))
"""


@pytest.mark.parametrize(
    ("module", "forbidden"), [pytest.param(m, f, id=m) for m, f in IMPORTS]
)
def test_importing_a_layer_does_not_load_what_it_does_not_use(
    module: str, forbidden: tuple[str, ...]
) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, module, json.dumps(list(forbidden))],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    loaded = json.loads(completed.stdout.strip().splitlines()[-1])
    assert loaded == [], (
        f"`import {module}` loaded {loaded}; move the module-level import that reaches "
        "it into the code that uses it (issue #823)"
    )
