"""Warehouse identifiers stel claims, and node-name validation.

Names become warehouse identifiers (table names, document_id scopes, dbt
export names), so they are restricted to a conservative charset up front.
SQL-side safety is handled separately by adapter quoting; this exists so a
typo'd or hostile name fails at config load with a clear message instead of
surfacing as a warehouse error mid-run.

The reserved prefixes and the warehouse defaults live together here because
they are the same kind of thing: names that already exist in users'
warehouses, which config models must agree on rather than each spelling out.

The `LEGACY_*` values are the pre-#313 spellings. They are still real objects
in warehouses built before the rename, so they stay here as data the migration
and the connect-time guards read — never as fallbacks the tool silently
adopts.
"""
from __future__ import annotations

import re

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Tables stel manages internally (state, staging views, test-failure tables)
# all live under these prefixes; user models must stay out of them.
#
# Frozen in both directions, and additive only. Existing projects rely on
# `dbt_ml_*` staying rejected, so it cannot be dropped now that internals have
# moved to `stel_*` — and reserving a prefix newly forbids model names that
# were legal before, so a later rename must append here rather than replace.
LEGACY_RESERVED_PREFIX = "dbt_ml_"
RESERVED_PREFIXES = (LEGACY_RESERVED_PREFIX, "stel_")

# Default warehouse schema/dataset, shared by the profile warehouse config,
# the legacy inline DuckDB config, and the BigQuery adapter — which each
# declared it separately. It holds every table a project materializes, so a
# deployment that never set `schema:` explicitly loses sight of all of them at
# once when this changes. #313 changed it, which is why connecting to a
# defaulted `stel` schema while a populated `dbt_ml` one exists is a hard
# error rather than a fresh start (see WarehouseAdapter.__enter__).
DEFAULT_SCHEMA_NAME = "stel"
LEGACY_SCHEMA_NAME = "dbt_ml"

# Default DuckDB database file for the legacy no-profile path. Same hazard: a
# new default silently points at an empty database, so config load raises when
# the legacy file is the only one present.
DEFAULT_DUCKDB_FILENAME = "stel.duckdb"
LEGACY_DUCKDB_FILENAME = "dbt_ml.duckdb"

# The project file, and the name it carried before the rename. This is the
# first thing an upgrading project trips over, before any of the warehouse
# guards get a chance to run, so the missing-file error looks for the old
# spelling next door and says so. Detection only, per this module's rule:
# nothing loads a `dbt_ml_project.yml`, because two filenames that both work is
# how the old one never dies (issue #324).
PROJECT_FILENAME = "stel_project.yml"
LEGACY_PROJECT_FILENAME = "dbt_ml_project.yml"

# Global profiles directory under the user's home, and its pre-rename name.
# Moving it is a manual upgrade step, so discovery names the legacy location
# when it is the only one present — and, again, never reads from it.
GLOBAL_PROFILES_DIRNAME = ".stel"
LEGACY_GLOBAL_PROFILES_DIRNAME = ".dbt_ml"


def validate_node_name(name: str, *, kind: str, reserve_internal: bool = False) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(
            f"{kind} name {name!r} is invalid: names must start with a letter or "
            "underscore and contain only letters, digits, and underscores"
        )
    if reserve_internal:
        lowered = name.lower()
        for prefix in RESERVED_PREFIXES:
            if lowered.startswith(prefix):
                raise ValueError(
                    f"{kind} name {name!r} is invalid: the '{prefix}' prefix is "
                    "reserved for stel internal tables"
                )
    return name
