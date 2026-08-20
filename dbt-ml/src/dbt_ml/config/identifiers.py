"""Warehouse identifiers dbt-ml claims, and node-name validation.

Names become warehouse identifiers (table names, document_id scopes, dbt
export names), so they are restricted to a conservative charset up front.
SQL-side safety is handled separately by adapter quoting; this exists so a
typo'd or hostile name fails at config load with a clear message instead of
surfacing as a warehouse error mid-run.

The reserved prefix and the warehouse defaults live together here because
they are the same kind of thing: names that already exist in users'
warehouses, which config models must agree on rather than each spelling out.
"""
from __future__ import annotations

import re

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Tables dbt-ml manages internally (state, staging views, test-failure tables)
# all live under this prefix; user models must stay out of it.
#
# Frozen in both directions. Existing projects depend on `dbt_ml_*` staying
# rejected, and reserving a *new* prefix newly forbids model names that were
# legal before — so a rename must add a prefix here, never move this one.
RESERVED_PREFIX = "dbt_ml_"

# Default warehouse schema/dataset, shared by the profile warehouse config,
# the legacy inline DuckDB config, and the BigQuery adapter — which each
# declared it separately. It holds every table a project materializes, so a
# deployment that never set `schema:` explicitly loses sight of all of them
# at once if this changes without a migration.
DEFAULT_SCHEMA_NAME = "dbt_ml"

# Default DuckDB database file for the legacy no-profile path. Same hazard:
# a new default silently points at an empty database.
DEFAULT_DUCKDB_FILENAME = "dbt_ml.duckdb"


def validate_node_name(name: str, *, kind: str, reserve_internal: bool = False) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(
            f"{kind} name {name!r} is invalid: names must start with a letter or "
            "underscore and contain only letters, digits, and underscores"
        )
    if reserve_internal and name.lower().startswith(RESERVED_PREFIX):
        raise ValueError(
            f"{kind} name {name!r} is invalid: the '{RESERVED_PREFIX}' prefix is "
            "reserved for dbt-ml internal tables"
        )
    return name
