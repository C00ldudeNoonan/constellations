"""One-time rename of stel's persisted warehouse objects (#313).

The rename moved every internal table from `dbt_ml_*` to `stel_*`. Those
tables are data, not code: the state table holds every incremental
fingerprint, and the serving ledger and leases hold live publication claims.
Creating fresh ones under the new names beside the old would look like a
brand-new project and reprocess every corpus at provider cost, so the
connect-time guard in `WarehouseAdapter._guard_legacy_names` refuses to run
until this has carried the existing objects over.

Deliberately narrow. Only tables stel owns are touched, and only within the
schema the target already points at — moving a whole schema is the operator's
call, made by pinning `schema:`, not something a migration guesses at.

**When this can be deleted (issue #321).** Not on a date, and not when the
migration is believed done — when it is *verified* that no warehouse stel
still connects to carries pre-rename names. The known consumer migrated its
three warehouses (astrolabe#291, 2026-08), but local DuckDB files under old
`target/` directories and scratch worktrees are warehouses too, and each one
that still holds `dbt_ml_state` is a corpus that gets silently reprocessed at
provider cost the moment the guard stops looking for it.

Removal is all-or-nothing: `_guard_legacy_names` exists to point at
`stel migrate`, so keeping the detection while dropping the command would
leave an error naming something that no longer runs. Delete this module, the
`LEGACY_*` name constants in `base.py`, the `LegacyWarehouseNamesError`
guards, and the CLI command in one change, or keep them all.
"""
from __future__ import annotations

from dataclasses import dataclass

from .base import (
    MIGRATED_TABLE_NAMES,
    MIGRATED_TABLE_PREFIXES,
    AdapterError,
    WarehouseAdapter,
)


class MigrationConflictError(AdapterError):
    """Both spellings of one object exist, so which holds the live data is not
    ours to decide."""


@dataclass(frozen=True)
class TableRename:
    """One planned rename, within the adapter's configured schema."""

    old: str
    new: str


def plan_name_migration(adapter: WarehouseAdapter) -> list[TableRename]:
    """Renames needed to bring `adapter`'s schema to the post-#313 names.

    Empty when there is nothing to do, which is the normal result on a
    warehouse created after the rename. Raises rather than guessing when both
    spellings of the same object are present: that means an interrupted
    migration or a hand-made table, and picking one would discard whichever
    holds the rows that matter.
    """
    present = set(adapter.list_all_tables())
    renames: list[TableRename] = []
    conflicts: list[tuple[str, str]] = []

    for legacy, current in MIGRATED_TABLE_NAMES:
        if legacy not in present:
            continue
        if current in present:
            conflicts.append((legacy, current))
            continue
        renames.append(TableRename(legacy, current))

    for legacy_prefix, current_prefix in MIGRATED_TABLE_PREFIXES:
        for name in sorted(present):
            if not name.startswith(legacy_prefix):
                continue
            target = current_prefix + name[len(legacy_prefix) :]
            if target in present:
                conflicts.append((name, target))
                continue
            renames.append(TableRename(name, target))

    if conflicts:
        detail = ", ".join(f"'{old}' and '{new}'" for old, new in conflicts)
        raise MigrationConflictError(
            f"Schema '{adapter.schema}' holds both the old and new name for "
            f"{len(conflicts)} stel object(s): {detail}. That is an interrupted "
            "migration or a hand-made table, and stel will not choose which "
            "one holds the live rows. Inspect both, drop or rename the one you "
            "do not want, then re-run `stel migrate`."
        )
    return renames


def apply_name_migration(
    adapter: WarehouseAdapter, renames: list[TableRename]
) -> list[TableRename]:
    """Perform `renames` in order, returning the ones that landed.

    No overall transaction: BigQuery has none to offer across DDL, and a
    partial run is recoverable because re-planning simply skips what already
    moved. A failure re-raises with the completed renames named, so the
    operator knows which half of the schema they are looking at.
    """
    done: list[TableRename] = []
    for rename in renames:
        try:
            adapter.rename_table(rename.old, rename.new)
        except Exception as error:
            if done:
                completed = ", ".join(f"{r.old} -> {r.new}" for r in done)
                error.add_note(
                    f"Renames already applied before this failure: {completed}. "
                    "Re-running `stel migrate` resumes from here."
                )
            raise
        done.append(rename)
    return done
