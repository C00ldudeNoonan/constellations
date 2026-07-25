# Compatibility inventory

Workstream E of the architecture-simplification program (issue #190). This is
the authoritative inventory of dbt-ml's live backward-compatibility paths, each
with an explicit keep/isolate/remove decision.

The guiding rule (from #190): **remove only paths whose documented
compatibility window has ended; otherwise isolate them behind a clearly marked
reader so they do not complicate current execution paths.** dbt-ml is pre-1.0
(current: `0.2.10`), and none of these paths had previously named a removal
version — they warned "will be removed" with no target. This inventory assigns
those targets. No path is removed here: no window had elapsed, so removing now
would break existing projects with zero deprecation notice.

Removal target for the deprecated paths below is **v1.0.0** — the conventional
single cutover for a pre-1.0 project. The user-facing `DeprecationWarning`s now
name that version.

| # | Path | Kind | Isolated in | Decision |
| - | --- | --- | --- | --- |
| 1 | Inline legacy `duckdb:` project config | Deprecated | `profile.py::_legacy_resolved` | Remove in **v1.0.0** |
| 2 | `DOCBT_PROFILES_DIR` env alias | Deprecated | `profile.py::_legacy_env_dir` | Remove in **v1.0.0** |
| 3 | Legacy LLM cache rows | Internal maintenance | `backends/llm_backend.py::_prune_legacy_entries` | Retain; droppable in **v1.0.0** |
| 4 | Legacy classic-ML artifact metadata | Versioned read-compat | `classic_ml/classifier.py`, `classic_ml/text.py` | Keep; revisit at next artifact-schema bump |
| 5 | Bare `@register` backend registration | Intentional extension surface | `backends/registry.py::register` | **Keep** — not debt |

## 1. Inline legacy `duckdb:` project configuration

Before profiles.yml, a project could declare an inline `duckdb:` block and no
`profile:`. `resolve_profile` falls back to `_legacy_resolved`, which emits a
`DeprecationWarning` and synthesizes a DuckDB `ResolvedProfile`.

- **Status:** deprecated, warned. Still exercised by fixtures/examples that
  predate profiles.yml.
- **Isolation:** fully contained in `_legacy_resolved`; the modern path
  (`project.profile` set) never touches it.
- **Decision:** remove in **v1.0.0**. The warning now names that version.
  Migration: declare a `profile:` in the project and a matching
  `warehouse:` in profiles.yml.

## 2. `DOCBT_PROFILES_DIR` environment alias

The profiles directory is discovered from `DBT_ML_PROFILES_DIR`;
`DOCBT_PROFILES_DIR` is honored as a deprecated alias (`_legacy_env_dir`).

- **Status:** deprecated, warned.
- **Isolation:** one function; the current env var is read separately.
- **Decision:** remove in **v1.0.0**. The warning now names that version.
  Migration: rename the env var to `DBT_ML_PROFILES_DIR`.

## 3. Legacy LLM cache rows

`_prune_legacy_entries` deletes pre-provider-contract cache rows
(`{model}|{content}|{schema}` keys) the first time a cache file is opened.
Every current key carries a `provider-v…|` prefix, so those rows can never be
read again and only grow the file.

- **Status:** internal self-healing maintenance, not a read path. No
  user-facing surface, so no warning.
- **Isolation:** one function, guarded by a per-path idempotency set.
- **Decision:** retain — it is cheap, idempotent, and keeps old cache files
  from bloating. Droppable in **v1.0.0** once caches are assumed clean; keeping
  it is harmless, so removal is optional rather than required.

## 4. Legacy classic-ML artifact metadata

Under the current `ARTIFACT_SCHEMA_VERSION = 2`, two readers accept older
metadata sub-structures: `classifier.py` promotes a legacy
`classifier_options.alpha` into `options`, and `text.py` reads `feature_count`
from a legacy `metrics` block for hashing artifacts.

- **Status:** versioned-artifact read-compat within schema v2. Preserving
  artifact readers and migrations is an explicit #190 contract.
- **Isolation:** localized to the family readers, behind the shared artifact
  envelope (`classic_ml/artifacts.py`).
- **Decision:** keep. These are read-only compatibility shims tied to the
  artifact schema version, not orchestration debt. Revisit when
  `ARTIFACT_SCHEMA_VERSION` next increments (a schema bump already rejects v1
  artifacts with a refit hint); a v1.0.0 review is a natural checkpoint.

## 5. Bare `@register` backend registration

`backends/registry.py::register` accepts a bare `@register` (no Pydantic
options model) and installs a pass-through option contract.

- **Status:** **intentional public extension surface**, not compatibility debt.
  It is the documented way a third-party backend registers without adopting the
  typed option contract; new first-party backends supply a model.
- **Isolation:** the pass-through is a single branch in `register`.
- **Decision:** **keep indefinitely.** Removing it would break third-party
  backends for no internal benefit. Listed here to record that the Phase 0
  "bare backend registration" line item was assessed and is deliberately
  retained, mirroring how capability sets stay independently declared.

## Summary

- Two genuinely deprecated paths (1, 2) are now version-stamped for **v1.0.0**.
- Two are compatibility shims kept behind isolated readers (3 internal, 4
  artifact-versioned).
- One (5) is an intentional extension contract, not debt.

No production behavior changes with this inventory beyond naming v1.0.0 in the
two existing deprecation warnings.
