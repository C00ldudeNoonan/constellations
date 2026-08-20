# Compatibility posture

Workstream E of the architecture-simplification program (issue #190). stel is
pre-1.0 with a single known user, so backward-compatibility shims for external
consumers earn their keep only if they still serve a live purpose. This records
what was retired and what was deliberately kept.

## Retired

Removed outright — no elapsed deprecation window to honor, and no current data
relies on them.

| Path | Was | Why removed |
| --- | --- | --- |
| `DOCBT_PROFILES_DIR` env alias | Deprecated alias for `STEL_PROFILES_DIR` | Only `STEL_PROFILES_DIR` remains |
| Legacy LLM cache-row pruning | Swept pre-`provider-v…` cache keys on write | No such caches exist; current keys are all versioned |
| Legacy classic-ML artifact metadata reads | Promoted a legacy `classifier_options.alpha` and read hashing `feature_count` from a legacy `metrics` block | Current artifacts carry `options` and `integrity.feature_count`; the fallbacks were dead read paths |

## Kept — deliberately

| Path | Kind | Why kept |
| --- | --- | --- |
| Implicit local DuckDB target (no `profile:`) | Feature | A project with no `profile:` runs against its inline `duckdb:` database. This is the supported zero-config path for local and test projects, not debt — the former `DeprecationWarning` was removed and it is now documented as `profile._implicit_local_profile`. Declare a `profile:` + profiles.yml for warehouse targets, credentials, retrieval, or LLM config. |
| Bare `@register` backend registration | Extension surface | `backends/registry.py::register` accepts a bare `@register` (no Pydantic options model) and installs a pass-through contract. It is the documented way a third-party backend registers; first-party backends supply a model. Not debt. |

## Notes

No production behavior changes for valid current inputs: the retired paths were
either unreachable (legacy artifact reads), self-healing maintenance that had
nothing to sweep (cache pruning), or a rarely-used env alias. Removing them
deletes dead branches and their tests; the implicit-local-DuckDB behavior is
unchanged apart from no longer emitting a deprecation warning.
