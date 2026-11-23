# ADR-0066: Built-in adapters and retrieval stores resolve by type on first use

- **Status:** accepted
- **Date:** 2026-11-23
- **Prompted by:** #823 (ALE-791), theme ALE-789

## Context

`stel ls` took 3.1 s and `stel compile` 3.75 s before doing any work. Once the CLI's own imports were deferred (#818), the layers
under the commands still imported everything: `stel.adapters.__init__` imported
the DuckDB and BigQuery adapters together so that `@register` ran as a side
effect, `stel.retrieval.__init__` did the same for the DuckDB and LanceDB
stores, the configuration layer imported an enum from a module that imports
polars, and the compiler reached the whole `stel.text` package for two option
parsers.

## Decision

**A built-in adapter or store is a row in a table of `type -> (module, class)`,
and its module is imported the first time that type is named.** The registry's
lookup reads the class from the module, caches it in `_REGISTRY`, and `register`
stays for third-party and test registrations. Neither package `__init__` imports
an implementation. A project names one warehouse and one store, so it imports
one of each.

What the configuration layer and the compiler validate against lives in a module
that imports no implementation: `agent_context_grain.py`,
`entity_link_options.py` and `relation_options.py`. The implementation modules
import from them and do not re-export, so each name has one import path.
Modules the compiler reaches import polars and pyarrow only under
`TYPE_CHECKING` or inside the function that builds a frame.

## Alternatives considered

- **Keep the side-effect import, make the adapter modules cheap.** The cost is
  pyarrow, numpy, a cloud SDK and 5,000 lines of one adapter. Making each import
  its dependencies lazily moves the same work into every method and leaves the
  other adapter's module imported for nothing.
- **Discover adapters through entry points.** Right for a plugin ecosystem, and
  the NLP providers use it (ADR-0064). Two first-party adapters do not need a
  metadata scan, and the table is checked by a test where an entry point is
  not.
- **Re-export the moved names from their old modules.** It would keep every
  caller working and put the dataframe import back on the path the moves exist to
  clear, since the old module is the one that imports it. This is a single-user
  project, so every caller was updated instead.
- **Read the class from `@register` after importing the module.** A registry
  that is cleared after the module was imported (the test fixture does exactly
  that) can never be filled again, because the module is cached. Reading the
  class by name does not depend on a side effect that runs once.

## Consequences

- An import error in an adapter or store module now surfaces when a project
  names that type, not at `import stel.adapters`.
- A new built-in adapter or store adds a row to the table in
  `adapters/registry.py` or `retrieval/registry.py`.
- `tests/test_cli_import_budget.py` pins the result as module sets, never
  timings: `stel.config` loads none of polars, pyarrow, duckdb or numpy;
  `stel.compiler` loads no polars, pyarrow, numpy, `stel.text` or store module;
  `stel ls` loads no dataframe library or adapter; and `stel compile` loads
  only the active adapter.
- Not done here: jinja2 (through `sql_models`) and the `duckdb` package (through
  the LLM extraction cache) still load on a project read.
