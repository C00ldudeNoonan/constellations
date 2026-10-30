# Vocabulary sources

One CSV seed supplies the enum labels for extraction's provider schema and the
warehouse field's derived `accepted_values` check. Identifiers retain leading
zeros. The offline deterministic provider produces schema-shaped example
values; it does not perform semantic company extraction.

From the repository root:

```bash
uv run stel compile --project-dir examples/vocabulary_sources
uv run stel build --project-dir examples/vocabulary_sources
uv run stel plan --project-dir examples/vocabulary_sources
```

The first build writes one row to `stel.classified_mentions`; the next plan is
unchanged. Add a company to `seeds/competitors.csv` and `plan` reports changed
configuration. The next build reprocesses that row with the new closed set.
`none` is appended through `extra_terms`. Reordering seed rows leaves the code
identity unchanged. These commands use no network or credentials.

For an existing warehouse dimension, replace `path: seeds/competitors.csv` with
`relation: warehouse://main.dim_competitors`. That relation must exist before
the invocation and expose string `competitor_id` and `name` columns. `compile`,
`plan`, `run`, and `build` read it through the active profile; missing inputs
fail before inference or stel schema/state creation. `max_terms` defaults to
10,000, including extra terms. Use inline terms or CSV for offline dbt export
and embedded execution.
