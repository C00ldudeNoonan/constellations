# Economic Entity Linking Example

This project links committed entity mentions (shaped like `nlp_entities`
output) to canonical economic identifiers — SEC CIK numbers, tickers, agency
IDs, and ISO 3166 country codes — through an operator-owned alias table. It is
fully deterministic: no optional extras, model downloads, credentials, or
network access.

```bash
uv run stel --project-dir examples/economic_entity_links build
```

`entity_links` emits one row per mention/namespace/canonical-ID outcome:

- `Apple Inc.` matches both the `cik` and `ticker` namespaces exactly;
- `the Fed` matches `The Fed` through the `normalized` method;
- `Mercury` maps to two tickers and is preserved as two `ambiguous` rows;
- `Acme Widgets` produces one explicit `unmatched` row.

Every row records the resolver identity, resolver version, and a fingerprint
of the complete alias set, so alias-table edits are visible downstream. The
matched mention text is not retained unless `include_mention_text: true` is
set explicitly.
