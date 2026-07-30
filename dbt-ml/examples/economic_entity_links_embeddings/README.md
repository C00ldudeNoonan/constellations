# Economic Entity Linking — Embedding Similarity Example

This project links entity mentions to canonical economic identifiers with the
`vector_similarity` resolver. Instead of matching mention text against alias
text, it matches a mention's **embedding** against each alias's embedding and
accepts candidates above a similarity threshold.

The embeddings come from the ordinary `embed` model kind: `mention_embeddings`
embeds the mention text and `alias_embeddings` embeds the alias surface forms.
The resolver stays a pure, offline transform over the resulting vector columns —
credentials, provider batching, and embedding identity all live in the `embed`
executor, not in the linker.

```bash
uv run dbt-ml --project-dir examples/economic_entity_links_embeddings build
```

This runs with no credentials, extras, or network access because both `embed`
models use the built-in `deterministic` provider.

## What the run produces

`entity_links` emits one row per mention/namespace/canonical-ID outcome, with a
`match_score` (cosine similarity) on every resolved row:

| mention | namespace | canonical_id | score | status |
|---|---|---|---|---|
| `fomc-minutes-2026-01-e0` (`the Fed`) | — | — | — | unmatched |
| `fomc-minutes-2026-01-e1` (`United States`) | iso3166 | US | 1.0 | matched |
| `fomc-minutes-2026-01-e2` (`Acme Widgets`) | — | — | — | unmatched |
| `sec-10k-apple-2025-e0` (`Apple Inc.`) | cik | 0000320193 | 1.0 | matched |
| `sec-10k-apple-2025-e0` (`Apple Inc.`) | ticker | AAPL | 1.0 | matched |
| `sec-10k-mercury-2025-e0` (`Mercury`) | ticker | MCY | 1.0 | ambiguous |
| `sec-10k-mercury-2025-e0` (`Mercury`) | ticker | MERC | 1.0 | ambiguous |

`Apple Inc.` resolves in both the `cik` and `ticker` namespaces; `Mercury` maps
to two tickers and is preserved as two `ambiguous` rows rather than guessed;
`Acme Widgets` has no alias and is an explicit `unmatched` row. As with the
alias-table resolver, every row records the resolver identity, resolver version,
and a fingerprint of the alias vector set, and mention text is withheld unless
`include_mention_text: true` is set.

## About the deterministic provider

The `deterministic` provider hashes text to a fixed vector, so **identical text
gets an identical vector (cosine 1.0) and any other text scores far lower**. It
demonstrates the pipeline mechanics reproducibly, not semantic-match quality.
That is why `the Fed` is `unmatched` here even though the alias table contains
`The Fed`: the two strings differ, so their deterministic vectors differ. A real
embedding model would score that pair highly — swap `provider`/`model` on both
`embed` models (and add the matching `embedding:` block to `profiles.yml`) to
see semantic matching, keeping the same `dimensions` on both so the vectors are
comparable, and lower `threshold` to an operator-calibrated value.
