# Context assembly experiment

Run from the repository root in the locked environment with the `text` extra:

```bash
uv run python examples/context_assembly/compare.py
```

This prints a deterministic `context_assembly_compare/v1` report for five
synthetic cases and four packing policies, with a 650-token budget using
`tiktoken/cl100k_base`. No warehouse, inference provider, credentials, or
production serving change is involved. The tokenizer may download its public
encoding vocabulary on first use. An encoding for a different model is an
estimate, not a promise about that provider's actual input limit.

The cases cover evidence split across neighboring chunks, overlapping spans,
long citation metadata, a chunk too large for the budget, and an absent scope.
All arms get identical ranked roots and the same neighbor snapshot. The
neighbors arm also changes ordering and overlap removal: it is a combined
policy, not an isolated estimate of expansion's causal effect.

`token_count` measures the entire rendered input, including instructions,
question, JSON escaping, citation envelope, and separators. `metadata_tokens`
counts the same input with empty text fields. `text_tokens` sums the isolated
chunk-text counts. BPE boundaries mean these counts are not additive. Packing
skips whole oversized chunks and continues; it never truncates an evidence
span or drops mandatory IDs to meet the budget. A budget unable to hold even
the empty prompt envelope is an error.

The report contains IDs, hashes, omissions and counts, with **null answer
scores**. It demonstrates packing mechanics; it does not measure a model or
establish position sensitivity. The explicit decision is to defer adoption.
The baseline and current MCP serving behavior remain unchanged.

Reproduced on 2026-11-03 with tiktoken 0.13.0 / cl100k_base:

| Case | Relevance tokens / chunks | Neighbors tokens / chunks |
| --- | ---: | ---: |
| Boundary-spanning evidence | 134 / 1 | 236 / 2 |
| Overlapping evidence | 245 / 2 | 241 / 2 |
| Long citation metadata | 133 / 1 | 133 / 1 |
| Insufficient budget | 33 / 0 | 33 / 0 |
| Absent scope | 37 / 0 | 37 / 0 |

The boundary case's empty-text envelopes use 129 and 223 tokens respectively.
The metadata case skips its first, oversized locator and admits the later
chunk. The overlap-only arm preserves the reversed retrieval order and
conservatively cannot trim a trailing overlap; the combined neighbors arm
orders by document before trimming. These are mechanical observations, with
zero generated answers and zero reviewed answer scores. Full identities and
omissions are emitted by the command, rather than copied into this table.

For a real benchmark, use the API and review protocol in
[the context assembly design](../../docs/architecture/context-assembly.md).
Do not relabel these synthetic development/holdout cases as the reviewed
ALE-281 holdout.
