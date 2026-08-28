# dbt documentation suggestions from the transcript corpus

The analysis half of `stel suggest` (issue #361). It turns the agent-transcript
corpus (#360) into rows that `stel suggest dbt` renders as a reviewable diff
against a dbt project's `schema.yml`.

The two halves are deliberately separate, and the seam is the point: deciding
*which* models are under-documented and *what to say* about them is an ordinary
stel project — with the provider, prompt-provenance, budget, and incremental
machinery every other model gets. The command only reads the relation these
models produce. Nothing about the analysis lives inside the CLI.

## The signal

An agent opening a model's SQL to answer a question is the documentation that
should have existed. The corpus records `files_touched` per exchange, so a
model whose file is read across several *separate sessions* is a gap with
evidence behind it.

**Distinct sessions, never distinct exchanges.** One long session where an
agent re-reads the same file is one person, one time — counting exchanges
would let a single afternoon manufacture its own evidence.

## The chain

| model | kind | what it does |
|---|---|---|
| `exchange_rows` | `warehouse://` + extraction | the corpus, as rows |
| `documentation_gaps` | SQL transform | models read across ≥3 sessions |
| `drafted_descriptions` | `llm:` | proposed prose, one call per gap |
| `dbt_doc_candidates` | SQL transform | the contract `stel suggest` reads |

No new model kind — the constraint #361 set. If this had needed a new
primitive, that would have been worth arguing before building one.

## Running it

The corpus comes first; this project has nothing to read without it.

```bash
# 1. Build the corpus (this example ships three demo sessions that carry the
#    signal — copy them into the transcripts project's landing directory, or
#    point the source at your own corpus).
cp fixtures/landing/*.json ../agent_transcripts/fixtures/landing/
(cd ../agent_transcripts && stel build)

# 2. Derive the candidates.
stel build

# 3. Render the diff. Nothing is written without --write, and nothing is
#    ever committed.
stel suggest dbt \
  --from suggestions.dbt_doc_candidates \
  --dbt-project ./dbt_project
```

The bundled `dbt_project/` is a two-model fixture: `fct_orders` has no
description (the gap), and `dim_customers` already has one — the patching half
never overwrites an existing description, so it must come out of the run
untouched.

Offline by default: the profile uses the `deterministic` provider, so the
example runs in CI with no credentials and no spend. Descriptions worth
reading need a real provider — swap the `llm:` block for `vertex` or
`anthropic`.

## Rules this example holds

- **The threshold lives in the analysis, not only the CLI.** `--min-evidence`
  is a second gate at the edge. If the first gate were the CLI's, this project
  would draft a description for every one-session file it ever saw and pay a
  provider per draft for candidates nobody reads.
- **Sensitivity travels.** The exchange *body* is never extracted here. Only
  headings — the human's own prompt — reach the drafting model, so a proposed
  description cannot quote free text a session opted out of capturing.
- **Provenance rides along.** Every candidate carries how many sessions
  support it and which ones, because a reviewer's first question is always
  "where did this come from?"
- **A draft is a draft.** The description is model-authored; the human reading
  and merging the diff is the confirmation. That is the recorded resolution of
  rule 1 — no auto-promotion, rather than no model-authored text — and it does
  not foreclose requiring a corpus acceptance signal later.

## What this does not do

Column-level suggestions. `files_touched` names files, not columns, so
`dbt_column` is always NULL here — the contract's way of saying "the model's
own description". A column-level gap needs a signal that names a column;
the query log (#329 phase 1) is the likelier source.
