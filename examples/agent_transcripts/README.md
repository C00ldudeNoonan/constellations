# agent_transcripts

Answers "what did I try last time I touched that file?" (issue #360): agent
sessions from Claude Code and Codex become an exchange-attributed, governed
search index, using only shipped primitives. The only transcript-specific code
in stel is the converter behind `stel transcripts convert` / `sync`; this
project is ordinary composition downstream of its landing directory.

```bash
uv sync --extra lancedb
uv run stel --project-dir examples/agent_transcripts run
uv run stel --project-dir examples/agent_transcripts search \
  --model transcript_search --query "rounding test" --mode text
```

No network credentials are required — a local duckdb warehouse, a local
LanceDB store, and the offline `deterministic` embed provider. The fixture
landing documents were produced by the real converter from synthetic
transcripts.

## Feeding it your own sessions

```bash
uv run stel transcripts sync --out examples/agent_transcripts/fixtures/landing
```

converts every settled session under `~/.claude/projects` and
`~/.codex/sessions` (a file modified in the last five minutes is a live
session and is skipped). In a real deployment the landing directory lives
outside the project — point the source at it with `external: true` — and a
SessionEnd hook runs `stel transcripts convert <transcript> --out <landing>`
per session.

## DAG

```
raw_transcripts          extraction: backend json over transcript/v1 landing files
        |
        +-----------------------------+
        v                             v
document_registry            transcript_chunks    chunk: recursive, headings '^## (\[\d+\] .+)$'
transform:, agent_context:           |            — each chunk's `section` names its exchange
  grain document_registry            |
        +-------------+--------------+
                      v
              document_chunks       transform:, agent_context: grain document_chunks;
                      |             incremental with the registry as a keyed
                      v             reference dep (issue #364)
              chunk_embeddings      embed: provider deterministic
                      v
              transcript_search     search: governed, filter/display attributes
                                    harness, exchange_heading, tools_used,
                                    files_touched

raw_transcripts                     #329 phase 3 — candidates a human
        |                           promotes, never goldens
        +-----------------------------+
        v                             v
retrieval_judgment_candidates   correction_inputs   transform: built-in —
transform: built-in — one row           |           exchange prose plus the
per returned id (issue #380)            v           ids it retrieved (#456)
                                drafted_corrections  llm: did the human
                                        |            correct a stated value
                                        v
                        classification_label_candidates
                                    transform: sql — the shape
                                    `eval.expected` reads
```

## Candidate retrieval judgments

When a session queries stel's own governed context, the transcript records
which ids came back and which the answer then named. `stel transcripts`
captures that (never the result bodies), and
`retrieval_judgment_candidates` reshapes it into one row per returned id:

| judgment | what it is | what it is not |
|---|---|---|
| `cited` | the answer named this id after the call returned it | |
| `returned_not_cited` | returned beside it, not named | **not** evidence of irrelevance — an agent may use a chunk without quoting its id |
| `zero_result` | the query matched nothing | |

Failed searches produce no rows at all: a denied or timed-out call returned
nothing because it failed, not because the corpus lacked a match.

Nothing here is read by `retrieval_tests:` or `eval:`, and nothing promotes
itself at any confidence — that is #329 rule 2. Promotion is a separate,
human step (issue #380); `stel promote` drafts a golden set from these rows
for a human to review.

## Candidate classification labels

The `eval:` half of the same phase (issue #456), in three models, and the
split is the point:

| model | kind | what it does |
|---|---|---|
| `correction_inputs` | `transform:` built-in | one row per exchange: its prose, and the context ids it retrieved |
| `drafted_corrections` | `llm:` | did the human correct a stated value, and to what |
| `classification_label_candidates` | `transform:` sql | the shape `eval.expected` reads |

**The ids are the constraint.** `eval:` joins expected labels to predictions
on a key, so a correction is only ground truth if it attaches to a *record*.
The ids an exchange retrieved are the only record identity a transcript
names — so an exchange that retrieved nothing produces no candidate, however
clearly it corrects something. That is a real limit on what this can derive,
not an oversight.

**Only an explicit correction counts.** Agreement, restatement, and a fresh
question are not corrections, and the prompt says so at length: an absent
label costs nothing, while a wrong one becomes ground truth a future model is
scored against.

That last rule lives in the prompt, so the offline `deterministic` provider
this example ships with cannot demonstrate it — it returns a stub for every
row. The example proves the chain and the contract; judging what is actually
a correction needs a real provider.

## The grain

The session is the document; the *exchange* — one user prompt plus everything
it caused — is the unit a chunk attributes to. The converter renders each
exchange under a `## [<ordinal>] <prompt>` heading, so the built-in chunker's
heading attribution (issue #332) maps every chunk back to the request that
produced it; the ordinal prefix keeps repeated prompts ("continue") from
colliding. Tool exhaust never reaches the index: each call is reduced to its
name, an argument fingerprint, its outcome, and the byte count of the output
that was dropped.

## Serving from DuckDB instead of LanceDB

`profiles.yml` carries a second target, `dev_duckdb`, that publishes the same
index to a DuckDB-native store (issue #371):

```bash
stel build --target dev_duckdb
```

Nothing under `models/` changes. The search model names a store *alias*, not a
store type, so which engine serves it is a target concern — which is the point
of the exercise: the switch is configuration, not model logic.

The DuckDB target points its retrieval store at `target/transcripts.duckdb` —
the warehouse file itself. Vector search (`vss`) and BM25 (`fts`) run in the
same database as the canonical rows, so this target stands up no second
system; `target/lancedb/` is never created.

Both targets return identical results for vector, text, and hybrid queries
over this corpus, which `tests/test_duckdb_search_example.py` asserts rather
than asserts about. Hybrid is worth noting: neither store computes it. Each
serves the two legs and stel fuses them with RRF, so the fused ranking
matching across stores is a property of the legs agreeing, not of a shared
implementation.
