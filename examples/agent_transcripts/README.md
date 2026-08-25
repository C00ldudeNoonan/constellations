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

raw_transcripts
        v
retrieval_judgment_candidates       transform: built-in, #329 phase 3 —
                                    candidates a human promotes, never goldens
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
human step (issue #380).

## The grain

The session is the document; the *exchange* — one user prompt plus everything
it caused — is the unit a chunk attributes to. The converter renders each
exchange under a `## [<ordinal>] <prompt>` heading, so the built-in chunker's
heading attribution (issue #332) maps every chunk back to the request that
produced it; the ordinal prefix keeps repeated prompts ("continue") from
colliding. Tool exhaust never reaches the index: each call is reduced to its
name, an argument fingerprint, its outcome, and the byte count of the output
that was dropped.
