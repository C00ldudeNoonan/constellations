# ADR-0036: Hybrid search fuses documents, not chunks

- **Status:** accepted
- **Date:** 2026-10-20
- **Prompted by:** #701 (found by #696)

## Context

Hybrid search runs a vector leg and a text leg, each up to `candidate_limit`
chunks, and fused them with reciprocal rank fusion (RRF, k=60) over chunk
ids. #696 benchmarked LanceDB against OpenSearch on astrolabe's 41 labelled
SEC golden questions (213,386 chunks, 1,192 filings). On both stores, hybrid
recall@5 at document grain came out at 0.54–0.58, below text alone at 0.62.
Fusion ran identically on the client for both stores, so the fusion was the
cause.

The mechanism is arithmetic. With k=60, a chunk ranked *r* in both legs scores
2/(60+r). A chunk ranked first in one leg and absent from the other scores
1/61. The first wins whenever r ≤ 61. With 80 candidates per leg, **any chunk
in the top 61 of both legs outranks each leg's single best chunk**. k=60
comes from fusing runs that are 1,000 documents deep. Over 50–80 chunk
candidates it makes "present in both legs" count far more than "ranked first
in one".

Nearly every golden question is scoped to one company, so both legs choose
among that company's filings. The relevant filing's best chunk is often a
different chunk in each leg. Boilerplate that ranks middling in both legs, from
the company's other filings, takes the top fused positions.
`abbv_annual_report` scored recall 1.0 from vector, 1.0 from text and 0.0
from hybrid.

## Decision

When the index declares a `document_id_field` (the default), fuse documents:

1. Collapse each leg to documents, at the rank of their best chunk.
2. Fuse the two document rankings with RRF, keeping k=60 and equal weights.
3. Expand back to chunks interleaved: every document's best chunk in document
   order, then every document's second chunk, and so on.

Within a document, chunks keep their chunk-grain RRF order. A chunk's `score`
is its document's fused score, normalized to 1.0 for a document first in both
legs. `contributing_ranks` stays the chunk's own rank per leg. A row with no
document id is a document of its own. An index with `document_id_field: null`
keeps chunk-grain RRF. Fusion across several `full_text.fields` stays at chunk
grain: those legs rank the same chunks by different text and do not have the
problem.

## Alternatives considered

Every candidate was scored offline against the same dumped legs (see
Evidence), at document grain @5, against shipped chunk-grain RRF at 0.558 /
0.639 / 0.567 (recall / MRR / nDCG).

### (b) Keep chunk grain; lower k or weight the text leg

This is the smallest change and keeps chunk grain. The best grid point,
k=20 with w_text=2.0, scored 0.623 / 0.641 / 0.590. At equal weights the best
k (20) reached only 0.577 nDCG, so most of the gain comes from weighting text
over vector. That weight is a property of this corpus and its embedder, not a
default to ship.

Tuned leave-one-out (pick the point on 40 questions, score the 41st), (b)
scored 0.599 / 0.630 / 0.574. That recovers about a fifth of document
fusion's nDCG gain, and it needs a per-index weight nobody can set without a
golden set. It lost because it treats a symptom of the k/depth mismatch, not
the grain mismatch.

### (a) with tuned k or weights

Document fusion barely moves with k at equal weights: nDCG@5 stays between
0.599 and 0.606 for k from 20 to 100. The best tuned point, k=100 with
w_text=3.0, scored 0.631 / 0.662 / 0.607, the same nDCG as the untuned k=60.
Leave-one-out tuning scored *lower* (0.583) than not tuning: across 41
questions it picks noise. So the RRF constants stay where they are.

### (a) expanded grouped instead of interleaved

Grouped means all of document 1's chunks, then document 2's, and so on. It has
the same document metrics as interleaved, because the document order is
identical. It differs for record-grain callers, MCP `search_context` above
all, which returns chunks (default `limit=10`). Grouped, the ten chunks span
2.46 filings on average and 29/41 questions get a relevant chunk in them.
Interleaved spans 6.44 filings, with 34/41. Shipped chunk RRF spans 5.59, with
30/41. Grouped also makes `stel eval`'s document widening fetch deeper
candidates than MCP sees, so its eval numbers would describe a different
pool.

### (c) Score-normalized fusion

This is a convex combination of per-leg min-max or z-score normalized raw
scores, at chunk grain or with each document's best chunk. The best grid points
(0.615 / 0.674 / 0.600, z-score at document grain, α=0.6) were at most level
with document RRF. Leave-one-out they were no better: 0.574–0.583 nDCG at document
grain, 0.562–0.566 at chunk grain. Raw scores mean different things per
store and per score kind: BM25 is unbounded, and DuckDB, LanceDB and
OpenSearch report different distance kinds. Min-max also moves with candidate
depth. Fusing on raw scores would tie ranking to the store, which
`retrieval/` exists to keep out of core.

## Consequences

- **`score` is not monotonic in rank under hybrid.** A document's second chunk
  scores above the next document's first. Rank was already the portable
  ordering (see `docs/architecture/semantic-retrieval.md`); now the
  difference is visible. Nothing in stel sorts by `score`. A consumer that
  does will see grouped order.
- **Hybrid record-grain results diversify.** MCP's top ten span more filings,
  and a question answered by two chunks of one filing gets the second chunk
  after every other filing's best. A caller that wants depth in one document
  filters on its id or reads it with `get_document`.
- **The gain is not uniform.** Against chunk RRF, document fusion wins nDCG@5
  on 6 questions and loses on 5. The mean rises because of the questions the
  mechanism hit hardest: `abbv_annual_report` 0.00→1.00,
  `mpc_annual_report` 0.00→0.50, `dow_net_loss_outlook` 0.20→0.65. The losses
  are rank 1↔2 swaps of the right filing, the largest being
  `ffiv_revenue_guidance` at 0.95→0.70. In MCP's top ten, 4 questions go from
  no relevant chunk to one, and 4 see their first relevant chunk move one or
  two places later.
- **`document_id_field` now changes hybrid ranking**, not only result fields
  and eval granularity. Setting it to null to drop the field also reverts to
  chunk fusion.
- **Existing indexes rank differently on upgrade, with no republish.** The
  change is in the query, not the collection. A pinned `stel eval` baseline
  on a hybrid model will move.

## Evidence

Measured 2026-10-20 on #696's slice (`sec_slice.parquet`, 213,386 chunks,
the stored 768-d `text-embedding-005` vectors) and its 41 labelled golden
questions. The harness, `fusion701.py`, is posted on #701.

- It builds a fresh no-ANN LanceDB 0.34.0 store and dumps each question's
  vector and text legs at depths 50, 80, 320 and 1000. It then replays
  `retrieval_eval._retrieve_ranking` offline with each fusion and scores it
  with stel's `retrieval_metrics`.
- The baseline calls stel's own `_rrf`. It reproduces #696's hybrid row at
  0.558 / 0.639 / 0.567 against 0.550 / 0.639 / 0.563. The vector leg
  is bit-identical. The text leg differs in near-tie order on 4 questions
  after rebuilding the FTS index, and every candidate is scored against the
  same rebuilt legs.
- The shipped `_rrf_by_document` was run against those legs too. It ranks
  identically to the harness's document-RRF k=60 interleaved candidate on all
  164 (question, depth) pairs, at 0.615 / 0.678 / 0.606. For reference, text
  alone scored 0.623 / 0.627 / 0.585 and vector alone 0.566 / 0.578 / 0.541.
- One question moves recall@5 by about 0.024. The differences between the
  better candidates are within a question or two, which is why the decision
  rests on the mechanism and on needing no tuning, not on the best grid
  point.
