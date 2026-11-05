# Context assembly experiments

Status: experimental library API (constellations#744, ALE-283). Production
MCP serving does not apply these policies. The executable synthetic comparison
is [examples/context_assembly](../../examples/context_assembly/README.md).

## Question and ownership

Hold retrieval candidates fixed and test whether ordering, duplicated overlap,
or bounded adjacent evidence changes a model's ability to use the evidence.
[Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/) motivates the
question; its position effects are not assumed to hold for today's model.

`context_assembly.py` packs governed evidence; `context_assembly_eval.py`
compares policies and scores reviewed answers. Existing `retrieval_eval.py`
and `retrieval_compare.py` continue to own retrieval metrics and comparisons.
This experiment neither reruns retrieval per arm nor changes its ranking.

## Evidence and authorization

Read roots and a bounded neighbor pool through the existing governed repository
under the same principal, warehouse identity, resource, temporal constraints,
and business/entity scope as the original search. Pin source versions and
record candidate rank. Feed **full text**, not truncated MCP snippets, to
`Evidence.from_rows(registry_row, chunk_row)`. The constructor checks stable
IDs, the document/version join, chunk index and exact text offsets when present.
It snapshots rows defensively; it does not replace the full agent-context
contract validator or perform repository access.

`assemble_context` requires the serving `AuthorizationProvider`, `Principal`
and resource `policy_attributes`. Like `ContextService._can_read_pair`, it
authorizes both registry and chunk. Every selected neighbor is checked before
prompt construction or token counting, including evidence later skipped for
budget. A denial aborts the experiment; authorization is never a ranking
metric. `compare_context_policies` assembles all cases/arms before any model
call so a later arm's denial cannot follow paid generation of an earlier one.

Neighbor expansion matches document **and version**, then an index distance
of at most two. It sorts by document/version/index/context ID and adds at most
the explicit `max_neighbors` (maximum 64) across all roots. Roots retain
retrieval order unless document order is selected. A snapshot contains at
most 1,000 roots and 1,000 neighbors. The caller must bound its repository
read too: the assembler does not make an unbounded read safe after the fact.
Business/temporal/entity scope must already be applied to that pool; record
its identity in benchmark provenance. Contradictory rows for one context ID
and duplicate retrieved IDs are errors.

## Policy and token budget

`AssemblyPolicy` is strict, frozen Pydantic configuration:

```python
AssemblyPolicy(token_budget=4096)                         # relevance baseline
AssemblyPolicy(token_budget=4096, order="document")
AssemblyPolicy(token_budget=4096, remove_overlap=True)
AssemblyPolicy(token_budget=4096, order="document", remove_overlap=True,
               neighbor_radius=1, max_neighbors=8)
```

All comparison arms must share the budget. The policy hash includes every
option, the explicit versioned tokenizer identity and renderer version. The
rendered prompt is `prefix + JSON evidence array + suffix`; comparison adds
instructions and question to that measured prefix. **The entire string is
counted**, including source IDs, version IDs, locators, JSON escapes and
separators. Any chat template, tool schema, provider framing or requested
output allowance outside this string must be reserved by the driver before
choosing `token_budget`. A proxy tokenizer enforces the proxy's limit only.

Packing greedily admits whole chunks in policy order, skips those that do
not fit and continues. It does not truncate evidence or omit citation fields
to make room. An empty-evidence envelope that cannot fit fails explicitly.
`metadata_tokens` separately counts the prompt with empty text fields;
`text_tokens` sums isolated body counts. Token-boundary effects mean these
numbers are **not additive**. Their purpose is to expose metadata overhead,
including the heavy provenance identified by ALE-108.

Overlap removal is conservative: only an identical, already covered prefix
at the same version's explicit character offsets is removed. Without exact
offsets the chunk stays intact. Similar wording, neighboring indexes or a
different document version never prove overlap. A trimmed span retains its
original context/chunk IDs and advances its citation start. A fully covered
chunk records the packed context IDs covering it in an `overlap` omission.
It does not remove internal or trailing duplicate substrings. Recovery uses
the original source IDs and the prompt's adjusted locator.

## Answer generation and human review

Call `compare_context_policies` with frozen `AssemblyCase` objects and explicit
benchmark/corpus/judgment versions. Preserve ALE-281's existing reviewed
group/split mapping; the API also rejects shared groups or source documents
across development and holdout. Freeze an atomic-fact inventory before
inspecting answers, including periods, units, tolerances and passage locators.

`generator=None` performs packing only. A trusted `AnswerGenerator` supplies
an implementation/model/prompt identity, a `live` flag and `generate(prompt)`.
A live driver additionally requires `allow_live=True`; that driver must use
the existing provider/budget contracts and enforce an explicitly chosen spend
cap. It returns `GeneratedAnswer` with raw text and optional actual usage and
cost. Unknown values remain null. Native generation errors are sanitized.
No automatic judge or new provider credential routing is introduced.

A reviewer inspects the exact answer and cited immutable passages, then
creates an `AnswerReview` keyed by `(case_id, policy_id)`. It must match the
prompt hash, `answer_identity(answer.text)` and frozen fact inventory ID.
The trusted driver can replay cached observations to score those reviews
without paying for a second generation. If the driver instead returns different
text, the review does not apply: that row is marked `answer_review: "stale"`
with null scores and reviewer, the other rows are kept, and the aggregate counts
`stale_reviews`. A review that mismatches the prompt or fact inventory is known
before any call and still raises first. Preserve answers privately under
operator-controlled storage if asynchronous review requires them; this API
never writes payloads, caches, or files.

Apply [ALE-281's companion rubric](https://github.com/C00ldudeNoonan/astrolabe/blob/main/constellations/evals/financial_qa/v1/PILOT_SCORING.md):

- Give each distinct asserted claim a stable normalized ID. Record duplicate
  assertions separately; reject duplicate claim IDs in a review.
- Fact precision is correct claims / asserted claims. Fact recall is distinct
  correctly recovered required facts / required facts. Additional true claims
  may improve precision, but never recall.
- Evidence support is human-confirmed passage entailment / asserted claims.
  A matching document ID alone is insufficient. Supported claims need citations,
  and all cited context IDs must actually occur in that prompt.
- Record human answer relevance separately. For scoped absences, success means
  explicit abstention within the correct scope with no asserted facts. Report
  false abstentions on answerable cases separately.
- With no asserted claims, precision/support are null; answerable recall is
  zero. Absent-scope recall is null. Missing reviews remain null, never zero.

Reports keep per-case denominators, reviewer identity, latency, provider token
counts and cost where measured. Aggregates separate policies and splits and
show scored/measured case counts. Descriptive means are not significance
tests; use the existing downstream grouped bootstrap protocol before making
a held-out adoption claim. Report no ARES/PPI or judge/human agreement claim
without the corresponding independent validation.

## Artifacts, reproducibility and decision

`AssembledContext.artifact()` is `context_assembly/v1`; the comparison is
`context_assembly_compare/v1`. Both are explicit allowlists. They contain
IDs, hashes, omissions and counts, **never raw questions, evidence, answers,
locators, authorization rows or credentials**. `render()` is an explicit,
sensitive in-memory model input. Benchmark/corpus/judgment, tokenizer and
generator identity strings must be non-secret version identifiers, not paths,
prompts or credential references. The snapshot hash binds ordered roots and
the neighbor pool; the prompt hash binds exact rendering. Policy semantics,
rendering and fingerprint domains are versioned contracts.

The shipped comparison always records **defer adoption**. The synthetic
example establishes packing mechanics and test boundaries, not real answer
quality or a new serving policy. As of 2026-11-03, ALE-281's reviewed pilot
contains retrieval golds, four companion cross-section answer criteria and
three absence golds; it explicitly does not contain completed held-out
answer/evidence judgments. The four new answer criteria are development
cases. Therefore a measured model-quality winner cannot be inferred from it.

To complete ALE-283: pin a governed full-text candidate/neighbor snapshot for
the reviewed benchmark; freeze reviewed fact inventories for the unchanged
holdout; choose model, prompt and spend cap; generate the same frozen arms;
obtain passage-level reviews; publish a reproducible held-out comparison and
explicit adopt/defer decision. Choose any winner on development data before
scoring the untouched holdout. A selected serving policy requires its own
reviewed integration; these library experiments never silently select one.
