# Summary page experiments

Status: experimental library API (constellations#826, ALE-285). MCP serving
does not read pages, and there is no `page:` model type. Decision: **defer**
(see [ADR 0067](../adr/0067-summary-pages-are-an-offline-experiment-and-adoption-is-deferred.md)).
The runnable synthetic comparison is [examples/summary_pages](../../examples/summary_pages/README.md).

## Question

Do source-linked summaries help broad and cross-section questions enough to
pay for their build, tokens and upkeep? The representation is a RAPTOR-inspired
one-level baseline ([ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html)),
shaped as a page per source document version, not a table across documents and
not a recursive cluster tree. Retrieval is not rerun.

## Ownership

`summary_pages.py` builds, identifies and reconciles pages.
`summary_pages_eval.py` runs two arms through `context_assembly.assemble_context`
under one token budget, tokenizer and authorization provider: passages, and
pages plus passages. It reuses `AssemblyCase` and the ADR 0049 evidence and
authorization checks. `retrieval_eval.py` and `retrieval_compare.py` still own
retrieval metrics.

## Contract

**Source links.** A page is a list of claims. Each claim is a context ID and a
quote that must be a verbatim substring of that chunk; anything else is dropped
at build and counted in `rejected_claims`. `resolve_claims` maps every claim
back to its source `Evidence`, and fails when the source no longer holds the
quote. A page is therefore a pointer to where to look, never a substitute
source.

**Authorization scope.** `build_page` requires one document ID, one version and
one value of every field in `SCOPE_FIELDS` across every registry and chunk row.
Anything else raises `PageError` *before the summarizer is called*, so text
cannot cross a boundary through a summary. The page's own row pair copies its
sources' scope fields, so `assemble_context` authorizes it with the same
provider as the sources, and a principal who cannot read the sources cannot
read the page. Custom policy attributes outside `SCOPE_FIELDS` are not covered;
an adopter must extend that list first.

**Identity and invalidation.** `page_id` fingerprints document, version, scope,
the ordered (context ID, text hash) of every source chunk, the summarizer
identity and the schema version. It is computable without summarizing
(`page_identity`). `reconcile(pages, current, summarizer_identity=...)` compares
stored pages with the live corpus and returns `add`, `rebuild` (with
`source_changed`, `permission_changed` or `summarizer_changed`), `prune` (source
deleted) and `keep`. Only `keep` is servable: a page whose source changed, whose
scope moved, or whose source is gone is withdrawn the moment the corpus
changes, not when it is rebuilt. A document whose chunks now span scopes
cannot be paged and is reported as `permission_changed`.

**Mock summarization.** `LeadSentenceSummarizer` is deterministic and offline.
A live `Summarizer` needs `allow_live=True` and must enforce its own spend cap.
Default tests call no provider.

## Report

`PageComparison.artifact()` (`summary_page_compare/v1`) carries IDs, counts and
timings, never a question, page or document text:

- quality proxy: `fact_coverage`, the share of required literals present in the
  prompt, by question class (`broad`, `multi_section`, `exact_identifier`);
  `page_only_facts` and `unresolved_page_facts` (the second must be 0);
- faithfulness: `claim_faithfulness`, kept / (kept + rejected) claims;
- cost: `token_count` per arm, `summarizer_calls`, `reused_builds` (identity
  cache hits), `page_to_source_ratio`, `build_seconds`, `assemble_seconds`;
- maintenance: a `Maintenance` from `reconcile` after a corpus change.

`fact_coverage` is evidence presence in the prompt, not answer quality. It
cannot say whether a model uses a page well. `decision` is always the `defer`
string until a reviewed held-out set scores answers.

## What the offline run shows, and does not

On the three-case synthetic fixture pages lift coverage on the broad and
multi-section cases, add nothing on the exact-identifier case, and cost about
2.7 times the prompt tokens. The page-to-source ratio is above 1 because the
fixture documents are tiny and each claim carries a context ID: the "grew past
its sources and abstracted nothing" failure the issue warns about, so the
ratio is reported rather than assumed below 1. These are mechanics, not
evidence about the SEC corpus. The measured filing run is blocked on the
benchmark: astrolabe#1884 recorded two held-out cross-section prompts and zero
held-out exact-identifier prompts, too few to support an adopt/defer call.
