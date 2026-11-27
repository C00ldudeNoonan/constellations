# Summary pages are an offline experiment, and adoption is deferred

Status: accepted (constellations#826, ALE-285)

The issue proposed source-linked hierarchical summaries (RAPTOR-style) for
broad questions. The plausible alternatives were to ship a `page:` model type
that materializes pages as warehouse tables and serves them through MCP, or to
build a recursive clustering tree. Either would change production evidence,
authorization and readiness work before any measured benefit on this corpus.

Instead one level of extractive pages, one per document version, is built and
compared as a library experiment on frozen governed evidence, reusing ADR 0049's
assembly, budget and authorization checks so a page spends tokens from the
budget it competes in. The two rules that make pages safe to consider are
enforced in code and tests: a page is built only inside one authorization scope
(refused before the summarizer sees text), and every claim is a verbatim quote
of a cited chunk. A page's identity covers every input, so invalidation is an
identity comparison, and a page is servable only while it is `keep` in
`reconcile`.

**Decision: defer.** The offline fixture shows pages raising evidence presence
for broad and multi-section prompts, nothing for exact identifiers, and more
than doubling prompt tokens, with a page-to-source ratio above 1 on tiny
documents. That is mechanics, not a measurement. The reviewed held-out set has
two cross-section and zero exact-identifier prompts (astrolabe#1884), and
answer-level scores need human review (ADR 0049). A later measured run that
shows a win on held-out broad and multi-section prompts without losing
exact-identifier recall would justify a `page:` model type, with its own ADR
and issue; this one records that it is not yet justified.

See [the experiment contract](../architecture/summary-pages.md).
