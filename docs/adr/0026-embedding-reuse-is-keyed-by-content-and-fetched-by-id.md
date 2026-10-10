# ADR-0026: embedding reuse is keyed by content and fetched by id

- **Status:** accepted
- **Date:** 2026-10-09
- **Prompted by:** #665

## Context

Embed reuse existed to stop a run re-paying a provider for text it had already
embedded. It looked the existing target up by row id — `WHERE <id_field> IN
UNNEST(?)` — and compared `embedding_input_hash` to the row's text hash
afterwards. The hash was a *verification*, not the key.

That is wrong whenever ids move independently of text. astrolabe's SEC corpus
moved its embedding identity onto `context_id` for the agent_context wrapper
hop: byte-identical text, identical `embedding_input_hash`, brand-new id
space. Every lookup missed, every vector was discarded, and 3.67M chunks were
embedded about 1.9 times — 7.78B input characters against a single pass's
~4.1B, roughly $195 against $100.

The same lookup is also the most expensive read in the embed path. On that
table `context_id` is neither the partition nor a cluster key, so each lookup
billed 21.3 GiB, almost all of it the 768-float vector column (#664).

## Decision

**The reuse key is `embedding_input_hash`. The row id is how a row is
fetched, not how it is found.**

One streamed, projected pass over the target builds two maps: the id column's
typed values by stringified id (unchanged, still needed to delete removed rows
and to push keyed predicates), and one representative typed id per distinct
text hash **among rows recorded under this run's `embedding_config_hash`**. A
window then asks for its text hashes, those are resolved to ids in memory, and
one keyed `IN` read per 10,000 ids pulls the reuse columns. The results come
back keyed by each row's own `embedding_input_hash`.

Three things follow:

- A re-keyed or re-chunked row whose text did not change reuses its vector.
- A hash the target does not hold is dropped before any read, so new text
  costs no vector read — the index answers it in memory.
- The caller's `embedding_input_hash == text_hash` check is now true by
  construction and was removed.

### Why the index filters by configuration

One representative id per hash is enough *within a configuration*: every row
carrying a hash under the same config holds a vector for the same input text,
so any of them answers.

Across configurations it is not. A target can hold the same text twice under
two configs — duplicate text, plus a config change, plus a publish interrupted
partway — and an index that kept one representative per hash without filtering
kept whichever row the unordered scan saw last. When that was the old-config
row, the lookup fetched it, the caller rejected it, and stel paid to re-embed
text the target already held a current vector for. Row-id keying never had
this problem, because each row found its own vector; collapsing to one
representative per hash is what introduced it (Codex review on #671).

Filtering at index time is therefore part of the decision, not a detail. It is
also strictly cheaper than the obvious repair: residency stays one value per
hash — fewer entries, since stale-config rows are now excluded entirely — and
a row that would be rejected is never fetched.

The call-site `embedding_config_hash` check is kept and is now **deliberately
redundant** with that filter. The index is a snapshot of a table the run keeps
writing to, so the check is a staleness guard rather than the primary rule,
and reusing a vector from another configuration would be a wrong answer rather
than a slow one. Being redundant it is not pinned by a test of its own, which
the code says in place; the configuration tests exercise the index filter.

## Alternatives considered

### Predicate on `embedding_input_hash` instead of resolving to ids

The obvious reading of "key it by the hash", and it would drop the in-memory
hash map. Declined for two reasons. The predicate contract and every adapter's
keyed-read path are built around the model's `id_field`; pointing them at a
second column means each adapter learns a new keyed-read shape for one caller.
And it loses the pre-filter: a hash the target does not hold would still be
sent to the warehouse, which is the read this change exists to avoid. On
BigQuery neither column is a cluster key, so the predicate column buys no
pruning by itself — that is #664's work, and it applies to whichever column
the predicate names.

### Keep the row-id lookup and add a content lookup as a fallback

Smaller diff: try the id, and on a miss try the hash. Declined because it
keeps the expensive case as the default. A row that exists by id but whose
text changed would still be fetched — vector column included — only to be
rejected on the hash check, which is precisely the wasted read on the largest
column. Deciding by hash first makes the common "text changed" case free.

### Index the hash truncated, to halve residency

The second map costs one entry per distinct text hash. A 64-bit prefix would
roughly halve its key bytes, and a prefix collision would be self-correcting
(the fetched row's full hash is what the result is keyed by, so a collision
costs a wasted read, never a wrong vector). Declined for now as an
optimization without a measurement behind it: the existing id map is the same
cardinality and has been acceptable since #401, and a truncation that nobody
has shown is needed is a mechanism that has to be explained forever. Revisit
if a corpus makes the key scan the memory ceiling.

### Dedupe identical text within a run as well

The index is built before the first window, so two rows with identical text in
the *same* run do not reuse each other's vector — the second one re-embeds. On
a corpus with boilerplate (SEC filings have plenty) that is real money.
Declined here as the wrong layer: this is a durable provider response cache,
the embed-side twin of `llm:`'s `cache_path`, which #665 asks for separately
and which also fixes the crash-before-flush case this decision does not touch.

### Retain every candidate id per hash until a matching config is found

The review's own suggestion for the shadowing bug, and it works. Declined for
cost: a list per hash turns the index from one value per distinct text into a
list object per distinct text — on a corpus of mostly-unique text that is a
one-element list and its overhead for every row, on the path that exists
because the corpus was already too large (#401). It also keeps fetching rows
that will be rejected, since the config is only known once the row is read.
Filtering at index time answers the same question with less memory than the
unfiltered version and no wasted fetch.

### Index on the `(text_hash, config_hash)` pair

The same correctness, and it would let one reader serve two configurations.
Declined as unused generality: a run has exactly one `embedding_config_hash`,
fixed before the reader is built, so the pair's second element would be
constant across every entry — a tuple key per row to express a value the
reader already holds as a field.

### Re-verify the text hash at the call site anyway

Harmless, and it would survive a future refactor that broke the keying.
Declined: a check that cannot fail is indistinguishable from one that does not
work, and the invariant belongs where it is established — the reader keys rows
by the hash they carry, and skips a row with no usable hash.

This is deliberately *not* the same judgement as keeping the redundant config
check above, and the difference is what the redundancy is against. The text
hash is established in the same function that returns the row: the reader
keys by the hash it just read off that row, so no state can intervene. The
config filter is established against a snapshot of a table the run keeps
writing to, which is a weaker guarantee — the kind a staleness check earns its
place against.
