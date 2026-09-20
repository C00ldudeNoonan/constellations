# ADR-0012: A failure's detail goes to a path the operator names, not to a log level

- **Status:** accepted
- **Date:** 2026-09-20
- **Prompted by:** #590 (found through astrolabe#705)

## Context

stel sanitizes store and provider failures at the boundary. `_operation_failed`
keeps the operation, the step, and the exception *type*, and drops the native
message, because LanceDB quotes object-store URIs and response bodies verbatim
and the sanitized message reaches `run_results.json` and the CLI. That is
right and this ADR does not revisit it.

The native exception is not discarded — it goes to `log.debug(..., exc_info=)`.
But `logging_setup` caps `-v` at INFO deliberately, for the same invariant, and
directs anyone needing more to attach their own handler.

**That hatch requires being an in-process Python caller.** A caller running
`stel build` as a subprocess — an orchestrator, which is how a long build
actually runs — cannot attach anything to a logger inside a process it only
spawns. So in the one situation where the detail matters most, nothing can
receive it.

The cost was measured, not hypothesized. `sec_chunk_search` failed in prod
twice, eight days apart (astrolabe#705):

```
lancedb_index_failed   BTree index on 'context_id', 3 retries exhausted
lancedb_upsert_failed  first batch of a resumed generation
```

Neither could be root-caused. The issue sat open for a week, and the second
failure was diagnosed only far enough to rule out memory (RSS 2.2 GB against a
27.9 GB limit). Root-causing either requires knowing what the native
`RuntimeError` said, and nothing available to the operator could answer.

## Decision

**The detail goes to a filesystem path the operator names**, via
`--diagnostics-file` or `STEL_DIAGNOSTICS_FILE`. Off unless a path is given.

**What is written is `redacted_exception_text`'s output, not native text** —
allowlisted exception type labels, stel source locations, and a count of
external frames. The same allowlist the provider debug path already ships.
Imported rather than reimplemented, so the two disclosure paths cannot drift
into disagreeing about what is safe.

**The sink absorbs its own failures.** An unwritable path is logged at DEBUG
and otherwise ignored; the error the caller was reporting is raised unchanged.
The file is created `0600`.

## Alternatives rejected

**Raise the `-v` cap to DEBUG.** The obvious one-line change, and the one a
contributor reaches for first. It re-opens exactly the hole
`logging_setup`'s docstring closes — every `exc_info=True` site in the tree,
in provider code and `execution/transform.py` too — and it opens it into a
channel the operator does not control. Logs fan out: a captured run ships
them to an aggregator, and a stel-shaped log record ends up somewhere nobody
chose. A path is a destination someone picked.

**Write the raw native text to the file instead.** Tempting, since the
operator chose the destination and the text is what actually answers the
question. Rejected because `redacted_exception_text` already settled this and
its reasoning does not weaken when the sink changes: "exact-value replacement
cannot safely redact repr-, JSON-, or URL-encoded request data." The
credential in a presigned object-store URI survives every redaction pass you
can write for it. A destination being deliberate does not make its content
safe, and a diagnostics file is exactly the artifact that gets pasted into an
issue.

**Classify known causes instead of disclosing anything**, extending the
`_PQ_MINIMUM_ROWS` precheck pattern. This is the better end state and should
follow. It was not viable *first*: we know none of the real causes, so it
could only be built against hypothesized ones, and it would not have helped
either failure above. The sink is what makes the classifications knowable
rather than guessed.

## Consequences

Every sanitized failure becomes diagnosable by an operator who opts in, not
just the two LanceDB codes this was found through — `record_failure` sits in
the shared error builder.

A future `_operation_failed`-shaped helper elsewhere in the tree does not get
this automatically; it has to call `record_failure` too.

`stel.diagnostics` imports from `stel.providers.base` for the redaction
helper, which is the wrong direction. The helper is provider-agnostic and
belongs in a leaf module; moving it is deferred so this change stays one idea.
