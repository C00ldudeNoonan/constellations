# ADR-0012: The native detail behind a sanitized failure goes to a file the operator named, never to a log level

- **Status:** accepted
- **Date:** 2026-09-20
- **Prompted by:** #590

## Context

A sanitized failure keeps the operation, the step and the native exception's
type, and drops the native text: LanceDB quotes object-store URIs and response
bodies verbatim, the message reaches `run_results.json` and the CLI, and
AGENTS.md requires sensitive exception text to stay out of logs and
artifacts. The native exception is written once, at DEBUG with `exc_info`,
and `logging_setup.py` caps `-v` at INFO by design so that record reaches
nothing. Its docstring names the hatch: a caller who needs DEBUG attaches
their own handler.

That hatch exists only for an in-process Python caller. Every orchestrated run
invokes `stel build` as a subprocess, which is how the code executes in
production, so for the operator who actually hits these failures the cause is
written to a logger nothing can be configured to receive. Two production
failures of one model, eight days apart (astrolabe#705), stopped at
`[RuntimeError] (code=lancedb_index_failed)` and
`(code=lancedb_upsert_failed)`; root-causing either needed the native text,
and nothing available to the operator could produce it.

## Decision

`--diagnostics-file PATH`, also `STEL_DIAGNOSTICS_FILE`, attaches a DEBUG
handler to the `stel` logger that writes only the records carrying an
exception, plus warnings, to that one file, appended and created owner-only.
Nothing else changes: the CLI stream, `run_results.json`, the `-v` handler
and every artifact stay as sanitized as before. While the handler is
installed the logger stops propagating, so the DEBUG records the file exists
for cannot also reach a parent handler such as an orchestrator's capture, and
a plain WARNING stderr handler stands in for `logging.lastResort` so warnings
still appear where they did. A `run` or `build` failure that wrote to the
file names it in the error message. The default, with nothing set, behaves
exactly as it did.

## Alternatives considered

### Raise the `-v` cap to DEBUG, or add `-vv`

The obvious knob, and the one `logging_setup.py` refuses on purpose. It would
open every `exc_info` site in the tree into a channel the operator does not
control: `-v` writes to stderr, and an orchestrator captures stderr into its
own event log, which is the exact fan-out the invariant exists to prevent. A
file the operator named is a deliberate act with a known destination; a log
level is a policy applied to whatever is listening.

### Classify the known causes instead

The `_PQ_MINIMUM_ROWS` precedent in `lancedb.py`: recognise a native error
and raise a safe, specific message for it. This is the right end state, and
this decision does not preclude it. It lost as the first step because none of
the real causes was known; a classifier could only be built for hypothesised
ones, and would have helped neither production failure. The diagnostics file
is what makes the classifications knowable rather than guessed, so it comes
first and feeds them.

### Carry the native text in the error's cause chain or in `run_results.json`

Would reach the operator with no new flag. It lost because both are exactly
the surfaces the sanitization protects: `run_results.json` is an artifact
that outlives the run and is read by things that are not the operator, and a
cause chain prints wherever the exception does.

### An explicit `diagnostics.record(error)` call at each sanitize site

Precise, and it would not touch logger levels or propagation at all. It lost
because the `log.debug(..., exc_info=...)` convention already marks every
site, and a handler on that convention covers the store, the index-build
retry, document fetch and extraction and transform code at once, including
sites added later that follow the convention without knowing about the file.
An explicit call is one more thing a new sanitize site must remember.

## Consequences

- The file holds what every other channel withholds. It is created `0600`,
  an existing file is re-moded to `0600` on open since `O_CREAT` cannot
  change an inode's mode, and it is documented as sensitive; an orchestrator
  that names it must treat the path the way it treats a credential file, and
  the file is never a stel-owned artifact that `stel clean` removes.
- The handler fails closed. Its first open and every write happen while the
  native exception is being handled, and the stdlib's own error report prints
  that exception's chain to stderr, so the handler contains both and writes
  one safe line instead. An unwritable destination is refused before the run
  starts, for the environment variable as much as the flag.
- The `stel` logger's level and propagation are now derived from which
  handlers are installed, in `_apply_channel_policy`. The two configure
  functions may be called in either order and any number of times; a
  contributor adding a third channel adds it there, not in either function.
- Propagation off without `-v` means a parent handler an in-process caller
  installed receives nothing from `stel` while the file is configured, not
  even warnings, which go to the stderr fallback instead. That is the point
  of the design, and it is a behaviour change only for a caller who sets the
  diagnostics file and expects propagation too.
- Provider errors are not in the file. The provider layer sanitizes before
  any logger sees the native exception and has its own allowlisted hatch,
  `STEL_DEBUG_PROVIDER_ERRORS`. Bringing provider detail under this file is a
  separate decision about that layer.
- The record filter is `exc_info is not None or level >= WARNING`. A new
  sanitize site that logs the native exception without `exc_info` will not
  reach the file; the convention is now load-bearing.
