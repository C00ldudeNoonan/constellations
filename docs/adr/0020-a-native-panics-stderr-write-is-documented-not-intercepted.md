# ADR-0020: A native panic's stderr write is documented as outside the sanitizer, not intercepted

- **Status:** accepted; amends [ADR-0012](0012-native-failure-detail-goes-to-an-operator-named-file.md)
- **Date:** 2026-10-03
- **Prompted by:** #648

## Context

ADR-0012 routes the native detail behind a sanitized store failure to one
operator-named file and promises that the CLI stream, `run_results.json`,
the `-v` handler and every artifact stay sanitized. That promise is kept by
Python code: every LanceDB call goes through `_operation_failed`, which
re-raises a `RetrievalError` carrying only the operation, the step and the
native exception's type.

A Rust panic inside LanceDB's native extension does not start in Python.
Rust's default panic hook writes the panic to the process's stderr on the
thread that panicked, and only then does the error cross into Python as a
`RuntimeError`. The #598 probe study noticed this in passing; #648 records it.
Nothing stel installs sees that write, so stderr is outside the invariant
AGENTS.md states, and an orchestrator that captures a subprocess's stderr
captures it.

## Decision

Document the gap and leave the channel alone. The AGENTS.md security
invariant names stderr as the known exception, the `--diagnostics-file`
reference says that file does not see the write, and a test pins the half
stel does own: a panic-born `RuntimeError` goes through `_operation_failed`
like any other native failure, with no panic text in the message or in any
INFO-or-louder record, while the same test observes the stderr write at the
file-descriptor level so a lancedb that stops producing it flags these notes
for removal.

## Alternatives considered

### Redirect file descriptor 2 around store calls

`os.dup2` the process's stderr onto the diagnostics file (when configured) or
`/dev/null` (when not) inside the store's context manager, so Rust output
Python cannot intercept lands where native text is allowed. It would work for
this write. It lost because it also swallows every other stderr write made in
the same window -- the WARNING fallback handler ADR-0012 installs, a
subprocess's output, Python's own fatal-error reporting -- and does so
process-wide, including for an in-process caller who did not ask. The
observed text (below) carries no store path, so the cost is not justified by
the evidence. If a panic site is ever shown to print one, this is the option
to revisit.

### Ask upstream for a configurable panic hook

The only fix that keeps the message *and* stel's control over where it goes.
`lancedb.Session` exposes cache sizes and nothing about panics;
`RUST_BACKTRACE` governs the backtrace, not the message line. Worth filing if
the evidence changes; not something this repository can decide.

### Treat it as already covered

ADR-0012's text reads as if every channel is sanitized. Leaving it that way
would have the next reader of a captured panic re-derive why the sanitizer
"failed". Writing the gap down is the whole point of the record.

## Consequences

- stderr is explicitly outside the sanitizer's guarantee. An orchestrator
  that captures a `stel build` subprocess's stderr into its own event log
  should treat it as native output, the way it treats the diagnostics file.
- The pin in `tests/test_lancedb_diagnostics.py` asserts a third-party
  behaviour on purpose. If it fails on a lancedb upgrade because no panic
  reaches stderr, remove the three notes rather than weaken the test.
- ADR-0012's promise is narrowed, not broken: everything Python emits is as
  sanitized as before.

## Evidence

lancedb 0.34.0 / lance 8.0.0, local filesystem store, 2026-10-03. A
stel-published 5,000-row collection, one data file, four corruption shapes,
each in a fresh subprocess with stderr captured:

| shape | Python received | stderr |
|---|---|---|
| truncate the last 16, 200 or 4096 bytes, or to 90% | `RuntimeError: lance error: LanceError(IO): ... failed to fill whole buffer` | nothing |
| 4 KB of zeros mid-file | `RuntimeError: task 100 panicked with message "range end index 4096 out of range for slice of length 0"` | `thread 'lancedb-tokio-worker' (570) panicked at .../lance-encoding-8.0.0/src/data.rs:334:50: ...` |
| 4 KB of garbage at the head | `RuntimeError: task 75 panicked with message "the offset + length of the sliced Buffer cannot exceed the existing length"` | two worker-thread panic blocks, `.../lance-encoding-8.0.0/src/buffer.rs:290:9` |
| truncated index files, then an indexed search | `RuntimeError: lance error: LanceError(IO): Generic memory error: Invalid range 0..1269 for object of size 634 bytes` | nothing |

No captured stderr named the store path or a data file; the text was the
thread name, the crate's source location under the cargo registry, and the
panic message. `RUST_BACKTRACE=1` added nothing. That corrects the #598 probe
study's assumption that the file path follows the panic, and it is the
measurement the decision above rests on.
