# ADR-0014: A debug switch owns its own destination, and marks the records it is for

- **Status:** accepted
- **Date:** 2026-09-20
- **Prompted by:** #599

## Context

`STEL_DEBUG_PROVIDER_ERRORS` is the hatch for diagnosing a provider failure.
It discloses the least of the three channels: provider errors are sanitized
before any logger sees them, so what it emits is `redacted_exception_text`'s
allowlist — exception types, stel frame locations, an external frame count —
and never native text. Its docstring says the switch exists for local
diagnosis.

It emitted nothing, under every combination of flags, since it was written.
Two separate gates had to open and the operator controlled only one. Each of
the nine call sites reads

```python
if provider_error_debug_enabled() and log.isEnabledFor(logging.DEBUG):
```

and `logging_setup` caps `-v` at INFO deliberately (ADR-0012), so the second
condition was false whenever the CLI configured logging — which is every
orchestrated run, since those invoke `stel build` as a subprocess. The named
alternative, attaching your own handler, is reachable only from in-process
Python.

ADR-0012 looked like it had closed this by accident, since `--diagnostics-file`
raises the logger to DEBUG. It had not, and the reason is the part worth
recording: that file's filter is `exc_info is not None or level >= WARNING`,
and these records carry no `exc_info` **because** `redacted_exception_text`
exists so native text never rides along. The one property that makes them safe
to emit is the property that made them unroutable. With both switches set the
branch fired and the record was then dropped.

## Decision

A switch that turns diagnostics on is responsible for their destination, not
only for their production. `STEL_DEBUG_PROVIDER_ERRORS` now raises the `stel`
logger to DEBUG itself, and — when no diagnostics file is configured —
installs a stderr channel filtered to these records alone. With a diagnostics
file configured, they go there instead, so an operator collecting diagnostics
still gets one artifact rather than two.

The records carry an explicit marker, `PROVIDER_DIAGNOSTICS_EXTRA`, the same
mechanism `REPORTER_ECHO` already uses. Routing is by declaration rather than
by inferring intent from a record's shape.

## Alternatives considered

### Drop the `isEnabledFor` guard and leave the destination alone

The smallest change, and it fixes nothing an operator would notice: with no
handler installed the DEBUG record still reaches `logging.lastResort` at
WARNING and is discarded. It also moves `redacted_exception_text(error)` from
"evaluated when someone is listening" to "evaluated on every provider error",
since it is an argument and Python evaluates it before `log.debug` decides.
Provider errors are retried in normal operation, so that is a real cost for
no benefit. The guard is kept for exactly this reason.

### Route to the diagnostics file only, and require both switches

The option the issue leaned toward, on the sound grounds that two diagnostics
destinations are worse than one. Rejected because it leaves the documented
behaviour false: the docstring promises local diagnosis, and an operator who
sets the variable alone would still get silence with nothing to indicate the
switch had not taken effect. It also does not avoid the marker — the filter
has to learn about these records either way.

### Its own file, mirroring `--diagnostics-file`

Keeps the two disclosure levels genuinely separate, which is a real property:
one carries native text, the other an allowlist, and an operator might want
the second without the first. Rejected on surface area — a second destination,
a second flag, a second set of permission and failure semantics — for a
channel whose whole output is a few lines of exception types. The single-file
behaviour preserves the useful half: with no file configured you get the
allowlist and nothing native.

### Emit at a level that is already enabled, such as INFO

Would have needed no level change and no channel. Rejected because it inverts
the default: these records would then appear under `-v`, which no one asked
for, and the switch would control nothing.

## Consequences

- **Raising the logger to DEBUG moves where native text is stopped.** It was
  the logger's level; it is now each handler's filter — the provider channel
  filters to marked records, `-v` sits at INFO, the fallback at WARNING, and
  propagation is off. The level is only ever raised alongside one of those
  filtering handlers. This is load-bearing and does not look it: a new
  unfiltered handler attached to the `stel` logger would now receive every
  native-text DEBUG record in the tree.
- **One case is not covered, deliberately.** A handler an in-process caller
  attached to the `stel` logger themselves will now receive DEBUG records
  they would previously have had to raise the level to see. Setting the
  variable is the opt-in.
- **ADR-0012's filter convention is no longer sufficient on its own.** "A new
  sanitize site that logs without `exc_info` will not reach the file" is still
  true, and the marker is now the way to say a record belongs there anyway.
- The stderr channel turns propagation off like the others, so the WARNING
  fallback that stands in for `logging.lastResort` had to become conditional
  on propagation rather than on the diagnostics file. Without that, enabling
  a debug switch would have silenced every warning in the run.

## Evidence

Measured on this branch before the change, exercising the real logging
configuration against a record shaped like the call sites', for each
combination of the two switches:

| switches | branch ran | reached stderr | reached diagnostics file |
|---|---|---|---|
| env var alone | no | no | — |
| env var + `-v` / `-vv` | no | no | — |
| env var + `--diagnostics-file` | yes | no | **no** |

The third row is why this is not simply a gating bug: the record was produced
and then filtered out. After the change, the same harness shows the allowlist
reaching stderr with the variable alone, the diagnostics file when one is
configured, and — in every row, including with the logger at DEBUG — no
native text on stderr.
