# ADR-0019: Recovery skips the owner-terminated confirmation only for a provably dead local owner

- **Status:** accepted
- **Date:** 2026-10-02
- **Prompted by:** #621

## Context

A publish claim recorded when it was acquired and nothing about by whom. After
a Docker Desktop VM crash killed a publish on 2026-10-01, `stel serving status`
printed `publisher: active` for a process that had been dead for forty minutes,
and deciding whether `recover --owner-terminated` was safe took a five-step
procedure entirely outside stel: a command-line-aware process scan on the host,
another inside every container that could run a build, the orchestrator's run
history, asking every human who might be publishing from another machine, and
a second `status` minutes later to see whether the row counts moved. The
confirmation is load-bearing -- `recover` advances the fence so a surviving
owner fails its *next* write, not before, so recovering under a live publisher
corrupts the index -- and it was supported by no data.

ADR-0001 and #449 already settled that there is no timeout-based stealing.
This record is about what the claim should carry, and the one case in which
carrying it changes the rule.

## Decision

The claim records its holder: hostname, PID, process start time, the kernel
boot id and PID namespace the PID is meaningful in, and an operator-supplied
label (`STEL_PUBLISHER_LABEL`, for the orchestrator run id). The page loop
touches a heartbeat once per page. `stel serving status` prints the readable
parts as the `publisher:` line, with the heartbeat's age.

`recover` without `--owner-terminated` proceeds in exactly one case: the row
names a publisher in *this* PID namespace on *this* kernel boot, and no
process with that PID and that start time exists. Everything else is refused,
and the refusal states what the ledger knows about the owner, so the
operator's confirmation is informed.

Heartbeat age is displayed and never acted on.

## Alternatives considered

- **Timeout-based stealing on heartbeat age.** The obvious use of a heartbeat,
  and wrong here: the index build after the last page is one native call of
  an hour or more with no page to beat on, and a resumed build's index drop is
  another. A publisher that has been silent for ninety minutes is the normal
  shape of the step that matters most, not a dead one. Fencing it mid-build
  is the corruption #449 ruled out, and a heartbeat makes it easier to commit,
  not harder.
- **Treat the heartbeat as a liveness proof in the other direction** -- a
  fresh heartbeat means alive, so refuse harder. True but useless: the
  dangerous case is the silent live publisher, which looks identical to the
  dead one. Liveness has to come from the process table.
- **`psutil` for a portable process probe.** It would give process start
  times on macOS and Windows too. Declined: the core installation stays lean
  (AGENTS.md), and the provable case this exists for -- an
  orchestrator-launched build on the same Linux host as the operator -- needs
  only `/proc`. Elsewhere liveness is unknown and the confirmation stays
  required, which is the pre-#621 behaviour with a better refusal message.
- **Record the identity on the lease table rather than the ledger row.** The
  lease table holds query pins; the publish claim lives on the ledger row as
  `publication_id`, and the identity describes that claim. Splitting them
  would make "who holds the claim" a join, and `recover` already rebuilds the
  row from the highest fence.
- **Hostname as the locality test.** The first version of this change looked
  the PID up whenever the recorded hostname matched this one. Review caught
  that a hostname proves nothing: two containers can be configured with the
  same one, a container can carry its host's, and the PID that is gone here
  may be alive there -- so a dead verdict would have skipped the confirmation
  under a live publisher, the exact corruption the confirmation exists to
  prevent. The identity therefore records the kernel boot id and PID
  namespace (`/proc/sys/kernel/random/boot_id`, `/proc/self/ns/pid`), and a
  PID is looked up only by a process that shares both. The hostname stays,
  for the operator to read.
- **TIMESTAMP columns.** `started_at` and `completed_at` are TIMESTAMPs written
  with `CURRENT_TIMESTAMP`, and nothing computes an age from them. DuckDB
  resolves `CURRENT_TIMESTAMP` through the session time zone and BigQuery does
  not, so an age computed from a TIMESTAMP would differ by the operator's UTC
  offset depending on the warehouse. The heartbeat and start markers are Unix
  seconds, written from Python, so an age is a subtraction everywhere.

## Consequences

- **Six new nullable ledger columns**, added by `_ensure_ledger_columns` to a
  ledger that predates them, the same way `active_collection` was.
- **One DML per page**, about a second on BigQuery against pages that take
  minutes. It doubles as the fence check `verify_publish` already made.
- **A row claimed by a version before #621** has a `publication_id` and no
  identity. `status` reports no publisher for it and `recover` refuses
  without confirmation, saying no publisher is recorded.
- **The provable case is narrow in the current deployment.** Prod builds run
  inside a container whose hostname is the container's, so from the host they
  read as unknown. The informed refusal is the part that applies every time.
- **`serving status` output changed shape** on the `publisher:` line; anything
  that parsed `active` needs the new form.

## Evidence

- #621, incident of 2026-10-01: run killed by a Docker Desktop VM crash at
  16:46 UTC, lease recovered by hand at 19:25 UTC after the five-step check.
- Prior instances the runbook was written after: #705, #722, #744, #757.
- `tests/test_publisher_identity.py` pins the three liveness verdicts,
  including PID reuse; `tests/test_serving_coordination.py` pins the recorded
  identity, the fenced heartbeat, the refusal text, the dead-local-owner
  recovery and that a real publish beats on every page.
