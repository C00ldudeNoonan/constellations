# ADR-0010: The warehouse identity a governed read runs as is a granted attribute, not a tenant

- **Status:** accepted
- **Date:** 2026-09-10
- **Prompted by:** #395 (per-tenant warehouse credentials)

## Context

stel is the enforcement point for governed context. #392/#396 moved
authorization to an operator-owned grants relation, so a forged header buys a
caller nothing — but the warehouse still executes whatever stel sends it. One
process holds one set of credentials for every caller, so a filter that is
dropped, mis-compiled, or skipped by a code path that forgot to apply one is
answered anyway. The blast radius of a single bug in the filter path is every
caller's data.

Closing that means some reads must run as a warehouse principal narrower than
the operator. Three things about the code as built constrain how:

**There is no tenant identity to key on.** `GrantStore` is keyed on
`subject_id` (`grants.py:42`). Tenant is not part of that key — it is one
possible `attribute` *value*, indistinguishable in the schema from
`access_groups` or any custom policy attribute, and a subject may hold several
values for it (`_granted_values` collects them into a tuple, compiled to `IN`
or `ARRAY_CONTAINS_ANY`). The service already refuses to name one:
`_authorizing_tenant` returns `None` for "a multi-tenant grant, where no
single value is the honest answer" (`service.py:1519`).

**The connection cannot simply be swapped.** The grants relation is read
through the same `WarehouseContextRepository` that reads governed context
(`service.py:543-551`, `grants.py:184`). Making that connection
caller-scoped is circular: you would need the caller's credentials to discover
which credentials the caller gets. The same held connection also carries the
serving lease and ledger (`search.py:760-825`) and the MCP query log — stel's
own infrastructure, which a caller-scoped principal must not write.

**Adapters differ in kind, not degree.** BigQuery can narrow a principal
without any new secret: `impersonate_service_account` is a service-account
email wrapped onto the operator's own credentials
(`bigquery.py:1281-1288`). MotherDuck cannot: its `token` is a
`CredentialReference` — an environment-variable name — so N callers means N
environment variables. Local DuckDB is a single-process file lock
(`duckdb.py:468`) and is not a hosted deployment at all.

## Decision

A governed read runs under a **warehouse identity**: an opaque,
operator-supplied name for the warehouse principal that read should execute
as. It is resolved from a **reserved `warehouse_identity` attribute in the
existing grants relation**, keyed by `subject_id` like every other grant.

Three rules make it a boundary rather than a hint:

1. **Exactly one, or refuse — but the two refusals differ.** Zero
   `warehouse_identity` grants is a **denial**, indistinguishable from having
   no grants at all; there is no fallback to the operator connection, because
   a missing row must never read as "unprotected". More than one is a
   **configuration error**, not a denial: a subject may legitimately hold
   several `tenant_id` grants, but cannot legitimately execute as two
   principals at once, and reporting a contradictory relation as "denied"
   would leave the operator with no signal — the reason
   `GrantConfigurationError` already exists in that module.
2. **Split by purpose, not by request.** The operator connection keeps the
   serving ledger and query lease, the grants relation, and the query log.
   Only governed context reads take an identity. The grant read is what
   resolves the identity, so it necessarily precedes it and runs as the
   operator.
3. **Capability, not behaviour.** `supports_identity_scoped_connection()`
   sits beside `supports_held_connection()`. Enforcement configured against an
   adapter that returns `False` is refused **at startup**, not downgraded.

This ADR records the seam. No adapter implements the capability yet; BigQuery
impersonation (#568) and MotherDuck per-caller tokens (#569) are separate work
under the same contract.

## Alternatives considered

### Key credentials on the tenant

The obvious reading of "per-tenant credentials", and what #395 proposed. It
requires a single tenant per caller, which the grants model does not provide
and deliberately does not: a subject can be granted several `tenant_id` values,
and `_authorizing_tenant` already declines to pick one because no single value
is honest. Adopting it would mean either forbidding multi-value tenant grants —
narrowing a capability that exists on purpose — or choosing arbitrarily, which
puts an arbitrary choice underneath a security boundary. Ruled out for that.

### Key credentials on the subject

Consistent with the grants store's own key, and needs no new attribute. But it
sizes the connection pool by *distinct callers* rather than distinct warehouse
principals, which is the dimension that actually grows without bound, and it
forces every caller to have their own warehouse principal provisioned before
they can read anything. Several callers legitimately share one principal; the
subject key cannot express that.

### Treat an ambiguous identity as a denial

Simpler, and consistent with how every other failed grant lookup surfaces. It
was the first draft of this ADR. Rejected on the same reasoning the module
already applies to malformed grant rows: a denial tells the caller nothing they
should learn and tells the operator nothing they can act on, and two identity
rows for one subject is never a legitimate state the way two tenant rows is.

### A separate credential-mapping relation

Keeps the grants relation purely about policy. It would have to re-earn
everything `WarehouseGrantStore` already has — operator ownership, TTL and
therefore bounded revocation delay, the per-subject cache, the sweep that keeps
that cache from growing for the life of the process (#466), and auditability
through the same mechanisms as every other operator-owned table. A second store
with weaker versions of all five, for the benefit of a cleaner column
vocabulary, is a bad trade.

### Put the mapping in the profile

Operator-controlled and file-shaped, like every other credential in stel. But a
profile is read at startup, so a revoked mapping keeps working until the server
restarts — for a *policy* value that is tolerable, for the identity a query
executes as it is not. It also splits caller identity across two stores that
must agree, which is the thing #395 asked to avoid.

### Fall back to the operator connection when no identity is granted

Backwards compatible, and it would let the capability be switched on before
every subject is provisioned. Rejected because it inverts the failure mode: a
forgotten grant row would silently downgrade that caller to today's behaviour,
and the deployment would believe it had warehouse-level enforcement it did not
have. That silent-success failure is the specific thing this theme exists to
prevent.

### Carry the identity in a contextvar rather than a parameter

The codebase already reads the principal and the HTTP request from contextvars
(`authorization.py:163-189`), so it would fit. But a contextvar that is not set
reads as absent, and absent would have to mean something — either operator
(the silent downgrade rejected above) or refuse (a hard-to-diagnose failure far
from the call that forgot it). A required keyword argument on `read_rows` makes
every call site state which connection it wants, and makes a new one that
forgets fail to compile rather than fail quietly.

## Consequences

**Every `read_rows` call site must name its connection.** That is the point,
and it is also the cost: adding a governed read now means making an explicit
security decision, and the ten existing call sites had to be classified.

**The refusal is strict, so provisioning is a prerequisite.** Turning
enforcement on before every authenticated subject has a `warehouse_identity`
grant denies those callers outright. That is the intended direction of failure,
but it means the rollout order is grants first, flag second.

**A held connection's cache is not the grant cache.** For a policy filter a TTL
is revocation delay. For a *held connection* a stale pool entry means a revoked
identity keeps a working client until eviction, which is a longer and less
obvious tail. The connection pool therefore carries its own bound and its own
sweep rather than inheriting the grant TTL.

**The pool is keyed by caller-derived input**, which is the shape that grew
without bound in #466. It is bounded and swept for that reason; anything later
keyed the same way should be too.

**`supports_identity_scoped_connection()` returns `False` everywhere today**,
so the seam is inert until an adapter implements it. That is deliberate — it is
also why the contract is exercised by a test adapter rather than by BigQuery,
so the refusal and pooling semantics are pinned before either real
implementation lands.

## Evidence

Read against `master` at 906544c on 2026-09-10:

- `grants.py:42` — `SUBJECT_COLUMN = "subject_id"`; `GRANT_COLUMNS` is
  `(subject_id, attribute, value)`, with no tenant column.
- `grants.py:232-259` — several grant rows for one attribute compile to `IN` /
  `ARRAY_CONTAINS_ANY`, so multi-valued is a supported state.
- `service.py:1519` — `_authorizing_tenant` returns `None` for a multi-tenant
  grant.
- `service.py:543-551`, `grants.py:184` — the grant store reads through the
  same repository instance as governed context.
- `search.py:760-825` — one `session.warehouse()` block spans the query lease,
  the store search and the row re-reads.
- `bigquery.py:333` and `:1281-1288` — `impersonate_service_account` is a
  plain `str | None` wrapped onto the operator's credentials by
  `impersonated_credentials.Credentials`; no new secret is introduced.
- `duckdb.py:237` — MotherDuck's `token` is a `CredentialReference`, i.e. an
  environment-variable name.
- `duckdb.py:468` — local DuckDB refuses a held connection because an open
  file is an exclusive lock.
