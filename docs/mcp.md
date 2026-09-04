# Governed context over MCP

stel exposes a read-only MCP stdio server for document context governed by
`agent_context/v1`. It complements the dbt MCP server rather than proxying it:

- use stel MCP for document discovery, retrieval, citations, and provenance;
- use dbt MCP for governed metrics, Semantic Layer queries, and dbt metadata.

The initial server exposes exactly four tools:

| Tool | Purpose |
| --- | --- |
| `list_context_models` | Discover only context models available to the caller. |
| `search_context` | Query the portable stel search API and return governed context. |
| `get_document` | Fetch one authorized document version as bounded, paginated chunks. |
| `get_context_lineage` | Trace an authorized document, context, chunk, or search result. |

There are no build, mutation, metric-query, answer-generation, path-read, or
source-URL-read tools.

## Prepare a project

Install the MCP extra plus the retrieval or warehouse extras used by the
project:

```bash
uv add 'stel[mcp,lancedb]'
```

The server reads `target/manifest.json` and `target/run_results.json`, then
reads authorized rows through the configured warehouse adapter. Run the
project before starting the server so those artifacts and relations agree:

```bash
uv run stel run
```

An exposed search index must descend from `agent_context/v1`
`document_registry` and `document_chunks` relations. Its `id_field` must be
`context_id` or `chunk_id`. Optional `context_entity_links` descendants add dbt
entity types and links to responses.

Those `document_registry`/`document_chunks` relations must come from
`transform: {type: python}` models that declare `agent_context:` — the
built-in `extraction:`/`chunk:` primitives cannot claim the contract, so a
search index built directly on them will not surface here even once fully
populated. `stel.agent_context.project_document_registry_row`/
`project_document_chunk_row` turn an existing `extraction:`/`chunk:` pipeline
into a contract-emitting transform in roughly 15-20 lines each. Construct the
output frame with an explicit schema from
`empty_agent_context_frame(grain).schema` (plus any extra columns) rather than
letting polars infer one — a batch that is all-null in an optional column such
as `citation_section_path` otherwise infers a Null-typed column and fails, or
drifts the schema, on a later mixed batch. See
`examples/agent_context_from_builtin_pipeline/` for a complete project
(`extraction:` → `chunk:` → two thin `transform:` wrappers → `embed:` →
`search:`) that compiles, runs, and is discoverable through
`list_context_models`, and `examples/google_drive_context/` for the same
shape over a Google Drive folder, from `gcloud` sign-in to a served answer. See [Agent context contract
v1](architecture/agent-context-v1.md#runtime-and-artifact-integration) for why
and how to wrap an existing extraction/chunk pipeline in a contract-emitting
transform.

## Local principal

The stdio MVP resolves one deterministic principal from operator-owned
environment variables:

| Variable | Meaning |
| --- | --- |
| `STEL_MCP_PRINCIPAL_ID` | Required opaque local subject identity. |
| `STEL_MCP_TENANT_ID` | Trusted tenant claim. |
| `STEL_MCP_ACCESS_GROUPS` | Optional comma-separated access groups. |
| `STEL_MCP_POLICY_CLAIMS` | Optional JSON object of additional trusted policy claims. |

Every call fails closed without `STEL_MCP_PRINCIPAL_ID`, including calls to
public context models. Authorization values are never accepted as tool
arguments. Governed search predicates are compiled from the principal, and
warehouse rows are independently rechecked after search and on every document
or lineage lookup.

This environment resolver is for deterministic local stdio use, where the
operator running the process *is* the principal. It is refused on a network
transport — see below.

## Serving over a network

```bash
stel mcp serve --transport streamable-http --host 0.0.0.0 --port 8000   --trust-proxy-principal-headers
```

A network transport serves many callers, so identity has to come per request
rather than from the process. The stdio environment resolver is **rejected at
startup** for these transports: it would serve every caller as whichever
identity the server started with, applying that identity's tenant filters to
everyone's queries. Nothing about the responses would look wrong, which is why
this is a refusal rather than a warning.

`--trust-proxy-principal-headers` reads each caller's identity from:

| header | meaning |
|---|---|
| `X-Stel-Principal-Id` | subject id; absent means no principal, and the call is refused |
| `X-Stel-Tenant-Id` | tenant, when the deployment is multi-tenant |
| `X-Stel-Access-Groups` | comma-separated groups |
| `X-Stel-Policy-Claims` | JSON object of trusted claims |

**This trusts its input completely.** It is sound only when something in front
of the server authenticates the caller and *overwrites* every one of these
headers on each request — an authenticating reverse proxy, an identity-aware
proxy, a service mesh. Exposed directly, it is not authentication: any caller
can claim any tenant by setting a header. Terminate auth in front of it, and
confirm the proxy overwrites rather than appends.

Note what the proxy is actually being trusted *for*. By default it
authenticates the caller **and** supplies their policy — the tenant and groups
in those headers decide what gets read. [Operator-owned
grants](#operator-owned-grants) narrow that to authentication alone: with
`--grants-relation` set, only `X-Stel-Principal-Id` is consulted and the rest
are ignored, so a mistake in the proxy's header handling stops being a
tenant-isolation failure. If you are exposing this over a network, use both.

Two limits worth knowing before a shared deployment:

- **Warehouse credentials are per process, not per caller.** A hosted server
  holds one set for everyone, so row-level governance is the only boundary
  between tenants. Per-tenant credentials are tracked in issue #395.
- **Rate limits are per process unless you say otherwise.**
  `--max-requests-per-minute` was sized for one local client; shared, it is a
  global cap one caller can exhaust. Set
  `--max-requests-per-minute-per-principal` to give each subject its own share
  under that ceiling. Requests with no resolvable principal share one anonymous
  bucket of the same size, so an unauthenticated flood counts against something
  rather than nothing. Unset, nothing changes. The refusals are worded apart so
  logs stay readable: `This caller is at its request rate limit` is one loud
  tenant, `Unauthenticated callers are at their shared request rate limit` is a
  flood with no identity, and `The context server is at its request rate
  limit` means the deployment is undersized. All three are `BUSY` and all
  three are retryable.

### Verifying tokens instead of trusting a proxy

When nothing authenticating sits in front of the server, have it verify each
caller's bearer token itself:

```bash
stel mcp serve --transport streamable-http --host 0.0.0.0 --port 8000   --jwt-issuer https://issuer.example   --jwt-audience https://stel.example/mcp   --jwt-jwks-uri https://issuer.example/.well-known/jwks.json
```

All three flags are required and none has a default — each is a security
boundary, and a default would be stel quietly choosing one. The JWKS URL must
be `https`: keys fetched in plaintext can be replaced in transit, which makes
verifying against them meaningless.

Every accepted token has passed all of these, and a token failing any of them
is refused with the same unauthenticated response — distinguishing "bad
signature" from "wrong audience" would be a probing oracle:

| check | why it is not optional |
|---|---|
| signature against the issuer's JWKS | the token is what it says it is |
| asymmetric algorithm (`RS*`, `ES*`, `PS*`) | accepting HMAC alongside RSA is the classic JWKS forgery: sign with the public key as the shared secret |
| `iss` matches `--jwt-issuer` exactly | a token from another issuer is another system's |
| `aud` matches `--jwt-audience` exactly | the confused-deputy case the MCP authorization spec calls out: without it, a token a caller legitimately holds for *another* service is valid here |
| `exp` and `nbf`, with 30s leeway | expiry is the only revocation this scheme has |
| a non-empty `sub` | grants are keyed by subject; a subjectless token could never be authorized |

**Key fetches are bounded.** A token naming a key id the cached JWKS lacks
triggers a refetch — that is how a key rotation works without a restart, and it
is also the one network call an unauthenticated caller can provoke before any
rate limit applies. So the refetch happens at most once a minute, concurrent
misses share it, each fetch times out after five seconds, and verification runs
off the event loop. A token signed with a key rotated in *during* that minute,
after someone's probe, is refused until the minute lapses — never accepted
wrongly. The JWT flags are refused on `--transport stdio`, where they would be
accepted and verify nothing.

**A token establishes identity, never authorization.** Only the subject is
taken from it. With [operator-owned grants](#operator-owned-grants), groups and
tenants are looked up by that subject — so a token that *claims* a tenant or a
group gains nothing by claiming it, and a legitimately issued token cannot be
turned into a privilege escalation by whoever minted it. Use `--grants-relation`
with token verification; without it, a verified caller has an identity and no
policy, and every governed read is refused.

`--trust-proxy-principal-headers` and the JWT flags are mutually exclusive.
Enabled together, the headers would decide and the token checking would be
decoration, so the server refuses to start rather than pick.

### Opaque tokens: introspection instead

A JWT verifier needs a token it can read. When the authorization server issues
**opaque** tokens, ask it about each one instead (RFC 7662):

```bash
export STEL_MCP_INTROSPECTION_SECRET=…
stel mcp serve --transport streamable-http --host 0.0.0.0 --port 8000 \
  --introspection-endpoint https://issuer.example/oauth2/introspect \
  --introspection-issuer https://issuer.example \
  --introspection-audience https://stel.example/mcp \
  --introspection-client-id stel-mcp \
  --introspection-client-secret-env STEL_MCP_INTROSPECTION_SECRET
```

All five are required, for the same reason the JWT flags are. The secret is
passed by **environment variable name**, never as a flag value: a secret on the
command line is visible in the process list and the shell history. Only the
name is held — the value is read at the moment each request is built, so no
long-lived object carries it into a dump, a debugger, or a crash report, and a
rotated secret takes effect without a restart. The server refuses to start if
the variable is unset. The endpoint must be `https` — the caller's token is
sent to it in the request body, so plaintext hands the credential itself to
anyone on the path.

| check | why it is not optional |
|---|---|
| `active` is JSON `true` | a truthy string is not a live session |
| `aud` matches `--introspection-audience` | **the one this scheme cannot skip.** `active` means the token is live *at the issuer*, not that it was minted for you; without this, a token a caller legitimately holds for another service is valid here |
| `iss` matches, when the response carries one | checked but not demanded — unlike a self-contained JWT, this answer came over TLS from the one endpoint you named, authenticated as a client that server registered |
| `exp` in the future, when present | `active` should have covered it; a server saying both is not one to read generously |
| a non-empty `sub` | grants are keyed by subject |

An authorization server that omits `aud` from its introspection response
**cannot be used**, because the confused-deputy check above becomes impossible.
Anything that is not a clean `200` — bad client credentials, a down issuer — is
"cannot vouch for this caller", never "let them in".

**The introspection call is bounded, because it sits on the unauthenticated
path.** Tokens are verified before any tool runs, so the per-principal and
global rate limits above are downstream of this: a caller who has proven
nothing decides how many outbound requests this server makes to your
authorization server. All introspections share one pooled client with a
connection ceiling, at most sixteen are in flight at once, and a caller that
cannot get a slot within two seconds is refused rather than queued — an
unbounded queue would itself be the resource worth attacking. "Too busy to
ask" is the same answer as every other failure here.

**A cache hit is a revocation delay, and that is the trade.** Every request
would otherwise cost a round trip on a path that already has a timeout budget,
so a verified token is reused for up to a minute, capped by its own `exp`. That
is also how long a token revoked at the issuer keeps working here. Failures are
never cached: it would blunt one repeated bad token while doing nothing about
distinct invented ones, and every cache keyed by attacker-supplied input is
memory an unauthenticated caller can grow. Entries are keyed by digest, so the
cache never holds a live credential in plaintext.

The introspection flags are mutually exclusive with both the JWT flags — which
one refused a caller would otherwise be ambiguous — and with
`--trust-proxy-principal-headers`, and are refused on `--transport stdio` for
the same reason the JWT flags are.

### Discovery: letting clients find the authorization server

Either verifier also needs `--public-url`, this deployment's externally
reachable URL:

```bash
stel mcp serve --transport streamable-http \
  --public-url https://stel.example/mcp \
  --jwt-issuer https://issuer.example …
```

It cannot be derived from `--host`/`--port`, which are what the process binds
behind a proxy rather than what a caller reaches. With it, the server publishes
**OAuth 2.0 protected resource metadata** (RFC 9728) at
`/.well-known/oauth-protected-resource/mcp`:

```json
{
  "resource": "https://stel.example/mcp",
  "authorization_servers": ["https://issuer.example/"],
  "bearer_methods_supported": ["header"]
}
```

and names that document in the `WWW-Authenticate` challenge it returns to an
unauthenticated caller. A spec-compliant client handed nothing but this
server's URL can then find out who issues tokens for it, instead of every
connecting team being told the issuer out of band.

stel is a **resource server**, not an authorization server: it validates tokens
somebody else issued and does not issue, refresh, or revoke any. The metadata
says exactly that and no more.

## Operator-owned grants

By default a principal's policy values *are* its claims: the tenant and access
groups the caller arrives with decide what the caller may read. Over stdio that
is correct, because the operator sets the environment and is the principal.
Over any transport where the claims come from the caller, it means a forged
`STEL_MCP_ACCESS_GROUPS` — or the header a proxy maps it from — is a policy
change, and a correctly configured proxy is the only thing separating one
tenant from another.

`--grants-relation` moves that decision to a relation you own:

```bash
stel mcp serve --grants-relation ops.context_grants
```

The relation supplies three string columns:

| Column | Meaning |
| --- | --- |
| `subject_id` | The authenticated subject the grant belongs to. |
| `attribute` | A policy attribute name, such as `tenant_id` or `access_group`. |
| `value` | One value the subject is permitted for that attribute. |

One row per permitted value; several rows for the same attribute compile to an
`IN` filter. With this set, `STEL_MCP_TENANT_ID`, `STEL_MCP_ACCESS_GROUPS`, and
`STEL_MCP_POLICY_CLAIMS` are **not consulted** — only `STEL_MCP_PRINCIPAL_ID`
is, as the subject to look up. That is the point: the caller proves who they
are, and you decide what that subject may read.

A subject with no grant for a required attribute is refused rather than given
an unfiltered read, and rows returned by search are rechecked against the same
grants, so a retrieval store that ignored a filter still cannot leak a row.

A grants relation that is malformed — a null, blank, or missing column — fails
as a configuration error rather than as a denial, so schema drift does not
present as "this subject has no grants".

Policy attributes declared as `array[string]` — including the `access_groups`
shape in the agent-context contract — are filtered by set overlap: the row's
groups must share at least one element with the caller's grants. One grant row
per group, exactly as for scalar attributes.

Grants are cached per subject for `--grant-ttl-seconds` (default 60). That TTL
is the ceiling on how long a revoked grant keeps working — restart the server
if a revocation must take effect immediately.

Two limits worth stating plainly. stel is still the enforcement point: grants
make policy central and auditable, but they do not make the warehouse refuse a
query stel should not have issued — per-tenant warehouse credentials are the
layer that does that, and they compose with this. And the grants relation is
as trustworthy as its write path; treat it as production access control, not
as a model that anything downstream may edit.

Queries are logged with the tenant the policy actually filtered to, not the
tenant the caller claimed, so the audit trail stays meaningful when those
differ.

## Generic client configuration

Most stdio MCP clients accept a configuration shaped like this. Replace the
project path and add the extras required by its active profile:

```json
{
  "mcpServers": {
    "stel-context": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "stel[mcp,lancedb]",
        "stel",
        "--project-dir",
        "C:/path/to/document-project",
        "mcp",
        "serve"
      ],
      "env": {
        "STEL_MCP_PRINCIPAL_ID": "local-analyst",
        "STEL_MCP_TENANT_ID": "economic-research",
        "STEL_MCP_ACCESS_GROUPS": "analysts,reviewers",
        "STEL_MCP_POLICY_CLAIMS": "{\"classification\":[\"internal\"]}"
      }
    },
    "dbt": {
      "command": "uvx",
      "args": ["dbt-mcp"]
    }
  }
}
```

Configure dbt MCP credentials and toolsets separately using the
[dbt Labs setup guide](https://docs.getdbt.com/docs/dbt-ai/about-mcp). A client
can then combine dbt metrics with stel citations without either server
assuming ownership of the other's domain.

For a source checkout, replace the `--with stel[...]` portion with
`--project C:/path/to/<checkout>/stel`.

### BigQuery with gcloud user credentials

With `method: oauth` (gcloud Application Default Credentials), stel loads the
ADC file directly and never shells out to `gcloud`, because a `gcloud` child
process would inherit the MCP stdio pipes and can hang the server on Windows.
Run `gcloud auth application-default login` once so the ADC file exists. stel
keeps the scopes granted to gcloud user ADC rather than replacing them with
the profile's dbt-compatible default list; credential types that require
scoping still receive the configured `scopes`. If your environment resolves
credentials another way, setting
`GOOGLE_APPLICATION_CREDENTIALS` to the ADC file path in the server's `env`
block forces the no-subprocess path explicitly:

```json
"env": {
  "GOOGLE_APPLICATION_CREDENTIALS": "C:/Users/<user>/AppData/Roaming/gcloud/application_default_credentials.json"
}
```

The server also opens the warehouse once at startup, so a broken credential
setup fails at boot with a real error instead of surfacing as per-call
`timeout` errors mid-session.

## Limits and errors

`stel mcp serve --help` lists controls for operation timeout, requests per
minute, concurrency, response bytes, and warehouse scan rows. Tool requests
also have bounded result and page sizes. Document cursors are tied to the
model, document ID, and exact document-version ID, so they cannot be replayed
against another document.

Every response uses the `mcp_context/v1` schema and returns a structured error
with a stable code. Unauthorized and nonexistent models or records both return
`not_found_or_denied`; the server does not reveal which condition occurred.
Timeout, request-rate, and concurrency errors are retryable. Response and scan
limit errors require a smaller request or an operator limit change.
