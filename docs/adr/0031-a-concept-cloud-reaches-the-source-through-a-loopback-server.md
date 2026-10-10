# ADR-0031: A concept cloud reaches the source through a loopback server, not a token in the artifact

- **Status:** accepted
- **Date:** 2026-10-10
- **Prompted by:** #678 (item 7 of #555)

## Context

The star map answers "which concepts, how often, by whom, when". It could not
answer "says who?" — a reader seeing a concept named in 300 documents had no
way to see one sentence that names it. The project already serves exactly that
sentence: `search_context` returns a `snippet` per hit, through authorization,
a pinned generation and a query log.

Joining the two is not obvious, because the artifact is a single static HTML
file and the MCP server is a separate process speaking JSON-RPC over stdio or
streamable-http. A browser page can only make HTTP calls, so something has to
give.

## Decision

**stel serves the artifact itself, on loopback, same-origin with one data
endpoint.** `stel concept-cloud --context-model <m> --serve` writes the
artifact as before and then serves it from `127.0.0.1`, answering
`GET /_stel/source?concept=<id>` out of `ContextService` — the same object
`stel mcp serve` wraps. The bundle (v6) carries `source_access: {model}`, a
name and nothing else; no sentence is ever written into the artifact.

Four properties follow, and each is enforced rather than documented:

**The bind address is not a parameter.** Identity here is the stdio model:
whoever ran the command is the principal, which is why `ContextService`'s
default `EnvironmentPrincipalResolver` is correct. A `--host` would turn a
single-operator tool into one answering for an identity it cannot establish —
the inversion `serve_network` refuses by making its principal resolver
required (#392). There is no safe value other than loopback, so there is no
option, and a test fails if one is added.

**No CORS header is ever emitted, and a foreign caller is refused twice.** The
page is same-origin, so the absence of `Access-Control-Allow-Origin` is what
stops any other page in the browser reading the reply. Two further checks
cover what that absence does not: the `Host` header must name loopback (DNS
rebinding resolves an attacker's domain here and the browser then treats their
page as same-origin), and a cross-origin `Origin` is refused outright, because
CORS stops another page *reading* the reply but the request would still have
run — and running it means a warehouse read and a query-log row attributed to
the operator.

**The endpoint takes a concept id, not query text.** The search string is
looked up in the bundle the server is serving, so the closed set of concepts
on the map is the closed set of things that can be asked. An id the bundle
does not contain is a 404, which means nothing else that reaches loopback can
drive the endpoint as a general search proxy over the corpus.

**A declared concept is scoped; an undeclared one is a text match, and the
card says which.** A concept whose `canonical_id` is a declared vocabulary
term is fetched with `entity_scope: {"term": id}`, so the chunk is one the
pipeline actually linked to it. A concept no vocabulary declares — a fuzzy
resolver's output, or a project with no `vocabularies:` — can only be matched
as text. "Linked to FERC" and "contains the word FERC" are different claims,
so the scope travels to the card and is rendered.

## Alternatives rejected

**Bake the sentences into the bundle.** Rejected by #555 item 7 itself, and
rightly: it would copy governed document text into a file that gets emailed
around, with no policy filter and no query log between the reader and the
corpus. It also scales with concepts × periods, in a bundle already carrying
per-period counts.

**Let the page call a remote `stel mcp serve` directly.** The page would need
a bearer token — forbidden, because credentials must never enter artifacts —
and the governed server would need CORS, which would let any page in the
browser reach it. Adding a security surface to the governed server to serve a
viewer convenience is the wrong trade. It would also mean implementing the MCP
session handshake in the viewer's JavaScript.

**A `stel concept-cloud serve <file>` subcommand** instead of a `--serve`
flag. `concept-cloud` is a command with twenty options, not a group, and
making it a group to gain a subcommand would change the existing invocation.
The cost is real and is recorded as a limit below rather than hidden: `--serve`
serves what the command just produced, so re-browsing an expensive export
means re-running it.

## Consequences

- A shared artifact degrades to exactly what it was before: the viewer gates
  its button on a same-origin probe, so a file opened from disk shows no
  affordance rather than a button that cannot work.
- A sentence a reader sees leaves a query-log row naming the viewer as the
  client (`stel concept-cloud`), distinct from an agent's — which is the
  distinction #528 added those fields for.
- The retrieval mode is resolved once at startup from what the model declares,
  so a model that cannot do `hybrid` is searched the way it can be rather than
  refusing every click.
- **A slow export cannot be re-browsed without re-exporting.** Accepted for
  now; if it bites, the fix is a way to serve a bundle already on disk, which
  needs the bundle written beside the HTML rather than only inside it.
- Nothing here helps a *published* artifact reach a governed server. That
  needs the token story this ADR rejects and would be its own decision.
