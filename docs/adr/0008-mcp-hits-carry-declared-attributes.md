# ADR-0008: A returnable attribute is additive within `mcp_context/v1`, not a v2

- **Status:** accepted
- **Date:** 2026-09-06
- **Prompted by:** #524

## Context

`SearchContextResult` carried no business attributes. A model declaring
`symbol`, `form_type` and `section` with `filter_role: user` and
`returned: true` could be filtered on all three through the MCP and returned
none of them, so an agent could scope a search to one ticker and still be
unable to say which ticker any hit belonged to.

The values were never missing. `_resolve_result_fields` already resolves the
declared `returned: true` set, `search()` already puts them on
`SearchResult.metadata`, and `stel search` already returns them. Only the MCP
contract had nowhere to put them.

That made the fix a schema question rather than a retrieval one.
`mcp_context/v1` is a declared contract, every response carries its marker, and
AGENTS.md requires that artifact shapes be treated as explicit contracts with
versioned schema changes. Adding a field to a live contract needed a decision
about what "versioned" obliges here.

## Decision

Add `attributes` to `SearchContextResult` as an optional field defaulting to
empty, and keep the schema at `mcp_context/v1`. Populate it from the hit's
resolved `returned: true` set, never from the warehouse row behind the hit.

An additive, defaulted field is a compatible change to this contract. Bumping
the version is reserved for changes that alter or remove what a response
already promises.

## Alternatives considered

### Bump to `mcp_context/v2`

The literal reading of "version schema changes", and rejected because it
inverts the cost. Every existing client pins the version marker, so a bump
breaks all of them in order to deliver a field they were already told about:
`list_context_models` advertises these attributes under `retrieval.filter_fields`
today. A version bump is worth its cost when a response stops meaning what it
meant; it is not worth it to start honouring a declaration the catalog already
publishes.

### Source the attributes from the warehouse row

The MCP path already re-reads each hit's chunk row for text, authorization and
lineage, so the columns were in hand and this needed no new plumbing. Rejected
because the row carries far more than the model declares, policy columns among
them. Sourcing from the row makes exposure a property of the table's shape
rather than of the model's declaration, so a new column on the chunk relation
would silently start appearing in agent-visible output. The hit's metadata is
exactly the declared set and cannot drift from it.

### Return every attribute regardless of `returned`

Simpler, and it would have answered the reporter's case. Rejected because
`returned: false` is the only control an operator has over what leaves the
governed boundary, and an attribute may be declared for filtering alone. The
flag has to keep meaning what it says.

### Filter policy-role attributes out of the response

Considered because a `filter_role: policy` attribute may also be declared
`returned: true`, and returning it tells a caller which tenant or group a row
belongs to. Rejected because the row has already passed `_can_read_pair` by the
time it is rendered, so the caller is entitled to it; because `stel search`
already returns it and a second, quieter rule for the MCP path would make the
two surfaces disagree about one declaration; and because `returned: false` is
the existing, explicit way to withhold it.

## Consequences

`returned: true` now means the same thing on both surfaces, which is the
property that was broken. An operator who declared an attribute returnable for
the CLI will find it appears over MCP as well, including a policy attribute
they may have assumed was CLI-only. That is a behaviour change for an existing
declaration, called out in the changelog for that reason.

The field is sourced from `SearchResult.metadata`, so anything that changes how
`_resolve_result_fields` resolves the declared set changes what agents see.
That coupling is deliberate and is the point, but it is now load-bearing in a
place it was not before.

One coercion serves both surfaces. `search.json_value` was made public for
this: `mcp_context/v1`'s value type admits no `date`, so a filing date has to
arrive as a string, and two implementations of that rule would drift into one
surface rendering a value the other rejects.

Deciding this once means the next additive field to these contracts does not
re-litigate it. A change that alters or removes an existing field still does.

## Evidence

Reported against the live `sec_chunk_search` index on 2026-09-05 (#524): a hit
filtered to `symbol eq AAPL` returned `document_id`, `chunk_id`, `snippet`,
`citation` and `lineage`, and none of `symbol`, `form_type`, `filing_date` or
`section`, all four of which the model declares `returned: true` and all four
of which `list_context_models` advertises as filterable.
