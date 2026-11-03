# Context assembly is an offline experiment before serving policy

Status: accepted (constellations#744, ALE-283)

Changing MCP result order or automatically loading neighboring sections would
change production evidence and authorization work before any measured benefit
existed. A plausible alternative was to put token packing directly into
`search_context` and compare real requests. That would confound retrieval with
assembly and risk promoting a policy on document-level relevance alone.

Instead, use frozen governed candidate/neighbor snapshots and explicit packing
policies through a library experiment. Reuse the existing serving authorization
provider for both registry and chunk, and fail before generation if any selected
evidence is denied. Keep the rendered input in memory; reports expose only
allowlisted identities, metrics and counts. Every arm uses the same complete
prompt budget and tokenization identity.

The 2026-10-30 ALE-281 pilot explicitly separates retrieval golds from unfinished
held-out answer/evidence judgments. Its companion scoring guide requires human
passage-entailment review. Consequently missing answer reviews remain null and
the experiment records defer adoption; synthetic packing results cannot select
a production serving policy. A later measured selection retains a versioned
policy identity and receives a separate serving integration.

See [the experiment contract](../architecture/context-assembly.md) and
[the reviewed scoring guide](https://github.com/C00ldudeNoonan/astrolabe/blob/main/constellations/evals/financial_qa/v1/PILOT_SCORING.md).
