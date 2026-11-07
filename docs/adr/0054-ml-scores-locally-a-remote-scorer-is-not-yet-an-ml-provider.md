# ADR-0054: `ml:` scores locally; a remote scorer is not yet an `ml:` provider

- **Status:** accepted
- **Date:** 2026-11-06
- **Prompted by:** #38 (ALE-307)

## Context

#38 asked for Vertex custom prediction endpoints. Such an endpoint is a remote
scorer: arbitrary JSON `instances` in, arbitrary JSON `predictions` out, with no
prompt, output schema or sampling parameters. That shape does not fit the
`InferenceProvider` seam, which is `InferenceRequest(content, system_prompt,
output_schema, ...)` to a mapping that conforms to a declared schema. A deployed
classifier returns a label and a score. The place a scorer belongs is `ml:`,
whose `mode: predict | load_pretrained` and `provider:` already describe
applying a model to rows.

Nothing under `ml:` scores remotely. Every provider is `builtin.*` and fits and
predicts locally against an artifact in `target/artifacts/`. Three properties
of `ml:` assume that:

- **Identity.** An `ml:` model's `code_version` hashes its declaration, and an
  artifact is the thing that was fitted. A remote endpoint's model can change
  under a stable endpoint id. The only version signal is whatever the endpoint
  reports (for Vertex, the deployed model id, which changes on redeploy), and
  stel has no term in `code_version` for it.
- **Cost.** `ml:` models are `materialization: full` only (`run_ml_model`
  refuses `incremental`). A local scorer is CPU-bound, so re-scoring every row
  is cheap. A remote one bills per call, and without incremental state every run
  would re-score and re-bill every row.
- **Determinism.** `load_pretrained` promises an offline, reproducible result
  from a persisted artifact. A network call cannot make that promise.

No consumer was named on #38, and none of the astrolabe work uses a custom
endpoint.

## Decision

`ml:` scores locally only. A remote scorer is not an `ml:` provider, and no
`InferenceProvider` or `EmbeddingProvider` is added for custom prediction
endpoints either.

The supported paths for a deployed model today are:

- **A model behind an OpenAI-compatible API**, including one deployed on Vertex:
  the built-in `vllm` provider (`supports_custom_base_url`, JSON-schema
  structured output).
- **Any other endpoint**: a Python transform that calls it. Transforms run as
  trusted code, so this adds no core surface, but it does not get the provider
  seam's retry, accounting or credential-protection behaviour by default; the
  transform author supplies them.

## Revisit when

Both of these hold:

1. A named consumer has a deployed custom model that cannot be put behind a
   compatible API.
2. `ml:` can publish incrementally, or the consumer accepts a full re-score on
   every run.

If so, the work is three separate pieces, not one: incremental `ml:`; an
endpoint-reported model version as a term in `code_version` (or an explicit
statement that drift is undetectable); and a `vertex.endpoint` `ml:` provider
with project, location, endpoint resource, timeout, a versioned
request/response mapping, bounded payload, batch and concurrency, retries, a
fake prediction client for tests and a credential-gated live test.

This condition governs `ml:` scoring. A custom endpoint that returns embeddings
is a different shape: it would reopen the `EmbeddingProvider` question
(vector count and dimension validation), not this one, and would need its own
decision.

## Alternatives rejected

- **An `InferenceProvider` for custom endpoints** (the 2026-07-24 suggestion on
  #38). It would have to invent prompt and schema semantics for an endpoint with
  neither, and covers only what `vllm` already covers.
- **A `vertex.endpoint` `ml:` provider now.** The implementation is small; the
  three costs above are not. Built without an answer to identity and
  incremental state, it would republish and re-bill every row on every run and
  could not tell when the model underneath it changed.
