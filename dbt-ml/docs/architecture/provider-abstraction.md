# Provider abstraction

dbt-ml treats hosted inference as an execution capability, not as a property of
the LLM backend. Models describe the transformation they need; a provider
translates that provider-neutral request into an SDK call. This boundary keeps
model compilation, caching, materialization, lineage, and usage accounting
independent from Anthropic or any future inference service.

The first built-in implementation is Anthropic. The contracts deliberately
cover inference and embeddings separately so a target can eventually select
different providers for generation and vectorization.

## Contract surface

`dbt_ml.providers` exposes frozen request and result envelopes:

- `InferenceRequest` contains the model, document content, system prompt,
  structured-output schema, temperature, and maximum output tokens.
- `InferenceResult` contains only the structured output, normalized usage, and
  an optional provider request ID.
- `BatchInferenceRequest` adds a dbt-ml-owned request ID. A
  `BatchInferenceResult` returns exactly one success or safe error per request
  and rejects duplicate IDs.
- `EmbeddingRequest` accepts one or more texts. `EmbeddingResult` validates the
  shape and finiteness of every vector.
- `ProviderRuntimeOptions` carries execution policy such as retry count without
  putting it in the semantic model request.
- `ProviderUsage` normalizes input, output, cache-read, and cache-creation token
  counts. A provider-reported cost may be included when the service supplies
  one.

All contract values validate at runtime. Booleans are not accepted as integer
token counts, non-finite costs and vector values are rejected, and batch IDs
must be non-empty and unique. A malformed provider response therefore cannot
silently enter a materialized relation.

`InferenceProvider.complete()` is the synchronous inference primitive.
`complete_batch()` has a safe sequential implementation, while providers with
a native batch API can override it and advertise `supports_native_batch`.
`EmbeddingProvider.embed()` is separately registered; its request is already
multi-input, so it is the embedding batch primitive. Implementations provide
`_embed()`, while the public wrapper validates that every input has exactly one
finite vector with the requested dimensions.

## Registry and selection

Inference and embedding providers use separate registries:

```python
from dbt_ml.providers import (
    InferenceProvider,
    register_inference_provider,
)


@register_inference_provider
class AcmeInferenceProvider(InferenceProvider):
    provider_name = "acme"
    implementation_version = "1"
    default_model = "acme-small"
    implementation_packages = ("acme-sdk",)
    # implement complete(...)
```

A provider name is a lowercase identifier containing letters, digits,
underscores, or hyphens. Registration rejects abstract implementations,
duplicate names within one capability, and invalid batch metadata. The same
name may intentionally exist in both registries.

Importing `dbt_ml.providers` registers built-in providers. Automatic discovery
of third-party packages is not part of this contract yet; an integration must
import its provider module before profile resolution.

Profiles select the default provider and model without storing credentials:

```yaml
outputs:
  dev:
    warehouse:
      type: duckdb
      path: target/dev.duckdb
    llm:
      provider: anthropic
      model: claude-sonnet-4-5
      api_key_env: ANTHROPIC_API_KEY
```

The target binds the provider to its operator-owned credential environment.
Provider selection always lives in the profile: a model may override the model
ID, but it cannot switch to a different provider — with or without an `llm:`
block, model YAML naming any provider other than the profile-effective one is
a profile error. Named multi-provider credentials are deliberately deferred
until the profile contract can represent them without sending one provider's
secret to another integration.

The selected provider is part of the transformation's semantic identity. Its
implementation identity hashes the contract version, provider class, explicit
provider `implementation_version`, and versions of provider-declared SDK
distributions. It deliberately excludes the dbt-ml release and module source
digests so response-cache entries survive unrelated package upgrades. Provider
integrations must bump `implementation_version` whenever request shaping or
response normalization changes; contract-wide changes bump
`PROVIDER_CONTRACT_VERSION`. Incremental model identity additionally includes
the LLM backend implementation, so current dbt-ml package upgrades still
invalidate model state.

The LLM response cache also includes the provider name, model, contract version,
provider implementation identity, schema, content, temperature, and output-token
limit. Legacy pre-contract entries can never be read under the versioned key
format, so they are pruned from the cache file on the next write.

Manifest model entries and per-model run results expose only the effective
provider, model, and hashed implementation identity. Credential names and values
are excluded from artifacts.

Python transforms that call an inference helper declare `uses_llm: true` under
their `transform:` block. That contract adds the profile-selected provider,
model, system-prompt fingerprint, provider implementation, and dbt-ml helper
implementation to model state identity. It also adds the safe provider
descriptor to manifest and run results artifacts. The transform passes
effective values from `ctx.llm` into the helper; credential values never enter
the context or artifacts.

## Credentials and errors

Provider contract v2 carries a `CredentialReference` through profile and
backend validation, then resolves it to a `ProviderCredential` only at the
provider boundary. `ProviderCredential` is an alias of `ProtectedCredential`:
it is constructed as `ProviderCredential(value)`, has no `.env_var` attribute,
and requires an explicit `reveal()` call at native SDK construction. Protected
values and credential-reference names are excluded from representation,
serialization, comparison, hashing, and fingerprints.
`resolve_llm_credential()` likewise returns this protected value or `None`,
rather than the v1 `(env_name, value)` tuple. Provider implementations must
never place the revealed value in logs, artifacts, usage metadata, cache keys,
or exceptions.

Errors crossing the provider boundary use the `ProviderError` hierarchy.
Unexpected SDK exceptions become `ProviderRequestError` values containing the
provider, operation, exception type, and retryability—not the upstream error
message. Unsafe provider error codes are replaced with `provider_error`.
Response errors must likewise use static, non-sensitive descriptions instead
of forwarding raw response bodies.

This gives run results a stable error vocabulary while leaving detailed SDK
diagnostics behind the credential boundary. Provider request IDs are allowed
for operational correlation but response bodies and headers are not.

For local diagnosis, setting `DBT_ML_DEBUG_PROVIDER_ERRORS=1` logs allowlisted
diagnostics at DEBUG level at the point of conversion. The compatibility helper
`redacted_exception_text()` never emits exception messages, provider source
paths, function names, or local values. It includes only recognized exception
categories, trusted dbt-ml module/line locations, and an external-frame count.
This avoids relying on fragile replacement of repr-, JSON-, URL-, or
metadata-encoded request data. Raised errors, run results, and artifacts carry
only the sanitized form in either mode.

## Anthropic mapping

`AnthropicInferenceProvider` maps the structured-output schema to a forced tool
call and rejects responses that are truncated, omit the requested tool, return
a non-mapping tool input, or contain invalid usage counts. It forwards retry
policy to the Anthropic client.

Native Message Batches are capped at Anthropic's 100,000-request limit. Results
are normalized back into input order. Missing, errored, duplicate, and unknown
result IDs are surfaced as safe provider errors rather than associated with the
wrong document. The provider advertises the native batch cost multiplier as
metadata so accounting code does not need an Anthropic-specific branch.

## Adding a provider

An inference integration should:

1. Subclass `InferenceProvider`, choose a stable `provider_name`, and declare
   credential, batch, `implementation_version`, and `implementation_packages`
   metadata. Bump the implementation version for every behavior change that
   can affect requests, normalized results, caching, or state.
2. Translate `InferenceRequest` without changing its output schema or semantic
   options.
3. Return `InferenceResult` and `ProviderUsage`; never return an SDK response
   object.
4. Convert upstream failures to sanitized `ProviderError` values with exception
   chaining suppressed.
5. If native batching is supported, validate IDs and limits, forward runtime
   retries, and return results in request order.
6. Register the class and add contract tests using a fake SDK client. Tests must
   include credential redaction, malformed responses, usage validation, and
   partial batch failures.

Embedding integrations follow the same rules with `EmbeddingProvider`, plus
strict one-vector-per-input and dimension validation.

## Deliberate boundaries

The provider layer owns SDK translation, provider response validation, retries,
and native batch mechanics. It does not own dbt model compilation, concurrency,
incremental state, caching, warehouse writes, pricing tables, or run budgets.
Those policies remain in the runner and backend, consuming only normalized
provider results and metadata.

Token, API-call, and cost budget enforcement can build on this contract, but is
not implemented by the provider registry itself. Likewise, LangChain can be an
integration surface later; it is not the core provider contract and provider
implementations do not expose LangChain-specific types.
