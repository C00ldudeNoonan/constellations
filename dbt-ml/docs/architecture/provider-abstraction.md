# Provider abstraction

Status: implemented, including entry-point plugin discovery, provider-owned
profile configuration, and failed-outcome accounting (issue #71). Built-ins
include Anthropic and vLLM inference plus deterministic and Vertex AI
embeddings; other vendor integrations remain their own issues.

dbt-ml treats hosted inference as an execution capability, not as a property of
the LLM backend. Models describe the transformation they need; a provider
translates that provider-neutral request into an SDK call. This boundary keeps
model compilation, caching, materialization, lineage, and usage accounting
independent from Anthropic or any future inference service.

The built-in inference implementations are Anthropic and vLLM. The built-in
embedding implementations are deterministic and Vertex AI. The contracts cover
inference and embeddings separately so a target can select different providers
for generation and vectorization.

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
  shape and finiteness of every vector and reports how many provider requests
  were needed to satisfy the logical batch.
- `ProviderRuntimeOptions` carries endpoint routing and execution policy such
  as request timeout and retry count without putting them in the semantic model
  request. Endpoint routing is still included separately in cache and model
  identity.
- `ProviderUsage` normalizes input, output, cache-read, and cache-creation token
  counts. A provider-reported cost may be included when the service supplies
  one.

All contract values validate at runtime. Booleans are not accepted as integer
token counts, non-finite costs and vector values are rejected, and batch IDs
must be non-empty and unique. A malformed provider response therefore cannot
silently enter a materialized relation.

`InferenceProvider.complete()` is the synchronous inference primitive.
Native batching (issue #149) is a set of resumable primitives: providers that
advertise `supports_native_batch` implement `submit_batch()` (returning a
validated provider job identifier), `poll_batch()` (returning an artifact-safe
`BatchJobStatus`), `fetch_batch_results()`, and `cancel_batch()`. The LLM
backend drives submit/poll/fetch itself so it can persist the job identifier
before polling, resume after an interrupt without resubmitting, bound polling
with exponential backoff, and cancel on timeout. `complete_batch()` remains as
a one-shot convenience driver over those primitives, with a sequential
per-request fallback for providers without a native batch API.
`EmbeddingProvider.embed()` is separately registered; its request is already
multi-input, so it is the embedding batch primitive. Implementations provide
`_embed()`, while the public wrapper validates that every input has exactly one
finite vector with the requested dimensions. Native embed-model batches also
carry stable input IDs; providers must return those IDs in the original order
before vectors can enter a cache or warehouse relation.

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

Importing `dbt_ml.providers` registers built-in providers. Separately packaged
providers load through the entry-point discovery contract below.

## Plugin discovery (issue #71)

Separately packaged providers are discovered through versioned Python
entry-point groups, so the stock `dbt-ml` CLI can load them without a wrapper
import:

```toml
[project.entry-points."dbt_ml.inference_providers.v3"]
acme = "acme_dbt_ml.provider:AcmeInferenceProvider"

[project.entry-points."dbt_ml.embedding_providers.v3"]
acme = "acme_dbt_ml.provider:AcmeEmbeddingProvider"
```

The group suffix is the provider contract major version
(`PROVIDER_CONTRACT_VERSION`). A plugin advertises the contract it was built
against by choosing the group; it never advertises a class under a contract it
does not implement.

Discovery rules, all enforced before any source, credential, or provider I/O:

- Discovery runs exactly once, at profile resolution, after built-in
  registration. Extraction and materialization never trigger imports.
- Entry points are processed in a deterministic order: sorted by entry-point
  name, then distribution name. Ordering never decides a conflict — it only
  makes error output stable.
- The entry-point name must equal the loaded class's `provider_name`. A
  mismatch is a registration error naming the distribution.
- Duplicate provider names — two distributions claiming one name, or a plugin
  claiming a built-in name — fail the run with both distribution names. There
  is no shadowing and no first-wins.
- A plugin that only advertises groups for a different contract version fails
  with a version-mismatch error naming the distribution and both versions,
  not with "provider not found".
- An entry point that raises on load fails the run with the distribution name
  and exception type, sanitized like every provider error. Broken plugins are
  never skipped silently.
- Loaded classes pass the same registration validation as built-ins
  (metadata, batch surface, constructor shape). `dbt-ml providers list` shows
  every discovered provider with its capability, distribution, and
  implementation identity, and is the supported way to debug discovery.

Discovery failures are configuration errors: they carry no retryability, and
they surface before a manifest is written, so a run can never bill provider
work under a misconfigured plugin set.

## Provider-owned profile configuration (issue #71)

The shared `llm:` block stays small and provider-neutral: provider selection,
model, credential reference, endpoint routing, timeout, cache, system prompt,
pricing, and budget. Everything else a provider needs is declared by the
provider itself, mirroring the model-level `warehouse_options` pattern the
warehouse adapters already use:

```yaml
llm:
  provider: acme
  model: acme-small
  api_key_env: ACME_API_KEY
  provider_options:        # opaque to core; validated by the selected provider
    region: eu-west-1
    tenant: research
```

Embedding providers use the parallel `embedding:` target block. The model ID
and dimensions remain in each `embed:` model; deployment routing, credentials,
provider options, and timeout remain operator-owned:

```yaml
embedding:
  provider: acme
  api_key_env: ACME_EMBEDDING_KEY
  timeout_seconds: 60
  provider_options:
    region: eu-west-1
```

- A provider may publish a strict Pydantic model
  (`profile_options_model()`, `extra="forbid"`,
  `hide_input_in_errors=True`). Core validates `provider_options:` against the
  selected provider's model at profile resolution; unknown keys are a profile
  error when a model is published, and any `provider_options:` content is a
  profile error when none is.
- The parsed options are delivered once, at instantiation, as a frozen model
  instance. Providers receive typed configuration at the boundary; they never
  read profile YAML, environment variables (other than through
  `resolve_credential`), or mutable dictionaries.
- Every declared field carries exactly one classification:
  - `credential` — must be typed as a `CredentialReference`; resolved to a
    `ProviderCredential` only at the provider boundary, and excluded from
    typed config repr, artifacts, hashes, logs, and errors like `api_key_env`.
  - `semantic` — changes what the provider returns for a request. Semantic
    fields enter the response-cache key and model identity, so tuning them
    reprocesses exactly what they affect.
  - `execution` — concurrency, timeouts, retry shaping. Excluded from
    identity; changing them never invalidates state or cache.
  - `artifact-safe` — non-secret descriptive fields that may additionally
    appear verbatim in manifest provider descriptors. `semantic` and
    `execution` fields stay out of artifacts unless also marked artifact-safe;
    `credential` fields never qualify.
- Classification is part of the field's declaration, and validation rejects a
  published model with an unclassified or doubly classified field at
  registration time, not at first use.

## Failed outcomes and usage accounting (issue #71)

Providers bill for work that does not produce a usable result: truncated
responses, schema-invalid tool output, and partially failed native batches all
consume tokens. The contract therefore lets a safe error and normalized usage
coexist instead of forcing a choice between raising and accounting:

- A frozen `InferenceFailure` envelope carries a safe error code, a
  `ProviderUsage`, the request count actually billed, and provenance:
  provider name, effective model, and implementation identity. It rides on
  the sanitized `ProviderError` (`error.failure`) so error control flow is
  unchanged.
- `BatchInferenceItem` keeps exactly-one-of `result`/`error` semantics, and
  the error side may carry the item's billed usage. Batch-level submissions
  and polling overhead stay on `BatchInferenceResult`.
- Synchronous failures still raise, and the raised `ProviderError` may carry
  an attached `InferenceFailure` for the backend to consume. Nothing about
  error control flow changes; only the accounting payload rides along.
- Budgets treat billed failures as spend: token, call, and cost budgets
  consume from failed work exactly as from successes, so a model that fails
  repeatedly cannot bill unbounded retries under an "only successes count"
  loophole.
- Failed work stays visible without payloads: billed-failure usage
  aggregates into run-result metrics (`failed_api_calls`, `failed_*` token
  and cost totals) beside the model's provider/model/implementation
  descriptor — never prompts, responses, headers, SDK message text, or
  credential names.

## Conformance proof (issue #71)

The three contracts above are proven by fixtures, not by a live vendor:

- A separately packaged fake provider — a real distribution (`dist-info`
  metadata plus entry points) with a published profile-options model covering
  all four field classifications — is materialized on the import path during
  tests and driven through discovery and the stock CLI.
- Deterministic malicious and faulty plugins (duplicate names, built-in name
  claims, wrong-contract groups, import-time failures, metadata violations,
  contract-violating results, secret-leaking errors) run without network
  access and must each produce their specified failure before any provider
  I/O.
- Vendor integrations (#15–#22) must be expressible as provider packages with
  no runner or backend branches; a vendor issue that needs a core change is a
  contract gap to fix here first.

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
limit. Custom endpoint deployments add the normalized base URL before the key
is hashed. Legacy pre-contract entries can never be read under the versioned
key format, so they are pruned from the cache file on the next write.

Manifest model entries expose only the effective provider, model, hashed
implementation identity, and, when configured, a one-way endpoint fingerprint.
Per-model run results omit endpoint routing. Credential names and values are
excluded from artifacts.

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

## vLLM mapping

`VLLMInferenceProvider` sends JSON-schema Chat Completions to an explicitly
configured OpenAI-compatible base URL. Local endpoints may omit authentication;
when `api_key_env` is configured, the provider sends the resolved value as a
bearer token. It normalizes usage, rejects truncated or malformed JSON, limits
response size, and retries only transport and transient HTTP failures. Native
vLLM batch submission is not advertised; normal concurrent requests let the
server own scheduling and continuous batching.

## Vertex AI embedding mapping

`VertexEmbeddingProvider` uses the optional `google-genai` SDK in explicit
Vertex mode with the stable `v1` API. Authentication is Application Default
Credentials; `api_key_env` is rejected. Profile options select the GCP project,
location, document task type, query task type, and truncation policy. Project
and location are execution routing and do not invalidate vectors; task types
and truncation behavior are semantic and enter embedding identity.

The provider forwards `EmbedConfig.dimensions` as `output_dimensionality` and
maps `max_retries`/profile timeout into SDK HTTP options. It forwards a runner
batch as one `embed_content` call when the selected model supports that shape;
`gemini-embedding-001` batches are split into one-input calls and reassembled
in the original input-ID order. Run metrics distinguish logical batches from
actual provider calls. Responses must contain one finite vector per input.
Token statistics are normalized into `ProviderUsage`; malformed billed
responses carry sanitized failed-outcome usage. Query helpers mark requests as
query inputs so inherited retrieval uses the separately configured query task
type.

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
5. If native batching is supported, implement `submit_batch`, `poll_batch`,
   `fetch_batch_results`, and `cancel_batch`; validate IDs and limits, forward
   runtime retries, and return results in request order. Job identifiers are
   persisted for resume, so they must be plain provider-issued tokens.
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

Token, API-call, and cost budgets build on this contract but live outside the
provider layer: `dbt_ml.budget` defines strictly typed per-model
(`extraction.options.budget`) and per-run (profiles.yml `llm.budget`) caps for
documents, per-file and total bytes, tokens, provider calls, and spend. The
runner and LLM backend check them before every provider call — exhaustion has
the distinct `budget_exceeded` run status and never bills further work. Likewise, LangChain can be an
integration surface later; it is not the core provider contract and provider
implementations do not expose LangChain-specific types.
