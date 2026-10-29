# A token packing rejection reports tuning rather than retrying

- **Status:** accepted
- **Date:** 2026-10-29
- **Prompted by:** constellations#728, ALE-456

## Context

Vertex text embeddings use a local three-characters-per-token estimate to pack
requests. It is efficient on ordinary prose, but a numeric SEC dev batch estimated
at 17,531 tokens contained 22,454 and exceeded the aggregate 20,000-token limit.
The resulting generic HTTP 400 hid the actionable cause. Each text was at most
1,599 characters; the aggregate request limit was the failure, not a text's limit.

## Decision

Recognize the measured aggregate-limit rejection inside the Vertex embedding SDK
boundary. Emit only an authored warning with numeric text/token counts and the
packing settings to reduce, then raise a non-retryable `ProviderRequestError`
with code `request_token_limit`. Never retain the native message or response in
the warning, error or cause chain. Unknown HTTP 400 errors keep their ordinary
classification. Keep existing packing, model identity and retry policy.

## Alternatives considered

### Automatically split the rejected request and retry

The service tells us the request was oversized, so smaller requests are an
attractive recovery. But execution reserves provider calls from the exact local
split before issuing them. Dynamic splitting would issue calls outside that
reservation and change request/failure accounting. A content-dependent retry
requires a separate admission/accounting design; it cannot safely be hidden in
this provider wrapper.

### Change the default budget to 6000

That budget worked on the bounded numeric slice, but remains an estimate and
cannot guarantee a limit for all text. It would also multiply request counts
for existing prose workloads. Keep it as documented tuning guidance, not a
claim of safety or an implicit throughput change for every operator.

### Log native provider diagnostics

The provider message may contain input text and response metadata. Only the
observed aggregate-limit signature and bounded decimal counts are used in memory.
The warning uses numeric conversions and static prose; nothing else from the
provider message is copied. The normal artifact formatter preserves request
codes, so the recognized cause also survives in run results without native text.

## Consequences and evidence

On 2026-10-29 a dry-run-capped, read-only BigQuery dev query selected 128 numeric
SEC chunks, totaling 204,034 characters. The default budget's first request
(33 texts, 52,521 characters) failed with HTTP 400 after 21.032 seconds. Its
17,531 estimated tokens were actually 22,454 against the 20,000-token limit.
Reducing the budget to 6000 processed all 128 texts in 12 sequential requests
with no truncation. These calls took 5.125 seconds combined; this is a small
diagnostic slice, not a throughput comparison with the cold rejected request.
Retries were disabled and no warehouse output/state was written.

The regression uses a protocol-compatible fake with a synthetic tokenizer and
generated SDK-shaped errors, not captured response bodies. It pins the warning,
artifact code, no hidden calls, exception-chain sanitization, execution-only
identity and the tuned split's output alignment. Disabling the classifier makes
the regression fail. Unrelated HTTP 400 and transient 429 cases keep their
previous classifications. Existing cache/publication/state paths are unchanged.

The classifier is deliberately narrow: a different service rejection format
falls back to the usual sanitized error. A per-text limit needs smaller chunks,
not a smaller aggregate request budget.
