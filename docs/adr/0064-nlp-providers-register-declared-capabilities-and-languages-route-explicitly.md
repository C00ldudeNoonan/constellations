# ADR-0064: NLP providers register declared capabilities, and languages route explicitly

- **Status:** accepted
- **Date:** 2026-11-23
- **Prompted by:** [#222](https://github.com/C00ldudeNoonan/constellations/issues/222) (ALE-309, under the NLP-depth theme ALE-52)

## Context

`nlp_tokens` and `nlp_entities` could only run spaCy: `NLPBaseOptions.provider`
was `Literal["spacy"]` and `get_nlp_provider` raised for anything else. A
second vendor would have meant a branch in that function, and a second
language would have meant a second transform model per language with the
caller deciding which documents go where. ADR-0060 had already settled that
language detection stays on `langdetect`, and named ALE-309 as the consumer.

## Decision

**A registry, one declaration per provider.** `stel.text.nlp_registry` holds
`NLPProviderRegistration(name, loader, capabilities, languages)`. spaCy
registers itself the same way a third party does: through
`register_nlp_provider`, or a versioned entry-point group
(`stel.nlp_providers.v1`) discovered once, in deterministic order, failing
closed on duplicates, built-in shadowing, name mismatches and load errors,
and never importing a plugin built for another contract version. Transform
execution asks the registry for a provider by name; nothing in `_nlp.py` or the
runner names a vendor.

**Capabilities are declared, not abstracted away.** A provider lists what it
can emit (`tokens`, `lemmas`, `pos`, `entities`, `sentences`, `confidence`) and
optionally the languages it supports. A transform declares what it needs
(`nlp_tokens`: tokens, lemmas, pos; `nlp_entities`: entities). The pairing is
checked when options validate, so an unknown provider, a missing capability or
an unsupported language fails in `stel compile`. A provider that cannot know its
languages ahead of loading a model (spaCy) declares `None`, and the language is
checked at load as before.

**Routing is a mapping the project writes.** `language_routing:` maps language
codes to `provider` + `model` routes. The language comes from an upstream column
(`language_field`) or, without one, from `detect_language`. Two policies cover
the failure cases separately (`on_undetected`: no language could be determined;
`on_unrouted`: a language with no route), each `fail`, `skip` or `fallback`, and
`fallback` is one explicit route. A route key matches the exact code first, then
its primary subtag, because `langdetect` reports `zh-cn` and `zh-tw`.
`fail` is decided for every document before any model loads. Each output row
records `nlp_route`, `nlp_input_language` and `nlp_language_source` beside the
five existing identity columns, only when routing is configured, so an unrouted
project's schema is unchanged.

## Alternatives considered

### Add vendors as branches in `get_nlp_provider`

The smallest change for one more vendor, and exactly the shape AGENTS.md
forbids: every vendor edits core execution, and capability and language support
is discovered by running it. It also gives a third party no way in.

### Reuse the LLM/embedding provider registry

`stel.providers` is built around request/result envelopes, credentials,
usage accounting and endpoint routing. An NLP provider has none of those and
returns tokens and spans. Sharing the registry would drag credential semantics
and a contract version into a local, offline code path, which the issue's
boundaries rule out. The entry-point and fail-closed discovery rules are
copied, not shared.

### One capability-erasing NLP interface

A lowest-common-denominator `analyze()` would hide that one provider has no
sentence boundaries or no entities. Declared capabilities keep the difference
visible and let a project reject a provider for a transform that needs more
than it offers. The protocol (`NLPProvider.pipe`) is unchanged.

### Choose the model from whatever is installed

Picking a model per language by scanning the environment would make output
depend on the machine. Routes are explicit and recorded, so the same project
produces the same routing everywhere and a missing model is an actionable
error naming that model.

### Fall back to detection when the declared language is null

Convenient, but it makes `nlp_language_source` ambiguous per row and hides bad
upstream data. A null declared value is `undetected` and the policy decides.

### Per-language transform models and a SQL/Python split

Works today with no change, but the caller must pre-partition the corpus,
cannot see a mixed-language document that fell through, and gets no recorded
decision. This stays possible and is not removed.

## Consequences

- `NLPBaseOptions.provider` is a string validated against the registry, so
  validating options now triggers entry-point discovery. It reads distribution
  metadata and imports only plugins in the `stel.nlp_providers.v1` group.
- Model-level problems (the model is not installed, or reports another
  language) still surface at load time, after documents are read, because
  checking them would mean importing the provider during compile. Providers
  that can answer ahead of time do so through `languages`.
- A routed run holds one analyzed result per document while it groups documents
  by route; the unrouted path now does too, which is the same memory the rows
  needed anyway.
- No second vendor ships here. Stanza or a project-owned provider can now be
  added as a separate package; evaluating one is deferred until a corpus needs it.
- A detector change (ADR-0060's revisit triggers) stays a one-function change:
  routing calls `detect_language` and nothing else.

## Evidence

- Preflight: `tests/test_nlp_routing.py` pins unknown provider, missing
  capability, unsupported language per route and for the fallback, each
  failing at `model_validate` with no provider loaded.
- Fail-before-load, input-order preservation, one load per route, and the
  separation of the two policies are pinned by tests that were mutated to
  fail (an early load, a disabled capability check, a disabled duplicate-claim
  check) and restored.
- Discovery is exercised against real `dist-info` fixtures on `sys.path`
  rather than a mock, as `tests/test_provider_plugins.py` does.
