# ADR-0042: Anthropic structured output is native JSON

- **Status:** accepted
- **Date:** 2026-10-29
- **Prompted by:** #714 (ALE-430)

## Context

The Anthropic provider always sent temperature and forced a tool call for
structured extraction. Current SDK signatures remove sampling keywords and
current models can reject forced tool selection. Both the document backend
and warehouse `llm:` models use this provider, including native batch requests.

## Decision

Use `output_config.format` with the SDK-transformed JSON schema and parse the
text block. Share parameters across single and batch requests. Omit temperature
unless configured; retain explicit legacy sampling through `extra_body` for
single calls and batch item parameters. Classify refusal before parsing and
retain billed usage without retaining refusal text. Keep automatic non-strict
tool selection only for schemas with free-form objects, which native JSON
would otherwise silently close and empty.

## Alternatives considered

### Automatic tool choice with a strict tool

This avoids forced selection but still permits a response without the tool.
Native JSON output expresses the actual single structured-result contract
directly and supports both synchronous and batch calls. Automatic non-strict
tools remain the compatibility path for free-form objects; absence of a
matching tool result is an error, never a published partial response.

### Change every provider's default sampling

Passing `None` to existing plugins would change the v3 contract they implement.
Instead, providers retain numeric `0.0` unless they opt into a service default;
only Anthropic opts in here. Explicit numeric requests keep their meaning.

## Consequences

Anthropic's implementation version advances to 2, invalidating its cached
responses and incremental model identities. The dependency floor moves to the
locked SDK version whose schema transformer and output configuration were
verified, while CI overlays the latest 2.x SDK without replacing the lock.
Native JSON object schemas are closed, so free-form schemas must retain their
tool path. Explicit sampling can still be rejected by the model/API
version; stel does not silently drop a configured value or choose another model.
Effort and fallback configuration are separate additions, outside this fix.

## Evidence

Read 2026-10-29:

- [Anthropic SDK migration guide](https://github.com/anthropics/anthropic-sdk-python/blob/main/MIGRATION.md)
  documents removed sampling keywords and the `extra_body` migration.
- [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
  documents native JSON, schema transformation, batches, and billed refusals.
- `test_current_sdk_accepts_single_and_batch_requests` executes the installed
  SDK's generated methods offline for Opus 5.5 and Haiku 4.5, including nested
  array/enum schemas, omitted sampling, and explicit zero.
