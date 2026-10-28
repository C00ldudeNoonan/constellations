# MCP array filters use explicit overlap

- **Status:** accepted
- **Date:** 2026-10-28
- **Prompted by:** constellations#723, ALE-433

## Context

The MCP catalog advertised `eq/ne/in` for `array[string]` attributes, while
the search core accepts only `array_contains_any` for them. The MCP request
enum did not accept that operator, making caller-supplied array filters
unusable. The issue proposed either exposing overlap or translating `in`.

## Decision

Expose `array_contains_any` in the MCP request contract and advertise it as
the only operator for array attributes. Like `in`, it requires a nonempty
JSON array of values. Scalar fields keep their existing operators, and
policy attributes remain server-controlled.

## Alternative considered

Translating `in` into overlap for array fields would make the operator mean
two things: scalar membership in a supplied set, or intersection between
two sets. The CLI and search core already name the second operation
`array_contains_any`. Keeping that name lets clients follow the catalog
without a transport-specific interpretation or alias.

## Consequences and evidence

This repairs the advertised array capability within `mcp_context/v1`; no
previously successful array request changes meaning. Existing manifests and
indexes need no migration. The regression in
`tests/test_agent_context_builtin_pipeline_example.py` discovers the
capability and executes real MCP membership requests against LanceDB in
text, vector and hybrid modes. Deliberately dropping business filters makes
the test fail. Array publication is covered separately by ADR-0039.
