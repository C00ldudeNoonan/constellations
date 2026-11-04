# ADR-0050: a backend's identity covers the options it reaches, not the whole options module

- **Status:** accepted
- **Date:** 2026-11-04
- **Prompted by:** #572 (ALE-302)

## Context

An `extraction:` model's `code_version` includes `BaseBackend.implementation_identity()`,
a digest of the backend's source and of every stel module that source
transitively imports (#363). That is deliberately automatic: nobody has to
remember to bump anything when backend-reachable code changes.

Every backend's option model lives in one file, `backends/options.py`, so each
backend imported it whole. The 0.24.0 release changed the LLM `temperature`
default and a validator in that file. The JSON backend never reads either, but
its identity moved, and in astrolabe `sec_document_registry` re-keyed on
20,289 of 20,289 rows. The change then cascaded through ten models, a paid
embed step among them. Recomputing the exact `code_version` payload under 0.23.0
and 0.24.0 showed one field moved, `backend_implementation`, and of the nine
modules behind it exactly one differed: `backends.options`.

The earlier theory in #572, that a parent's `code_version` cascades into its
children, did not hold. The cause is hash granularity.

## Decision

Keep the automatic source hash and narrow what it covers in the one module that
aggregates unrelated backends. The dependency walk no longer hashes
`backends/options.py` whole. It contributes one entry, built from:

- the backend's own option model, when that module defines it;
- every name a visited module imports from it, followed through the
  definitions that name other definitions in the file (base classes, helpers,
  constants, decorators, annotations), and into the stel modules those
  definitions import;
- every module-level statement that is not a definition or an import, since
  those run whichever model is requested;
- the validation path all backends share (`validate_backend_options` and the
  contract registry), by its own text only.

The last item is by text only because `validate_backend_options` names
`LLMBackendOptions`; following that reference would hand every backend the LLM
model again. Editing the shared path therefore still re-keys every backend,
which is right, since that is code they all run.

A module that binds the options module itself (`from . import options`,
`import ...options`, `import *`) falls back to hashing it whole.

## Alternatives

**A declared version per backend**, bumped by hand like `ALIAS_RESOLVER_VERSION`.
Simple and exact, but it fails in the dangerous direction: a forgotten bump
leaves stale rows reading as current, silently, whereas an over-broad hash costs
a recompute that `stel plan` shows first. #363 chose source digests for this
reason, and a contributor adding a backend would otherwise carry a rule the
tests cannot check.

**Splitting `options.py` per backend.** Gives the same granularity with no
analysis, but the validation dispatcher imports every model, so it needs the
registry inverted and every backend's options moved. A larger change than the
problem justifies, and the shared-path rule above would still be needed.

**Hashing only the options model class.** Misses helpers and bases it names, so
a change to `_validate_css_selector` would leave HTML rows looking current.

## Consequences

- Shipping this changes every backend's identity once, because the algorithm
  differs. Every `extraction:` model re-keys on the first run after upgrading,
  as on any release that touches backend-reachable code. An operator whose rows
  are already stale from 0.24.0 should take this release before running the
  rebuild, so the corpus is rewritten once.
- Afterwards an edit to one backend's options, or to a module only that
  backend's options import, re-keys only that backend.
- The closure follows static names in one file. A definition reached by
  `getattr` or a string is not seen, the same limit the module-level walk
  already has.
- Frozen payload keys of `implementation_identity` are unchanged: the options
  entry sits inside `dependency_sources`.
