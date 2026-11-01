# Class ancestry is a compiled declaration

Status: accepted (issue #738, ALE-420)

Relation pairings, MCP class scopes and concept-cloud descriptions need to
agree when one entity class is a subclass of another. Vocabulary terms already
have a `broader` hierarchy, but that describes terms, not their classes. Using
it to infer class ancestry would conflate two independently authored claims.

Each class may declare one parent through `subclass_of`; undeclared parents,
duplicate names and cycles fail validation. Bare class names keep their prior
meaning. Relation checks and MCP scopes include transitive descendants, while
the concept cloud retains the declared class and displays its parent chain.

The serving manifest stores the class declarations and moves to v4, so the
catalog resolves scopes against the compiled hierarchy rather than live YAML.
The concept-cloud bundle moves to v8 and carries the class ancestors. Older
artifacts must be regenerated. This retains the compiled-declaration boundary
in ADR-0028; it does not add transitive graph-relation inference or multiple
inheritance.
