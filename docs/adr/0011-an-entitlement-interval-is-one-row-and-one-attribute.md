# ADR-0011: An entitlement interval is one row, one attribute, and never a one-sided bound

- **Status:** accepted
- **Date:** 2026-09-12
- **Prompted by:** #582

## Context

A grant is a row of `(subject_id, attribute, value)`, and several rows for one
attribute mean OR — `_granted_values` collects them and the provider compiles
`IN`. That invariant is intuitive and load-bearing: more grants, more access.

It only expresses membership. `GrantAuthorizationProvider` emitted `EQUAL`,
`IN` and `ARRAY_CONTAINS_ANY` and nothing else, so an entitlement like "the
2024–2025 filings" had no representation — even though `SearchFilterOperator`
already defines `lt`/`le`/`gt`/`ge` and both retrieval stores compile them.
ALE-73 assumed date-range entitlements worked; they did not exist.

Two facts about the surrounding code decided the shape more than the feature
request did.

**Search filters are AND-ed.** Both stores build their predicate with
`" AND ".join(clauses)` (`duckdb.py:1062`, `lancedb.py:1286`). The filter list
is a conjunction, with no way to express a disjunction between two of its
members.

**Policy attributes can already be ordered.** `SearchAttributeConfig.data_type`
admits `date`, `timestamp`, `integer` and `float`, so nothing but the grant
compiler was missing.

## Decision

**A `between` grant is one row whose value is a closed interval,
`<lower>/<upper>`**, with `..` for an open end. A fourth `operator` column says
how to read `value`; absent or null means `eq`, which is what every existing
row already meant.

**One interval per attribute per subject.** A second one is a
`GrantConfigurationError`, not a denial.

**An interval and a literal on the same attribute is also a configuration
error.** Different attributes are unaffected.

**No one-sided operators.** There is no `gte` grant. An open-ended range is
`2024-01-01/..`.

## Alternatives considered

### One-sided bounds as their own operators (`gte`, `lte`)

The obvious shape, and the one to write down as wrong, because it fails in the
dangerous direction.

Rows for an attribute OR together. An operator wanting the 2024–2025 window
writes what reads naturally:

```
(analyst, filing_date, 2024-01-01, gte)
(analyst, filing_date, 2025-12-31, lte)
```

Under OR that is **every row in the corpus**. Each row is correct in
isolation, the pair reads as a narrowing, and the result is a total
over-grant — the silent-success failure this layer exists to remove, and one
that would pass review.

A closed interval in a single row cannot be written that way: there is no
second row to combine wrongly with.

### Several intervals per attribute, OR-ed

What #582 originally proposed, and what an operator would reasonably expect
two interval rows to mean. Not implementable: filters AND, so two intervals
would silently narrow to their overlap — or to nothing, when they are
disjoint, which is exactly when the operator most clearly meant "either".

Rejected in favour of refusing. A grant that means something other than what
was written is worse than one that will not load, and
`GrantConfigurationError` already exists to distinguish "the relation is
wrong" from "this caller may see nothing".

Expressing the union properly needs disjunction in the filter language, which
is a change to every store and to the user-facing filter contract. If that
ever lands, this decision can be revisited; nothing here forecloses it.

### Discretize instead: publish `filing_year` and grant `IN (2024, 2025)`

Costs a column in the corpus and no change to the grant model, and it remains
the better answer when entitlement boundaries always land on year or quarter
lines. Rejected as *the* answer because boundaries are not always calendar
aligned — a contract starting mid-month has no discretization that does not
either over- or under-grant — and because it makes the corpus grow a column
per entitlement axis.

### Encoding the operator in the attribute name (`filing_date__gte`)

Keeps the three-column schema. Rejected: it is a DSL parsed out of a column
that is otherwise compared literally, it collides with any attribute whose
real name contains the separator, and it puts authorization semantics
somewhere no schema documents.

## Consequences

The grants relation gains a column. Relations written before this keep working
untouched — null reads as `eq` — and `stel grants` widens them with an
`ALTER TABLE`, the same way `_ensure_ledger_columns` handles a ledger
predating #355. The store now reads every column rather than projecting
`GRANT_COLUMNS`, because a projection naming `operator` would turn an
un-widened relation into a configuration error on upgrade.

`can_read` has to understand intervals too. It is the second look that catches
a store ignoring a filter, so a recheck that only understood equality would
admit precisely the rows a range filter was meant to exclude. That duplication
is deliberate and is pinned by tests.

Interval bounds are compared as strings. This is correct for ISO dates and
timestamps, which sort lexicographically and are compared the same way by the
filter the store receives. It is **not** correct for a numeric policy
attribute — `"10" < "9"` — so a numeric interval is a latent defect. Nothing
currently declares one, and the honest fix is to type the comparison against
the attribute's declared `data_type` when something does.

The operator becomes part of a grant's identity for idempotence, so the same
text under `eq` and under `between` are two distinct grants rather than one
suppressing the other's insert.

## Evidence

Read at `8fb4b26`: `_granted_values` collected rows into a tuple compiled to
`IN`; `search_policy_filters` emitted only `EQUAL`, `IN` and
`ARRAY_CONTAINS_ANY`; `SearchFilterOperator` defined `lt`/`le`/`gt`/`ge`;
`duckdb.py:1062` and `lancedb.py:1286` both join predicate clauses with
`" AND "`; `SearchAttributeConfig.data_type` admitted `date` and `timestamp`;
`GRANT_COLUMNS` was projected explicitly by `WarehouseGrantStore._read`.
