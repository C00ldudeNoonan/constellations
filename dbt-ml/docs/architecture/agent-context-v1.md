# Agent context contract v1

`agent_context/v1` is the warehouse contract between dbt-ml document pipelines,
dbt entities, and governed retrieval. It describes auditable warehouse relations;
it is not an identity provider, authorization decision service, metrics semantic
layer, entity-resolution system, or answer-generation protocol.

The machine-readable source of truth is `dbt_ml.agent_context`. A warehouse
transform model opts in with:

```yaml
agent_context:
  contract: agent_context/v1
  grain: document_chunks
```

The model must materialize every field for its grain. Nullable fields must still
exist as columns. Timestamps are timezone-aware UTC values. Intervals are
half-open: `from <= t < to`; a null upper bound is open-ended.

Only a `transform: {type: python}` model may declare `agent_context:` —
`ModelConfig` rejects it on every other kind. This contract *conceptually*
extends the document-registry, chunk, extraction, and embedding grains that the
built-in `extraction:`/`chunk:` primitives (#86) produce, but those primitives
cannot themselves claim it: see [Runtime and artifact
integration](#runtime-and-artifact-integration) below for why, and how to wrap
one in a custom transform instead.

## Relational shape

### `document_registry`

One row per immutable logical document version. Primary key:
`document_version_id`.

| Field | Logical type | Null | Meaning |
| --- | --- | --- | --- |
| `document_id` | string | no | Stable logical document identity across source versions. |
| `document_version_id` | string | no | Immutable source/content-version identity. |
| `source_system` | string | no | Stable source-system namespace. |
| `source_key` | string | no | Stable logical key inside the source system. |
| `source_uri` | string | yes | Human-resolvable source URI when safe to retain. |
| `source_version` | string | no | Immutable source-native version; use `source_content_hash` when no native version exists. |
| `source_content_hash` | string | no | BLAKE2b-128 hash of canonical source text or bytes represented as text. |

The relation also includes every bitemporal, policy, freshness, and provenance
field below.

### `document_chunks`

One row per indexable chunk. Primary key: `context_id`. Foreign key:
`document_version_id -> document_registry.document_version_id`.

| Field | Logical type | Null | Meaning |
| --- | --- | --- | --- |
| `context_id` | string | no | Stable retrieval-record identity for this document version. |
| `chunk_id` | string | no | Stable canonical chunk identity. |
| `document_id` | string | no | Stable logical parent document identity. |
| `document_version_id` | string | no | Exact parent document version. |
| `chunk_index` | integer | no | Zero-based canonical chunk ordinal. |
| `text` | string | no | Canonical indexable chunk text. |
| `chunk_content_hash` | string | no | BLAKE2b-128 hash of `text`. |
| `source_uri` | string | yes | Human-resolvable parent source URI. |
| `citation_page_number` | integer | yes | One-based source page. |
| `citation_section_path` | array[string] | yes | Ordered heading path. |
| `citation_char_start` | integer | yes | Inclusive offset in canonical source text. |
| `citation_char_end` | integer | yes | Exclusive offset in canonical source text. |
| `citation_speaker` | string | yes | Transcript speaker identity. |
| `citation_start_seconds` | float | yes | Inclusive transcript or audio offset. |
| `citation_end_seconds` | float | yes | Exclusive transcript or audio offset. |
| `citation_locator` | JSON | yes | Supplemental locator metadata; it cannot replace common typed fields. |
| `chunker_identity` | string | no | Safe chunker implementation and configuration identity. |

The relation also carries every bitemporal, policy, freshness, and provenance
field below. Carried policy values must equal the parent registry row. This
intentional denormalization permits mandatory in-store policy prefilters while
the registry remains the source of record.

### `context_entity_links`

One row per context-to-entity relationship. Primary key:
`context_entity_link_id`. Foreign key:
`context_id -> document_chunks.context_id`. Multiple chunks may link to multiple
entities without duplicating registry rows.

| Field | Logical type | Null | Meaning |
| --- | --- | --- | --- |
| `context_entity_link_id` | string | no | Stable link identity. |
| `context_id` | string | no | Referenced context identity. |
| `entity_namespace` | string | no | Namespace shared with the dbt entity contract. |
| `entity_name` | string | no | dbt Semantic Layer-compatible entity name. |
| `entity_id` | string | no | Stable namespaced identity of the typed entity key. |
| `entity_key` | JSON | no | Canonical type-preserving serialized key. |
| `dbt_unique_id` | string | yes | Originating dbt resource `unique_id`. |
| `relationship_type` | string | no | Stable relationship semantic, for example `applies_to`. |
| `link_method` | string | no | Deterministic or inferred link-method identity. |
| `confidence` | float | yes | Inferred-link confidence in the closed interval `[0, 1]`. |
| `recorded_from` | timestamp | no | Inclusive UTC system-time boundary. |
| `recorded_to` | timestamp | yes | Exclusive UTC system-time boundary; null means current. |
| `link_provenance_fingerprint` | string | no | One-way identity of link derivation provenance. |

#### Projecting `link_entities` output into this grain

The `link_entities` transform (see the main README) resolves entity mentions to
canonical IDs on a different grain — one row per mention/candidate, keyed by
`entity_link_id`. `dbt_ml.agent_context.project_entity_link` bridges a matched
link into a `context_entity_links` row: the link's `canonical_id` becomes the
row's `entity_key`, so `entity_id` is `fp("…-entity", {entity_namespace,
entity_name, entity_key})` — the *same* id a governed dbt metric keyed on that
namespace/name/canonical value resolves to. That shared `entity_id` is the join
key that combines documentary evidence with structured metrics across the two
MCP planes; neither plane has to understand the other's schema.

Only resolved links belong in the governed context: `project_entity_link`
requires a non-empty `canonical_id`, so `unmatched` mentions are never
published, and callers typically drop `ambiguous` rows rather than record a
guess. Record the deriving resolver with
`entity_link_method(resolver, resolver_version)` (for example
`entity_link:fuzzy:1`) so the link's method identity is auditable and
invalidates the row when the resolver changes.

## Shared fields

### Bitemporal fields

These fields occur on registry and chunk rows.

| Field | Logical type | Null | Meaning |
| --- | --- | --- | --- |
| `validity_known` | boolean | no | Whether business-time validity is known. |
| `valid_from` | timestamp | yes | Inclusive business-time boundary. |
| `valid_to` | timestamp | yes | Exclusive business-time boundary; null means open-ended. |
| `recorded_from` | timestamp | no | Inclusive system-time boundary. |
| `recorded_to` | timestamp | yes | Exclusive system-time boundary; null means current. |

When `validity_known` is false, both validity bounds must be null. This is
different from a known open-ended period, which has a non-null `valid_from` and
null `valid_to`. Unknown-validity rows are excluded from as-of selection unless
the caller explicitly includes them. Recorded intervals for versions of the
same `document_id` cannot overlap; adjacent boundaries are valid.

A reproducible bitemporal selection uses both predicates:

```sql
where validity_known
  and valid_from <= :valid_at
  and (valid_to is null or :valid_at < valid_to)
  and recorded_from <= :recorded_at
  and (recorded_to is null or :recorded_at < recorded_to)
```

### Policy fields

These fields occur on registry and chunk rows.

| Field | Logical type | Null | Meaning |
| --- | --- | --- | --- |
| `tenant_id` | string | yes | Trusted tenant partition identifier. |
| `is_public` | boolean | no | Explicit public-access marker. |
| `access_groups` | array[string] | no | Sorted, duplicate-free trusted group identifiers. |
| `classification` | string | yes | Optional classification label. |
| `policy_ref` | string | yes | Safe governing-policy identifier. |
| `policy_version` | string | yes | Safe governing-policy version. |
| `authorization_resolved` | boolean | no | Whether policy metadata came from a trusted source. |

A row is eligible for retrieval publication only when
`authorization_resolved = true` and it is either explicitly public or has at
least one policy attribute (`tenant_id`, `access_groups`, `classification`, or
`policy_ref`). Unresolved rows remain auditable in the warehouse but must not be
indexed. This is publication eligibility, not caller authorization.

At query time the retrieval server derives mandatory filters from trusted
caller claims. User or model relevance filters are composed separately and
cannot remove or weaken mandatory policy filters. dbt-ml does not authenticate
callers or decide their permissions.

`policy_fingerprint()` normalizes `access_groups` before hashing.
`retrieval_projection_fingerprint()` includes all policy, temporal, freshness,
citation, content, and provenance fields. An ACL-only change therefore changes
the projected record fingerprint and must trigger the delete/upsert behavior of
record-level retrieval publication.

### Freshness fields

These fields occur on registry and chunk rows.

| Field | Logical type | Null | Meaning |
| --- | --- | --- | --- |
| `source_updated_at` | timestamp | yes | Update time reported by the source. |
| `source_observed_at` | timestamp | yes | Time this source version was observed. |
| `ingested_at` | timestamp | no | Ingestion time. |
| `materialized_at` | timestamp | no | Warehouse materialization time; never before ingestion. |
| `freshness_checked_at` | timestamp | yes | Latest successful freshness check; never before ingestion. |
| `refresh_due_at` | timestamp | yes | Deadline for the next successful pipeline refresh. |
| `stale_after` | timestamp | yes | Time after which this source version is old. |

`freshness_status()` reports `pipeline_stale` when `refresh_due_at` has passed,
`source_stale` when `stale_after` has passed, `unknown` when no successful check
is known, and `fresh` otherwise. Pipeline failure takes precedence when both
deadlines have passed, preserving the distinction between old-but-observed
source content and a stopped refresh pipeline.

### Provenance fields

These fields occur on registry and chunk rows. Values are safe identities or
one-way fingerprints, never credentials, caller claims, prompt text, or source
content.

| Field | Logical type | Null | Meaning |
| --- | --- | --- | --- |
| `upstream_unique_id` | string | no | Upstream dbt or dbt-ml resource `unique_id`. |
| `invocation_id` | string | no | Materialization invocation identity. |
| `parser_identity` | string | yes | Parser implementation and version. |
| `transform_identity` | string | yes | Transform implementation and version. |
| `prompt_fingerprint` | string | yes | One-way prompt identity. |
| `schema_fingerprint` | string | yes | One-way extraction-schema identity. |
| `provider_identity` | string | yes | Provider contract and implementation identity. |
| `model_identity` | string | yes | Safe provider model identifier. |
| `provenance_fingerprint` | string | no | BLAKE2b-128 identity of the complete safe provenance chain. |

## Identifier algorithms

All v1 IDs are adapter-independent lowercase 32-character hexadecimal strings.
Simple hashes use BLAKE2b with a 16-byte digest over UTF-8 bytes.

`fp(domain, value)` below is `canonical_fingerprint(value, domain=domain,
version=1, digest_size=16)`. It prefixes `dbt-ml-canonical-fingerprint`, then
length-frames the domain, version, and type-preserving canonical JSON value.
Mapping entries are key-sorted and scalar types remain distinct. The
implementation in `dbt_ml.hashing` is normative.

| ID | Exact v1 input |
| --- | --- |
| `document_id` | `fp("dbt-ml-agent-context-document", {source_system, source_key})` |
| `document_version_id` | `fp("dbt-ml-agent-context-document-version", {document_id, source_version, source_content_hash})` |
| `chunk_id` | `blake2b16(utf8(document_id + "|" + decimal(chunk_index) + "|" + text))` |
| `context_id` | `fp("dbt-ml-agent-context-record", {document_version_id, chunk_id})` |
| `entity_key` | `canonical_json(typed_key_value)` |
| `entity_id` | `fp("dbt-ml-agent-context-entity", {entity_namespace, entity_name, entity_key})` |
| `context_entity_link_id` | `fp("dbt-ml-agent-context-entity-link", {context_id, entity_id, relationship_type})` |

For v1, `chunk_index` is the canonical locator identity inside a logical
document. Citation metadata does not rewrite `chunk_id`; changing it does
change the retrieval projection fingerprint and therefore republishes the
record. A source/content version change always creates a new `context_id`, even
when unchanged chunks retain their `chunk_id`.

`canonical_entity_key()` preserves the difference between values such as the
integer `1` (`["int","1"]`) and string `"1"` (`["string","1"]`). SQL
producers must emit the same compact, key-sorted JSON representation.
Non-canonical JSON is rejected.

## Citation resolution

A result's `context_id` joins to `document_chunks`, then
`document_version_id` joins to `document_registry`; `context_id` also joins to
zero or more `context_entity_links`. The typed citation columns plus the exact
document version are sufficient for a human-verifiable location.
`citation_locator()` parses JSON-backed fields and returns a typed locator for
API consumers.

## Runtime and artifact integration

For Python-backed transform models, dbt-ml validates contract columns,
nullability, types, IDs, interval rules, freshness chronology, citation ranges,
and policy shape before materialization. Built-in extraction and chunk models
do not claim this contract because they cannot derive trusted policy and
bitemporal fields — their generated columns (`chunk_id`, `document_id`,
`chunk_index`, ...) are a different, narrower shape than the contract's
(`context_id`, `document_version_id`, the bitemporal/policy/freshness/
provenance fields, ...), and neither primitive resolves authorization or
bitemporal validity on a document's behalf. `validate_agent_context_relations()`
additionally checks foreign keys and carried-policy equality for a complete
relation set. The relations contain ordinary scalar, timestamp, array, and JSON
columns so a future SQL transform from issue #141 can construct them without a
Python data round trip.

A project built on the `extraction:`/`chunk:` primitives that wants to become
MCP-discoverable needs a `transform: {type: python}` model in front of the
warehouse relation it wants to expose: read the extraction/chunk output,
compute the contract's identity, temporal, policy, freshness, and provenance
columns (`dbt_ml.agent_context` provides the id-generation helpers), and
declare `agent_context:` on that transform. The transform can `ref()` the
existing extraction/chunk models, so no data pipeline is discarded — only the
contract-shaped projection is new.

The manifest exposes the contract version, grain, fields, primary and foreign
keys, model `unique_id`, lineage, and safe fully qualified relation. Generated
docs render the same descriptor. `emit-dbt-sources` emits all contract columns
and places contract metadata under `meta.dbt_ml.agent_context`; JSON and array
logical types are represented as portable dbt `string` columns where necessary.
None of these artifacts contains credentials or sensitive caller claims.

Retrieval projections may denormalize required filters, citations, and lineage,
but the three warehouse relations remain the auditable source of record. Store
schemas, runtime answer generation, and agent orchestration are outside this
contract.
