# Semantic retrieval architecture

Status: accepted design for issue #133; the bounded local public LanceDB slice
was implemented in issue #134. Sections covering governed authorization,
portable query serving, replacement, and coordinated readiness remain the
target contract for follow-up issues and are not current guarantees.

This decision builds on the accepted
[warehouse adapter capability boundary](adapter-capabilities.md) and the
[inference/embedding provider boundary](provider-abstraction.md).

## Decision

dbt-ml will represent a serving index as a distinct `search:` DAG resource and
will call the backend role `RetrievalStore`.

A search index is a serving projection of exactly one warehouse-backed model.
The warehouse remains the system of record for canonical rows, lineage,
schema tests, and publication state. A retrieval store owns collection and
index lifecycle plus vector, text, filtered, and hybrid retrieval. It never
implements `WarehouseAdapter`, exposes arbitrary SQL, or becomes the only copy
of a row.

The first proving implementation is local LanceDB; turbopuffer remains planned.
Features are
capabilities, not assumptions: a vector-only store can implement this contract
without pretending to support BM25, hybrid search, online schema evolution, or
atomic replacement.

The end-to-end boundary is:

```text
source -> warehouse models -> search index -> portable retrieval API
                  |                  |
                  |                  +-- serving projection
                  +-- canonical rows, lineage, tests, and publication state
```

The following rules are non-negotiable:

- A search index never has a fabricated warehouse relation.
- Core sends only typed records and requests to a retrieval store; SDK request
  objects, raw SQL/filter strings, and arbitrary provider payloads do not cross
  the boundary.
- Credentials and physical endpoints are operator-owned profile configuration.
- Mandatory policy predicates remain separate from user relevance predicates
  until core combines them. They run inside the store before ranking and
  projection; post-filtering is not authorization.
- Access mode is explicit and defaults to governed. Publishing a public index
  additionally requires a profile-owned operator opt-in.
- Publication is at-least-once. Store mutation happens before warehouse state
  advancement, and every retry is safe by stable record ID.
- A collection with stale or failed governed publication state is unavailable
  through the dbt-ml query surface until a successful reconcile marks it ready.
  A query holds a generation-pinned read lease through result validation so a
  publisher cannot race that check.

## Resource grammar

### A distinct `search:` resource

Search indexes stay in the existing `version: 2` `models:` files so they keep
YAML provenance, one global node namespace, selectors, tags, and DAG lineage.
The `search:` block is a fifth mutually exclusive resource block beside
`extraction:`, `transform:`, `chunk:`, and `ml:`. Its internal and artifact
resource type is `search_index`.

`search:` is intentionally not named `index:`: a collection may contain vector,
full-text, and scalar indexes, while the dbt-ml resource represents the whole
queryable serving projection. It is also not a `search_index` materialization
on a warehouse model. Publication has different capabilities, tests, state,
failure modes, and artifacts from warehouse materialization.

```yaml
version: 2
models:
  - name: economic_context_search
    description: Governed passages for economic-data retrieval
    depends_on: [ref('economic_context_embeddings')]
    materialization: incremental
    tags: [retrieval, economic-data]

    search:
      access: governed
      store: primary
      collection: economic_context
      id_field: chunk_id
      document_id_field: document_id
      chunk_id_field: chunk_id
      text_fields: [text]
      return_text_fields: [text]

      vector:
        field: embedding
        dimensions: 384
        metric: cosine
        search: approximate
        embedding: inherit

      full_text:
        fields: [text]

      attributes:
        - name: series_id
          data_type: string
          filter_role: user
          sortable: true
          returned: true
        - name: tenant_id
          data_type: string
          filter_role: policy
        - name: access_groups
          data_type: array[string]
          filter_role: policy

      display_fields: [title, source_uri, page]
      query:
        modes: [vector, text, hybrid]
        consistency: strong

      on_index_change: fail
```

The example is declarative. `store: primary` is a logical alias, not a URL or
credential. `embedding: inherit` requires the safe identity emitted by the
direct upstream `embed:` model from #138; index-time vectors and query-time
embedding must match that compiled identity.

### Configuration sketch

The implementation uses strict, frozen Pydantic v2 models. This sketch records
the public shape rather than prescribing module layout:

```python
class EmbeddingIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: str
    model: str
    provider_contract_version: int
    provider_implementation: str
    semantic_config_fingerprint: str
    dimensions: int = Field(gt=0)


class SearchVectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    field: str
    dimensions: int = Field(gt=0)
    metric: Literal["cosine", "euclidean", "dot"]
    search: Literal["exact", "approximate"] = "approximate"
    embedding: Literal["inherit"] | EmbeddingIdentity = "inherit"


class SearchAttributeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    data_type: AttributeType
    nullable: bool = False
    filter_role: Literal["none", "user", "policy", "user_and_policy"] = "none"
    sortable: bool = False
    returned: bool = False


class TextAnalyzer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["store_default"] = "store_default"


class FullTextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    fields: tuple[str, ...]
    analyzer: TextAnalyzer = Field(default_factory=TextAnalyzer)


class SearchQueryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    modes: frozenset[Literal["vector", "text", "hybrid", "filter"]]
    consistency: Literal["strong", "eventual"] = "strong"


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    access: Literal["governed", "public"] = "governed"
    store: str | None = None
    collection: str | None = None
    id_field: str
    document_id_field: str | None = "document_id"
    chunk_id_field: str | None = None
    text_fields: tuple[str, ...]
    return_text_fields: tuple[str, ...] = ()
    vector: SearchVectorConfig | None = None
    full_text: FullTextConfig | None = None
    attributes: tuple[SearchAttributeConfig, ...] = ()
    display_fields: tuple[str, ...] = ()
    query: SearchQueryConfig
    on_index_change: Literal["fail", "rebuild", "online"] = "fail"
    index_options: Mapping[str, JsonValue] = Field(default_factory=dict)
```

`index_options` is not an escape hatch. After the logical store is resolved,
the registered adapter validates it with an adapter-owned, `extra="forbid"`
Pydantic model before credentials, network access, or mutation. Its normalized,
safe semantic form participates in index configuration identity. It is an
explicitly non-portable extension: projects using it must provide a block valid
for every selected target or accept compile failure. Native SDK request
dictionaries are never accepted.

At least one text field is required. A vector block is optional so a store can
serve a text-only index, but `vector` and `hybrid` query modes require it. A
`text` or `hybrid` query mode requires `full_text`. `full_text.fields` must be a
non-empty subset of `text_fields`; `return_text_fields` must also be a subset
and marks existing text columns for projection without copying them into the
display payload.

`embedding: inherit` is the normal path and is compile-checked against the
upstream #138 descriptor. Externally produced vectors require an explicit full
`EmbeddingIdentity`; this is a trusted declaration, runtime still validates
every vector, and query vectors must carry the identical identity. A free-form
model/version label is insufficient.

### Dependencies and selectors

A search index:

- has exactly one direct upstream warehouse-backed model;
- cannot depend directly on a source or another search index;
- is a leaf serving sink: warehouse models and search indexes cannot `ref()` it;
- shares the global source/model/search-index name namespace; and
- must resolve to a unique `(safe store target identity, physical collection)`
  within one compiled project target.

The DAG gains `NodeKind.SEARCH_INDEX`. Internal helpers currently named
`select_models()` and `execution_order()` should generalize to runnable
resources rather than classifying a serving sink as a warehouse model.

Name, tag, `state:modified`, and graph selectors behave uniformly:

| Selector | Behavior |
| --- | --- |
| `economic_context_search` | publish only the index from an already materialized upstream relation |
| `+economic_context_search` | run all ancestors, then publish the index |
| `economic_context_embeddings+` | run the model and its downstream index |
| `tag:retrieval` | select matching runnable resources |
| `state:modified+` | include changed resources and their descendants |

`ls --resource-type` adds `search_index`; `model` continues to mean a
warehouse-backed model. `show` remains a warehouse-table operation and reports
that a search index has no relation, with guidance to the index metadata/search
commands delivered by the query workstream.

Search resources must explicitly declare `materialization` in v0.2 so the
existing warehouse-model default of `full` cannot accidentally request an
unsupported lifecycle. `materialization: incremental` reconciles stable IDs.
`materialization: full` replaces the entire collection on every run and
therefore requires atomic full replacement.
The loader enforces explicit presence through Pydantic's model-fields-set data;
the existing default remains unchanged for the other four resource kinds.
`--full-refresh` requests the same replacement for a selected incremental
search index. It never means drop-and-recreate.

## Configuration ownership and precedence

### Profile-owned targets

Profile targets own physical routing, credentials, and operational policy:

```yaml
economic_data:
  target: dev
  outputs:
    dev:
      warehouse:
        type: duckdb
        path: target/economic_data.duckdb
        schema: economic_data

      retrieval:
        default: primary
        allow_public_indexes: false
        authorization:
          resolver: economic_data_acl
          test_principals: [economic_data_smoke]
        stores:
          primary:
            type: lancedb
            path: target/lancedb
            collection_template: '{project}__{target}__{collection}'
            timeout_seconds: 30
            minimum_consistency: strong

      embedding:
        provider: local
        model: bge-small-en-v1.5
        api_key_env: LOCAL_EMBEDDING_API_KEY
```

The exact embedding-profile expansion is shared with issues #71 and #138. The
boundary decided here is stable: query-time embedding occurs before the
`RetrievalStore` call, uses a profile-owned provider and credential, and must
match the index's compiled provider contract, provider implementation, semantic
configuration fingerprint, model, and dimensions.

Project defaults are intentionally small and portable:

```yaml
# dbt_ml_project.yml
search:
  default_store: primary
  default_query:
    consistency: strong
  on_index_change: fail
```

They use a strict `ProjectSearchDefaults` Pydantic model. Field mappings,
collection names, access mode, embedding identity, and adapter-specific index
options remain resource declarations rather than project-wide implicit state.

Configuration resolves in these domains:

1. Core supplies validation and bounded-query defaults.
2. The profile selects the logical store aliases and owns connection,
   credential, physical collection template/namespace, timeout, retry, batch
   maxima, public-index permission, trusted policy resolver/test-principal
   aliases, and minimum consistency.
3. `ProjectSearchDefaults` chooses the default logical alias, default requested
   consistency, and default index-change policy.
4. The model `search:` block replaces those project scalar defaults when set
   and declares collection/field/query semantics plus typed index options.

Merging is field-by-field, not a recursive dictionary merge. Model scalar
values replace project scalar values; model tuples and `index_options` replace
their complete project-level value (there is no project value for either in
v0.2). The resolved request must still satisfy profile minima and maxima.

A model can require a stronger portable guarantee than the profile default; it
cannot silently weaken profile policy. Project/model YAML cannot set an
endpoint, resolved namespace prefix, credential environment-variable name, or
physical authentication option. A model cannot switch the query embedding
provider; it only records and validates the expected safe identity.

`retrieval.authorization.resolver` names an operator-registered trusted policy
resolver; it is required when the selected target contains a governed index.
`test_principals` allowlists opaque aliases that resolver recognizes for
governed smoke tests. The aliases contain no claims or credentials, project YAML
can only reference them, and neither aliases nor resolved policy context enter
artifacts or logs. Public-only targets may omit authorization.

The logical collection defaults to the resource name. The profile-owned
template defaults to `{project}__{target}__{collection}`, preventing two
projects or targets that share a store from colliding by default. Store-specific
identifier validation happens at compile. The resolved physical collection is
artifact-visible; the raw template and unrelated profile routing fields are
not. Profiles must not derive names from secrets or sensitive caller claims.

`access: governed` is the default. It requires at least one policy field,
strong consistency, and a trusted policy resolver. `access: public` requires
`retrieval.allow_public_indexes: true` in the active profile. A project author
cannot turn an operator's target public by omission or model config alone.

### Secret handling

Retrieval configuration follows the provider credential boundary, not generic
recursive environment interpolation:

- Profiles store an explicit environment-variable reference, never a secret.
- Runtime resolves the value only when constructing the SDK client and wraps it
  in a value excluded from representation, comparison, serialization, and
  hashing. Access requires an explicit reveal at the SDK boundary.
- Credential-bearing URL user information is rejected.
- A store adapter exposes positive-allowlisted `safe_descriptor()` and
  `state_descriptor()` values. Core never serializes or fingerprints a general
  config dump and tries to remove known secret fields afterward.
- Safe errors contain only a stable error code, store type, operation, and
  retryability. Native exception messages, request/response bodies, headers,
  vectors, text, filter literals, caller claims, and record IDs do not enter
  logs or artifacts.
- Credential values and credential-variable names are excluded from manifests,
  docs, run results, cache keys, target identity, and code/config identity.

#154 hardens the existing warehouse-profile path to this same boundary. A
retrieval adapter must use the protected credential-reference contract from its
first implementation and must not copy generic pre-validation secret
interpolation into ordinary Pydantic strings or mappings.

A future query cache lookup occurs only while holding a ready-generation read
lease. Its key includes the physical generation and config fingerprint,
embedding identity, query mode/value fingerprint, effective policy fingerprint,
user-filter fingerprint, requested projection, `top_k`, `candidate_k`, `rrf_k`,
filter-query sort/null ordering, and consistency. Inapplicable fields use a
domain-separated sentinel rather than being omitted ambiguously. Literal
claims, filters, vectors, and text never enter cache metadata. Otherwise an
authorized, differently shaped, or pre-revocation result can be reused across
tenants, requests, or generations.

## Canonical collection and row contracts

### Collection specification

`CollectionSpec` is immutable and contains only portable semantics:

- logical and physical collection names;
- explicit access mode and policy-field requirements;
- stable ID and document-ID mappings;
- text fields and full-text analyzer settings;
- optional vector field, dimensions, distance metric, search kind, and safe
  embedding identity;
- typed attributes with user-filter, policy-filter, sort, and return roles;
- display-field projection;
- required query modes and consistency;
- normalized adapter-validated index options;
- `schema_version`, `config_fingerprint`, and store implementation identity.

Reserved physical fields use a versioned dbt-ml namespace. User mappings that
collide with reserved fields fail compile. A store may choose different native
field names internally, but that mapping is not observable through the portable
API.

The configuration fingerprint includes ID/field mappings, field types and
roles, vector dimensions/metric/search mode, embedding identity, text analyzer,
access/query guarantees, normalized semantic index options, and the retrieval-
store contract/implementation version. Timeouts, retry counts, batch size,
credentials, and log settings are execution-only and excluded.

Every Pydantic contract shown in this decision uses `extra="forbid"` and
`frozen=True`, including envelopes where the repeated model config is elided.
Because Pydantic freezing is otherwise shallow, validators canonicalize nested
mappings/lists into immutable, deterministically ordered value objects before a
fingerprint, cache, state, or store call. `Mapping` in the sketches denotes that
read-only interface, not permission to retain caller-owned mutable dictionaries.

### Indexed rows

Core validates warehouse rows into an immutable `IndexedRow` before passing
them to a store:

```python
class IndexedRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    document_id: str | None
    chunk_id: str | None
    text: Mapping[str, str]
    vector: tuple[float, ...] | None
    attributes: Mapping[str, AttributeValue]
    display: Mapping[str, JsonValue]
    provenance: WarehousePointer
    input_fingerprint: str
    config_fingerprint: str
    record_contract_version: Literal[1] = 1
```

The constraints are:

- `id` is a non-empty UTF-8 string. Optional `document_id` and `chunk_id` values
  are also non-empty when present. IDs containing NUL are rejected. A store
  advertises its byte limit; over-limit values fail with a safe row position,
  never truncation or silent hashing. Projects intended for LanceDB and
  turbopuffer portability should keep IDs at or below 64 UTF-8 bytes.
- IDs are unique within a collection. The complete key domain is validated
  before the first store mutation through a typed warehouse key-domain check.
  The eager read in the local proof of concept may do this in memory;
  production work requires a warehouse-native/null-and-uniqueness check plus
  paged state/diff operations. Other row values are validated one projected
  batch at a time before that batch mutates the store.
- Every configured text field is a string. At least one is non-empty. Only
  declared text and display fields are copied; a search index never copies the
  whole upstream row by default.
- A configured vector is required on every row, has exactly the declared
  dimensions, contains finite real values, and matches the recorded embedding
  identity. Booleans are not numeric vector values. Stores must use their
  reject-on-bad-vector mode; dropping, filling, or nulling malformed vectors is
  forbidden.
- Filter attributes support `string`, signed 64-bit `integer`, finite 64-bit
  `float`, `boolean`, `date`, UTC `timestamp`, and homogeneous arrays of those
  scalar types. Nested objects and arbitrary JSON are display-only. Null is
  accepted only when the field declares `nullable`; null or missing policy
  attributes never grant access.
- `display` is an explicit JSON-safe projection with store-advertised row and
  field byte limits. Text in `return_text_fields` is referenced once rather
  than duplicated in `display`. Any attribute with `policy` in its filter role
  is non-returnable and cannot be a text/display field; violating that rule is
  a compile error. An upstream model can deliberately project a separate safe
  display value when disclosure is required.
- `provenance` contains a safe upstream resource identifier and record ID. It
  may include declared document/chunk IDs and safe locators. The authoritative
  large document stays in the warehouse; dereferencing it is a query-service
  operation subject to the same policy context, not a store-adapter operation.
- `input_fingerprint` covers every serving-visible or policy-relevant vector,
  text, attribute, display, provenance, and identity value using #139's typed
  canonical fingerprinting plus the `config_fingerprint`. It is exactly the
  value written to `StateRecord.input_fingerprint`; the search node's compiled
  code/config identity is exactly `StateRecord.code_version`.
- `config_fingerprint` is the collection specification fingerprint described
  above. `record_contract_version` domain-separates canonical row encoding.
  These values are core publication metadata and warehouse state. An adapter
  persists them in reserved store fields only when its schema contract declares
  that behavior; callers cannot filter or project them.

Mapping key order alone is not a change; list order and typed value changes are.

Chunk models already provide deterministic `chunk_id` and `document_id` values.
They are the preferred record and document identities. Non-document rows may
omit both optional values while retaining the stable record ID and provenance
pointer; the contract does not hard-code column names.

### Large text and warehouse pointers

Text used for BM25 must be present in the store. Text returned directly in a
hit must be declared in `return_text_fields`. Large raw documents and unused
source columns remain warehouse-only and are represented by a frozen pointer:

```python
class WarehousePointer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    upstream_unique_id: str  # model.<project>.<name>
    record_id: str
    document_id: str | None = None
    chunk_id: str | None = None
    locator: Mapping[str, JsonScalar] = Field(default_factory=dict)
```

The pointer is descriptive, not SQL. Only the warehouse adapter/query service
may dereference it. A retrieval store cannot query the warehouse or return an
SDK-native pointer. A physical relation identity is neither required nor stored
in every record. This makes chunk-sized text the normal serving unit without
forcing every raw source body into a remote index.

## Typed filter and authorization contract

### Expression tree

Filters are strict, frozen, tagged expression nodes. The caller-visible grammar
is:

```python
UserFilter = (
    Eq | NotEq | Lt | Lte | Gt | Gte | Between | In |
    IsNull | IsNotNull | ContainsAny | ContainsAll |
    StartsWith | Contains | And | Or | Not
)

EffectiveFilter = UserFilter | MatchAll
```

`MatchAll` is an internal root-only identity node. It cannot be parsed from a
public request, nested by a caller, or emitted by project YAML; core uses it only
for an operator-approved public index and simplifies conjunctions before the
adapter call.

There is no raw string, SQL fragment, JSON object, callable, or native filter
escape hatch. Every expression is validated against `CollectionSpec` before a
store call:

| Operator | Valid field types |
| --- | --- |
| `eq`, `not_eq` | scalar or same-typed homogeneous array |
| `lt`, `lte`, `gt`, `gte` | integer, float, date, timestamp; string only with matching binary collation |
| `between` | a store-advertised ordered scalar type |
| `in` | scalar with a non-empty same-typed value list |
| `is_null`, `is_not_null` | nullable field when advertised |
| `contains_any`, `contains_all` | homogeneous array when advertised |
| `starts_with`, `contains` | string when advertised |
| `and`, `or`, `not` | expressions whose leaves are all supported |

`Eq(field, None)` is rejected in favor of explicit null operators. Empty
boolean groups and empty membership lists are rejected. Implementations impose
capability-advertised limits for tree depth, node count, membership values, and
serialized bytes before translation. Error paths may name a field/operator and
expected type, but never include filter literal values.

`Between(field, lower, upper)` is inclusive at both ends, requires non-null
same-typed endpoints, and rejects `lower > upper`; callers use `gt`/`lt` for an
open end. String ranges require the same advertised binary collation as other
ordered string comparisons.

The AST is a core vocabulary, not a promise that every store supports every
node/type pair. Capabilities publish an operator-to-types matrix. Compile checks
configured policy requirements; query setup checks the concrete user tree.

Portable evaluation is two-valued. Missing fields are treated as null;
comparison (including `not_eq`), membership, string, and array predicates on
null evaluate false. `is_null` is true for missing or null and `is_not_null` is
its inverse. Boolean nodes then apply ordinary two-valued logic, so
`not(eq(field, value))` is deliberately distinct from `not_eq(field, value)`
when the field is null. Dates and timestamps compare after exact type validation
and UTC normalization. Arrays are ordered values for equality, while containment
treats their elements as a set for membership. For a policy expression, every
referenced policy field must be present and non-null before the expression is
evaluated; otherwise the entire policy is false even below `not` or `or`. This
prevents negation from turning missing ACL metadata into authorization. Adapter
conformance tests run the same truth table against every native translator.

Regex, glob, fuzzy match, arbitrary full-text syntax, geospatial predicates,
and backend functions are outside the v1 portable grammar. They can be added as
versioned typed operators only when at least one capability can describe their
semantics precisely.

### Policy separation and fail-closed behavior

The public service accepts only `user_filter`. An internal factory authenticates
the caller, invokes a registered trusted policy resolver, and creates the
otherwise non-constructible `AuthorizedSearchRequest`. Resolver absence,
failure, or an empty result for a governed index denies before embedding or
store I/O. Core alone creates:

```text
effective_filter = AND(policy_filter, user_filter)
```

A governed index declares at least one policy field, requires strong
read-after-write consistency, and requires a non-empty policy filter. User
filters cannot reference policy-only fields. A user cannot provide, replace,
weaken, negate, or remove the policy expression. Eventual consistency is
available only to explicitly public indexes because a stale read could
resurrect revoked access.

For a public index, the internal factory supplies a core-owned `MatchAll`
policy node after verifying the profile's public-index permission. Callers
cannot supply that node themselves.

The store receives only a resolved request whose effective filter has already
been checked. The adapter must apply that filter inside the store before vector
ranking, full-text ranking, both hybrid legs, and returned-field projection.
Client-side filtering and provider post-filter modes are forbidden for policy.
If a field, operator, prefilter guarantee, or required consistency is
unsupported, the request fails before any result is read.

Diagnostics and traces retain only node/operator/field shape plus scoped,
non-reversible fingerprints. They never contain policy literals, caller claims,
user-filter literals, query text, or vectors.

Namespaces can reduce the filter surface but are not treated as authentication
or an identity-provider boundary. Profile-owned namespace routing and mandatory
row policy may be used together.

## Portable query and result contracts

Query-time embedding sits above the store:

1. Authenticate, resolve/validate mandatory policy and user filters, validate
   requested projection, and reject unsupported capabilities.
2. Acquire a shared query lease that pins the ready physical generation and is
   held until the result has been validated.
3. For vector or hybrid search from query text, resolve the profile-owned
   embedding provider and require its complete `EmbeddingIdentity` to match the
   index descriptor. Text-only and filter-only requests skip this step.
4. When step 3 applies, convert text to one validated finite
   `EmbeddedQueryVector` while the generation remains pinned.
5. Execute typed store primitives, validate the result/generation, then release
   the lease and expose the result.

`TextQuery.text` is literal UTF-8 query text, never provider query syntax,
Lucene/SQL, or a backend expression. The adapter must bind, escape, or otherwise
translate it as data and rejects unsupported size/encoding before native I/O.

The internal authorized request is a discriminated union. Every variant carries
a core-created `AuthorizedSearchContext` containing the index and physical
collection, pinned physical generation, config fingerprint, effective filter,
validated projection, and consistency. That context cannot be constructed by
the public API and binds the store call to the held query lease. Raw caller
vectors must be wrapped with the identity that produced them; this identity is a
trusted assertion, not inferred from dimensions alone:

```python
EmbeddedQueryVector(values, embedding_identity)
VectorQuery(context, vector, candidate_k, top_k)
TextQuery(context, text, candidate_k, top_k)
HybridQuery(context, vector, text, candidate_k, top_k, rrf_k)
FilterQuery(context, top_k, sort)
```

`Sort(field, direction, nulls)` is available only on attributes declared
sortable and only in `FilterQuery` for v1; relevance queries remain ranked by
their retrieval score. Sorting happens inside the store before `top_k`, uses
explicit `nulls_first`/`nulls_last`, and ends with UTF-8 record ID as a stable
tie-break. String sort requires an adapter that guarantees binary UTF-8
collation. `candidate_k >= top_k` is bounded and controls each hybrid leg.

`top_k`, `candidate_k`, timeout, field projection, filter complexity, and query
bytes are bounded by core, profile, and store limits. A request cannot return a
vector or any attribute with a policy filter role. The adapter returns a
portable result, never an SDK response:

```python
class SearchHit(BaseModel):
    id: str
    document_id: str | None
    chunk_id: str | None
    rank: int = Field(ge=1)
    score: float | None
    score_kind: Literal["similarity", "bm25", "rrf", "attribute", "none"]
    raw_score: float | None
    raw_score_kind: Literal["distance", "similarity", "bm25", "rrf"] | None
    component_ranks: Mapping[Literal["vector", "text"], int]
    text: Mapping[str, str]
    attributes: Mapping[str, AttributeValue]
    display: Mapping[str, JsonValue]
    provenance: WarehousePointer


class SearchResult(BaseModel):
    index: str
    mode: Literal["vector", "text", "hybrid", "filter"]
    config_fingerprint: str
    physical_generation: str
    consistency: Literal["strong", "eventual"]
    hits: tuple[SearchHit, ...]
```

When present, portable scores are finite and higher-is-better. Cosine distance
maps to `1 - distance`, Euclidean-squared distance maps to
`-sqrt(max(distance, 0))`, and dot-product similarity is unchanged. BM25 and
RRF use their positive native/core scores. Filter-only results have no score.
`raw_score` is optional and numeric only; provider payloads and explanations are
excluded. Rank is the primary portable ordering, and scores are not comparable
across stores, collections, physical generations, or query modes.

`SearchHit.text` contains only requested `return_text_fields`;
`SearchHit.attributes` contains only requested `returned: true` non-policy
attributes; and `display` contains only additional declared display fields.
Vectors and reserved fingerprints are never returned by the v1 query contract.

Hybrid semantics are reciprocal rank fusion with a default rank constant of
60 and one-based component ranks:

```text
rrf_score(id) = sum(1 / (rrf_k + rank_in_component))
```

Core owns candidate deduplication, that formula, and the UTF-8 ID tie-break.
It calls typed vector and text primitives by default. A store's optional native
hybrid method is used only when `SERVER_SIDE_HYBRID_RRF` promises identical
semantics and component ranks. The identical mandatory prefilter and pinned
snapshot/generation apply to both legs.

## RetrievalStore interface

`RetrievalStore` is registered independently from warehouses and providers.
All envelopes are immutable core types:

```python
class SafeRetrievalTarget(BaseModel):
    store_type: str
    safe_target_identity: str


class StateRetrievalTarget(BaseModel):
    store_type: str
    routing_identity_fingerprint: str
    physical_collection: str


class StoreHealth(BaseModel):
    status: Literal["healthy", "degraded", "unavailable"]
    safe_code: str | None = None


class IndexMetadata(BaseModel):
    kind: Literal["vector", "full_text", "scalar"]
    fields: tuple[str, ...]
    status: Literal["building", "ready", "failed"]
    analyzer: TextAnalyzer | None = None
    options_fingerprint: str | None = None


class ObservedFieldSchema(BaseModel):
    name: str
    role: Literal["id", "text", "attribute", "display", "reserved"]
    data_type: AttributeType | Literal["vector", "json", "unknown"]
    nullable: bool


class ObservedVectorSchema(BaseModel):
    field: str
    dimensions: int
    metric: DistanceMetric | None


class ObservedCollectionSchema(BaseModel):
    fields: tuple[ObservedFieldSchema, ...]
    vector: ObservedVectorSchema | None
    record_contract_version: int | None
    unknown_native_field_count: int


class CollectionMetadata(BaseModel):
    status: Literal["building", "ready", "failed"]
    ownership: Literal["dbt_ml", "external", "unknown"]
    config_fingerprint: str | None
    physical_generation: str
    row_count: int | None
    schema: ObservedCollectionSchema
    indexes: tuple[IndexMetadata, ...]


class ReadinessRequirements(BaseModel):
    config_fingerprint: str
    physical_generation: str
    required_features: frozenset[RetrievalFeature]


class PublicationContext(BaseModel):
    publication_id: str
    fencing_token: int
    expected_base_generation: str | None
    snapshot_fingerprint: str
    config_fingerprint: str
    code_version: str


class UpsertBatch(BaseModel):
    ordinal: int
    rows: tuple[IndexedRow, ...]
    mutation_digest: str


class DeleteItem(BaseModel):
    id: str
    prior_input_fingerprint: str


class DeleteBatch(BaseModel):
    ordinal: int
    items: tuple[DeleteItem, ...]
    mutation_digest: str


class MutationOutcome(BaseModel):
    id: str
    status: Literal["applied", "deleted", "absent", "failed"]
    safe_error_code: str | None = None


class MutationReceipt(BaseModel):
    publication_id: str
    mutation_digest: str
    atomic: bool
    outcomes: tuple[MutationOutcome, ...]


class PublishReceipt(BaseModel):
    publication_id: str
    request_digest: str
    activated_generation: str
    row_count: int


class RetrievalStore(ABC):
    @classmethod
    def store_type(cls) -> str: ...

    @classmethod
    def config_model(cls) -> type[RetrievalStoreConfig]: ...

    @classmethod
    def index_options_model(cls) -> type[BaseModel] | None: ...

    @classmethod
    def capabilities(cls) -> RetrievalCapabilities: ...

    @classmethod
    def implementation_identity(cls) -> str: ...

    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc, traceback) -> None: ...

    def safe_descriptor(self) -> SafeRetrievalTarget: ...
    def state_descriptor(self, collection: str) -> StateRetrievalTarget: ...

    def health(self) -> StoreHealth: ...
    def inspect_collection(self, name: str) -> CollectionMetadata | None: ...
    def create_collection(
        self, spec: CollectionSpec, context: PublicationContext
    ) -> CollectionMetadata: ...
    def evolve_collection(
        self,
        collection: str,
        desired: CollectionSpec,
        context: PublicationContext,
    ) -> CollectionMetadata: ...
    def drop_collection(self, name: str, *, confirmation: DropConfirmation) -> None: ...

    def replace_all(
        self,
        spec: CollectionSpec,
        batches: Iterable[UpsertBatch],
        context: PublicationContext,
    ) -> PublishReceipt: ...

    def upsert(
        self,
        collection: str,
        batch: UpsertBatch,
        context: PublicationContext,
    ) -> MutationReceipt: ...

    def delete(
        self,
        collection: str,
        batch: DeleteBatch,
        context: PublicationContext,
    ) -> MutationReceipt: ...

    def wait_until_ready(
        self,
        collection: str,
        requirements: ReadinessRequirements,
        timeout: timedelta,
    ) -> CollectionMetadata: ...

    def vector_search(self, request: AuthorizedVectorQuery) -> SearchResult: ...
    def text_search(self, request: AuthorizedTextQuery) -> SearchResult: ...
    def filter_search(self, request: AuthorizedFilterQuery) -> SearchResult: ...
    def native_hybrid_search(
        self, request: AuthorizedHybridQuery
    ) -> SearchResult: ...
```

The object is lifecycle-managed and closes local files, clients, streams, and
temporary staging resources on both success and failure. `health()` and schema
inspection return sanitized typed metadata. Core compares the full observed
schema/index descriptor with `CollectionSpec` before classifying a change;
`config_fingerprint` alone is not schema inspection. An existing collection
without the reserved dbt-ml ownership/contract metadata is external or unknown
and is never adopted or mutated implicitly. Unknown native fields may be counted
for fail-closed diagnostics but their names are not logged or artifact-visible.

Keyed upsert means whole-record replacement for an ID, not a partial patch.
Keyed delete is idempotent when an ID is absent. `drop_collection` is never part
of normal reconcile or `dbt-ml clean`; only a separately confirmed destructive
command can construct `DropConfirmation`.

`PublicationContext` is core-owned and is checked against the active publish
lease before every create, evolution, or row-mutation call. Each mutation has a
canonical digest over safe target
and collection identity, publication ID/fencing token, expected base generation,
immutable warehouse snapshot fingerprint, operation/batch ordinal, config/code
version, and the ordered `(record ID, input fingerprint)` items. Delete items
carry the expected prior fingerprint. Core stores each pending
`(publication_id, operation, ordinal, digest)` under the current fence in the
warehouse publication ledger before native I/O and records its receipt there
afterward. Reusing a publication ID and ordinal with a different digest is a
hard error. The adapter independently recomputes the digest and may forward it
as a native idempotency key, but a remote retrieval store is not required to
host dbt-ml's idempotency ledger. Stable whole-record upsert/delete semantics
make a retry safe when the prior outcome is unknown. A full replacement
receipt's `request_digest` covers the pre-I/O context and ordered batch digests.
The store-generated activated generation is returned and validated separately;
it is never part of the retry key.

`MutationReceipt` contains complete, unique internal ID outcomes. An ID is
acknowledged only when the store gives the adapter a durable success guarantee.
Count-only partial acknowledgement is insufficient. A store must provide exact
per-ID durable outcomes or prove `ATOMIC_BATCH_MUTATION` and return an
all-success atomic receipt. Otherwise it raises, core advances no state for the
indeterminate batch, and retry safely replays it.

Retrieval failures use a stable hierarchy:

```text
RetrievalError
├── RetrievalConfigurationError
├── RetrievalCapabilityError
├── RetrievalMutationError
├── RetrievalReadinessError
└── RetrievalQueryError
```

Every error has a core-owned safe code, store type, operation, and retryability.
Native exception messages are suppressed at conversion and never used as
portable diagnostics.

No method accepts or returns Polars/Pandas frames, SDK clients, provider-native
filters, arbitrary option mappings, or raw exceptions.

## Capability model and preflight

Capabilities describe tested guarantees in the pinned integration version,
not features mentioned in provider marketing or an SDK method that has not
been wired into dbt-ml.

```python
class RetrievalFeature(StrEnum):
    EXACT_VECTOR_SEARCH = "exact_vector_search"
    APPROXIMATE_VECTOR_SEARCH = "approximate_vector_search"
    METADATA_FILTERING = "metadata_filtering"
    MANDATORY_PREFILTER = "mandatory_prefilter"
    FILTER_ONLY_SEARCH = "filter_only_search"
    ATTRIBUTE_SORT = "attribute_sort"
    FULL_TEXT_BM25 = "full_text_bm25"
    SERVER_SIDE_HYBRID_RRF = "server_side_hybrid_rrf"
    LOGICAL_NAMESPACES = "logical_namespaces"
    ENFORCED_TENANT_ISOLATION = "enforced_tenant_isolation"
    ATOMIC_FULL_REPLACE = "atomic_full_replace"
    KEYED_UPSERT = "keyed_upsert"
    KEYED_DELETE = "keyed_delete"
    ONLINE_SCHEMA_EVOLUTION = "online_schema_evolution"
    INDEX_READINESS = "index_readiness"
    OBSERVABLE_READ_GENERATION = "observable_read_generation"
    PINNED_GENERATION_READS = "pinned_generation_reads"
    DURABLE_WRITE_ACK = "durable_write_ack"
    EXACT_MUTATION_RECEIPTS = "exact_mutation_receipts"
    ATOMIC_BATCH_MUTATION = "atomic_batch_mutation"


class RetrievalCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    features: frozenset[RetrievalFeature]
    leaf_filter_support: Mapping[LeafFilterOperator, frozenset[AttributeType]]
    boolean_filter_operators: frozenset[BooleanFilterOperator]
    sort_types: frozenset[AttributeType]
    full_text_analyzers: frozenset[TextAnalyzer]
    consistency_modes: frozenset[ConsistencyMode]
    distance_metrics: frozenset[DistanceMetric]
    max_id_bytes: int | None
    max_dimensions: int | None
    max_batch_rows: int | None
    max_batch_bytes: int | None
    max_row_bytes: int | None
    max_text_field_bytes: int | None
    max_display_field_bytes: int | None
    max_query_bytes: int | None
    max_top_k: int
    max_candidate_k: int
    max_projection_fields: int | None
    max_filter_bytes: int
    max_filter_depth: int
    max_filter_nodes: int
    max_membership_values: int
```

`MANDATORY_PREFILTER` means the adapter can enforce the effective filter inside
the store before ranking. It is required for every governed index and is
stronger than generic `METADATA_FILTERING`; a client/post-filter feature never
satisfies it.

`LOGICAL_NAMESPACES` means a store can address isolated collection groups.
`ENFORCED_TENANT_ISOLATION` is stronger: the adapter must prove that a caller
credential/routing context cannot access another tenant's group. A store never
receives the stronger capability merely because its API calls something a
namespace. v0.2 authorization continues to require mandatory row prefilters.

`SearchResult.physical_generation` is an observed result-generation identity,
not a copy of the expected lease value. Strong reads are safe while the
exclusive/shared coordination contract prevents publication. Eventual reads
additionally require either `PINNED_GENERATION_READS`, which binds the native
query to the leased generation, or `OBSERVABLE_READ_GENERATION`, which lets core
reject a response from any other generation. For core-composed hybrid search,
both legs must prove the same leased generation; a mismatch fails/retries the
whole request and never exposes partial hits. An adapter that cannot make either
guarantee rejects eventual mode instead of labeling an unverified result.

Compile resolves registries, strict profile/model config, static capabilities,
and collisions without opening a store or reading credentials. `compile`
checks every resource; `run`, `build`, and `test` check the selected workload.
Missing requirements are reported together with the resource, store, and
reason. Runtime rechecks capabilities after creating the adapter so a direct
library call cannot bypass preflight.

The requirements include:

- incremental publication: keyed upsert, keyed delete, durable acknowledgement,
  exact mutation receipts or atomic batch mutation, plus mandatory prefilter
  for governed fields;
- full materialization or `--full-refresh`: atomic full replacement;
- configured exact/approximate vector mode and distance metric;
- text mode: BM25 over all declared fields with the exact normalized analyzer;
- hybrid mode: vector and text features; server-side RRF is optional and used
  only when its semantics match the core contract;
- filter mode and sort requests: their exact features and supported sort types;
- every configured filter operator/type and policy prefilter;
- requested strong/eventual consistency;
- observable or natively pinned generation guarantees for eventual reads and
  both hybrid legs;
- configured dimensions, ID size, batch/query/result limits, and filter
  complexity/serialized size; and
- `on_index_change: online`: online evolution for the exact classified change.

No requirement silently downgrades to another search mode, consistency level,
filter placement, schema-change behavior, or replacement strategy.

## Publication state and lifecycle

### Warehouse-owned state

Issue #139 defines record-scoped state in the active warehouse. Search
publication uses it as follows:

```text
StateScope.model_name      = search resource name
StateScope.stage           = "retrieval_publish"
StateScope.target_identity = fingerprint(store routing identity + physical collection)
StateRecord.record_key     = IndexedRow.id
StateRecord.input_fingerprint = serving-relevant row fingerprint
StateRecord.code_version   = search configuration/implementation version
```

The target descriptor is a positive-allowlisted, non-secret physical identity:
store type, safe connection identity, resolved namespace, and physical
collection. Semantic schema/index configuration does not go in target identity;
it belongs in code/config version. Otherwise a metric or embedding change would
create a fresh state scope and lose the IDs needed to reconcile the current
collection.

Changing the physical store or collection intentionally creates a new scope.
dbt-ml does not automatically delete the old collection or state because it
may still serve another deployment. Remote cleanup must be an explicit,
destructive command in a later issue; `dbt-ml clean` remains local-artifact
cleanup only.

In addition to per-record state, the warehouse owns a small publication ledger
for each scope:

```text
generation, owner/publication ID, status, expected code/config version,
started_at, completed_at, safe counts, last safe error code, plus fenced child
records for mutation operation/ordinal/digest and receipt status
```

The status is `publishing`, `ready`, or `failed`. This ledger is readiness
metadata, not a lock. Issue #139 does not provide it or serving coordination;
implementation must add the following separate warehouse-owned contract:

```python
class PublishLease(BaseModel):
    scope: StateScope
    publication_id: str
    fencing_token: int


class QueryLease(BaseModel):
    scope: StateScope
    physical_generation: str
    config_fingerprint: str
    fencing_token: int


acquire_publish(scope, publication_id) -> exclusive session-owned PublishLease
acquire_query(scope) -> shared session-owned QueryLease for a ready generation
mark_ready(lease, generation, config, counts) -> None
mark_failed(lease, safe_error_code, counts) -> None
```

Publish acquisition is an atomic compare-and-set that increments a monotonic
fencing token. It waits for or rejects active query leases; query acquisition
rejects an active publisher. A query holds its shared lease until after native
results and generation metadata are validated. Every ledger/state transition
and mutation digest verifies the current token. There is no automatic
time-based lease stealing: a paused publisher must not resume after a newer
one. Locks are released by the owning warehouse session/transaction. Explicit
administrative recovery requires the operator to terminate the old owner first.

The LanceDB proof of concept can implement session ownership with an enforced
local reader/writer file lock. A warehouse fencing token alone cannot prevent a
partitioned process from continuing to call an independent remote SDK. A
distributed warehouse/store pair must therefore implement provider-enforced
fencing or write only to an immutable per-publication generation followed by
conditional atomic activation. Losing the publish session aborts before any
further store I/O; administrative recovery terminates the old process as well
as its warehouse session. #152 owns this follow-on contract. v0.2 does not claim
fault-tolerant multi-writer publication merely because a ledger exists.

### Initial and incremental publication

The runner sequence is:

1. Resolve the warehouse and selected logical retrieval target, validate the
   credential reference, and construct a protected credential without revealing
   it to core.
2. Validate DAG/config/capabilities and reject physical collection collisions.
3. Open one immutable `WarehouseReadSnapshot`. Validate field mappings/schema
   and run a typed null/uniqueness key-domain check before any store mutation.
4. Acquire the exclusive publication lease and mark the scope `publishing`.
5. Reveal the credential only at SDK construction, open the retrieval store,
   inspect collection metadata, and classify any
   config/schema change.
6. Diff paged warehouse state against projected upstream batches. Validate all
   row values in a batch before mutating that batch.
7. Mutate the store first using a digest containing the snapshot/generation and
   input fingerprints. Buffer state changes only for IDs in an exact durable
   receipt.
8. Delete stale store IDs and buffer their state deletions after acknowledgement.
9. Wait for every required index to be ready and validate safe schema/count
   invariants. Verify the upstream materialization generation has not changed.
10. Apply the buffered warehouse state changes, mark the publication ledger
    `ready` with the physical generation, release resources, and return counts.

A no-op run still verifies the physical collection metadata and ready ledger;
state alone cannot prove that a collection exists.

New IDs are upserted. Changed IDs include any change to vector, text, display,
provenance, filter/policy attributes, safe identity, or search code/config
version. Unchanged IDs are skipped. IDs present in state but absent upstream are
deleted remotely before their state rows are deleted.

For a governed collection, changed existing IDs are deleted before their new
whole record is upserted. If a policy revocation publish fails, the old more
permissive row is therefore absent rather than left queryable. The scope remains
non-ready until the full reconcile succeeds. Direct access to the backing store
bypasses this readiness policy and is outside dbt-ml's authorization boundary;
production credentials should restrict such access to the service.

### Config and schema changes

Changes are classified before mutation:

| Change | Behavior |
| --- | --- |
| row vector/text/display/provenance/attribute/policy value | keyed republish for that ID |
| ID mapping, vector field/dimensions/metric/embedding identity | whole-index invalidation |
| text fields/analyzer, attribute type/role, display projection | whole-index invalidation |
| store contract or semantic implementation version | whole-index invalidation |
| adapter-declared compatible additive index change | allowed only with `on_index_change: online` and the exact capability |
| timeout, retry, batch size, logging | no semantic invalidation |
| physical target or collection | new state scope and independent publication |

`on_index_change: fail` is the default and leaves the previous collection
untouched. If atomic activation is unavailable, the recovery guidance is to
publish under a new collection name/target, validate it, and deliberately cut
consumers over; it must not recommend an impossible `--full-refresh`. `rebuild`
performs an atomic full replacement only when advertised. `online` is allowed
only for an adapter-classified compatible change; a broad
`ONLINE_SCHEMA_EVOLUTION` flag cannot make an incompatible dimension or type
change safe.

### Full replacement

Atomic replacement has these semantics:

1. Build a new private generation that cannot receive production queries.
2. Validate schema, count, required indexes, policy fields, and readiness.
3. Atomically activate the new generation under the logical collection.
4. Atomically replace the warehouse state snapshot.
5. Mark the publication ledger ready, then retire the previous generation
   according to adapter retention policy.

Step 4 requires an independently advertised warehouse capability for an atomic,
fence-checked replacement of every state row in a scope. Per-record CRUD from
#139 is insufficient. #153 owns that operation together with ordered/paged state
reconciliation. Full replacement is rejected unless both the retrieval store's
atomic activation and the warehouse's atomic state-snapshot replacement are
available.

If a store cannot prove atomic activation, compile rejects full materialization,
`--full-refresh`, and `on_index_change: rebuild`. It does not drop/recreate,
temporarily empty a collection, or expose a half-built generation. Initial
incremental publication may create a missing collection because no prior
serving generation exists.

### Failure, recovery, and idempotency

Every mutation is stable-ID idempotent and publication is at-least-once:

- Store failure before a durable receipt advances no state for indeterminate
  IDs. A retry replays the batch.
- An exact partial receipt advances state only for the acknowledged IDs, records
  safe counts, marks the node/run failed, and retries the remainder next time.
- Store success followed by state failure leaves remote rows ahead of state.
  The next run repeats the same whole-record mutation and then advances state.
- Delete success followed by state failure repeats an absent-ID delete safely.
- Full activation success followed by state failure repeats an atomic rebuild;
  it never assumes state advanced.
- Index-readiness timeout after writes leaves the scope failed/non-ready. A
  retry inspects metadata, republishes stale state as necessary, and waits again.
- A warehouse snapshot/generation change prevents `ready`; the next run reads a
  new immutable snapshot and reconciles it.
- A search publication failure does not roll back its upstream warehouse model,
  but it fails the overall invocation and blocks retrieval tests and any
  downstream serving readiness.

Run results preserve inserted, updated, skipped, deleted, failed, and total
counts known from exact receipts. `total = inserted + updated + skipped + failed`
for the current upstream snapshot; `deleted` counts stale prior-generation IDs
and is outside `total`. Indeterminate terms remain null rather than guessed from
a provider response.

## Compiler, runner, test, and export integration

### Compiler and runner

Current warehouse capability preflight assumes every executable node writes a
relation. Implementation branches by DAG resource type:

- Warehouse models retain warehouse materialization capability checks.
- A search index requires the warehouse's typed read capability for its one
  upstream relation and retrieval-store capabilities for its sink.
- The #140 warehouse contract provides immutable snapshot reads, projected
  typed batches, predicate pushdown, and a typed key-domain check. #153 provides
  ordered/paged state iteration, bounded
  state/upstream diffing, and atomic scope-state replacement; the current #139
  `fetch_state()` full dictionary is not sufficient.
- The runner lifecycle-manages one warehouse and only the selected retrieval
  stores. It does not put store-specific branches in the general orchestration
  path.
- A search node produces a structured publication result even on partial
  failure so run artifacts can report safe status/counts.

The implementation touchpoints are `config/model.py`, `config/profile.py`,
`profile.py`, a new `retrieval/` registry and contract package, `dag.py`,
`compiler.py`, `runner.py`, `versioning.py`, `manifest.py`, `docs.py`,
`dbt_export.py`, test routing, and relevant CLI resource dispatch.

### Build and tests

`build` preserves topological semantics:

```text
upstream warehouse run -> upstream warehouse tests -> index publish -> retrieval tests
```

Current SQL schema tests are never issued against a search index. Attaching a
warehouse-only test to a search resource is a compile error, not a silent skip.
Search-resource tests route through a retrieval-test registry and portable query
contract. Contract validation of IDs, vector shape, fields, and types occurs
before publication and is not implemented as SQL against the serving store.

The #134 grammar is fixed to three search-only tests:

```yaml
tests:
  - index_ready
  - row_count: {min: 1, max: 1000000}
  - query_smoke:
      mode: text
      query: unemployment outlook
      top_k: 5
      min_results: 1
      principal: economic_data_smoke
```

`index_ready` validates the pinned serving generation and required indexes.
`row_count` compares safe store metadata with optional inclusive bounds.
`query_smoke` uses the portable request/result path and accepts only a declared
mode, bounded text or identity-bearing vector fixture, typed user filter,
`top_k`, `min_results`, and a profile-owned principal alias. `principal` is
required for a governed smoke test and forbidden for a public one; the active
profile maps the alias to the trusted policy resolver, while project YAML cannot
embed caller claims. Query literals are project data but do not enter run
artifacts/logs. Deterministic ranking evaluation, labeled relevance sets, and
metrics belong to #137. Custom Python retrieval tests are a reserved typed
capability after the portable client stabilizes, not a #134 requirement.

Manifest v2 must not reuse the current warehouse model's wholesale `tests`
serialization for search resources. It emits only a safe projection of each
retrieval test: test type, mode, numeric bounds, and status-independent schema.
Query text/vector fixtures, principal aliases, user/policy filter literals, and
native options are runtime-only and omitted from manifest/docs/run results.
Run results identify the test by resource-qualified ID and safe type/mode, then
record status and a core-owned safe error code without echoing the fixture.

`test --select <index>` runs only retrieval tests against a ready publication.
A failed/non-ready publication fails closed. Failure skips dependent retrieval
tests, while already completed upstream warehouse results remain visible.

### dbt export

`emit-dbt-sources` never emits a search index or its retrieval tests as a dbt
relation. After resolving any selector form (exact name, tag, state, or graph),
export projects every selected search index to its one canonical direct
upstream warehouse model, deduplicates with already selected warehouse models,
then applies explicit warehouse exclusions. An exclusion therefore wins. The
remaining warehouse resources emit in DAG order.

Dagster/dbt relation metadata applies only to warehouse outputs. A serving sink
is a separate asset/output descriptor and must not receive a fabricated dbt
source or relation key.

## Artifact contract and migration

### Manifest v2

Adding a non-relation resource requires `manifest_version: 2`. The existing
`models` list remains the ordered list of model-file DAG resources for a small
migration surface, but every entry gains an explicit discriminator and output.
v2 moves resolved target data out of v1's `project` object into a canonical
top-level `target` sibling of `project`, `models`, and `sources`; the v2
`project` object contains project name and version only:

```json
{
  "target": {
    "profile": "economic_data",
    "name": "dev",
    "warehouse": {
      "adapter_type": "duckdb",
      "safe_target_identity": "...",
      "catalog": "economic_data",
      "schema": "economic_data"
    },
    "retrieval": [
      {
        "alias": "primary",
        "store_type": "lancedb",
        "safe_target_identity": "..."
      }
    ]
  }
}
```

The retrieval list contains only aliases actually referenced by compiled
resources. A warehouse model entry gains a newly defined v2 output descriptor:

```json
{
  "name": "economic_context_embeddings",
  "unique_id": "model.economic_data.economic_context_embeddings",
  "resource_type": "model",
  "output": {
    "type": "warehouse_relation",
    "relation": {
      "catalog": "economic_data",
      "schema": "economic_data",
      "name": "economic_context_embeddings",
      "fully_qualified": "economic_data.economic_data.economic_context_embeddings"
    }
  }
}
```

Manifest v1 did not contain a per-model relation; this is an explicit new v2
field rather than a claim that such a descriptor already exists. A search
entry uses the other discriminator:

```json
{
  "name": "economic_context_search",
  "unique_id": "search_index.economic_data.economic_context_search",
  "resource_type": "search_index",
  "kind": "search",
  "access": "governed",
  "depends_on": ["model.economic_data.economic_context_embeddings"],
  "code_version": "...",
  "output": {
    "type": "serving_resource",
    "serving_resource": {
      "kind": "retrieval_index",
      "store_type": "lancedb",
      "store_implementation": "...",
      "safe_target_identity": "...",
      "logical_collection": "economic_context",
      "physical_collection": "economic_data__dev__economic_context",
      "scope_fingerprint": "...",
      "materialization": "incremental",
      "schema_version": 1,
      "config_fingerprint": "...",
      "text_fields": ["text"],
      "return_text_fields": ["text"],
      "full_text": {
        "fields": ["text"],
        "analyzer": {"kind": "store_default"}
      },
      "vector": {
        "field": "embedding",
        "dimensions": 384,
        "metric": "cosine",
        "search": "approximate",
        "embedding": {
          "provider": "local",
          "model": "bge-small-en-v1.5",
          "provider_contract_version": 1,
          "provider_implementation": "...",
          "semantic_config_fingerprint": "...",
          "dimensions": 384
        }
      },
      "attributes": [
        {
          "name": "series_id",
          "data_type": "string",
          "filter_role": "user",
          "sortable": true,
          "returned": true
        },
        {
          "name": "tenant_id",
          "data_type": "string",
          "filter_role": "policy",
          "sortable": false,
          "returned": false
        },
        {
          "name": "access_groups",
          "data_type": "array[string]",
          "filter_role": "policy",
          "sortable": false,
          "returned": false
        }
      ],
      "display_fields": ["title", "source_uri", "page"],
      "query": {
        "modes": ["vector", "text", "hybrid"],
        "consistency": "strong"
      },
      "capabilities": {
        "required": [
          "approximate_vector_search",
          "metadata_filtering",
          "mandatory_prefilter",
          "full_text_bm25",
          "keyed_upsert",
          "keyed_delete",
          "index_readiness",
          "durable_write_ack",
          "atomic_batch_mutation"
        ],
        "available": [
          "exact_vector_search",
          "approximate_vector_search",
          "metadata_filtering",
          "mandatory_prefilter",
          "full_text_bm25",
          "keyed_upsert",
          "keyed_delete",
          "index_readiness",
          "observable_read_generation",
          "durable_write_ack",
          "atomic_batch_mutation"
        ],
        "consistency_modes": ["strong", "eventual"],
        "distance_metrics": ["cosine", "euclidean", "dot"]
      },
      "upstream": "model.economic_data.economic_context_embeddings"
    }
  }
}
```

Search entries have no `relation` key. `config_fingerprint` is the expected
semantic index specification, not evidence that a physical generation exists.
Static manifests do not claim an active generation, last-run counts, readiness,
or publish success.

`capabilities.required` is the concrete requirement set after resolving
alternatives such as exact receipts versus atomic batch mutation;
`capabilities.available` is the adapter's complete tested feature set for its
pinned implementation. The artifact also emits the safe filter/type matrix,
analyzer support, consistency/metric sets, and finite limits from
`RetrievalCapabilities`; they are abbreviated above only to keep the example
readable. An artifact never lists an untested native feature as available.

Manifest v2 dependency values and DAG node/edge identifiers use canonical
project-qualified `unique_id` values. They do not preserve raw `ref()` syntax or
rely on a globally meaningful bare model name. Human-facing selectors continue
to use resource names.

The top-level target block is generic and contains allowlisted warehouse and
referenced retrieval descriptors. V1's `project.profile`, `project.target`,
`project.duckdb_path`, and `project.duckdb_schema` do not appear in the
canonical v2 writer; the v1 compatibility reader maps them into the new shape.
Raw file paths, endpoints, organization/project identifiers not explicitly
classified safe, raw collection templates/prefix settings, credentials,
credential-variable names, content, vectors, and filter/policy values are
excluded. The resolved physical collection is intentionally visible and
validated safe.

`state:modified` compares code versions for both resource types. For a search
index it also compares `scope_fingerprint`, the non-reversible fingerprint of
the safe store-routing identity and physical collection. A target/template
change is therefore modified even though routing is correctly excluded from
semantic `config_fingerprint`. A v1 reader knows only warehouse models. A v2
reader must branch on `resource_type` and `output.type`; readers reject unknown
manifest versions instead of guessing.

The in-process v1 compatibility adapter classifies every v1 `models[]` entry as
`resource_type: model`, derives its warehouse relation from the v1 resolved
target fields plus model name (catalog may be null because v1 did not record a
generic catalog), and supplies no retrieval descriptors. It never
reclassifies a v1 row as a search index. Artifact writers emit only v2 after the
implementation lands; external v1-only consumers must continue using a v1
artifact or upgrade rather than silently parsing v2.

### Run results v2

Run results add top-level `run_results_version: 2` and use the same output
discriminator. Every result row has `name`, project-qualified `unique_id`,
`resource_type`, `status`, and `output`; warehouse rows place their existing
relation descriptor under `output.type: warehouse_relation`. A successful or
failed search result contains:

```json
{
  "name": "economic_context_search",
  "unique_id": "search_index.economic_data.economic_context_search",
  "resource_type": "search_index",
  "status": "success",
  "output": {
    "type": "serving_resource",
    "serving_resource": {
      "store_type": "lancedb",
      "safe_target_identity": "...",
      "logical_collection": "economic_context",
      "physical_collection": "economic_data__dev__economic_context",
      "scope_fingerprint": "..."
    }
  },
  "publish": {
    "mode": "incremental",
    "status": "ready",
    "inserted": 10,
    "updated": 2,
    "skipped": 980,
    "deleted": 3,
    "failed": 0,
    "total": 992,
    "schema_version": 1,
    "config_fingerprint": "...",
    "physical_generation": "42",
    "ready": true
  }
}
```

`total` follows the count invariant defined above and excludes `deleted`.
Unknown/indeterminate terms are `null`, not zero. `status: success` requires
`publish.status: ready` and `ready: true`; a partial receipt produces resource
`status: error`, publish status `failed`, and `ready: false` even when some
counts are known. Native request IDs, errors, responses, and physical endpoints
are absent. Safe correlation belongs in provider/store logs, not portable
artifacts.

For migration, absence of `run_results_version` means the existing v1 shape.
The v1 adapter maps `model_name` to `name`, assigns `resource_type: model`, and
wraps the existing `relation` as a warehouse output. It populates a canonical
`unique_id` only when the companion manifest or explicit project context is
available; its internal compatibility type otherwise permits `unique_id: null`
instead of fabricating a project name. New v2 artifacts always require a
non-null project-qualified ID. Writers emit only v2 after the implementation
lands, and readers reject unknown explicit versions. The one-release
compatibility layer may retain `model_name` as a deprecated alias for warehouse
rows, but new consumers use `name` and `unique_id`; search rows never pretend to
have a model relation.

Docs read the v2 discriminator and render separate warehouse-model and search-
index sections. A search page shows upstream lineage, safe serving descriptor,
schema, embedding/index versions, capabilities, and the last run's safe publish
status/counts. It never offers SQL or labels the index as a DuckDB/BigQuery
relation. Existing warehouse pages and v1 manifests continue to render through
the explicit version adapter.

## Proving-store mappings

This mapping was checked against official documentation on 2026-07-15. It is
implementation guidance, not a declaration of shipped capabilities. Each
adapter must pin a supported SDK range and advertise only behavior covered by
contract tests.

| Portable contract | LanceDB mapping | turbopuffer mapping |
| --- | --- | --- |
| collection | Lance table within the configured database/catalog | namespace |
| inspect/create | table schema and index metadata; explicit Arrow schema | namespace metadata and explicit schema on write |
| keyed upsert | `merge_insert(id)` with matched update + unmatched insert | whole-document upsert by `id` |
| keyed delete | typed ID predicate translated by the adapter to `delete` | `deletes` by document ID |
| exact vector | flat/kNN search | filtered `kNN` only; do not advertise general exact search |
| approximate vector | Lance vector index/search | ANN/SPFresh |
| distance metrics | `cosine`, `l2`, and `dot` when supported by chosen index | `cosine_distance` and `euclidean_squared`; dot is unsupported |
| mandatory prefilter | translated typed AST with `prefilter=True`; scalar/list indexes as configured | native typed filters evaluated with vector/text queries |
| full text | native FTS/BM25 indexes over declared string fields | schema `full_text_search` plus BM25 ranking |
| optional native hybrid | use only when Lance RRF matches the core formula/component-rank contract; otherwise core calls both primitives | multi-query plus `rerank_by=RRF` only when compatible; otherwise core calls both primitives |
| strong reads | connection configured to refresh every read; local adapter verifies read-after-write | strong consistency request, the documented default |
| eventual reads | provider mode exists, but advertise only after the adapter proves an observed/pinned table version | provider mode exists, but advertise only after the adapter proves an observed/pinned namespace generation |
| schema/index evolution | provider supports several operations, but only adapter-classified compatible changes may advertise online behavior | limited index-attribute changes are online; type/most schema changes require rebuild |
| atomic full replace | not advertised until a tested atomic activation primitive exists | not advertised without a tested atomic namespace activation primitive |
| readiness | index status/listing and a bounded wait | metadata plus handling for indexes that return not-ready/202 |

Relevant official references:

- LanceDB: [updates and merge insert](https://docs.lancedb.com/tables/update),
  [metadata filtering](https://docs.lancedb.com/search/filtering),
  [full-text search](https://docs.lancedb.com/search/full-text-search),
  [hybrid search](https://docs.lancedb.com/search/hybrid-search),
  [indexing](https://docs.lancedb.com/indexing),
  [schema evolution](https://docs.lancedb.com/tables/schema), and
  [consistency](https://docs.lancedb.com/tables/consistency).
- turbopuffer: [writes, schema, and mutation](https://turbopuffer.com/docs/write),
  [vector/text/hybrid queries, filters, RRF, and consistency](https://turbopuffer.com/docs/query),
  and [namespace metadata](https://turbopuffer.com/docs/metadata).

LanceDB SQL-like predicates remain an adapter implementation detail. Core emits
the typed AST and the adapter quotes/parameterizes or otherwise translates it;
project/query callers never supply a predicate string. LanceDB's bad-vector
drop/fill/null modes are forbidden by the canonical row contract.

turbopuffer string IDs currently have a 64-byte limit, uses one distance metric
for a namespace's vector fields, and does not support dot-product distance in
the baseline mapping. Its native embedding feature is not used by this
contract: dbt-ml publishes explicit, warehouse-recorded vectors so index- and
query-time embedding identity remains governed outside the store.

Logical namespaces and collections help isolation and cost management, but
neither proving adapter is treated as dbt-ml's identity provider. Authorization
still requires profile-controlled routing and mandatory prefilters.

## Migration for existing vector-store issues

Issues #28-#35 should be re-estimated as `RetrievalStore` integrations rather
than `WarehouseAdapter` implementations:

| Issue | Store | Migration rule |
| --- | --- | --- |
| #28 | Pinecone | implement only proven vector/filter/lifecycle capabilities; no SQL emulation |
| #29 | Qdrant | map collections, payload filters, keyed mutation, and supported query modes |
| #30 | Weaviate | keep schema/client-native objects inside the adapter; disable unproven modes |
| #31 | Milvus/Zilliz | map collection/index readiness and exact capability limits explicitly |
| #32 | Postgres/pgvector | implement a separate retrieval role even when it shares a physical database with a warehouse |
| #33 | Chroma | advertise the minimal tested local feature set; reject unsupported production guarantees |
| #34 | OpenSearch/Elasticsearch | map vector/BM25/hybrid semantics without exposing Query DSL to core |
| #35 | Redis vector search | map index lifecycle and typed filters without exposing Redis commands to core |

Every integration must provide:

1. a strict profile config, strict semantic index-options model, safe/state
   descriptors, implementation identity, and sanitized errors;
2. an exact capability/limit descriptor and compile-time rejection tests;
3. the common canonical row, mutation receipt, filter, search result, policy
   prefilter, state/failure, cleanup, and optional-dependency contract tests;
4. provider-specific schema/query translation tests using fakes plus
   credential-gated integration smoke tests; and
5. documentation of unsupported guarantees and non-atomic behavior.

A vector-only implementation is valid. It rejects resource configs requesting
text or hybrid modes. No adapter has to emulate SQL, a warehouse relation,
another provider's schema, or arbitrary portable functionality it cannot
guarantee.

## Implementation and acceptance map

Issue #134 delivered the core contract, artifacts, incremental publication,
optional LanceDB adapter, integration tests, and local example as one bounded
proof-of-concept slice. The remaining production-serving guarantees stay in
their owning follow-ups:

- serving/publication coordination in #152, split from adapter delivery because
  #139 does not provide the ledger or query/publish lease contract.
- #140 adds immutable projected snapshots and key-domain validation; #153 adds
  bounded state reconciliation and atomic scope-state replacement before
  production-scale publication is claimed.
- #135 builds the portable query service, Python API, and `dbt-ml search` over
  the authorization factory, read-lease, request/result/filter, score, and
  core-RRF contracts here.
- #136 implements turbopuffer without changing resource grammar or core request
  types.
- #154 supplies the protected profile-credential representation before #136 or
  any other hosted retrieval adapter accepts secrets.
- #137 adds deterministic golden-query evaluation through the portable query
  surface.
- #145 specializes provenance, entity/time, permissions, and citation fields;
  it may add typed fields but must preserve mandatory prefilter separation.
- #28-#35 conform to this contract and common adapter test suite.

Online schema evolution, enforced tenant isolation, optional string operators,
and custom Python retrieval tests remain typed reserved capabilities. The
implemented LanceDB proof of concept rejects those requests.

The bounded #134 slice covers:

- strict config, one-upstream/leaf rules, collection collisions, and selectors;
- capability aggregation without opening a store;
- missing/null/duplicate/overlong IDs, non-finite/wrong-dimension vectors, and
  invalid text/attribute/display values before mutation;
- typed scalar predicate translation with no raw predicate-string API;
- initial, no-op, changed-row, deleted-row, config-change, empty-input,
  failed-mutation, and idempotent retry behavior;
- stable mutation digests and exact atomic-batch receipt rejection;
- build/test routing with no SQL test against a serving store;
- manifest/run-results v1 migration, v2 discriminators, no fabricated relation,
  safe docs, and credential/content/filter/vector redaction;
- dbt export's all-selector upstream projection, deduplication, and exclusion.

#152 owns shared/exclusive leases, governed ACL delete-before-upsert,
concurrent-publisher, abandoned-session, snapshot-generation, durable readiness,
and readiness-race tests. #153 owns paged state reconciliation and atomic full
state replacement. #135 owns the complete filter AST, public/governed query
authorization, identity-bearing query vectors, score normalization, cache
isolation, and deterministic RRF/tie-breaking tests.

## Rejected alternatives

### `search_index` warehouse materialization

Rejected because the sink has no warehouse relation, uses different
capabilities and tests, and can fail independently after the canonical model is
successfully materialized. Treating it as a materialization would recreate the
SQL-shaped adapter problem resolved in #70.

### Extending `VectorStore`

Rejected as the public role name because LanceDB, turbopuffer, OpenSearch, and
similar targets can serve BM25, typed filters, and hybrid retrieval without a
vector in every collection. `RetrievalStore` names the broader role while
vector search remains an optional capability.

### Provider-native configuration and filters in core

Rejected because they make model grammar non-portable, bypass field/policy
validation, leak SDK types into artifacts/tests, and prevent compile-time
capability checks. Adapter-owned strict options are the only extension point.

### Client-side authorization filtering

Rejected because it can expose unauthorized rows to the process, breaks top-k
semantics, creates cache/timing footguns, and cannot safely cover both hybrid
legs. Mandatory policy is always an in-store prefilter.

### Non-atomic drop-and-recreate full refresh

Rejected because it can expose an empty, partial, or stale-policy collection.
Stores without atomic activation support incremental reconcile only until they
can prove a safe replacement primitive.
