# dbt_ref_roundtrip — the bidirectional dbt ↔ dbt-ml round trip (#177)

[`examples/dbt_embed_duckdb`](../dbt_embed_duckdb) proves the forward direction:
dbt-ml runs inside a `dbt build` and feeds a downstream dbt SQL model. This
example closes the loop — a dbt-ml transform reads a **dbt-built** table back,
via `source: dbt_ref('<dbt_model>')`, and feeds a *further* dbt SQL model, all
in one `dbt build`:

```
models/dbt_ml/raw_vendors.py          ← dbt-ml extraction (no upstream)
models/marts/invoice_facts.sql        ← dbt SQL: select ... from ref('raw_vendors')
models/dbt_ml/flagged_invoices.py     ← dbt-ml transform: source: dbt_ref('invoice_facts')
models/marts/flagged_invoice_summary.sql ← dbt SQL: select ... from ref('flagged_invoices')
```

One `dbt build` runs all four levels in dependency order — **dbt-ml → dbt →
dbt-ml → dbt** — as one lineage graph. `flagged_invoices` never enters dbt-ml's
own graph for `invoice_facts` (dbt resolves that `ref()`, not dbt-ml); dbt-ml
only sees it arrive as an injected upstream frame, exactly like a `depends_on:`
dependency.

The dbt-ml project (`../dbt_ref_roundtrip`, the source of truth) declares
`flagged_invoices` as:

```yaml
# ../dbt_ref_roundtrip/models/flagged_invoices.yml
models:
  - name: flagged_invoices
    source: dbt_ref('invoice_facts')   # reads a dbt-built table
    transform: { type: python, module: transforms.flag_high_value }
```

The `models/dbt_ml/*.py` and `schema.yml` files here are **generated** by
`dbt-ml codegen` from that project; they're checked in so this example runs
without the codegen step.

## Run it

From `examples/dbt_ref_roundtrip_dbt/`:

```bash
# Seed the tiny source corpus into the colocated dbt-ml project (once) — three
# one-line vendor-spend documents, small enough to write directly rather than
# through a generator
mkdir -p ../dbt_ref_roundtrip/data/vendors
cat > ../dbt_ref_roundtrip/data/vendors/acme.json <<'EOF'
{"vendor": "Acme", "spend": 1200.0}
EOF
cat > ../dbt_ref_roundtrip/data/vendors/globex.json <<'EOF'
{"vendor": "Globex", "spend": 450.0}
EOF
cat > ../dbt_ref_roundtrip/data/vendors/initech.json <<'EOF'
{"vendor": "Initech", "spend": 780.0}
EOF

# (Re)generate both dbt-ml nodes (raw_vendors + flagged_invoices) from the
# dbt-ml project's YAML
uv run --project ../.. dbt-ml --project-dir ../dbt_ref_roundtrip \
  codegen --output models/dbt_ml

# Install this dbt project's env (dbt-duckdb + editable dbt-ml) and build
uv sync
DBT_ML_PROJECT_DIR="$(cd ../dbt_ref_roundtrip && pwd)" uv run dbt build --profiles-dir .
```

Expected: `PASS=11` — the extraction model, the reverse-direction transform,
seven schema tests, and both SQL marts all succeed, materializing
`raw_vendors`, `invoice_facts`, `flagged_invoices`, and
`flagged_invoice_summary` into one `target/embedded.duckdb`.

## Caveat: one dbt-ml project per `dbt build`

Every generated shim resolves its dbt-ml project via a single shared
`DBT_ML_PROJECT_DIR` environment variable. That is why this example is
self-contained — both `raw_vendors` and `flagged_invoices` live in the *same*
dbt-ml project (`../dbt_ref_roundtrip`) rather than reusing
[`../invoice_pipeline`](../invoice_pipeline) (the source for
`dbt_embed_duckdb`). Embedding models from two different dbt-ml projects in one
`dbt build` isn't supported yet — that would need codegen to bake in a
per-model project path instead of reading one shared environment variable.

See [`examples/dbt_embed_duckdb`](../dbt_embed_duckdb) for the forward-only
three-level DAG, and the "Composing with dbt → Embedded" section of the
[main README](../../README.md) for the full write-up.
