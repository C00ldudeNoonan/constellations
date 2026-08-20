# dbt_ref_roundtrip — the bidirectional dbt ↔ stel round trip (#177)

[`examples/dbt_embed_duckdb`](../dbt_embed_duckdb) proves the forward direction:
stel runs inside a `dbt build` and feeds a downstream dbt SQL model. This
example closes the loop — a stel transform reads a **dbt-built** table back,
via `source: dbt_ref('<dbt_model>')`, and feeds a *further* dbt SQL model, all
in one `dbt build`:

```
models/stel/raw_vendors.py          ← stel extraction (no upstream)
models/marts/invoice_facts.sql        ← dbt SQL: select ... from ref('raw_vendors')
models/stel/flagged_invoices.py     ← stel transform: source: dbt_ref('invoice_facts')
models/marts/flagged_invoice_summary.sql ← dbt SQL: select ... from ref('flagged_invoices')
```

One `dbt build` runs all four levels in dependency order — **stel → dbt →
stel → dbt** — as one lineage graph. `flagged_invoices` never enters stel's
own graph for `invoice_facts` (dbt resolves that `ref()`, not stel); stel
only sees it arrive as an injected upstream frame, exactly like a `depends_on:`
dependency.

The stel project (`../dbt_ref_roundtrip`, the source of truth) declares
`flagged_invoices` as:

```yaml
# ../dbt_ref_roundtrip/models/flagged_invoices.yml
models:
  - name: flagged_invoices
    source: dbt_ref('invoice_facts')   # reads a dbt-built table
    transform: { type: python, module: transforms.flag_high_value }
```

The `models/stel/*.py` and `schema.yml` files here are **generated** by
`stel codegen` from that project; they're checked in so this example runs
without the codegen step.

## Run it

From `examples/dbt_ref_roundtrip_dbt/`:

```bash
# Seed the tiny source corpus into the colocated stel project (once) — three
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

# (Re)generate both stel nodes (raw_vendors + flagged_invoices) from the
# stel project's YAML
uv run --project ../.. stel --project-dir ../dbt_ref_roundtrip \
  codegen --output models/stel

# Install this dbt project's env (dbt-duckdb + editable stel) and build
uv sync
STEL_PROJECT_DIR="$(cd ../dbt_ref_roundtrip && pwd)" uv run dbt build --profiles-dir .
```

Expected: `PASS=11` — the extraction model, the reverse-direction transform,
seven schema tests, and both SQL marts all succeed, materializing
`raw_vendors`, `invoice_facts`, `flagged_invoices`, and
`flagged_invoice_summary` into one `target/embedded.duckdb`.

## Caveat: one stel project per `dbt build`

Every generated shim resolves its stel project via a single shared
`STEL_PROJECT_DIR` environment variable. That is why this example is
self-contained — both `raw_vendors` and `flagged_invoices` live in the *same*
stel project (`../dbt_ref_roundtrip`) rather than reusing
[`../invoice_pipeline`](../invoice_pipeline) (the source for
`dbt_embed_duckdb`). Embedding models from two different stel projects in one
`dbt build` isn't supported yet — that would need codegen to bake in a
per-model project path instead of reading one shared environment variable.

See [`examples/dbt_embed_duckdb`](../dbt_embed_duckdb) for the forward-only
three-level DAG, and the "Composing with dbt → Embedded" section of the
[main README](../../README.md) for the full write-up.
