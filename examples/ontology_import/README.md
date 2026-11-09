# Ontology import

`stel_project.yml` imports classes, a relation type and two vocabularies from
`ontologies/fibo_subset.ttl` instead of retyping them. The file is a small,
hand-written subset modelled on FIBO and OMG LCC IRIs; its triples are
illustrative. Each allow-listed IRI keeps its source IRI on the imported term,
class or relation pairing, and only allow-listed IRIs are imported (Japan is in
the file and not in the project). The offline deterministic provider produces
schema-shaped example values; it does not perform semantic extraction.

From the repository root (needs the `rdf` extra, which `uv sync --all-extras`
installs):

```bash
uv run stel compile --project-dir examples/ontology_import
uv run stel build --project-dir examples/ontology_import
uv run stel plan --project-dir examples/ontology_import
```

The first build writes one row to `stel.classified_filings`, with `domicile`
constrained to the two allow-listed countries and `sector` to the two
allow-listed sectors. The next plan is unchanged. Re-serializing the file, or
reordering its statements, leaves it unchanged too: the allow-list's order is
the order terms take. Change a label in the file (`"Germany"@en`) and `plan`
reports changed configuration; the next build reprocesses the row with the new
closed set. Name an IRI in `include:` that the file does not declare and
`compile` fails before discovering any document. These commands use no
network or credentials.
