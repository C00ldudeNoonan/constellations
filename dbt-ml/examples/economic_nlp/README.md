# Economic NLP Example

This project turns a small committed economic-policy corpus into normalized
token and named-entity child tables. It uses the optional spaCy integration;
CI validates the same provider contract with a deterministic fake and does not
download a language model.

```bash
uv sync --extra nlp
uv run python -m spacy download en_core_web_sm
uv run dbt-ml --project-dir examples/economic_nlp build
```

`document_tokens` emits one row per non-space token. `document_entities` emits
one row per named entity, but deliberately omits the matched entity text.
Both tables carry stable child IDs, model identity, and the explicitly allowed
`publisher` and `published_at` fields. Neither table retains the source
document's raw `text` column.

`entity_links` resolves entity mentions to canonical agency and ISO 3166
identifiers through the committed `aliases/` table. Linking needs the matched
surface text, which `nlp_entities` withholds by default — so `entity_mentions`
is a separate model that opts in with `include_text: true` and feeds only the
linking step, leaving `document_entities` text-free for analysis.

The alias table carries determiner-prefixed surface forms (`The Federal
Reserve`, `the United States`) because that is what spaCy's NER actually spans —
a reminder that an operator-owned alias table has to match real mention text,
not the tidy canonical name. The corpus exercises all three link outcomes:
`The Federal Reserve` and `the United States` are `matched`, `Washington` is
`ambiguous` across `US-DC` and `US-WA` and is preserved as two rows rather than
guessed, and the date mentions are explicitly `unmatched`.

`document_features` is the analysis-ready table: one row per document with token,
sentence, and entity counts, lexical diversity, stop-word and alphabetic ratios,
configured POS and entity-label counts, and counts of distinct canonical IDs per
namespace. Because it declares `raw_documents` as its document universe, a
document that produced no tokens still gets a row with zero counts and null
ratios rather than disappearing.

```sql
select document_id, publisher, token_count, lexical_diversity,
       linked_agency_count, link_ambiguous_count
from economic_nlp.document_features
order by document_id;
```

`document_tone` scores each document against the operator-owned `tone_lexicon`
(term, category, weight). It is deterministic and reads the token table, so it
needs no language model. General polarity (`positive`/`negative`) and domain
signals (`uncertainty`, `dovish`) are separate lexicon categories, each emitted
as a normalized score and a hit count, with the lexicon fingerprinted as
`lexicon_version` and publisher/release date carried through for analysis.

```sql
select document_id, publisher, published_at, positive_score, negative_score,
       uncertainty_score, dovish_score, coverage, status
from economic_nlp.document_tone
order by document_id;
```
