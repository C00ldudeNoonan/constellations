-- The contract relation `stel suggest dbt --from` reads (#361/#381).
--
-- The join exists because an `llm:` model emits its id field plus the fields
-- it declared — not the input row's other columns. The evidence (how many
-- sessions, which ones) lives on the gaps model and has to be carried across
-- rather than re-derived, so the count a reviewer sees in the PR body is the
-- same count that decided the candidate was worth drafting.
select
    g.dbt_model,
    -- Model-level documentation only. Column-level gaps need a signal that
    -- names a column, and `files_touched` names files; emitting NULL here is
    -- the contract's way of saying "the model's own description", not a
    -- placeholder for work skipped.
    cast(null as varchar) as dbt_column,
    d.suggested_description,
    g.evidence_count,
    g.evidence_sessions
from {{ ref('documentation_gaps') }} as g
join {{ ref('drafted_descriptions') }} as d
  on d.gap_id = g.gap_id
-- A drafting call that returned nothing usable is not a suggestion. Better an
-- absent candidate than a PR proposing an empty description.
where d.suggested_description is not null
  and trim(d.suggested_description) <> ''
