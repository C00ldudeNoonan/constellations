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
-- One project's evidence only. `stel suggest --dbt-project` patches a single
-- repository, so a candidate built from another repository's sessions would
-- be a description applied to a model that merely shares a filename. Set this
-- to the project whose sessions this analysis should read; the corpus holds
-- every project's sessions under the documented global sync (#361 review).
--
-- `project_key` is the repository directory name, so this is stable across
-- the machines a session was recorded on.
where g.project_key = 'analytics'
-- A drafting call that returned nothing usable is not a suggestion. Better an
-- absent candidate than a PR proposing an empty description.
  and d.suggested_description is not null
  and trim(d.suggested_description) <> ''
