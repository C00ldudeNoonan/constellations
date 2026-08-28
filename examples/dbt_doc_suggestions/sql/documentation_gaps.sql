-- Which dbt models does the corpus show agents repeatedly reading?
--
-- The signal (#361): an agent opening a model's SQL to answer a question is
-- the doc that should have existed. `files_touched` carries the paths each
-- exchange touched, so a model named across several *separate sessions* is a
-- documentation gap with evidence behind it.
--
-- Distinct sessions, never distinct exchanges. One long session where an
-- agent re-reads the same file is one person, one time — counting exchanges
-- would let a single afternoon manufacture its own evidence.
--
-- The threshold lives HERE, not only in `stel suggest --min-evidence`. That
-- flag is a second gate at the edge; if the first gate were the CLI's, this
-- model would draft a description for every one-session file it ever saw and
-- pay a provider per draft for candidates nobody will read (#361).
with touched as (
    select
        e.upstream_document_id as session_key,
        e.exchange_heading,
        -- Paths arrive exactly as the harness recorded them: `file_path` is
        -- stored verbatim, so real sessions carry absolute paths and, on
        -- Windows, backslashes. Matching `models/%` against those finds
        -- nothing, and the analysis would run clean while producing zero
        -- candidates forever (#361 review). Normalize before anything reads
        -- the path.
        replace(f.file_path, '\', '/') as file_path
    from {{ ref('exchange_rows') }} as e,
         unnest(from_json(e.files_touched, '["VARCHAR"]')) as f(file_path)
    where e.files_touched is not null
),
dbt_models as (
    select
        -- The repository directory immediately above `models/`. Sessions
        -- from every project land in one corpus under the documented global
        -- sync, so without this three unrelated repos each touching
        -- `models/marts/fct_orders.sql` would pool their prompts, clear the
        -- threshold together, and produce a description applied to whichever
        -- project `stel suggest` was pointed at (#361 review).
        --
        -- The directory name rather than the whole prefix: the same repo
        -- cloned to `/home/dev/repos/analytics` and
        -- `C:/Users/dev/repos/analytics` is one project, and grouping on the
        -- full path would split its evidence across machines -- which fails
        -- the same threshold from the opposite direction.
        regexp_extract(file_path, '([^/]+)/models/', 1) as project_key,
        -- `.../models/marts/fct_orders.sql` -> `fct_orders`. Anchored to the
        -- dbt convention the patching half already enforces: it only ever
        -- edits `models/**/*.yml`, so only paths under models/ can produce a
        -- suggestion it could apply.
        regexp_extract(file_path, 'models/.*/([^/]+)\.sql$', 1) as dbt_model,
        session_key,
        exchange_heading
    from touched
    where regexp_extract(file_path, 'models/.*/([^/]+)\.sql$', 1) <> ''
)
select
    -- The gap is per project *and* model: evidence never crosses repository
    -- boundaries, and the key records which repository it came from.
    project_key || '::' || dbt_model as gap_id,
    project_key,
    dbt_model,
    count(distinct session_key) as evidence_count,
    -- Sorted so the provenance a reviewer reads is stable across runs, and
    -- so the fingerprint of this row does not churn on warehouse row order.
    string_agg(distinct session_key, ',' order by session_key) as evidence_sessions,
    -- The only thing the drafting model is allowed to see: the human's own
    -- prompt headings. Not the exchange body, which may contain free text a
    -- session opted out of capturing -- a proposed description must never
    -- quote it (#361 rule 4, sensitivity travels). Headings are also the
    -- strongest signal: what the person actually asked is what the missing
    -- description should have answered.
    string_agg(distinct exchange_heading, E'\n' order by exchange_heading)
        as evidence_prompts
from dbt_models
group by 1, 2, 3
-- The analysis-side threshold. Three separate sessions before a gap is worth
-- a provider call; raise it for a busy corpus, never lower it to zero.
having count(distinct session_key) >= 3
