-- Structured answers from unstructured text, in plain SQL: for each system
-- the docs name, how many postmortems and tickets mention it, and whether
-- it has a runbook. Runs inside the warehouse; no rows pass through stel.
with mentions as (
    select distinct
        l.canonical_id as system,
        r.kind,
        r.source_key,
        r.published_at
    from {{ ref('entity_links') }} as l
    join {{ ref('document_chunks') }} as c on c.context_id = l.document_id
    join {{ ref('document_registry') }} as r on r.document_id = c.document_id
    where l.status = 'matched'
)
select
    system,
    count(distinct source_key) filter (where kind = 'postmortem') as postmortems,
    count(distinct source_key) filter (where kind = 'ticket') as tickets,
    count(distinct source_key) filter (where kind = 'runbook') > 0 as has_runbook,
    max(published_at) filter (where kind = 'postmortem') as last_postmortem
from mentions
group by system
order by postmortems desc, tickets desc, system
