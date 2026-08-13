-- The dbt-built table `flagged_invoices` reads back via `dbt_ref('invoice_facts')`
-- (#177). This is native dbt SQL, not dbt-ml: the reverse direction starts from
-- an ordinary dbt node, same as it would for any dbt-built table.
select
    vendor,
    spend as total_spend
from {{ ref('raw_vendors') }}
