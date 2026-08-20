-- A native dbt SQL model that ref()s a stel *transform* model
-- (invoice_summary), which itself was built from the stel extraction model
-- (raw_invoices) via an injected upstream frame. One dbt build runs all three
-- levels in order — extraction -> transform -> SQL — as one lineage graph.
select
    vendor,
    total_spend
from {{ ref('invoice_summary') }}
order by total_spend desc
limit 5
