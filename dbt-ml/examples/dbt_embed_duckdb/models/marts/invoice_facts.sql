-- A native dbt SQL model that ref()s the embedded dbt-ml model. Because
-- raw_invoices is a real dbt node (not an external source), this is one dbt DAG
-- with one lineage graph — `dbt build` runs extraction and SQL in order.
select
    vendor,
    count(*)   as invoice_count,
    sum(total) as total_spend
from {{ ref('raw_invoices') }}
group by 1
order by total_spend desc
