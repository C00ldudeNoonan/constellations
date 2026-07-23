select *
from (
    values
        ('2026-01-01'::date, 'enterprise', 100, 4),
        ('2026-04-01'::date, 'enterprise', 100, 7)
) as quarterly_refunds(metric_time, customer_segment, total_orders, refunded_orders)
