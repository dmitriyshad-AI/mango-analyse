# Snapshot / rollback

- Snapshot источника: `/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline/product_data/customer_timeline/canonical_readonly_20260619_migration_20260618_002_copy/customer_timeline.sqlite`.
- Источник открыт read-only через `CustomerTimelineReadApi`.
- Новых таблиц или persistent-хранилищ не создавалось.
- Live AMO/Tallanto writeback не выполнялся, поэтому rollback внешних систем не требуется.
- Откат поведения в коде: держать флаги OFF или revert этого коммита до мерджа.
