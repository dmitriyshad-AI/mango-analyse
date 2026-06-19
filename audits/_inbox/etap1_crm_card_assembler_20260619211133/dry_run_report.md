# Dry-run

Live/dry-run writeback в AMO не запускался: текущий блок строит preview и xlsx, запись запрещена.

Вместо writeback dry-run выполнена read-only сборка workbook:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/build_crm_customer_card_workbook.py \
  --timeline-db "/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline/product_data/customer_timeline/canonical_readonly_20260619_migration_20260618_002_copy/customer_timeline.sqlite" \
  --allowed-root "/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline/product_data/customer_timeline/canonical_readonly_20260619_migration_20260618_002_copy" \
  --out-xlsx "/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-19_Etap1_Block4_crm_customer_cards_preview_codex.xlsx" \
  --tenant-id foton \
  --sample-size 200
```

Результат: 200 строк, 0 ready, 200 blocked.
