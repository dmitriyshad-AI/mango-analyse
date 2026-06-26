# D1 AMO Card Dry-Run After Source Label Cleanup

Дата: 2026-06-26

Режим: read-only dry-run, live write запрещён.

## Безопасность

- AMO write: 0
- Tallanto write: 0
- CRM write: 0
- AMO read path: `foton-crm-readonly`, только `amo_api_get`
- Секреты и токены в отчёт не выводились

## Вход

- Preview: `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260626_source_label_cleanup_prod_ro/crm_cards_preview.csv`
- Предыдущий dry-run для тех же 5 карточек: `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_dry_run_writeback_20260625_5cards_mcp/dry_run_writeback_5cards_mcp.json`

Проверялись те же 5 `customer_id`, что и в dry-run 25.06.

## Итог

- Проверено карточек: 5
- `payload_noise` / `mango_processed_summary` в payload: 0
- Прошли бы к записи после отдельного разрешения: 3/5
- Заблокированы anti-clobber: 2/5

`anti-clobber` сработал корректно: текущее значение поля `Авто история общения` в AMO уже отличается от expected-before SHA из прошлого dry-run, поэтому такие карточки нельзя перетирать без отдельного ручного решения.

## Artifacts

- JSON: `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_dry_run_writeback_20260626_5cards_source_label_cleanup/dry_run_writeback_5cards_source_label_cleanup.json`
- Markdown: `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_dry_run_writeback_20260626_5cards_source_label_cleanup/dry_run_writeback_5cards_source_label_cleanup.md`

## Semantic Review

Verdict: `FORMAL_PASS_WITH_WRITEBACK_GATES`.

Служебная метка `mango_processed_summary` убрана из manager-visible payload. Механика защиты записи работает: часть карточек пропущена, часть остановлена из-за изменения текущего AMO-поля после предыдущего снимка. Live-write карточек не запускался и не должен запускаться без отдельного списка разрешённых карточек и свежего readback-плана.
