# Semantic review

Verdict: `PASS_WITH_NOTES`.

Что прошло:

- CRM ready пакет остаётся export-only: `ready_rows=3`, blockers пустые только у 3 строк.
- Golden-negative fixture закрепляет найденные смысловые риски карточек: сырой email thread, маски/debug, чужой бренд, реквизиты, weak next step, family ambiguity, `[сжато]`, stale date, payment conflict.
- В отчётах не заявлено “готово к проду”; явно указаны human gates для SWAP и AMO write.

Неблокирующие риски:

- Н6 проверяет readiness diff, не readback после записи.
- Wappi workbook может создать ложную уверенность, если `medium_text` трактовать как точный match; это отдельно помечено.
- AMO incremental ограничен page cap и не доказывает полный боевой nightly.

Регрессионные правила:

- Новые golden-negative cases добавлены в `tests/fixtures/customer_timeline_crm_export_negative_gate_cases.jsonl`.
- Проверка подключена к `tests/test_customer_timeline_crm_export_package.py`.
