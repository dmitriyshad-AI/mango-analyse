# ADR-003 F2e Existence Report Uses Product Catalog

## Что сделано

`scripts/report_adr003_existence_fact_verification.py` больше не использует временный ad-hoc matcher по строкам KB для доказательства существования продукта/формата.

Теперь отчёт строит `product_existence_axes_catalog` из KB snapshot и вызывает `verify_product_format_exists(...)`.

## Что изменилось в логике отчёта

- `exists` и `not_offered` из каталога считаются `kb_exact`.
- `unknown` и `needs_slot` не считаются доказательством.
- В строке отчёта появился `product_existence_check`, чтобы Claude #1 мог регрейдить по сырью.
- Старые helper-функции временного matcher-а удалены из report script.

## Дополнительный фикс

F2e на реальном 36ea110 показал, что proof-layer не понимал обычную фразу `5-й класс`. Нормализация grade расширена на формы вроде `5-й класс`, закреплено тестом.

После независимого аудита добавлены ещё две защиты:

- enrollment-лексемы (`регистрация`, `набор`, `зачисление`, `заявка`) не становятся proof существования продукта;
- grade parser больше не принимает произвольное слово между числом и `класс` (`5 дней класс`, `9 кабинет класс`, `5 смена класс`).

## Реальный пересчёт 36ea110

См. `real_36ea110/adr003_existence_fact_verification_report.md`:

- existence/format rows: 10;
- current handoff rows: 2;
- handoff with exact KB evidence: 2;
- handoff without exact KB evidence: 0;
- already self with exact KB evidence: 6;
- already self without exact KB evidence: 1;
- excluded danger/money/P0: 1.

Это не включает поведение. Это только показывает, что следующий active/shadow-кандидат существует: 2 safe handoff на ЛВШ 5 класс с точным KB-proof.
