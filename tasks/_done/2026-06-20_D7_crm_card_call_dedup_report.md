# D7 CRM card call dedup report

Дата: 2026-06-20
Ветка: `codex/etap1-crm-card-assembler`
Live-write: не запускался; AMO/Tallanto/CRM не трогались.

## Что изменено

- `Последняя AI-сводка` остаётся единственным местом, где полный текст последнего содержательного звонка выводится целиком.
- `Авто история общения` больше не дублирует полный текст последней AI-сводки: вместо него ставится ссылка на поле `Последняя AI-сводка`.
- `AI-сводка по сделке` больше не вываливает полный текст того же звонка, а даёт краткую привязку к сделке и ссылку на AI-сводку.
- `AI-история по сделке` компактно показывает звонки: полный текст не повторяется, остаются дата/источник и структурные поля разбора.
- Для исторических полей используется мягкое обрезание `…` вместо маркера `[сжато]`.

## Preview

Пересобран preview на 200 клиентов Фотона:

- XLSX: `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260620_dedup/crm_cards_preview.xlsx`
- CSV: `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260620_dedup/crm_cards_preview.csv`
- Summary: `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260620_dedup/crm_cards_preview.summary.json`

Сводка preview:

- rows: 200
- ready_yes: 68
- ready_no: 132
- live_network_calls: false
- write_amo/write_tallanto/write_customer_timeline: false

## Машинная проверка

- Служебные снимки в итоговых CRM-полях: 0 (`Read-only AMO contact snapshot`, `exact_phone_single`, `no_exact_phone_match`).
- Полный текст последней AI-сводки в `Авто история общения`: 0.
- Полный текст последней AI-сводки в `AI-сводка по сделке`: 0.
- Полный текст последней AI-сводки в `AI-история по сделке`: 0.
- Строки, где полный текст последней AI-сводки повторяется в общем AMO-preview больше одного раза: 0.
- Поля с маркером `[сжато]` в автоистории/истории сделки: 0.

## Тесты

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_crm_card_aggregator.py`
  - `7 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`
  - `3377 passed, 5 skipped, 1 warning`

## Смысловой контроль

Вердикт: `PASS_WITH_NOTES`.

Что проверено:

- Карточка стала компактнее по целевому классу регрейда: последний звонок не повторяется полным текстом в нескольких разделах.
- Информация не потеряна: полный последний разбор остался в `Последняя AI-сводка`, старые содержательные звонки остаются в контактной автоистории, а раздел сделки хранит компактные ссылки и структурные поля.
- Служебные события не вернулись в клиентскую/менеджерскую ленту.

Остаточный риск:

- Это preview-регрессия на выборке 200 клиентов Фотона. Финальный смысловой регрейд XLSX остаётся за архитектором.
