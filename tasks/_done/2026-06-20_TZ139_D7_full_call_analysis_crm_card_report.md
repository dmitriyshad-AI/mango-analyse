# TZ139 + D7: full call analysis -> CRM card preview

Дата: 2026-06-20

## Что сделано

### Часть A — customer_timeline

- Убран прежний короткий разбор звонка: `mango_call.summary` теперь берётся из полного `analysis_json.history_summary`, без лимита 500 символов.
- В `record.call_analysis` добавлены структурные поля из `canonical_calls.analysis_json`: `objections`, `next_step`, `pain_points`, `interests`, `target_product`, `budget`, `structured_fields`, `quality_flags`, `call_type`, `call_history_eligible`.
- Фильтр живой истории сделан по структурному `quality_flags.call_type` из реального источника. В исходной БД нет поля `call_quality_current.call_type`; фактический источник — `analysis_json.quality_flags.call_type`.
- В `read_api.customer_profile` добавлена внутренняя `manager_projection` с немаскированными AMO contact/lead id. UI/bot-проекции остаются маскированными.
- Для полного импорта исправлен узкий performance-дефект FTS: синхронизация теперь ведёт ключи `timeline_event_fts_keys` и не делает полный `DELETE` по FTS на каждый новый event.

Коммит части A: `9ad5f76 TZ139: preserve full call analysis in customer timeline`.

### Часть B — D7 CRM-card assembler

- Карточка берёт историю общения из полного `call_analysis.history_summary` и добавляет отдельные строки: `Возражения`, `Следующий шаг`, `Боли/ограничения`, `Интересы`, `Целевой продукт`, `Бюджет`.
- В историю карточки попадают только живые разговоры: `sales_call`, `existing_client_progress`, `technical_call`; `non_conversation` и сервисные недозвоны отфильтрованы через `call_history_eligible`.
- Связка с AMO для карточного писателя идёт через `manager_projection` с немаскированными id.

Коммиты D7:

- `4dc90be D7: use manager projection and full call analysis in CRM cards`
- `fdd4cbe TZ139: preserve full call analysis in customer timeline` (cherry-pick контракта read_api)

## Свежая timeline-БД

Путь:

`/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline/product_data/customer_timeline/canonical_readonly_20260619_full_call_analysis_migration_20260618_002_copy_v3/customer_timeline.sqlite`

Проверка импорта:

- `mango_call`: 71 962
- средняя длина `summary`: 515.5
- max `summary`: 1423
- `summary > 500`: 36 901
- `call_history_eligible=true`: 41 736
- `call_history_eligible=false`: 30 226

`call_type` в импортированной timeline:

- `sales_call`: 38 743
- `non_conversation`: 20 860
- `service_call`: 9 366
- `existing_client_progress`: 2 152
- `technical_call`: 841

## Preview XLSX

Папка:

`/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260620_full_call_analysis/`

Файлы:

- `crm_cards_preview.xlsx`
- `crm_cards_preview.csv`
- `crm_cards_preview.summary.json`

Итог на 200 клиентах Фотона:

- `ready_yes`: 68
- `ready_no`: 132
- строки с `Следующий шаг`: 147
- строки с `Возражения`: 162
- максимальная длина блока `ЧТО ПОЙДЁТ В AMO`: 8 928 символов

Пример карточки без ПДн:

- `customer_id`: `customer:0056897822baaaf456fc9a7b8219d9f8`
- `Готово`: да
- ссылка AMO есть
- блок `ЧТО ПОЙДЁТ В AMO`: 3 837 символов
- есть отдельные строки `Следующий шаг` и `Возражения`

## Тесты

Точечные:

- D7 связка `customer_timeline + crm_card_aggregator`: `42 passed`
- TZ139 customer_timeline: `35 passed`

Полный pytest:

- `/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline`: `3362 passed, 5 skipped, 1 warning`
- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards`: `3377 passed, 5 skipped, 1 warning`

Предупреждение одинаковое и внешнее: `urllib3 NotOpenSSLWarning` из системного Python.

## Что важно для регрейда

- Live-write не запускался: AMO/Tallanto/CRM не тронуты.
- Источники `stable_runtime` читались только read-only.
- Preview CSV/JSON/XLSX оставлены как локальные артефакты; в кодовые коммиты не добавлялись.
- Семантический регрейд карточек остаётся за архитектором: formal_pass есть, semantic_pass не заявляю.
