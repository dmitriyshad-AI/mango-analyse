# CRM card minimal internal preview: 2026-06-20

Статус: formal_pass, semantic_pass = PASS_WITH_NOTES.

## Задача

По регрейду Block 2 карточка была слишком широкой: 19 полей, старые AI-поля, телефон маскирован. Сделана минимальная manager-only карточка.

## Что изменено

- `src/mango_mvp/crm_card_aggregator.py`
  - контактный payload сужен до: `Запрос`, `Последняя сводка`, `История общения`;
  - сделочный payload сужен до: `Статус сделки`, `Возражения`, `Следующий шаг`, `Tallanto`, `Предупреждения`;
  - убраны из видимого payload старые поля `AI-приоритет`, `AI-основание рекомендации`, `AI-качество привязки`, `AI-дата обновления`, `AI-бюджет*`, `AI-история по сделке`;
  - `Следующий шаг` и `Предупреждения` остаются пустыми, если нет факта; заглушка `проверить вручную` не подставляется;
  - Tallanto остаётся отдельным manager-only полем.
- `src/mango_mvp/customer_timeline/read_api.py`
  - в `manager_projection` добавлен raw `primary_phone`/`phone_numbers` для внутренней карточки;
  - ui/bot-проекция остаётся маскированной;
  - summary `bot_context` переведён с чтения JSON chunks на `COUNT(*)`, чтобы preview не зависал на большой БД.
- `src/mango_mvp/crm_card_workbook.py`
  - видимая таблица начинается с `Имя`, `Телефон`, `Бренд`;
  - таблица показывает 8 смысловых полей карточки и контрольные колонки `Готово`, `Блокеры`, `Вердикт`, `Комментарий`, `customer_id`;
  - телефон берётся из manager-only projection, без маски.

## Preview

Новая папка:

`/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260621_minimal_internal_pii_r3/`

Артефакты:

- `crm_cards_preview.xlsx`
- `crm_cards_preview.csv`
- `crm_cards_preview.summary.json`
- `build_stdout.json`

Сводка preview:

- rows: 50
- ready_yes: 18
- ready_no: 32
- blocker_counts:
  - `amo_contact_id_not_available_in_profile`: 10
  - `amo_lead_id_not_available_in_profile`: 21
  - `p9_ambiguous_identity_manual_review`: 19
- safety:
  - write_amo=false
  - write_tallanto=false
  - write_customer_timeline=false
  - live_network_calls=false

Почему 50, не 200: сборка на 200 строк упёрлась в медленное чтение профилей из большой channel-БД. Процессы были остановлены вручную без записи в источники; неудачные папки оставлены как локальные артефакты, не добавлены в git. Для регрейда структуры полей собран валидный preview на 50 строк.

## Машинные проверки preview

- старые AI-поля в видимой шапке: 0;
- строки с маскированным телефоном `***`: 0/50;
- строки с полным телефоном: 50/50;
- payload с полями вне разрешённой восьмёрки: 0;
- `Tallanto` непустой: 50/50;
- `Предупреждения` непустые только при наличии блокеров/конфликтов/сигналов: 31/50;
- заглушка `проверить вручную` в `Следующий шаг`: 0.

## Тесты

Targeted:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_crm_card_aggregator.py tests/test_amo_writeback_guards.py
```

Результат: `42 passed in 1.29s`.

Full:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
```

Результат: `3377 passed, 5 skipped, 1 warning in 50.84s`.

## Semantic review

Вердикт: PASS_WITH_NOTES.

Что стало лучше:

- карточка стала компактнее: менеджер видит запрос, сделку, возражения, следующий шаг, сводку, Tallanto, предупреждения и историю без дублирующих AI-полей;
- телефон открыт только во внутренней manager-only карточке;
- Tallanto вынесен отдельно и не смешивается с автоисторией;
- пустые поля не заменяются фальшивыми заглушками.

Остаточный риск:

- preview построен на 50 строках из-за скорости чтения большой БД; для финального принятия нужен регрейд архитектора глазами;
- raw phone теперь доступен в `manager_projection`, поэтому downstream должен сохранять разделение: bot/ui используют маскированную проекцию, manager-card использует внутреннюю.
