# Что сделано

- Добавлен CRM-слой сборщика карточки клиента: `src/mango_mvp/crm_card_aggregator.py`.
  - Читает уже готовый `read_api.customer_profile` как входной профиль, сам `customer_timeline` не меняет.
  - Одно ядро собирает две проекции: `contact_card` и `deal_card`.
  - Даты обновления берутся из `snapshot_as_of` / `last_event_at`, без `now()`.
  - `manager_facts` из старого analyze используются как fallback для возражений, вероятности, бюджета, скидки и базового next step.
  - Ambiguous/shared/family identity блокирует сделочные и ученические поля: `Готово=нет`, blocker `p9_ambiguous_identity_manual_review`.
  - Длинные возражения сжимаются до лимита с явным маркером `[сжато]`.
- Добавлен transient workbook builder: `src/mango_mvp/crm_card_workbook.py` и CLI `scripts/build_crm_customer_card_workbook.py`.
  - Пишет только `.xlsx`, `.csv`, `.summary.json` во внешнюю папку Foton.
  - Не создаёт новых таблиц и не пишет в timeline/AMO/Tallanto.
  - Summary-дата workbook стабильна: максимальная дата `Дата` из строк или явный override, не время запуска.
- Подключены потребители за флагами default OFF:
  - `CRM_CARD_AGGREGATOR_ENABLED` в `scripts/write_amo_ready_contacts.py` и `src/mango_mvp/deal_aware/deal_text_builder.py`.
  - `CRM_AUTO_HISTORY_CHRONOLOGY_TEXT` для текстовой хронологии в поле auto history.
  - `CRM_DEAL_OBJECTION_EXPLICIT_COMPACT` для маркера `[сжато]` вместо старого ellipsis.
  - `CRM_CONTACT_WRITEBACK_AI_ALLOWLIST` для P8 allowlist нижнего AMO-helper.
- Усилен P8 guard в `src/mango_mvp/amocrm_runtime/amo_integration.py`:
  - при включенном флаге разрешены только целевые AI-поля контакта;
  - телефон, ФИО, email и ручная `История общения` остаются запрещены.

# Реальный workbook

- XLSX: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-19_Etap1_Block4_crm_customer_cards_preview_codex.xlsx`
- CSV: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-19_Etap1_Block4_crm_customer_cards_preview_codex.csv`
- Summary JSON: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-19_Etap1_Block4_crm_customer_cards_preview_codex.summary.json`

Числа:

- rows: 200
- ready_yes: 0
- ready_no: 200
- generated_at: `2026-05-21T08:57:42+00:00`
- blockers:
  - `amo_contact_id_masked_in_read_api`: 162
  - `amo_contact_id_not_available_in_profile`: 38
  - `amo_lead_id_masked_in_read_api`: 126
  - `amo_lead_id_not_available_in_profile`: 74
  - `open_conflicts_require_manager_review`: 10
  - `p9_ambiguous_identity_manual_review`: 91

Интерпретация: таблица собрана корректно, но все 200 строк fail-closed. Основной блокер - `read_api` отдаёт AMO ids маскированными или не отдаёт их; это правильно блокирует будущую запись до контракта с владельцем истории/AMO snapshot.

# Как проверялось

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_amo_writeback_guards.py tests/test_crm_card_aggregator.py tests/test_deal_aware_structured_objections.py` -> `56 passed`.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q` -> `3376 passed, 5 skipped, 1 warning`.
- `git diff --check` -> без вывода.
- `openpyxl.load_workbook(...)` на итоговом xlsx:
  - sheets: `['Сводка', 'Клиенты']`;
  - rows: 201 на листе `Клиенты`;
  - cols: 13.

# Read-only подтверждение

- Источник timeline открыт через `CustomerTimelineReadApiConfig(..., allowed_root=...)`.
- Реальные live-write действия не запускались.
- AMO/Tallanto/CRM write не выполнялся.
- ASR/analyze/тяжёлые batch-прогоны не запускались.
- `customer_timeline` worktree не менялся.

# Что осталось

- Для будущего Этапа 2 нужен немаскированный read-only join-key AMO contact/lead id или отдельный approved join snapshot. Пока P9/P8 блокирует запись.
- Бренд в менеджерской карточке информативный, не стоп-гейт. Для будущего бота нужно отдельное bot-safe разделение.
