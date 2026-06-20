# Block2 — CRM card preview with channels

Дата: 2026-06-20/21  
Ветка/worktree: `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards` (`codex/etap1-crm-card-assembler`)  
Live-write: не запускался. AMO/Tallanto/CRM не трогались.

## Входная timeline-БД

Использована новая БД из Block1:

`/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline/product_data/customer_timeline/canonical_readonly_20260621_with_channels/customer_timeline.sqlite`

Fallback на старую БД не понадобился.

## Preview

Основной preview:

`/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260621_with_channels/`

Файлы:

- `crm_cards_preview.xlsx`
- `crm_cards_preview.csv`
- `crm_cards_preview.summary.json`
- `build_stdout.json`

Повторный preview для детерминизма:

`/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260621_with_channels_repeat/`

Текущий preview `crm_card_preview_20260620_dedup` не перезаписывался.

## Счётчики

- rows: 200
- ready_yes: 68
- ready_no: 132
- blockers:
  - `amo_contact_id_not_available_in_profile`: 38
  - `amo_lead_id_not_available_in_profile`: 74
  - `open_conflicts_require_manager_review`: 1
  - `p9_ambiguous_identity_manual_review`: 91

Safety из summary:

- live_network_calls: false
- write_amo: false
- write_tallanto: false
- write_customer_timeline: false

## Детерминизм карточки

Два прогона на одной БД дали идентичные CSV row hashes:

- `csv_deterministic`: true
- rows compared: 200
- first_diff: null

## Проверки качества preview

- `Read-only AMO contact snapshot` в итоговом AMO preview: 0
- `exact_phone_single` / `no_exact_phone_match` в итоговом AMO preview: 0
- `[сжато]` в итоговом AMO preview: 0
- полный текст последней AI-сводки не дублируется в автоистории/сделке.

## Источники в выборке

По 200 customer_id из preview:

- `mango_processed_summary`: 200
- `master_contacts_snapshot`: 200
- `tallanto_snapshot`: 200
- `amocrm_snapshot`: 162
- `mail_archive`: 35
- `channel_snapshot` (WhatsApp): 11

Максимум источников на клиента в выборке: 6 (`amocrm_snapshot`, `channel_snapshot`, `mail_archive`, `mango_processed_summary`, `master_contacts_snapshot`, `tallanto_snapshot`).

Telegram-касания в этой выборке не попали в топ-200 Фотона; это не ошибка сборки, а свойство sample. Для регрейда Telegram рядом со звонком нужен отдельный targeted sample по `source_system='telegram_history'`.

## Смысловой статус

`formal_pass` по сборке preview и машинным проверкам. `semantic_pass` не заявляю: финальный регрейд XLSX по сырью остаётся за архитектором.
