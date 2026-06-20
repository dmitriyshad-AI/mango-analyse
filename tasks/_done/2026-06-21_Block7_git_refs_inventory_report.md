# Block7 git refs inventory — night run

Дата: 2026-06-20/21
Режим: read-only. `git`-команды для выводов не использовались; ветки/файлы не удалялись и не переключались.

## Карта refs/HEAD

- Основное дерево `/Users/dmitrijfabarisov/Projects/Mango analyse`: `HEAD -> codex/tz135-direct-wow-tone @ 9e8fb3b...`, не `main`.
- Локальный `main`: `0e0c7b70...`.
- `origin/main`: `43134ae1...`; отличается от локального `main`.
- `codex/tz139-customer-timeline`: `6c07ba64...`; активного worktree не найдено.
- `codex/tz139-customer-timeline-integrate`: `fba43e40...`; worktree `/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline`.
- `codex/etap1-crm-card-assembler`: `0356723d...`; worktree `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards`.

Важно: у `tz139-integrate` и `etap1-crm-card-assembler` loose refs актуальнее `packed-refs`; старые packed значения не использовать как истину.

## Слитость / архивирование

По refs-only данным нельзя доказать, что целевые ветки слиты в `main`: ни одна целевая ветка не совпадает с локальным `main`, а история объектов не анализировалась. Утром нужно сначала решить расхождение `main` vs `origin/main`, затем отдельно проверять слитость обычным git-аудитом.

Кандидаты на утренний разбор:

- `codex/tz139-customer-timeline-integrate @ fba43e40...` — главный кандидат timeline.
- `codex/etap1-crm-card-assembler @ 0356723d...` — главный кандидат карточки.
- `codex/tz139-customer-timeline @ 6c07ba64...` — проверять на включённость в integrate; refs-only архивировать нельзя.

## Источники canonical builder

Сборщик:
`/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline/scripts/build_canonical_readonly_customer_timeline.py`

Дефолтные источники через `canonical_readonly_import.py`:

- `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/CURRENT_RUNTIME.json`
- `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/sales_master_export_20260523_audio_working_store_v1/master_contacts_ru.csv`
- `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/sales_master_export_20260523_audio_working_store_v1/master_calls_ru.csv`
- `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/deal_aware_amo_live_snapshot_20260513_v2/amo_contacts_snapshot.csv`
- `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/deal_aware_amo_live_snapshot_20260513_v2/amo_deals_snapshot.csv`
- `/Users/dmitrijfabarisov/Projects/Mango analyse/_external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/customer_history_handoff_full_all_mail/mail_customer_history_handoff.sqlite`
- `/Users/dmitrijfabarisov/Projects/Mango analyse/_external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/mango_bridge_preview_full_all_mail_extended_phone_index/mail_mango_bridge_preview.sqlite`

Замечание: в worktree `/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline/stable_runtime/` нет `CURRENT_RUNTIME.json`, поэтому дефолтный запуск из этого worktree без явных source-путей не подходит.
