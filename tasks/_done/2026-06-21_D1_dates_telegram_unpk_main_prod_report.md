# D1: даты + Telegram УНПК в main и боевую память

Дата: 2026-06-21
Рабочее дерево: `/Users/dmitrijfabarisov/Projects/Mango_main_merge`

## Git

- Даты: `294fd3bd timeline: repair mail stage2 event dates` уже был `HEAD` локального `main` до начала Telegram-слияния.
- Telegram УНПК: влит `codex/tz1-telegram-aprel` merge-коммитом `41ff87f Merge Telegram UNPK timeline ingest`.
- Ветка Telegram добавила один содержательный коммит: `f5b4680 timeline: ingest telegram export as manager-only`.
- Запрещённые ветки `codex/etap3-botsafe-layer`, `codex/etap3-faza1-botsafe-bot`, `codex/foton-next-step-resolver` в рамках этой задачи не вливались.

## Pytest

Команда:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
```

Результат:

```text
3483 passed, 5 skipped, 1 warning in 57.83s
```

## Бэкап боевой памяти

Бэкап создан ДО Telegram-записи.

- Боевая БД: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`
- Бэкап: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/backups/before_telegram_unpk_ingest_20260621T090700Z/customer_timeline.sqlite`
- Манифест: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/backups/before_telegram_unpk_ingest_20260621T090700Z/backup_manifest.json`
- backup_sha256: `d7c0528635c539da8adb80844ee89c9c1ef4829be93af00b0d12793c7a233f20`
- source_sha256 на момент бэкапа: `9a394f6b0afa281a8d57122f1045f7bf2ffe753943ce470290a4ddcc32fea2b7`
- `PRAGMA integrity_check`: `ok` для source и backup.
- Контрольные счётчики `timeline_events`, `bot_context_chunks`, `timeline_conflicts`, `customer_identities`, `identity_links`, `ingestion_runs` совпали между source и backup.

Примечание: SQLite online backup консистентен транзакционно, но файл может не быть байт-в-байт равен исходнику из-за перепаковки страниц, поэтому file sha отличается; откатный файл проверен через `integrity_check` и контрольные таблицы.

## Telegram apply

Команда применялась к боевой БД:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/import_telegram_export_to_timeline.py \
  --export-dir "/Users/dmitrijfabarisov/Projects/Mango analyse/telegram_exports (2)/local_vm_2024-04-01_max" \
  --allowed-root "/Users/dmitrijfabarisov/Projects/Mango analyse" \
  --timeline-db "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite" \
  --tenant-id foton \
  --brand unpk \
  --apply \
  --actor d1_telegram_unpk_prod_ingest_20260621 \
  --idempotency-key telegram_unpk_local_vm_2024_04_01_max_prod_20260621
```

Отчёты:

- Первый apply: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/reports/telegram_unpk_apply_20260621T0910.json`
- Повтор apply: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/reports/telegram_unpk_apply_repeat_20260621T0910.json`

Первый apply:

- `validation_ok=true`
- `records_accepted=7085`
- `normalized_total=13201`
- `counters.imported=7085`
- `write_status_counts`: `created=11448`, `updated=1749`, `duplicate=4`

Повтор apply:

- `validation_ok=true`
- `counters.imported=0`
- `counters.duplicates=7085`
- новых событий/дублей по SQL: `0`

## SQL-сверка боевой БД

- `timeline_events WHERE source_system='telegram_history'`: `1377`
  - `telegram_dialog`: `147`
  - `telegram_message`: `1230`
  - все `match_status=strong_unique`
- `bot_context_chunks WHERE source_system='telegram_history'`: `1230`
- Плохие чанки (`allowed_for_bot != 0 OR requires_manager_review != 1 OR brand != unpk`): `0`
- Telegram events с brand != `unpk`: `0`
- `timeline_conflicts`:
  - `pending_attribution`: `5667`
  - `telegram_identity_ambiguous`: `188`
- `bad_unmatched_events`: `0`
- Telegram-created customers: `0`
- Event dedupe duplicates: `0`
- `event_at LIKE '1970-%'`: `0`

## Примеры привязанных Telegram-событий

1. `customer:93f8236ca33c9aef645694c236abb51f`, `2026-04-15T06:38:26+00:00`, `telegram:5241710232:16877`, summary: `Спасибо!`
2. `customer:380c5c713efee1322d9f884b62ae9829`, `2026-04-15T06:31:56+00:00`, `telegram:439944652:16869`, summary: `Администратора предупредили🙏🏼`
3. `customer:93f8236ca33c9aef645694c236abb51f`, `2026-04-15T06:30:31+00:00`, `telegram:5241710232:16868`, summary: `Добрый день! Работы на проверке, на этой неделе будут готовы результаты.🙏`
4. `customer:380c5c713efee1322d9f884b62ae9829`, `2026-04-14T13:39:35+00:00`, `telegram:439944652:16863`, summary: `Постарается дойти. Надо предупредить преподавателей?`
5. `customer:380c5c713efee1322d9f884b62ae9829`, `2026-04-14T13:39:14+00:00`, `telegram:439944652:16862`, summary: `О! Спасибо!`

## NEG

- `telegram_history` чанков с `allowed_for_bot=1`: `0`.
- Повтор apply: `imported=0`, event duplicates по `dedupe_key`: `0`.
- Бэкап создан до записи, манифест и `integrity_check=ok` есть.
- Записей в AMO/Tallanto/CRM не выполнялось; скрипт пишет только SQLite customer_timeline.
- `stable_runtime` не менялся.
- Перегон/Фаза 1/резолвер в рамках этой задачи не вливались.

