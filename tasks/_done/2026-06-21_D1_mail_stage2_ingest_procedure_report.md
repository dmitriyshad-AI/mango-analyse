# D1: процедура боевого ingest Stage2-почты в customer_timeline

Дата: 2026-06-21  
Ветка/worktree: `codex/etap2-step1-address-book`, `/Users/dmitrijfabarisov/Projects/mango-tz33-perf`  
Статус: formal_pass; semantic_pass = pass_with_notes, боевой запуск заблокирован до отдельного "да" Дмитрия.

## Что сделано

Подготовлена безопасная процедура для будущего боевого ingest почтовых Stage2-событий в `customer_timeline`.

Новые файлы:

- `src/mango_mvp/customer_timeline/mail_stage2_ingest.py`
- `scripts/run_mail_stage2_timeline_ingest_procedure.py`
- `tests/test_customer_timeline_mail_stage2_ingest.py`

Процедура поддерживает четыре шага:

1. `backup` — SQLite backup API, без простого копирования WAL-БД.
2. `dry-run` — считает дифф по `dedupe_key`, БД не мутирует.
3. `apply` — не стартует без `--backup-manifest`; пишет только новые `dedupe_key`.
4. `restore` — откат тестовой БД из backup manifest.

## Привязка и гейты

- Привязка: свежий `bacdd96f` relink через `tallanto_email_identity_map.sqlite`.
- Union-книга не используется; в отчётах явно пишется `union_identity_db_used=false`.
- Старый `customer_id` из Stage2 не принимается как клиентский ключ; он остаётся только хэшом в аудите.
- Linked-событие получает `customer_id=tallanto:<id>`.
- Unmatched-событие получает `customer_id=null`, `match_status=unmatched`, `metadata.pending_attribution=true`.
- Каналовые email-чанки создаются только для linked-событий и всегда:
  - `allowed_for_bot=false`
  - `requires_manager_review=true`

## Контрольный прогон на тестовой БД

Тестовая БД:

`/Users/dmitrijfabarisov/Projects/mango-tz33-perf/_external_handoffs/mail_stage2_ingest_procedure_20260621/test_timeline/customer_timeline.sqlite`

Источник Stage2:

`/Users/dmitrijfabarisov/Projects/Mango analyse/_external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/stage2_email_ingest_20260620/candidates.jsonl`

Identity DB:

`/Users/dmitrijfabarisov/Projects/mango-tz33-perf/_external_handoffs/tallanto_contacts_export_2026-06-20/identity_map/tallanto_email_identity_map.sqlite`

Лимит контрольного прогона: 250 событий.

Артефакты:

`/Users/dmitrijfabarisov/Projects/mango-tz33-perf/_external_handoffs/mail_stage2_ingest_procedure_20260621/`

Результаты:

- `dry-run`: planned=250, would_create=250, linked=199, unmatched=51.
- `apply #1`: created_events=250, created_chunks=199, pending_attribution_events=51.
- `apply #2`: selected_new_events=0, created_events=0, created_chunks=0, skipped_existing_events=250.
- `restore`: тестовая БД восстановлена; после отката `timeline_events=0`, `bot_context_chunks=0`.

## Проверки

Узкие тесты:

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_mail_stage2_ingest.py`

Результат: `2 passed`.

Профильные тесты:

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_*.py tests/test_productization_mail_archive.py`

Результат: `251 passed`.

Полный pytest:

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`

Результат: `3442 passed, 5 skipped, 1 warning`.

## Что не делалось

- Боевой ingest не запускался.
- AMO/Tallanto/CRM не вызывались.
- Live-write не выполнялся.
- Stable runtime не менялся.
- Union-книга не использовалась.

## Остаточные риски

- Перед боевым запуском нужно отдельное подтверждение Дмитрия и свежий backup текущей целевой БД.
- Контрольный прогон был процедурным на 250 событиях; массовый ingest всего корпуса должен идти отдельным боевым запуском с теми же гейтами.
- Для писем без свежего уникального relink сигналов события остаются `pending_attribution`; это ожидаемое безопасное поведение.
