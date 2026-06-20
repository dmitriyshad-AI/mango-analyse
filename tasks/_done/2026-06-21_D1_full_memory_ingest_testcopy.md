# D1 full memory ingest: test-copy procedure

Дата: 2026-06-21
Ревизия канона по ТЗ: af3fa2cc
Режим: test-copy only, production apply не выполнялся

## Что добавлено

- Добавлена безопасная процедура `scripts/run_customer_timeline_full_memory_ingest_procedure.py test-copy`.
- Production-команды в CLI нет: процедура только строит тестовую копию.
- Назначенная production-БД:
  `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`.
- Перед первым импортёром создаётся полный backup тестовой БД с sha-манифестом.
- Порядок импортёров проверен строго последовательно:
  1. backup пустой test-copy DB;
  2. mail_stage2 dry-run;
  3. canonical import: contacts/calls/AMO/Tallanto/mail aggregate;
  4. mail_stage2 apply: full emails via fresh relink;
  5. повтор canonical + mail_stage2 для идемпотентности;
  6. restore из backup.

## Артефакты проверки

- Test-copy root:
  `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/canonical_readonly_full_memory_testcopy_20260620T225459Z`
- Итоговый JSON:
  `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/canonical_readonly_full_memory_testcopy_20260620T225459Z/full_memory_ingest_test_report.json`
- Backup manifest:
  `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/canonical_readonly_full_memory_testcopy_20260620T225459Z/backups/before_full_memory_ingest_20260620T225459Z/backup_manifest.json`
- Backup sha256:
  `ab5546a73e513127a40da98a4b14eb0a896a57fa5641d168af92c24f80f2d9b9`

## Числа по каналам

События после первого полного прохода:

- `master_contacts_snapshot`: 16 901
- `mango_processed_summary`: 71 962
- `amocrm_snapshot`: 16 277
- `tallanto_snapshot`: 16 901
- `mail_archive`: 4 168
- `mail_archive_stage2`: 30 093

Итого после canonical: 126 209 events, 85 189 bot_context_chunks.
Итого после mail_stage2: 156 302 events, 85 189 bot_context_chunks.

## Идемпотентность

Повторный проход:

- `repeat_delta.timeline_events`: 0
- `repeat_delta.bot_context_chunks`: 0
- `mail_repeat.created_events`: 0
- `mail_repeat.skipped_existing_events`: 30 093
- дубликаты `dedupe_key`: 0

Restore из backup вернул тестовую БД к исходным пустым счётчикам.

## Safety

- `unsafe_bot_context_chunks`: 0
- `mail_stage2_unmatched_with_customer`: 0
- `mail_stage2_interim_customer_ids`: 0
- `mail_stage2_pending_attribution_events`: 30 093
- `source_systems_do_not_overlap`: true
- `production_apply_not_performed`: true

Важно: mail_stage2 full-events JSONL содержит только старые `customer_id/customer_name`, но не содержит `from/to/participants/extracted_text_path` с email/phone-сигналами. Fresh relink по контракту не доверяет старому `customer_id`, поэтому все 30 093 письма безопасно ушли в `unmatched/pending`. Это безопасный результат, но покрытие привязки писем = 0 до появления входа с сигналами relink.

## Лента клиентов

Read API samples получены в `full_memory_ingest_test_report.json`, поле `read_api_samples`.
Для 5 клиентов проверены смешанные источники; в примерах есть `mango_processed_summary`, readiness показывает `bot_allowed_chunks=0`, `bot_review_required_chunks>0`, то есть каналовая память остаётся manager-only.

## Проверки

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_store.py tests/test_customer_timeline_full_memory_ingest.py tests/test_customer_timeline_mail_stage2_ingest.py`
  - результат: 23 passed
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`
  - результат: 3470 passed, 5 skipped

## Semantic pass

Вердикт: `PASS_WITH_NOTES`.

Процедура безопасна для production, потому что production apply не реализован в CLI и не выполнялся. Главный остаточный риск не в записи, а в качестве входа mail_stage2: текущие JSONL не дают fresh-relink сигналов, поэтому письма пока сохраняются как pending, а не привязанные к клиентам.

Apply в боевую БД без отдельного «да» Дмитрия не делал.
