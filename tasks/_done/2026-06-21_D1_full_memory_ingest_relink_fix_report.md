# D1 full memory ingest: mail relink join fix

Дата: 2026-06-21
Предыдущая проблема: `mail_stage2 linked=0`, потому что JSONL несёт старый `customer_id`, но не несёт email/phone-сигналов; fresh-relink ему правильно не доверял.

## Что изменено

- `mail_stage2_ingest` теперь умеет принимать CSV решений relink:
  - `mail_stage2_customer_relink_preview_decisions.csv`;
  - join по `message_sha256`;
  - `linked/already_linked` резолвится в авторитетный `tallanto_id` через fresh identity DB;
  - старый `customer_id` из JSONL по-прежнему не используется как авторитет.
- Если в canonical timeline уже есть customer по `tallanto_student_id`, письмо ложится на существующий `customer_id`.
- Если canonical customer ещё нет, создаётся минимальный `tallanto:<id>` identity.
- `mail_archive_stage2` остаётся отдельным `source_system`; canonical `mail_archive` не смешивается с полными письмами.

## Проверка на тестовой копии

Test-copy root:
`/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/canonical_readonly_full_memory_relink_testcopy_20260620T232243Z`

Итоговый JSON:
`/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/canonical_readonly_full_memory_relink_testcopy_20260620T232243Z/full_memory_ingest_test_report.json`

Backup manifest:
`/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/canonical_readonly_full_memory_relink_testcopy_20260620T232243Z/backups/before_full_memory_ingest_20260620T232244Z/backup_manifest.json`

Backup sha256:
`18b6ef8dd49a9619e73af5dd133c2055041952ecad9343d10b431878c02f23ba`

## Числа

Mail stage2:

- input: 30 093
- relink decisions loaded: 30 093
- linked: 22 491
- unmatched/pending: 7 602
- linked share: 74.7%
- created email chunks: 22 397

Это соответствует D4-мосту (`22 491 resolved`, `7 602 pending`) и исправляет прежние `0 linked`.

События по каналам после первого полного прохода:

- `master_contacts_snapshot`: 16 901
- `mango_processed_summary`: 71 962
- `amocrm_snapshot`: 16 277
- `tallanto_snapshot`: 16 901
- `mail_archive`: 4 168
- `mail_archive_stage2`: 30 093

Итого после canonical: 126 209 events, 85 189 bot_context_chunks.
Итого после mail_stage2: 156 302 events, 107 586 bot_context_chunks.

## Идемпотентность и safety

- `repeat_delta.timeline_events`: 0
- `repeat_delta.bot_context_chunks`: 0
- `mail_repeat.created_events`: 0
- `mail_repeat.skipped_existing_events`: 30 093
- `duplicate_event_dedupe_keys`: 0
- `unsafe_bot_context_chunks`: 0
- `mail_stage2_pending_attribution_events`: 7 602
- `mail_stage2_unmatched_with_customer`: 0
- `mail_stage2_interim_customer_ids`: 0
- restore вернул тестовую БД к backup-состоянию.

Production apply не выполнялся; CLI по-прежнему поддерживает только test-copy.

## Проверки

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_mail_stage2_ingest.py tests/test_customer_timeline_full_memory_ingest.py tests/test_customer_timeline_store.py`
  - результат: 25 passed
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`
  - результат: 3472 passed, 5 skipped

## Semantic review

Вердикт: `PASS`.

Смысловой контракт теперь корректный: письма не доверяют старому JSONL `customer_id`, но используют авторитетные relink-решения по `message_sha256`. Неподтверждённые письма не теряются и остаются pending, а привязанные письма остаются manager-only (`allowed_for_bot=0`).
