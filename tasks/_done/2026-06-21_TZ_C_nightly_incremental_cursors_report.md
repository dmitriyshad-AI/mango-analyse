# TZ C: nightly customer_timeline cursors + incremental import

Дата: 2026-06-21
Ветка: `codex/tz-c-nightly-cursors`
Worktree: `/Users/dmitrijfabarisov/Projects/Mango_tzC_nightly_cursors`

## Что сделано

- Добавлена таблица `ingestion_cursors` в customer_timeline SQLite:
  - ключ: `(tenant_id, source_system)`;
  - поля: `last_cursor_ts`, `updated_at`, `metadata_json`;
  - методы store: `get_ingestion_cursor`, `upsert_ingestion_cursor`, `list_ingestion_cursors`.
- Добавлен инкрементальный раннер:
  - `src/mango_mvp/customer_timeline/nightly_incremental.py`;
  - CLI: `scripts/run_customer_timeline_nightly_incremental.py`.
- Инкрементальный выбор событий идёт по `updated_at OR created_at OR event_at`, то есть изменения существующих сделок не теряются из-за `created_at`-only.
- Курсор двигается с нахлёстом `max_source_ts - safety_margin_seconds`; повторы опираются на существующий `dedupe_key`.
- Добавлен single-run lock: `<timeline_db>.nightly.lock`.
- Запись в БД идёт через существующий store, который уже держит writer `flock` и WAL.
- Добавлен JSONL-журнал каждого прогона:
  - источники, курсоры, затронутые клиенты, реально изменившиеся клиенты;
  - времена фаз `ingest` / `rebuild`;
  - статусы источников и safety-блок.
- Недоступный источник не валит прогон:
  - источник пропускается;
  - пишется `source_unavailable`;
  - после 2 подряд пропусков в cursor metadata ставится `alert=true`.
- Пересчёт затронутых реализован по текущему интерфейсу:
  - `CustomerProfileBuilder` получает `customer_ids`;
  - `build_bot_safe_summaries` теперь умеет ограничение `customer_ids`;
  - если финальный сборщик Фазы 0 ещё не подключён, отчёт явно пишет `deferred_pending_phase0_builder` и список клиентов.

## Проверка критериев

- Курсоры тянут только новое: тест `test_nightly_incremental_uses_overlap_and_repeat_adds_no_duplicates`.
- Повтор без новых данных даёт `changed_customer_ids=[]` и не запускает пересчёт: тот же тест + локальное демо.
- `updated_at` ловится независимо от старого `created_at`: тест `test_nightly_incremental_uses_updated_at_not_only_created_at`.
- Недоступный источник пропускается и не останавливает цикл: тест `test_nightly_incremental_unavailable_source_skips_and_alerts_after_two_failures`.
- Два прогона одновременно: второй ждёт lock: тест `test_single_run_lock_waits_for_existing_holder`.
- Фаза пересчёта трогает только `changed_customer_ids`: локальное демо, второй прогон `changed_customer_count=0`, `rebuild.selected_customer_count=0`.

## Локальное демо

Путь: `/tmp/mango_tzC_nightly_demo_20260621T133543`

Первый прогон:

```json
{
  "affected_customer_count": 1,
  "changed_customer_count": 1,
  "source_statuses": {"ok": 1},
  "phase_seconds": {"ingest": 0.008, "rebuild": 0.013}
}
```

Повторный прогон на тех же данных:

```json
{
  "affected_customer_count": 1,
  "changed_customer_count": 0,
  "source_statuses": {"ok": 1},
  "phase_seconds": {"ingest": 0.008, "rebuild": 0.0}
}
```

Журнал: `/tmp/mango_tzC_nightly_demo_20260621T133543/nightly_journal.jsonl`.

## NEG

- Повторный apply не растит дубли: покрыто тестом, в демо второй прогон дал `write_status_counts.duplicate=1`, новых изменений 0.
- События на границе курсора не выпадают: курсор хранится с нахлёстом, тест проверяет `last_cursor_ts = max_ts - 60s`.
- `updated_at`-изменение существующей записи подхватывается: покрыто тестом.
- Недоступный источник не падает весь прогон: покрыто тестом, статус `source_unavailable`, alert после 2 провалов.
- Параллельный запуск не ломает запись: покрыто тестом single-run lock.
- В AMO/Tallanto/CRM нет записи: новый раннер работает только с локальными JSONL-источниками и customer_timeline SQLite; safety-блок журнала фиксирует `writes_amo=false`, `writes_tallanto=false`, `network_calls=false`.
- `stable_runtime` не изменялся.

## Тесты

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_nightly_incremental.py tests/test_customer_timeline_store.py tests/test_customer_timeline_bot_safe_summary.py
30 passed in 1.28s
```

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
3488 passed, 5 skipped, 1 warning in 56.93s
```

## Остаточный стык с D3

Финальный пересчёт выжимки Фазы 0 и шага D8 подключается через список `changed_customer_ids`.
Сейчас код уже передаёт этот список в текущие профильные и bot-safe сборщики. После влития D3 нужно заменить/добавить конкретный вызов финального сборщика Фазы 0 в `rebuild_affected_outputs`, не меняя слой курсоров и dedupe.
