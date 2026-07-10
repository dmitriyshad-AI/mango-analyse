# D4: regular Mango capture enablement

Дата: 2026-07-10 04:26 MSK  
Ветка: `codex/calls-two-processes`  
База: после `136b61ac Finish calls two-process pipeline gate`

## Что сделано

- Проверено ТЗ `2026-07-10_CODEX_D4_TZ_enable_regular_capture.md` v4.1 и фактический код.
- Исправлен локальный config: `poll_overlap_minutes` уменьшен с `4320` до `30`.
- Первый реальный `cycle` запущен без `--skip-capture`, с Mango API и живыми env-кредами.
- Подтверждено: workers идут строго последовательно, не параллельно:
  `transcribe -> backfill-second-asr -> resolve -> analyze`.
- Выполнены два быстрых повторных `cycle --skip-workers`, потому что новых скачанных аудио не было; это проверило cursor/idempotence без лишнего ASR.
- Установлен launchd сервис `com.mango.calls-two-processes` с интервалом `1800` секунд.
- Обновлены runbook и дефолт инсталлера: near-real-time default `900` секунд, пример `poll_overlap_minutes=30`.

## Первый реальный capture

Команда:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /usr/bin/python3 scripts/run_mango_calls_pipeline.py \
  --config .codex_local/mango_calls_two_processes/config.json \
  cycle \
  --since '2026-07-09T03:00:00+03:00' \
  --until '2026-07-10T04:00:00+03:00'
```

Report:

- Process A: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_calls_two_processes/reports/20260710T005829Z_process_a.json`
- Process B: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_calls_two_processes/reports/20260710T011802Z_process_b.json`

Счётчики capture:

- `api_requests`: 3
- `api_rows_total`: 665
- `api_events_total`: 407
- `api_events_without_recording`: 166
- `total_events`: 241
- `already_manifested`: 241
- `downloaded`: 0
- `failed`: 0

Вывод: Mango API реально опрошен, креды работают. Новых записей с готовой записью в этом окне не было; все 241 события уже были в локальном manifest.

## Повторные прогоны

Повтор 1, `cycle --skip-workers`:

- Process A: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_calls_two_processes/reports/20260710T012301Z_process_a.json`
- Process B: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_calls_two_processes/reports/20260710T012305Z_process_b.json`
- `api_requests`: 1
- `api_rows_total`: 0
- `downloaded`: 0
- `failed`: 0
- Process B: только `duplicate=459`

Повтор 2, `cycle --skip-workers`:

- Process A: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_calls_two_processes/reports/20260710T012402Z_process_a.json`
- Process B: `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_calls_two_processes/reports/20260710T012409Z_process_b.json`
- `api_requests`: 1
- `api_rows_total`: 0
- `downloaded`: 0
- `failed`: 0
- Process B: только `duplicate=459`

Причина `--skip-workers`: после первого no-skip прогона `downloaded=0`, поэтому повторный полный ASR-дренаж был бы пустой 20-минутной нагрузкой на M4. Аудитор подтвердил этот выбор.

## Staging invariants

Staging DB:

`/Users/dmitrijfabarisov/Projects/Mango_email_pipeline_restore/.codex_local/staging/customer_timeline_staging.sqlite`

Проверки после всех прогонов:

- `PRAGMA quick_check`: `ok`
- `PRAGMA integrity_check`: `ok`
- `mango_call` events: 75 265
- `COUNT(DISTINCT dedupe_key)`: 75 265
- `source_system`: только `mango_processed_summary`
- `open_call_chunks`: 0

Записей в prod timeline, `stable_runtime`, AMO, CRM, Tallanto не было.

## Launchd

Установлено:

- Label: `com.mango.calls-two-processes`
- Plist: `/Users/dmitrijfabarisov/Library/LaunchAgents/com.mango.calls-two-processes.plist`
- Interval: 1800 секунд
- WorkingDirectory: `/Users/dmitrijfabarisov/Projects/Mango_calls_two_processes`
- Runner: `scripts/run_mango_calls_cycle.sh`
- Env file: `/Users/dmitrijfabarisov/.mango_secrets/mango_office.env`

Старый `com.mango.customer-timeline-mango-capture` не установлен и не тронут.

## Тесты

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /usr/bin/python3 -m pytest -q \
  tests/test_mango_calls_two_processes.py \
  tests/test_parallel_pipeline.py \
  tests/test_customer_timeline_nightly_service.py
```

Результат: `42 passed, 1 warning`.

Safe collect:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /usr/bin/python3 -m pytest --collect-only -q
```

Результат: `4155 tests collected`.

## ПДн-гейт

Проверены 6 дневных отчётов `Foton/_daily/20260710T*_process_*_calls.json`.

- phone: 0
- email: 0
- secrets: 0

## Остаток

Следующая естественная проверка: первый scheduled цикл, в котором `capture.downloaded > 0`.
После него нужно проверить:

- staging count вырос;
- Process B не только `duplicate`;
- `source_system` остался `mango_processed_summary`;
- `quick_check/integrity_check=ok`;
- `open_call_chunks=0`.

Открытие звонков боту не делалось и остаётся отдельным решением.

## Audit pack

`/Users/dmitrijfabarisov/Projects/Mango_calls_two_processes/audits/_inbox/enable_regular_mango_capture_20260710_20260710042727`
