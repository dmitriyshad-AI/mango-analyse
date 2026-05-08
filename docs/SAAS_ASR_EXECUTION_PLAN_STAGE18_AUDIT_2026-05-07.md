# SaaS Stage 18: ASR Execution Plan Builder Audit

Дата: 2026-05-07
Ветка работ: SaaS/productization
Статус: завершено как execution plan dry-run, без ASR execution

## Цель

Stage 18 добавляет ASR execution plan builder.

Задача этапа:

- взять approved scheduler dry-run plan из Stage 17;
- проверить approval, scheduler status и hard guards;
- прочитать Stage 13 ASR worker pack manifest;
- сверить manifest SHA, audio paths, size, SHA256 и planned output paths;
- разложить 21 запись в детальный per-item execution plan;
- не запускать ASR и не dispatch'ить worker.

## Граница безопасности

Stage 18 не делает:

- ASR;
- R+A;
- scheduler dispatch;
- запись в runtime DB;
- запись в product DB;
- запись в asset DB;
- запись в CRM/AMO/Tallanto;
- скачивание, копирование или hardlink audio;
- изменение `stable_runtime/`;
- изменение текущих batch/start/run-ui scripts.

Stage 18 пишет только JSON artifacts в product appliance:

- execution plan;
- execution plan audit;
- idempotency audit;
- product DB integrity audit.

## Реализация

Добавлены:

- `src/mango_mvp/productization/asr_execution_plan.py`
- `scripts/mango_office_asr_execution_plan.py`
- `tests/test_productization_asr_execution_plan.py`

Обновлено:

- `src/mango_mvp/productization/__init__.py`

CLI:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 scripts/mango_office_asr_execution_plan.py
```

## Real Run

Вход:

- Stage 17 approved scheduler plan:
  `_local_archive_mango_api_downloads_20260507/product_appliance/asr_scheduler_approved_dry_run_stage17/asr_scheduler_approved_dry_run_plan_stage17.json`
- Stage 13 ASR worker pack manifest:
  `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_pack_stage13/asr_worker_input_manifest_stage13.jsonl`

Результат:

- `schema_version`: `asr_execution_plan_v1`
- `status`: `planned_not_dispatched`
- `approval_ref`: `stage17-approved-dry-run-20260507`
- `manifest_rows`: `21`
- `planned_items`: `21`
- `blocked_items`: `0`
- `skipped_items`: `0`
- `technical_blocked`: `0`
- `warnings`: `0`
- `run_asr`: `false`
- `scheduler_dispatch`: `false`
- `execution_allowed`: `false`
- `validation_ok`: `true`
- action: `PLAN_ASR_EXECUTION_ITEM` x `21`

Execution plan:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_execution_plan_stage18/asr_execution_plan_stage18.json`
- SHA256: `f8a0e23f4286f1f0e376327961b6a9b83874c65e0435daf8b218b259736544a0`

Audit:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_execution_plan_stage18/asr_execution_plan_stage18_audit.json`

## Source Integrity

Approved scheduler plan:

- SHA256: `faed86bbe5390e9235837b7cc5aba90c3b271426be3505a2a360f264e24174bf`

Stage 15 job plan:

- SHA256: `127be248628fc0cba7a16f761aeacc877910af8f3acc650e28bfd56976958b8e`

Stage 13 ASR worker pack manifest:

- SHA256: `3311cce58b1fe956d5aa74d579ad229631b09d74e8089ace17a532ae1a51d4e3`

Pack workload:

- audio files: `21`
- total bytes: `6706368`

## Idempotency

Повторный запуск с idempotency output дал тот же execution plan SHA256:

- `f8a0e23f4286f1f0e376327961b6a9b83874c65e0435daf8b218b259736544a0`

Idempotency artifact:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_execution_plan_stage18/asr_execution_plan_stage18_idempotency_audit.json`

## Product DB Integrity

После Stage 18 проверена product appliance DB.

Результат:

- `validation_ok`: `true`
- `blocked`: `0`
- `capture_inbox_items`: `21`
- `capture_inbox_ready`: `21`
- `product_calls`: `297`
- `job_runs`: `5`
- `schema_migrations`: `4`
- `warnings`: `3`

Warnings ожидаемые: `pending_owner_mappings`.

Audit artifact:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_execution_plan_stage18/product_db_integrity_stage18_audit.json`

## Тесты

Компиляция:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache python3 -m py_compile \
  src/mango_mvp/productization/asr_execution_plan.py \
  scripts/mango_office_asr_execution_plan.py \
  src/mango_mvp/productization/__init__.py
```

Focused tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 -m pytest -q \
  tests/test_productization_asr_execution_plan.py \
  tests/test_productization_asr_approval_record.py \
  tests/test_productization_asr_scheduler_dry_run.py \
  tests/test_productization_asr_execution_approval_gate.py
```

Результат:

- `24 passed`

Full productization tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 -m pytest -q tests/test_productization_*.py
```

Результат:

- `189 passed, 1 warning`

Warning не связан с Stage 18: стандартное предупреждение `urllib3`/LibreSSL окружения.

## Вывод

Stage 18 закрыт. SaaS/productization ветка теперь имеет безопасный end-to-end planning path до ASR execution boundary:

1. Worker pack готов и verified.
2. Approval gate заблокировал запуск без approval.
3. Scheduler dry-run увидел pending approval.
4. Approval record создан и validated.
5. Scheduler перешел в approved dry-run.
6. Execution plan builder разложил 21 item в детальный plan, но не запустил ASR.

Следующий безопасный шаг: добавить ASR worker execution dry-run adapter, который прочитает Stage 18 execution plan и построит per-item worker command envelope / resource estimate, все еще без ASR execution и без runtime DB writes.
