# SaaS Stage 19: ASR Worker Execution Dry-Run Adapter Audit

Дата: 2026-05-07
Ветка работ: SaaS/productization
Статус: завершено как worker command envelope dry-run, без ASR execution

## Цель

Stage 19 добавляет ASR worker execution dry-run adapter.

Задача этапа:

- прочитать Stage 18 ASR execution plan;
- проверить, что plan находится в статусе `planned_not_dispatched`;
- построить per-item worker command envelope;
- оценить ресурсы по audio duration/bytes;
- оставить worker dispatch, ASR, transcript writes, runtime DB и CRM writes выключенными.

## Граница безопасности

Stage 19 не делает:

- ASR;
- R+A;
- worker dispatch;
- запись transcripts;
- запись в runtime DB;
- запись в product DB;
- запись в asset DB;
- запись в CRM/AMO/Tallanto;
- скачивание, копирование или hardlink audio;
- изменение `stable_runtime/`;
- изменение текущих batch/start/run-ui scripts.

Stage 19 пишет только JSON artifacts в product appliance:

- worker dry-run plan;
- worker dry-run audit;
- idempotency audit;
- product DB integrity audit.

## Реализация

Добавлены:

- `src/mango_mvp/productization/asr_worker_execution_dry_run.py`
- `scripts/mango_office_asr_worker_dry_run.py`
- `tests/test_productization_asr_worker_execution_dry_run.py`

Обновлено:

- `src/mango_mvp/productization/__init__.py`

CLI:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 scripts/mango_office_asr_worker_dry_run.py
```

## Real Run

Вход:

- Stage 18 execution plan:
  `_local_archive_mango_api_downloads_20260507/product_appliance/asr_execution_plan_stage18/asr_execution_plan_stage18.json`

Результат:

- `schema_version`: `asr_worker_execution_dry_run_v1`
- `status`: `worker_envelopes_planned_not_dispatched`
- `approval_ref`: `stage17-approved-dry-run-20260507`
- `source_items`: `21`
- `envelopes`: `21`
- `blocked_envelopes`: `0`
- `skipped_items`: `0`
- `technical_blocked`: `0`
- `warnings`: `0`
- `dispatch_allowed`: `false`
- `run_asr`: `false`
- `write_outputs`: `false`
- `validation_ok`: `true`
- action: `PLAN_ASR_WORKER_ENVELOPE` x `21`

Worker dry-run plan:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_dry_run_stage19/asr_worker_dry_run_plan_stage19.json`
- SHA256: `106a8d9082b086acda33713db75ce02d36f569a7f630a871e55c5a345a1066cc`

Audit:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_dry_run_stage19/asr_worker_dry_run_stage19_audit.json`

## Worker Command Envelope

Каждый item получает envelope с контрактом:

- `contract_version`: `asr_worker_command_envelope_v1`
- `executable`: `null`
- `argv`: `[]`
- `dry_run_only`: `true`
- `dispatch_allowed`: `false`
- `run_asr`: `false`
- `write_outputs`: `false`
- `write_runtime_db`: `false`
- `write_crm`: `false`
- `adapter_contract`: `future_asr_worker_adapter`

Это намеренно не runnable command. Это безопасный envelope contract для будущего worker adapter.

## Resource Estimate

Итоговая оценка по 21 envelope:

- total audio duration sec: `1676.592`
- total audio bytes: `6706368`
- estimated tmp bytes: `20807232`
- estimated timeout sec: `7639`
- estimated CPU seconds min: `840.844`
- estimated CPU seconds max: `5029.776`

Size class counts:

- `short`: `13`
- `medium`: `7`
- `long`: `1`

## Source Integrity

Stage 18 execution plan:

- SHA256: `f8a0e23f4286f1f0e376327961b6a9b83874c65e0435daf8b218b259736544a0`

Stage 13 ASR worker pack manifest:

- SHA256: `3311cce58b1fe956d5aa74d579ad229631b09d74e8089ace17a532ae1a51d4e3`

## Idempotency

Повторный запуск с idempotency output дал тот же worker plan SHA256:

- `106a8d9082b086acda33713db75ce02d36f569a7f630a871e55c5a345a1066cc`

Idempotency artifact:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_dry_run_stage19/asr_worker_dry_run_stage19_idempotency_audit.json`

## Product DB Integrity

После Stage 19 проверена product appliance DB.

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

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_dry_run_stage19/product_db_integrity_stage19_audit.json`

## Тесты

Компиляция:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache python3 -m py_compile \
  src/mango_mvp/productization/asr_worker_execution_dry_run.py \
  scripts/mango_office_asr_worker_dry_run.py \
  src/mango_mvp/productization/__init__.py
```

Focused tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 -m pytest -q \
  tests/test_productization_asr_worker_execution_dry_run.py \
  tests/test_productization_asr_execution_plan.py \
  tests/test_productization_asr_approval_record.py
```

Результат:

- `17 passed`

Full productization tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 -m pytest -q tests/test_productization_*.py
```

Результат:

- `194 passed, 1 warning`

Warning не связан с Stage 19: стандартное предупреждение `urllib3`/LibreSSL окружения.

## Вывод

Stage 19 закрыт. SaaS/productization ветка теперь имеет безопасный ASR worker boundary:

1. Stage 18 дает execution plan на 21 item.
2. Stage 19 превращает его в 21 worker command envelope.
3. Все envelopes dry-run-only и не содержат runnable command.
4. ASR, transcript writes, runtime DB и CRM остаются заблокированы.

Следующий безопасный шаг: создать worker sandbox readiness gate, который проверит наличие ASR engine/config как capabilities report, но не будет запускать ASR и не будет писать transcripts/runtime DB.
