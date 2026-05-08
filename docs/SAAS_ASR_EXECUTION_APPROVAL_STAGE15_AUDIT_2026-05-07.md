# SaaS Stage 15: ASR Execution Approval Gate Audit

Дата: 2026-05-07
Ветка работ: SaaS/productization
Статус: завершено как dry-run approval gate

## Цель

Stage 15 добавляет изолированный шлюз разрешения перед любым будущим ASR-исполнением по SaaS/productization worker pack.

Задача шага не запускать ASR, а создать проверяемый job plan, который:

- читает Stage 14 readiness audit;
- подтверждает, что worker pack технически готов;
- явно блокирует ASR execution без recorded approval;
- фиксирует hard guards против runtime DB/audio/transcripts, R+A и CRM/Tallanto writes.

## Граница безопасности

Stage 15 не делает:

- ASR;
- R+A;
- запись в runtime DB;
- запись в product DB;
- запись в asset DB;
- запись в AMO/CRM/Tallanto;
- скачивание или копирование audio;
- изменение `stable_runtime/`;
- изменение текущих batch/start/run-ui scripts.

Вход только read-only:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_pack_stage13/asr_worker_pack_verify_stage14_audit.json`

Выход только в изолированную productization-папку:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_execution_approval_stage15/`

## Реализация

Добавлены:

- `src/mango_mvp/productization/asr_execution_approval_gate.py`
- `scripts/mango_office_asr_execution_approval_gate.py`
- `tests/test_productization_asr_execution_approval_gate.py`

CLI:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 scripts/mango_office_asr_execution_approval_gate.py
```

## Результат dry-run

Создан approval-gated job plan:

- `schema_version`: `asr_execution_approval_gate_v1`
- `job_type`: `asr_execution`
- `mode`: `approval_gate_dry_run`
- `status`: `blocked_pending_explicit_approval`
- `execution_allowed`: `false`
- `approval_present`: `false`
- `approval_required`: `true`
- `readiness_ok`: `true`
- `source_manifest_rows`: `21`
- `ready_items`: `21`
- `technical_blocked`: `0`
- `approval_blocked`: `1`
- `validation_ok`: `true`
- `warnings`: `1`
- action: `BLOCK_ASR_EXECUTION_PENDING_APPROVAL`

Job plan:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_execution_approval_stage15/asr_execution_job_plan_stage15.json`
- SHA256: `127be248628fc0cba7a16f761aeacc877910af8f3acc650e28bfd56976958b8e`

Stage 14 worker pack input refs preserved:

- manifest rows: `21`
- ready items: `21`
- pack audio files: `21`
- pack total bytes: `6706368`
- pack manifest SHA256: `3311cce58b1fe956d5aa74d579ad229631b09d74e8089ace17a532ae1a51d4e3`

## Idempotency

Повторный запуск с idempotency output дал тот же job plan SHA256:

- `127be248628fc0cba7a16f761aeacc877910af8f3acc650e28bfd56976958b8e`

Idempotency artifact:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_execution_approval_stage15/asr_execution_approval_stage15_idempotency_audit.json`

## Product DB integrity audit

После Stage 15 отдельно проверена целостность product appliance DB.

Результат:

- `validation_ok`: `true`
- `blocked`: `0`
- `capture_inbox_items`: `21`
- `capture_inbox_ready`: `21`
- `product_calls`: `297`
- `job_runs`: `5`
- `schema_migrations`: `4`
- `warnings`: `3`

Warnings остаются ожидаемыми для текущей ветки: `pending_owner_mappings`.

Audit artifact:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_execution_approval_stage15/product_db_integrity_stage15_audit.json`

## Тесты

Компиляция:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache python3 -m py_compile \
  src/mango_mvp/productization/asr_execution_approval_gate.py \
  scripts/mango_office_asr_execution_approval_gate.py
```

Focused tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 -m pytest -q \
  tests/test_productization_asr_execution_approval_gate.py \
  tests/test_productization_asr_worker_pack_verifier.py
```

Результат:

- `13 passed`

Full productization tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 -m pytest -q tests/test_productization_*.py
```

Результат:

- `171 passed, 1 warning`

Warning не связан с Stage 15: стандартное предупреждение `urllib3`/LibreSSL окружения.

## Вывод

Stage 15 доводит SaaS/productization ветку до безопасной границы перед вычислительным ASR-исполнением.

Текущий статус правильный: данные готовы к будущему worker execution, но запуск заблокирован без явного approval и без отдельного согласования runtime target.

Следующий безопасный шаг: добавить отдельный approval artifact contract и scheduler dry-run, который сможет видеть blocked job plan, показывать причину блокировки и не запускать execution до появления валидного approval record.
