# SaaS Stage 16: ASR Scheduler Dry-Run Approval Contract Audit

Дата: 2026-05-07
Ветка работ: SaaS/productization
Статус: завершено как scheduler dry-run, без ASR execution

## Цель

Stage 16 добавляет следующий безопасный слой после Stage 15: scheduler dry-run, который умеет читать approval-gated ASR job plan и проверять optional operator approval record.

Задача этапа:

- зафиксировать контракт approval artifact;
- показать scheduler-visible статус будущего ASR job;
- блокировать dispatch без валидного approval record;
- не запускать ASR даже при валидном approval, потому что Stage 16 остается dry-run этапом.

## Граница безопасности

Stage 16 не делает:

- ASR;
- R+A;
- dispatch worker job;
- запись в runtime DB;
- запись в product DB;
- запись в asset DB;
- запись в CRM/AMO/Tallanto;
- скачивание, копирование или hardlink audio;
- изменение `stable_runtime/`;
- изменение текущих batch/start/run-ui scripts.

Входы read-only:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_execution_approval_stage15/asr_execution_job_plan_stage15.json`
- optional approval record JSON, если оператор явно передаст `--approval`

Выход только в изолированную productization-папку:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_scheduler_dry_run_stage16/`

## Реализация

Добавлены:

- `src/mango_mvp/productization/asr_scheduler_dry_run.py`
- `scripts/mango_office_asr_scheduler_dry_run.py`
- `tests/test_productization_asr_scheduler_dry_run.py`

Обновлено:

- `src/mango_mvp/productization/__init__.py`

CLI:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 scripts/mango_office_asr_scheduler_dry_run.py
```

## Approval Artifact Contract

Новый контракт:

- `schema_version`: `asr_execution_approval_record_v1`
- `decision`: `approved`
- `approval_ref`: обязательный non-empty ref
- `approved_by`: обязательный оператор
- `approved_at`: ISO datetime
- `job_plan_sha256`: должен совпасть с Stage 15 job plan SHA
- `pack_manifest_sha256`: должен совпасть с ASR worker pack manifest SHA
- `approved_approvals`: должен покрывать все required approvals из Stage 15
- `scope.allowed_item_count`: должен совпасть с количеством ready items
- `acknowledgements`: обязательные acknowledgement flags

Required approvals:

- `explicit_asr_execution_approval`
- `runtime_target_db_approval`
- `stable_runtime_write_policy_acknowledgement`
- `asr_worker_resource_approval`

Required acknowledgements:

- `explicit_asr_execution`
- `runtime_target_db_selected`
- `stable_runtime_write_policy_acknowledged`
- `asr_worker_resource_approved`
- `no_crm_or_tallanto_writes`

## Реальный dry-run результат

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 scripts/mango_office_asr_scheduler_dry_run.py
```

Результат:

- `schema_version`: `asr_scheduler_dry_run_v1`
- `ready_items`: `21`
- `technical_blocked`: `0`
- `approval_present`: `false`
- `approval_valid`: `false`
- `pending_approval`: `1`
- `invalid_approval`: `0`
- `scheduler_may_dispatch`: `false`
- `execution_allowed`: `false`
- `validation_ok`: `true`
- action: `BLOCK_ASR_SCHEDULER_PENDING_APPROVAL`

Scheduler plan:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_scheduler_dry_run_stage16/asr_scheduler_dry_run_plan_stage16.json`
- SHA256: `fe64416881bb37d0777914f588296d01cbed83167f338933b56722f0874870ce`

Source Stage 15 job plan:

- SHA256: `127be248628fc0cba7a16f761aeacc877910af8f3acc650e28bfd56976958b8e`

Source Stage 13 pack manifest:

- SHA256: `3311cce58b1fe956d5aa74d579ad229631b09d74e8089ace17a532ae1a51d4e3`

## Idempotency

Повторный запуск с idempotency output дал тот же scheduler plan SHA256:

- `fe64416881bb37d0777914f588296d01cbed83167f338933b56722f0874870ce`

Idempotency artifact:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_scheduler_dry_run_stage16/asr_scheduler_dry_run_stage16_idempotency_audit.json`

## Product DB integrity audit

После Stage 16 проверена product appliance DB.

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

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_scheduler_dry_run_stage16/product_db_integrity_stage16_audit.json`

## Тесты

Компиляция:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache python3 -m py_compile \
  src/mango_mvp/productization/asr_scheduler_dry_run.py \
  scripts/mango_office_asr_scheduler_dry_run.py \
  src/mango_mvp/productization/__init__.py
```

Focused tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 -m pytest -q \
  tests/test_productization_asr_scheduler_dry_run.py \
  tests/test_productization_asr_execution_approval_gate.py \
  tests/test_productization_scheduler_runtime.py
```

Результат:

- `25 passed, 1 warning`

Full productization tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 -m pytest -q tests/test_productization_*.py
```

Результат:

- `177 passed, 1 warning`

Warning не связан с Stage 16: стандартное предупреждение `urllib3`/LibreSSL окружения.

## Вывод

Stage 16 закрыт. SaaS/productization ветка теперь имеет явный scheduler-visible approval contract для будущего ASR execution.

Текущее состояние правильное: `21` recording asset готов к будущему worker, но scheduler не может dispatch'ить ASR job без валидного approval record. Даже при валидном approval Stage 16 все равно остается dry-run и не запускает ASR.

Следующий безопасный шаг: добавить approval record writer/validator CLI, который сможет создать операторский approval artifact в controlled location, а затем повторно прогнать Stage 16 в режиме `PLAN_ASR_SCHEDULER_APPROVED_DRY_RUN`, все еще без ASR execution.
