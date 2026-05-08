# SaaS Stage 17: ASR Approval Record Writer/Validator Audit

Дата: 2026-05-07
Ветка работ: SaaS/productization
Статус: завершено как approval artifact + approved scheduler dry-run

## Цель

Stage 17 добавляет controlled approval record writer/validator для ASR scheduler dry-run.

Задача этапа:

- создать валидный approval artifact для Stage 15 ASR job plan;
- привязать approval к SHA job plan и SHA ASR worker pack manifest;
- проверить approval validator;
- повторно прогнать Stage 16 scheduler dry-run с approval record;
- получить `PLAN_ASR_SCHEDULER_APPROVED_DRY_RUN`, но не dispatch'ить ASR.

## Граница безопасности

Stage 17 не делает:

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

Stage 17 пишет только JSON artifacts в product appliance:

- approval record;
- approval audit;
- approval validation audit;
- approved scheduler dry-run audit.

## Реализация

Добавлены:

- `src/mango_mvp/productization/asr_approval_record.py`
- `scripts/mango_office_asr_approval_record.py`
- `tests/test_productization_asr_approval_record.py`

Обновлено:

- `src/mango_mvp/productization/__init__.py`

## Approval Record

Команда write:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 scripts/mango_office_asr_approval_record.py write \
  --approval-ref stage17-approved-dry-run-20260507 \
  --approved-by codex-stage17-dry-run \
  --approved-at 2026-05-07T19:20:00+00:00 \
  --reason stage17_scheduler_approved_dry_run_only \
  --replace
```

Результат:

- `schema_version`: `asr_approval_record_writer_v1`
- `approval_ref`: `stage17-approved-dry-run-20260507`
- `approval_valid`: `true`
- `written`: `1`
- `ready_items`: `21`
- `validation_ok`: `true`
- `approval_sha256`: `f6f2d8b3023367441490a459fe7fc962fd37224b5a79a286dd04d3b024a1f6ba`

Approval artifact:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_approval_record_stage17/asr_execution_approval_record_stage17.json`

Audit artifact:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_approval_record_stage17/asr_approval_record_stage17_audit.json`

## Approval Record Safety Scope

Approval record специально ограничен dry-run контекстом:

- `scope.scheduler_dry_run_only`: `true`
- `scope.execution_dispatch_allowed`: `false`
- `stage_contract.valid_for_scheduler_dry_run`: `true`
- `stage_contract.valid_for_asr_execution_dispatch`: `false`
- `stage_contract.must_not_run_asr`: `true`
- `stage_contract.must_not_write_runtime_db`: `true`
- `stage_contract.must_not_write_crm`: `true`

Это означает: record валиден для Stage 16 scheduler dry-run, но не является разрешением на реальное ASR execution.

## Validation

Команда validate:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 scripts/mango_office_asr_approval_record.py validate
```

Результат:

- `approval_valid`: `true`
- `written`: `0`
- `validation_ok`: `true`
- `writes_approval_record`: `false`

Validation artifact:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_approval_record_stage17/asr_approval_record_stage17_validation_audit.json`

## Approved Scheduler Dry-Run

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 scripts/mango_office_asr_scheduler_dry_run.py \
  --approval _local_archive_mango_api_downloads_20260507/product_appliance/asr_approval_record_stage17/asr_execution_approval_record_stage17.json \
  --out-dir _local_archive_mango_api_downloads_20260507/product_appliance/asr_scheduler_approved_dry_run_stage17 \
  --scheduler-plan _local_archive_mango_api_downloads_20260507/product_appliance/asr_scheduler_approved_dry_run_stage17/asr_scheduler_approved_dry_run_plan_stage17.json \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/asr_scheduler_approved_dry_run_stage17/asr_scheduler_approved_dry_run_stage17_audit.json
```

Результат:

- action: `PLAN_ASR_SCHEDULER_APPROVED_DRY_RUN`
- `approval_present`: `true`
- `approval_valid`: `true`
- `pending_approval`: `0`
- `invalid_approval`: `0`
- `technical_blocked`: `0`
- `ready_items`: `21`
- `scheduler_may_dispatch`: `false`
- `execution_allowed`: `false`
- `validation_ok`: `true`
- `warnings`: `0`

Scheduler approved dry-run plan:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_scheduler_approved_dry_run_stage17/asr_scheduler_approved_dry_run_plan_stage17.json`
- SHA256: `faed86bbe5390e9235837b7cc5aba90c3b271426be3505a2a360f264e24174bf`

Approved dry-run audit:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_scheduler_approved_dry_run_stage17/asr_scheduler_approved_dry_run_stage17_audit.json`

Idempotency audit:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_scheduler_approved_dry_run_stage17/asr_scheduler_approved_dry_run_stage17_idempotency_audit.json`

Повторный запуск дал тот же scheduler plan SHA256:

- `faed86bbe5390e9235837b7cc5aba90c3b271426be3505a2a360f264e24174bf`

## Source Integrity

Stage 15 job plan:

- SHA256: `127be248628fc0cba7a16f761aeacc877910af8f3acc650e28bfd56976958b8e`

Stage 13 ASR worker pack manifest:

- SHA256: `3311cce58b1fe956d5aa74d579ad229631b09d74e8089ace17a532ae1a51d4e3`

## Product DB Integrity

После Stage 17 проверена product appliance DB.

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

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_scheduler_approved_dry_run_stage17/product_db_integrity_stage17_audit.json`

## Тесты

Компиляция:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache python3 -m py_compile \
  src/mango_mvp/productization/asr_approval_record.py \
  scripts/mango_office_asr_approval_record.py \
  src/mango_mvp/productization/asr_scheduler_dry_run.py \
  scripts/mango_office_asr_scheduler_dry_run.py \
  src/mango_mvp/productization/__init__.py
```

Focused tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 -m pytest -q \
  tests/test_productization_asr_approval_record.py \
  tests/test_productization_asr_scheduler_dry_run.py \
  tests/test_productization_asr_execution_approval_gate.py \
  tests/test_productization_scheduler_runtime.py
```

Результат:

- `31 passed, 1 warning`

Full productization tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
  python3 -m pytest -q tests/test_productization_*.py
```

Результат:

- `183 passed, 1 warning`

Warning не связан с Stage 17: стандартное предупреждение `urllib3`/LibreSSL окружения.

## Вывод

Stage 17 закрыт. SaaS/productization ветка теперь имеет полный approval artifact path:

1. Stage 15 строит ASR job plan и блокирует execution без approval.
2. Stage 16 показывает scheduler-visible pending approval.
3. Stage 17 создает dry-run-scoped approval record.
4. Stage 16 с этим approval переходит в approved dry-run, но все еще не dispatch'ит ASR.

Следующий безопасный шаг: создать ASR execution plan builder, который на вход принимает approved dry-run scheduler plan и строит детальный execution plan по 21 item, но все еще не запускает ASR и не пишет runtime DB.
