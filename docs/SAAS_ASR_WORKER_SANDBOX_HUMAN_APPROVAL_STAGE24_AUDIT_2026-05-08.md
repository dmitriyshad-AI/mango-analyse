# SaaS ASR Worker Sandbox Human Approval Stage 24 Audit

Дата: 2026-05-08

## Цель

Stage 24 добавляет writer/validator для human approval record поверх Stage 23 approval packet. Этот слой умеет записать approval record только при строгих условиях: точная approval phrase, все acknowledgements `true`, совпадение Stage 23 packet SHA, Stage 22 preflight SHA и Stage 21 contract SHA.

В реальном прогоне approval record не создавался, потому что человек не давал явное approval. Был выполнен только safe `requirements` audit.

## Что добавлено

- `src/mango_mvp/productization/asr_worker_sandbox_human_approval_record.py`
  - `requirements`: проверяет Stage 23 packet и сообщает требования к ручному approval.
  - `write`: пишет human approval record только при строгом совпадении phrase/acknowledgements/SHA.
  - `validate`: проверяет уже записанный human approval record.
  - Даже валидный record не дает immediate dispatch в Stage 24.
- `scripts/mango_office_asr_worker_sandbox_human_approval.py`
  - CLI с командами `requirements`, `write`, `validate`.
- `tests/test_productization_asr_worker_sandbox_human_approval_record.py`
  - Requirements без approval.
  - Write + validate happy path в тестовой среде.
  - Блокировка wrong phrase.
  - Блокировка missing acknowledgement.
  - Блокировка tampered record.
  - Path guards.
  - CLI requirements/write/validate.
- `src/mango_mvp/productization/__init__.py`
  - Экспортирует Stage 24 summary/functions.

## Реальный прогон

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 scripts/mango_office_asr_worker_sandbox_human_approval.py requirements
```

Основной артефакт:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_human_approval_stage24/asr_worker_sandbox_human_approval_requirements_stage24_audit.json`

Результат:

- `operation=requirements`
- `validation_ok=true`
- `approval_packet_valid=true`
- `approval_record_present=false`
- `approval_record_valid=false`
- `written=0`
- `execution_approved=false`
- `dispatch_allowed=false`
- `run_asr=false`
- `write_transcripts=false`
- `tasks=21`
- `selected_engine=mlx`
- `required_acknowledgements=10`
- `acknowledgement_true_count=0`

Проверка отсутствия approval record:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_human_approval_stage24/asr_worker_sandbox_human_approval_record_stage24.json` отсутствует.

Stage 23 packet SHA:

- `8dfe987f0afd19e696710545311efbbbb49d5b460068af1a770f924c47aa6845`

Requirements audit SHA after same-path rerun:

- `ac9e1acc2d5f726f5f267e96ec79b17ced7ce759626d9d66d7821ddd9c3f48c1`

## Approval requirements

Required approval phrase:

```text
APPROVE_ASR_SANDBOX_EXECUTION_STAGE23
```

Required acknowledgements:

- `explicit_asr_sandbox_execution`
- `selected_engine_acknowledged`
- `audio_sha_preflight_acknowledged`
- `expected_outputs_acknowledged`
- `resource_limits_acknowledged`
- `sandbox_output_root_acknowledged`
- `no_runtime_db_writes`
- `no_crm_or_tallanto_writes`
- `stable_runtime_not_touched`
- `human_operator_accepts_execution_risk`

## Record Semantics

Если в будущем будет создан валидный approval record, он будет означать только разрешение на следующий execution-request слой, а не запуск ASR в Stage 24.

Валидный record должен иметь:

- `decision=approved`
- exact approval phrase
- all acknowledgements `true`
- Stage 23 packet SHA match
- Stage 22 preflight SHA match
- Stage 21 contract SHA match
- `valid_for_asr_sandbox_execution_request=true`
- `valid_for_immediate_worker_dispatch=false`
- `valid_for_runtime_db_writes=false`
- `valid_for_crm_writes=false`

## Safety boundary

В real requirements audit:

- `writes_approval_record=false`
- `reads_audio=false`
- `creates_sandbox_output_dirs=false`
- `creates_sandbox_tmp_dirs=false`
- `imports_asr_modules=false`
- `loads_models=false`
- `dispatch_worker=false`
- `run_asr=false`
- `write_transcripts=false`
- `runtime_db_writes=false`
- `stable_runtime_writes=false`
- `write_crm=false`
- `write_tallanto=false`

## Product DB integrity

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 scripts/mango_office_product_db_admin.py integrity \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_human_approval_stage24/product_db_integrity_stage24_audit.json
```

Результат:

- `validation_ok=true`
- `product_calls=297`
- `raw_payload_refs_present=297`
- `capture_inbox_items=21`
- `capture_inbox_ready=21`
- `capture_inbox_blocked=0`
- `job_runs=5`
- `failed_job_runs=0`
- `running_job_runs=0`
- `warnings=3`

## Тесты

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache python3 -m py_compile \
  src/mango_mvp/productization/asr_worker_sandbox_human_approval_record.py \
  scripts/mango_office_asr_worker_sandbox_human_approval.py \
  src/mango_mvp/productization/__init__.py
```

Результат: успешно.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 -m pytest -q \
  tests/test_productization_asr_worker_sandbox_human_approval_record.py \
  tests/test_productization_asr_worker_sandbox_approval_packet.py \
  tests/test_productization_asr_worker_sandbox_preflight.py
```

Результат: `19 passed`.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 -m pytest -q \
  tests/test_productization_asr_worker_sandbox_human_approval_record.py \
  tests/test_productization_asr_worker_sandbox_approval_packet.py \
  tests/test_productization_asr_worker_sandbox_preflight.py \
  tests/test_productization_asr_worker_sandbox_execution_contract.py \
  tests/test_productization_asr_worker_sandbox_readiness.py \
  tests/test_productization_asr_worker_execution_dry_run.py \
  tests/test_productization_asr_execution_plan.py
```

Результат: `41 passed`.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 -m pytest -q tests/test_productization_*.py
```

Результат: `224 passed, 1 warning`.

Warning внешний: `urllib3` сообщает, что локальный Python собран с `LibreSSL 2.8.3`, а `urllib3 v2` предпочитает OpenSSL 1.1.1+.

## Аудит

Stage 24 достиг потолка полезности для record-only слоя:

- Writer и validator реализованы.
- Real run не создал approval record без явного человеческого approval.
- Strict phrase и 10 acknowledgements покрыты тестами.
- Tamper protection по SHA покрыт тестами.
- Runtime DB, `stable_runtime`, CRM/Tallanto не затронуты.
- Product DB integrity зеленый.
- Полный productization test suite зеленый.

## Следующий безопасный шаг

Stage 25: ASR sandbox execution request dry-run.

Нужно построить слой, который читает Stage 24 approval record, если он появится, и создает execution request envelope. До появления валидного human approval record Stage 25 должен блокироваться. Даже после появления record Stage 25 должен остаться dry-run/request-only: без ASR, без dispatch, без создания sandbox dirs, без transcript writes, без runtime DB и без CRM.
