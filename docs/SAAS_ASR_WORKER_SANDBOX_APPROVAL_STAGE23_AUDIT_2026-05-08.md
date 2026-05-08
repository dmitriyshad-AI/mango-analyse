# SaaS ASR Worker Sandbox Approval Stage 23 Audit

Дата: 2026-05-08

## Цель

Stage 23 готовит explicit approval packet для возможного будущего ASR sandbox execution. Пакет собирает в одном месте проверенные Stage 22 preflight данные: список 21 аудио, выбранный engine `mlx`, expected outputs, resource limits, SHA preflight result и обязательные acknowledgements.

Важно: Stage 23 не является разрешением на запуск. Пакет имеет статус `pending_human_approval`, `execution_approved=false`, `dispatch_allowed=false`, `run_asr=false`.

Этот этап не запускает ASR, не dispatch-ит worker, не импортирует ASR engine-модули, не загружает модели, не создает `sandbox_outputs/` или `sandbox_tmp/`, не читает аудио, не пишет транскрипты, не пишет runtime DB, не пишет CRM/Tallanto и не трогает `stable_runtime`.

## Что добавлено

- `src/mango_mvp/productization/asr_worker_sandbox_approval_packet.py`
  - Валидирует Stage 22 preflight report.
  - Валидирует связанный Stage 21 contract и SHA match.
  - Формирует deterministic approval packet.
  - Включает список approval items с audio SHA, expected outputs и resource limits.
  - Формирует acknowledgement template, где все значения остаются `false`.
  - Формирует approval record template со статусом `pending`, невалидный для dispatch.
- `scripts/mango_office_asr_worker_sandbox_approval_packet.py`
  - CLI для генерации Stage 23 approval packet/audit JSON.
- `tests/test_productization_asr_worker_sandbox_approval_packet.py`
  - Happy path: pending human packet.
  - Блокировка unsafe preflight.
  - Блокировка contract SHA mismatch.
  - Path guards.
  - CLI audit output.
- `src/mango_mvp/productization/__init__.py`
  - Экспортирует `AsrWorkerSandboxApprovalPacketSummary` и `build_asr_worker_sandbox_approval_packet`.

## Реальный прогон

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 scripts/mango_office_asr_worker_sandbox_approval_packet.py
```

Основные артефакты:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_approval_stage23/asr_worker_sandbox_approval_stage23_audit.json`
- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_approval_stage23/asr_worker_sandbox_execution_approval_packet_stage23.json`

Результат:

- `validation_ok=true`
- `status=pending_human_approval`
- `approval_status=pending_human_approval`
- `approval_packet_ref=stage23-pending-approval-d90a5c883a3fd692`
- `selected_engine=mlx`
- `tasks=21`
- `audio_files_checked=21`
- `audio_sha_ok=21`
- `output_collisions=0`
- `required_acknowledgements=10`
- `approval_required=true`
- `execution_approved=false`
- `dispatch_allowed=false`
- `run_asr=false`
- `write_transcripts=false`

Input SHAs:

- Stage 22 preflight report:
  - `d90a5c883a3fd6922674afc733d7704a1bf6936c07f5b850efc8c001d83732f6`
- Stage 21 contract:
  - `984320bbc5370b5cea90e400426b3c48cf43397a42b205c5bbc63c547e46f204`

Stage 23 approval packet SHA:

- `8dfe987f0afd19e696710545311efbbbb49d5b460068af1a770f924c47aa6845`

## Approval requirements

Required approval phrase for a future separate approval-record stage:

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

В `acknowledgement_template` все значения оставлены `false`. В `approval_record_template`:

- `decision=pending`
- `approved_by=null`
- `approved_at=null`
- `approval_phrase=null`
- `valid_for_asr_execution_dispatch=false`

## Workload and limits

Workload:

- `tasks=21`
- `audio_files_checked=21`
- `audio_sha_ok=21`
- `output_collisions=0`

Batch resource limits:

- `total_audio_bytes=6706368`
- `total_duration_sec=1676.592`
- `estimated_tmp_bytes=20807232`
- `estimated_timeout_sec=7639`
- `max_single_timeout_sec=1279`
- `size_class_counts={"long": 1, "medium": 7, "short": 13}`

Preflight summary:

- `preflight_ready=true`
- `engine_preflight_ok=true`
- `disk_space_ok=true`
- `dir_preflight_ok=true`
- `required_free_bytes=94622464`
- `available_free_bytes=282259881984`

## Safety boundary

В approval packet:

- `approval_required=true`
- `execution_approved=false`
- `dispatch_allowed=false`
- `run_asr=false`
- `execution_allowed=false`
- `write_outputs=false`
- `write_transcripts=false`
- `write_runtime_db=false`
- `write_crm=false`

В hard guards:

- `dispatch_worker=false`
- `load_asr_model=false`
- `run_asr=false`
- `run_ra=false`
- `write_transcripts=false`
- `write_runtime_db=false`
- `write_product_db=false`
- `write_asset_db=false`
- `write_crm=false`
- `write_tallanto=false`
- `touch_stable_runtime=false`

В audit safety:

- `reads_preflight_report=true`
- `reads_contract=true`
- `reads_audio=false`
- `creates_sandbox_output_dirs=false`
- `creates_sandbox_tmp_dirs=false`
- `imports_asr_modules=false`
- `loads_models=false`
- `runtime_db_writes=false`
- `stable_runtime_writes=false`

## Идемпотентность

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 scripts/mango_office_asr_worker_sandbox_approval_packet.py --idempotency-out
```

Idempotency audit:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_approval_stage23/asr_worker_sandbox_approval_stage23_idempotency_audit.json`

Повторный прогон выдал тот же approval packet SHA:

- `8dfe987f0afd19e696710545311efbbbb49d5b460068af1a770f924c47aa6845`

## Product DB integrity

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 scripts/mango_office_product_db_admin.py integrity \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_approval_stage23/product_db_integrity_stage23_audit.json
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
  src/mango_mvp/productization/asr_worker_sandbox_approval_packet.py \
  scripts/mango_office_asr_worker_sandbox_approval_packet.py \
  src/mango_mvp/productization/__init__.py
```

Результат: успешно.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 -m pytest -q \
  tests/test_productization_asr_worker_sandbox_approval_packet.py \
  tests/test_productization_asr_worker_sandbox_preflight.py \
  tests/test_productization_asr_worker_sandbox_execution_contract.py
```

Результат: `18 passed`.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 -m pytest -q \
  tests/test_productization_asr_worker_sandbox_approval_packet.py \
  tests/test_productization_asr_worker_sandbox_preflight.py \
  tests/test_productization_asr_worker_sandbox_execution_contract.py \
  tests/test_productization_asr_worker_sandbox_readiness.py \
  tests/test_productization_asr_worker_execution_dry_run.py \
  tests/test_productization_asr_execution_plan.py
```

Результат: `34 passed`.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 -m pytest -q tests/test_productization_*.py
```

Результат: `217 passed, 1 warning`.

Warning внешний: `urllib3` сообщает, что локальный Python собран с `LibreSSL 2.8.3`, а `urllib3 v2` предпочитает OpenSSL 1.1.1+.

## Аудит

Stage 23 достиг потолка полезности для approval-only слоя:

- Stage 22 preflight и Stage 21 contract связаны по SHA.
- Approval packet содержит полный список 21 item.
- Для каждого item есть audio SHA, expected outputs и resource limits.
- Approval template намеренно не дает права на execution.
- Runtime DB, `stable_runtime`, CRM/Tallanto не затронуты.
- Product DB integrity зеленый.
- Полный productization test suite зеленый.

## Следующий безопасный шаг

Stage 24: human approval record writer/validator.

Нужно сделать отдельный слой, который сможет принять approval packet и явно записать human approval record только при строгих условиях: точная approval phrase, все acknowledgements `true`, совпадение Stage 23 packet SHA, Stage 22 preflight SHA и Stage 21 contract SHA. Даже Stage 24 должен оставаться record-only: без ASR, без dispatch, без создания sandbox dirs, без transcript writes, без runtime DB и без CRM.
