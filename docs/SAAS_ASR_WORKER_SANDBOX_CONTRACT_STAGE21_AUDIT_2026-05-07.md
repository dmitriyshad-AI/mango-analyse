# SaaS ASR Worker Sandbox Contract Stage 21 Audit

Дата: 2026-05-07

## Цель

Stage 21 строит dry-run execution contract для будущего ASR worker sandbox. Он читает Stage 20 readiness report и Stage 19 worker plan, выбирает конкретный ASR engine и раскладывает 21 worker envelope в sandbox task contracts.

Этот этап не запускает ASR, не dispatch-ит worker, не импортирует ASR engine-модули, не загружает модели, не создает transcript/output/tmp директории, не пишет транскрипты, не пишет runtime DB, не пишет CRM/Tallanto и не трогает `stable_runtime`.

## Что добавлено

- `src/mango_mvp/productization/asr_worker_sandbox_execution_contract.py`
  - Валидирует Stage 20 readiness contract.
  - Повторно валидирует Stage 19 worker plan через existing readiness validator.
  - Выбирает engine: `--engine auto|mlx|gigaam|openai`.
  - Для `auto` использует порядок `mlx`, `gigaam`, `openai`.
  - Строит sandbox task contracts с expected transcript/audit/log paths, resource limits и final preflight requirements.
  - Блокирует unsafe paths, `stable_runtime`, duplicate sandbox paths и unexpected execution/write flags.
- `scripts/mango_office_asr_worker_sandbox_contract.py`
  - CLI для генерации Stage 21 contract/audit JSON.
- `tests/test_productization_asr_worker_sandbox_execution_contract.py`
  - Проверяет happy path.
  - Проверяет auto engine selection.
  - Проверяет блокировку неготового preferred engine.
  - Проверяет блокировку unsafe readiness.
  - Проверяет path guards.
  - Проверяет CLI audit output.
- `src/mango_mvp/productization/__init__.py`
  - Экспортирует `AsrWorkerSandboxExecutionContractSummary` и `build_asr_worker_sandbox_execution_contract`.

## Реальный прогон

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 scripts/mango_office_asr_worker_sandbox_contract.py
```

Основные артефакты:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_contract_stage21/asr_worker_sandbox_contract_stage21_audit.json`
- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_contract_stage21/asr_worker_sandbox_execution_contract_stage21.json`

Результат:

- `validation_ok=true`
- `selected_engine=mlx`
- `engine_selection_reason=auto_preference_order`
- `action_counts={"PLAN_ASR_SANDBOX_TASK": 21}`
- `source_items=21`
- `tasks=21`
- `blocked_tasks=0`
- `total_duration_sec=1676.592`
- `total_audio_bytes=6706368`
- `estimated_tmp_bytes=20807232`
- `estimated_timeout_sec=7639`
- `max_single_timeout_sec=1279`
- `size_class_counts={"long": 1, "medium": 7, "short": 13}`

Input SHAs:

- Stage 20 readiness report:
  - `26881be4530b9e9429b62c6f0f66c74e8ff1f8ccad071ec098ad8716c2c12c7c`
- Stage 19 worker plan:
  - `106a8d9082b086acda33713db75ce02d36f569a7f630a871e55c5a345a1066cc`

Stage 21 contract SHA:

- `984320bbc5370b5cea90e400426b3c48cf43397a42b205c5bbc63c547e46f204`

## Sandbox paths

Контракт содержит будущие paths:

- `sandbox_outputs`
- `sandbox_tmp`

В этом этапе они только записаны строками в JSON. Проверка `find` показала, что физически создана только директория Stage 21 с JSON-файлами; `sandbox_outputs/` и `sandbox_tmp/` не создавались.

Для каждого task есть:

- `transcript_json`
- `transcript_txt`
- `asr_audit_json`
- `engine_stdout_log`
- `engine_stderr_log`
- `task_tmp_dir`

Task-level preflight flags:

- `audio_file_exists=true`
- `audio_file_size_matches_plan=true`
- `sandbox_paths_under_output_root=true`
- `no_stable_runtime_paths=true`
- `preflight_must_verify_sha256_before_execution=true`

Stage 21 проверяет existence/size, но не читает аудио для SHA verification. Это оставлено на final preflight перед фактическим execution.

## Safety boundary

В contract hard guards:

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

- `creates_sandbox_output_dirs=false`
- `creates_sandbox_tmp_dirs=false`
- `imports_asr_modules=false`
- `loads_models=false`
- `downloads_audio=false`
- `copies_audio=false`
- `hardlinks_audio=false`
- `runtime_db_writes=false`
- `stable_runtime_writes=false`

## Идемпотентность

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 scripts/mango_office_asr_worker_sandbox_contract.py --idempotency-out
```

Idempotency audit:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_contract_stage21/asr_worker_sandbox_contract_stage21_idempotency_audit.json`

Повторный прогон выдал тот же contract SHA:

- `984320bbc5370b5cea90e400426b3c48cf43397a42b205c5bbc63c547e46f204`

## Product DB integrity

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 scripts/mango_office_product_db_admin.py integrity \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_contract_stage21/product_db_integrity_stage21_audit.json
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
  src/mango_mvp/productization/asr_worker_sandbox_execution_contract.py \
  scripts/mango_office_asr_worker_sandbox_contract.py \
  src/mango_mvp/productization/__init__.py
```

Результат: успешно.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 -m pytest -q \
  tests/test_productization_asr_worker_sandbox_execution_contract.py \
  tests/test_productization_asr_worker_sandbox_readiness.py \
  tests/test_productization_asr_worker_execution_dry_run.py
```

Результат: `16 passed`.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 -m pytest -q \
  tests/test_productization_asr_worker_sandbox_execution_contract.py \
  tests/test_productization_asr_worker_sandbox_readiness.py \
  tests/test_productization_asr_worker_execution_dry_run.py \
  tests/test_productization_asr_execution_plan.py
```

Результат: `22 passed`.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 -m pytest -q tests/test_productization_*.py
```

Результат: `205 passed, 1 warning`.

Warning внешний: `urllib3` сообщает, что локальный Python собран с `LibreSSL 2.8.3`, а `urllib3 v2` предпочитает OpenSSL 1.1.1+.

## Аудит

Stage 21 достиг потолка полезности для contract-only слоя:

- Engine выбран явно и воспроизводимо: `mlx`.
- Каждый из 21 worker envelope превращен в sandbox task.
- Все task paths находятся внутри Stage 21 product appliance output root.
- Output/tmp директории не созданы физически.
- Фактический ASR execution невозможен из-за hard guards и пустого command envelope.
- Product DB integrity зеленый.
- Полный productization test suite зеленый.

## Следующий безопасный шаг

Stage 22: ASR sandbox final preflight verifier.

Нужно прочитать Stage 21 contract и проверить фактические условия перед возможным будущим запуском: SHA256 всех audio files, отсутствие existing output collisions, доступность места на диске, доступность выбранного engine capability без загрузки модели, возможность создать sandbox dirs. Stage 22 должен остаться dry-run/preflight only: без ASR, без transcript writes, без runtime DB, без CRM.
