# SaaS ASR Worker Sandbox Preflight Stage 22 Audit

Дата: 2026-05-08

## Цель

Stage 22 выполняет final preflight для Stage 21 ASR worker sandbox contract. Он проверяет фактическую готовность sandbox-контракта перед возможным будущим запуском: SHA256 аудио, output collisions, disk space, writable parent directories и capability выбранного ASR engine.

Этот этап не запускает ASR, не dispatch-ит worker, не импортирует ASR engine-модули, не загружает модели, не создает `sandbox_outputs/` или `sandbox_tmp/`, не пишет транскрипты, не пишет runtime DB, не пишет CRM/Tallanto и не трогает `stable_runtime`.

## Что добавлено

- `src/mango_mvp/productization/asr_worker_sandbox_preflight.py`
  - Валидирует Stage 21 contract.
  - Проверяет все task command guards: `executable=None`, `argv=[]`, `run_asr=false`, `write_outputs=false`.
  - Пересчитывает SHA256 всех audio files.
  - Проверяет output collisions для planned sandbox output paths.
  - Проверяет selected engine через `importlib.util.find_spec`, без импорта engine-кода и без model load.
  - Проверяет disk space с 64 MiB safety margin и bucketed free-space value для воспроизводимого отчета.
  - Проверяет, что parent directories для будущих sandbox roots writable, но сами sandbox dirs не создает.
- `scripts/mango_office_asr_worker_sandbox_preflight.py`
  - CLI для генерации Stage 22 preflight/audit JSON.
- `tests/test_productization_asr_worker_sandbox_preflight.py`
  - Happy path без execution.
  - SHA mismatch block.
  - Output collision block.
  - Missing engine block.
  - Insufficient disk block.
  - Path guards.
  - CLI audit output.
- `src/mango_mvp/productization/__init__.py`
  - Экспортирует `AsrWorkerSandboxPreflightSummary` и `build_asr_worker_sandbox_preflight`.

## Реальный прогон

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 scripts/mango_office_asr_worker_sandbox_preflight.py
```

Основные артефакты:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_preflight_stage22/asr_worker_sandbox_preflight_stage22_audit.json`
- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_preflight_stage22/asr_worker_sandbox_preflight_report_stage22.json`

Результат:

- `validation_ok=true`
- `preflight_ready=true`
- `selected_engine=mlx`
- `action_counts={"PASS_ASR_SANDBOX_PREFLIGHT_TASK": 21}`
- `tasks=21`
- `passed_tasks=21`
- `blocked_tasks=0`
- `audio_files_checked=21`
- `audio_sha_ok=21`
- `output_collisions=0`
- `engine_preflight_ok=true`
- `disk_space_ok=true`
- `dir_preflight_ok=true`
- `required_free_bytes=94622464`
- `available_free_bytes=282259881984`

Input Stage 21 contract SHA:

- `984320bbc5370b5cea90e400426b3c48cf43397a42b205c5bbc63c547e46f204`

Stage 22 preflight report SHA:

- `d90a5c883a3fd6922674afc733d7704a1bf6936c07f5b850efc8c001d83732f6`

## Проверки preflight

Audio:

- Проверено `21` audio files.
- Все `21` SHA256 совпали с Stage 21 contract.
- Все sizes совпали.

Output collisions:

- `0` collisions.
- Planned transcript/audit/log files отсутствуют.

Engine:

- `selected_engine=mlx`
- `module=mlx_whisper`
- `module_available=true`
- `checks_import_specs_only=true`
- `loads_models=false`
- `runs_asr=false`

Disk:

- Требуется `94622464` bytes.
- Доступно bucketed `282259881984` bytes.
- Проверка места: ok.

Directories:

- `sandbox_outputs/` не существует.
- `sandbox_tmp/` не существует.
- Parent directory writable.
- `creates_dirs=false`.

## Safety boundary

В preflight report:

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

- `reads_audio_for_sha256=true`
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
python3 scripts/mango_office_asr_worker_sandbox_preflight.py --idempotency-out
```

Idempotency audit:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_preflight_stage22/asr_worker_sandbox_preflight_stage22_idempotency_audit.json`

Повторный прогон выдал тот же preflight report SHA:

- `d90a5c883a3fd6922674afc733d7704a1bf6936c07f5b850efc8c001d83732f6`

## Product DB integrity

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 scripts/mango_office_product_db_admin.py integrity \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_preflight_stage22/product_db_integrity_stage22_audit.json
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
  src/mango_mvp/productization/asr_worker_sandbox_preflight.py \
  scripts/mango_office_asr_worker_sandbox_preflight.py \
  src/mango_mvp/productization/__init__.py
```

Результат: успешно.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 -m pytest -q \
  tests/test_productization_asr_worker_sandbox_preflight.py \
  tests/test_productization_asr_worker_sandbox_execution_contract.py \
  tests/test_productization_asr_worker_sandbox_readiness.py
```

Результат: `18 passed`.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 -m pytest -q \
  tests/test_productization_asr_worker_sandbox_preflight.py \
  tests/test_productization_asr_worker_sandbox_execution_contract.py \
  tests/test_productization_asr_worker_sandbox_readiness.py \
  tests/test_productization_asr_worker_execution_dry_run.py \
  tests/test_productization_asr_execution_plan.py
```

Результат: `29 passed`.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 -m pytest -q tests/test_productization_*.py
```

Результат: `212 passed, 1 warning`.

Warning внешний: `urllib3` сообщает, что локальный Python собран с `LibreSSL 2.8.3`, а `urllib3 v2` предпочитает OpenSSL 1.1.1+.

## Аудит

Stage 22 достиг потолка полезности для preflight-only слоя:

- Контракт Stage 21 проверен.
- Audio SHA256 реально пересчитаны.
- Output collisions отсутствуют.
- Доступность места и parent dirs подтверждена.
- Выбранный engine `mlx` подтвержден без загрузки модели.
- `sandbox_outputs/` и `sandbox_tmp/` не созданы.
- Runtime DB, `stable_runtime`, CRM/Tallanto не затронуты.
- Product DB integrity зеленый.
- Полный productization test suite зеленый.

## Следующий безопасный шаг

Stage 23: explicit ASR sandbox execution approval packet.

Нужно подготовить approval packet для человека: точный список 21 аудио, выбранный engine `mlx`, expected outputs, resource limits, SHA preflight result и явные acknowledgement flags. Этот этап должен по-прежнему быть approval-only: без ASR, без создания sandbox dirs, без transcript writes, без runtime DB и без CRM.
