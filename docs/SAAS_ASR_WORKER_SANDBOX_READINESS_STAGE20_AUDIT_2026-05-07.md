# SaaS ASR Worker Sandbox Readiness Stage 20 Audit

Дата: 2026-05-07

## Цель

Stage 20 добавляет readiness gate для будущего ASR worker sandbox. Шаг читает Stage 19 worker dry-run plan, проверяет контракт безопасности и capability surface для доступных ASR engine, затем пишет JSON-отчет готовности.

Этот этап не запускает ASR, не импортирует ASR engine-модули через runtime-код, не загружает модели, не пишет транскрипты, не пишет runtime DB, не пишет CRM/Tallanto и не трогает `stable_runtime`.

## Что добавлено

- `src/mango_mvp/productization/asr_worker_sandbox_readiness.py`
  - Валидирует Stage 19 worker plan перед sandbox-слоем.
  - Проверяет, что worker envelopes остаются dry-run only: без executable, argv, dispatch, ASR, transcript/runtime/CRM writes.
  - Строит read-only capability report по engine-кандидатам `mock`, `mlx`, `gigaam`, `openai`.
  - Для Python-модулей использует только `importlib.util.find_spec`, без импорта ASR engine-кода и загрузки моделей.
  - Для бинарей проверяет только путь через `shutil.which`.
  - Редактируемые секреты не выводит: `OPENAI_API_KEY` отражается только boolean-флагом `openai_api_key_present`.
- `scripts/mango_office_asr_worker_sandbox_readiness.py`
  - CLI для генерации Stage 20 readiness report и audit JSON.
  - По умолчанию читает Stage 19 plan из product appliance архива.
- `tests/test_productization_asr_worker_sandbox_readiness.py`
  - Проверяет готовый sandbox при найденном real engine.
  - Проверяет блокировку при отсутствии real engine без провала source validation.
  - Проверяет блокировку unsafe worker plan.
  - Проверяет path guards против `stable_runtime` и output за пределами product root.
  - Проверяет CLI audit output.
- `src/mango_mvp/productization/__init__.py`
  - Экспортирует `AsrWorkerSandboxReadinessSummary` и `build_asr_worker_sandbox_readiness`.

## Реальный прогон

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 scripts/mango_office_asr_worker_sandbox_readiness.py
```

Основной audit:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_readiness_stage20/asr_worker_sandbox_readiness_stage20_audit.json`
- readiness report:
  - `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_readiness_stage20/asr_worker_sandbox_readiness_report_stage20.json`

Результат:

- `validation_ok=true`
- `source_plan_valid=true`
- `worker_sandbox_ready=true`
- `asr_engine_ready=true`
- `action_counts={"PLAN_ASR_WORKER_SANDBOX_READY_DRY_RUN": 1}`
- `source_items=21`
- `envelopes=21`
- `blocked_envelopes=0`
- `total_duration_sec=1676.592`
- `total_audio_bytes=6706368`
- Stage 19 worker plan SHA:
  - `106a8d9082b086acda33713db75ce02d36f569a7f630a871e55c5a345a1066cc`
- Stage 20 readiness report SHA:
  - `26881be4530b9e9429b62c6f0f66c74e8ff1f8ccad071ec098ad8716c2c12c7c`

## ASR capability report

Read-only проверка нашла:

- `mock`: ready, но `counts_as_real_asr=false`; только для тестов.
- `mlx`: ready, `counts_as_real_asr=true`.
- `gigaam`: ready, `counts_as_real_asr=true`, `ffmpeg=/opt/homebrew/bin/ffmpeg`.
- `openai`: module available, но `ready=false`, потому что `OPENAI_API_KEY` не задан.

Важно: наличие `mlx` и `gigaam` означает только готовность к следующему sandbox-planning этапу. Stage 20 не выбирает engine для execution и не запускает распознавание.

## Safety boundary

В readiness report hard guards:

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
python3 scripts/mango_office_asr_worker_sandbox_readiness.py --idempotency-out
```

Idempotency audit:

- `_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_readiness_stage20/asr_worker_sandbox_readiness_stage20_idempotency_audit.json`

Повторный прогон выдал тот же readiness SHA:

- `26881be4530b9e9429b62c6f0f66c74e8ff1f8ccad071ec098ad8716c2c12c7c`

## Product DB integrity

Команда:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 scripts/mango_office_product_db_admin.py integrity \
  --out _local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_sandbox_readiness_stage20/product_db_integrity_stage20_audit.json
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
  src/mango_mvp/productization/asr_worker_sandbox_readiness.py \
  scripts/mango_office_asr_worker_sandbox_readiness.py \
  src/mango_mvp/productization/__init__.py
```

Результат: успешно.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 -m pytest -q \
  tests/test_productization_asr_worker_sandbox_readiness.py \
  tests/test_productization_asr_worker_execution_dry_run.py
```

Результат: `10 passed`.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 -m pytest -q \
  tests/test_productization_asr_worker_sandbox_readiness.py \
  tests/test_productization_asr_worker_execution_dry_run.py \
  tests/test_productization_asr_execution_plan.py
```

Результат: `16 passed`.

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src \
python3 -m pytest -q tests/test_productization_*.py
```

Результат: `199 passed, 1 warning`.

Warning внешний: `urllib3` сообщает, что локальный Python собран с `LibreSSL 2.8.3`, а `urllib3 v2` предпочитает OpenSSL 1.1.1+.

## Аудит

Stage 20 достиг потолка полезности для безопасного dry-run слоя:

- Контракт Stage 19 проверяется до допуска sandbox planning.
- Все dangerous flags явно заблокированы.
- Доступность ASR engine фиксируется без запуска engine.
- Секреты не выводятся.
- Артефакты идемпотентны.
- Product DB integrity зеленый.
- Полный productization test suite зеленый.

## Следующий безопасный шаг

Stage 21: ASR worker sandbox execution contract.

Нужно сделать отдельный planning layer, который выберет конкретный engine (`mlx` или `gigaam`) и построит sandbox execution contract: где будут лежать временные outputs, какие resource limits применяются, как проверить expected transcript filenames, как остановиться перед фактическим запуском. Stage 21 по-прежнему должен быть dry-run only: без ASR, без transcript writes, без runtime DB, без CRM.
