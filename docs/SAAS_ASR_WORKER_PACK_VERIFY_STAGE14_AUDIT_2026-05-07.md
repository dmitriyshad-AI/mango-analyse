# SaaS Stage 14: ASR Worker Pack Verify Audit

Дата: 2026-05-07

## Цель этапа

Stage 14 добавляет read-only readiness gate для переносимого ASR worker pack:

```text
asr_worker_pack_stage13/ -> verify/readiness audit
```

Граница этапа:

- читаем только pack manifest и audio files;
- пишем только verify audit JSON под `product_appliance`;
- не копируем audio;
- не скачиваем audio;
- не запускаем ASR;
- не запускаем R+A;
- не пишем runtime DB;
- не трогаем `stable_runtime`;
- не пишем AMO/Tallanto/CRM.

## Что добавлено

- `src/mango_mvp/productization/asr_worker_pack_verifier.py`
  - проверяет `asr_worker_pack_v1` manifest;
  - проверяет `queue_status=ready_for_asr`;
  - проверяет relative `audio_rel_path`;
  - блокирует path traversal;
  - проверяет relative `planned_outputs_rel`;
  - проверяет audio exists, file, non-zero, size, sha256;
  - ловит duplicate `queue_item_id` и `audio_rel_path`;
  - ловит unreferenced audio files;
  - блокирует любые `stable_runtime` references;
  - возвращает readiness gate, но не разрешает запуск ASR без отдельного runtime approval.
- `scripts/mango_office_asr_worker_pack_verify.py`
  - CLI для read-only verify.
- `tests/test_productization_asr_worker_pack_verifier.py`
  - clean pack accepted;
  - idempotent verification;
  - checksum mismatch block;
  - missing audio block;
  - path traversal block;
  - outside/stable path guards;
  - CLI audit.

## Safety Contract

Stage 14 reports:

- `read_only`: true;
- `product_db_writes`: false;
- `asset_db_writes`: false;
- `runtime_db_writes`: false;
- `stable_runtime_writes`: false;
- `downloads_audio`: false;
- `copies_audio`: false;
- `hardlinks_audio`: false;
- `run_asr`: false;
- `run_ra`: false;
- `write_crm`: false;
- `write_tallanto`: false.

## Real Verify Run

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_asr_worker_pack_verify.py
```

Result:

- `manifest_rows`: 21
- `ready_items`: 21
- `pack_audio_files`: 21
- `pack_total_bytes`: 6,706,368
- `blocked`: 0
- `warnings`: 0
- `validation_ok`: true
- `manifest_sha256`: `3311cce58b1fe956d5aa74d579ad229631b09d74e8089ace17a532ae1a51d4e3`

Readiness gate:

- `ready_for_worker`: true
- `worker_may_run_asr`: false
- `requires_explicit_runtime_target_approval`: true

Verified checks:

- manifest schema;
- queue status;
- relative audio paths;
- relative planned output paths;
- audio file exists;
- audio size;
- audio sha256;
- duplicate queue ids;
- unreferenced audio files.

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_pack_stage13/asr_worker_pack_verify_stage14_audit.json
```

## Idempotency Verify

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_asr_worker_pack_verify.py --idempotency-out
```

Result:

- `manifest_rows`: 21
- `ready_items`: 21
- `blocked`: 0
- `warnings`: 0
- `validation_ok`: true
- `manifest_sha256`: `3311cce58b1fe956d5aa74d579ad229631b09d74e8089ace17a532ae1a51d4e3`

Output:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_pack_stage13/asr_worker_pack_verify_stage14_idempotency_audit.json
```

## Product DB Integrity

Main product appliance DB was checked after Stage 14. Stage 14 did not write to it.

Result:

- `validation_ok`: true
- `blocked`: 0
- `capture_inbox_items`: 21
- `capture_inbox_ready`: 21
- `capture_inbox_blocked`: 0
- `product_calls`: 297
- `job_runs`: 5
- `schema_migrations`: 4
- known warnings: 3 pending owner mappings

## Tests

Focused gate:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_asr_worker_pack_verifier.py \
  tests/test_productization_asr_worker_pack.py
```

Result: `12 passed`.

Full productization gate:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Result: `165 passed, 1 warning`.

The warning is the existing external urllib3/LibreSSL warning.

## Вывод

Stage 14 закрыт. Переносимый ASR worker pack на 21 Mango recording прошел read-only readiness gate и готов как артефакт для будущего worker, но запуск ASR все еще заблокирован до отдельного explicit runtime approval.

Следующий безопасный этап: добавить product appliance job plan для `asr_worker_pack_verify -> ASR execution approval gate`, пока только как dry-run job definition/status, без запуска ASR.
