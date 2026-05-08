# SaaS Stage 13: ASR Worker Pack Audit

Дата: 2026-05-07

## Цель этапа

Stage 13 собирает переносимый dry-run input pack для будущего ASR worker:

```text
asr_handoff_manifest_stage12.jsonl -> asr_worker_pack_stage13/
```

Граница этапа:

- читаем Stage 12 handoff manifest;
- копируем audio только внутри `product_appliance/asr_worker_pack_stage13/audio`;
- пишем только pack manifest и audit JSON под `product_appliance`;
- не запускаем ASR;
- не запускаем R+A;
- не пишем runtime DB;
- не трогаем `stable_runtime`;
- не пишем AMO/Tallanto/CRM.

## Что добавлено

- `src/mango_mvp/productization/asr_worker_pack.py`
  - читает `ready_for_asr` JSONL manifest;
  - проверяет source audio path под `product_appliance`;
  - блокирует `stable_runtime` references;
  - проверяет `.mp3`, file existence, zero-size, checksum;
  - строит deterministic worker audio names;
  - копирует или hardlink-ит audio в pack root;
  - пишет portable worker manifest с relative paths;
  - проверяет idempotency через `SKIP_ALREADY_PACKED`;
  - не запускает ASR и не пишет DB.
- `scripts/mango_office_asr_worker_pack.py`
  - CLI для Stage 13 pack build.
- `tests/test_productization_asr_worker_pack.py`
  - copy + idempotency;
  - dry-run without audio copy;
  - checksum mismatch block;
  - outside/stable path guards;
  - CLI audit.

## Safety Contract

Initial materialization reports:

- `copies_audio`: true;
- `hardlinks_audio`: false;
- `product_db_writes`: false;
- `asset_db_writes`: false;
- `runtime_db_writes`: false;
- `stable_runtime_writes`: false;
- `downloads_audio`: false;
- `run_asr`: false;
- `run_ra`: false;
- `write_crm`: false;
- `write_tallanto`: false.

Idempotency run reports `copies_audio: false` because all 21 files were already present.

## Real Pack Build

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_asr_worker_pack.py
```

Result:

- `source_manifest_rows`: 21
- `selected_items`: 21
- `PACK_ASR_WORKER_ITEM`: 21
- `manifest_rows`: 21
- `copied`: 21
- `pack_audio_files`: 21
- `pack_total_bytes`: 6,706,368
- `blocked`: 0
- `warnings`: 0
- `validation_ok`: true
- `manifest_sha256`: `3311cce58b1fe956d5aa74d579ad229631b09d74e8089ace17a532ae1a51d4e3`

Pack root:

```text
_local_archive_mango_api_downloads_20260507/product_appliance/asr_worker_pack_stage13
```

Key files:

```text
asr_worker_input_manifest_stage13.jsonl
asr_worker_pack_stage13_audit.json
audio/*.mp3
```

## Idempotency Run

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 scripts/mango_office_asr_worker_pack.py --idempotency-out
```

Result:

- `SKIP_ALREADY_PACKED`: 21
- `copied`: 0
- `already_present`: 21
- `manifest_rows`: 21
- `pack_audio_files`: 21
- `pack_total_bytes`: 6,706,368
- `blocked`: 0
- `warnings`: 0
- `validation_ok`: true
- `manifest_sha256`: `3311cce58b1fe956d5aa74d579ad229631b09d74e8089ace17a532ae1a51d4e3`

## Worker Manifest Contract

Each worker manifest row uses relative pack paths:

- `audio_rel_path`: `audio/<queue-prefix>__<source-filename>.mp3`
- `planned_outputs_rel.transcript_json`
- `planned_outputs_rel.transcript_txt`
- `planned_outputs_rel.asr_audit_json`

Worker must verify before any real processing:

- `audio_rel_path` exists inside pack;
- `audio_sha256` matches;
- runtime target approval is explicit.

Pack build itself must not:

- run ASR;
- write runtime DB;
- write CRM.

## Product DB Integrity

Main product appliance DB was checked after Stage 13. Stage 13 did not write to it.

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
  tests/test_productization_asr_worker_pack.py \
  tests/test_productization_processing_handoff.py
```

Result: `11 passed`.

Full productization gate:

```bash
PYTHONPYCACHEPREFIX=/tmp/mango_pycache PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Result: `158 passed, 1 warning`.

The warning is the existing external urllib3/LibreSSL warning.

## Вывод

Stage 13 закрыт. У нас есть переносимый ASR worker input pack на 21 Mango recording, но ASR/R+A по-прежнему не запускается и runtime DB не меняется.

Следующий безопасный этап: добавить read-only pack verifier/worker readiness gate, который проверяет переносимый pack как самостоятельный артефакт перед любым будущим запуском ASR.
