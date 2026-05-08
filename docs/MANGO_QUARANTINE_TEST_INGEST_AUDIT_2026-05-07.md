# Mango quarantine test ingest audit

Date: 2026-05-07

Goal: validate that the materialized Mango quarantine package can enter the
current `call_records` model in a disposable SQLite DB, without touching the
runtime DB and without starting ASR/R+A.

## Safety boundaries

The command writes only under:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest
```

The command does not:

- write to `mango_mvp.db`;
- write to `stable_runtime`;
- copy files into `2026-03-09--26`;
- start ASR/R+A;
- write to AMO/Tallanto;
- change batch/start/run-ui scripts.

The runner refuses:

- runtime-looking DB names such as `mango_mvp.db`;
- DB paths outside the allowed output root;
- DB paths under `stable_runtime`.

## Code added

- `src/mango_mvp/productization/test_ingest.py`
- `scripts/mango_office_quarantine_test_ingest.py`
- `tests/test_productization_test_ingest.py`
- `tests/test_productization_quarantine_test_ingest_script.py`

The runner wraps the existing `mango_mvp.services.ingest.ingest_from_directory`
without changing it. This keeps the check representative of the current
pipeline model while keeping all writes in a disposable DB.

## Command

Clean disposable DB run:

```zsh
PYTHONPATH=src python3 scripts/mango_office_quarantine_test_ingest.py \
  --audio-dir _local_archive_mango_api_downloads_20260507/quarantine_import/audio \
  --metadata-csv _local_archive_mango_api_downloads_20260507/quarantine_import/metadata.csv \
  --db _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/quarantine_test_ingest.sqlite \
  --out-root _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest \
  --out _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/test_ingest_audit.json \
  --replace
```

Idempotency run on the same disposable DB:

```zsh
PYTHONPATH=src python3 scripts/mango_office_quarantine_test_ingest.py \
  --audio-dir _local_archive_mango_api_downloads_20260507/quarantine_import/audio \
  --metadata-csv _local_archive_mango_api_downloads_20260507/quarantine_import/metadata.csv \
  --db _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/quarantine_test_ingest.sqlite \
  --out-root _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest \
  --out _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/test_ingest_idempotency_audit.json \
  --allow-existing
```

## Outputs

Disposable DB:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/quarantine_test_ingest.sqlite
```

Clean audit:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/test_ingest_audit.json
```

Idempotency audit:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/test_ingest_idempotency_audit.json
```

## Clean ingest result

```text
validation_ok = true
metadata_rows = 297
audio_files = 297
ingest_processed = 297
ingest_inserted = 297
ingest_skipped = 0
db_call_records = 297
blocked = 0
warnings = 0
```

All inserted rows are still unprocessed:

```text
transcription_status.pending = 297
resolve_status.pending = 297
analysis_status.pending = 297
sync_status.pending = 297
```

Direction distribution:

```text
inbound = 72
outbound = 225
```

Manager reference distribution:

```text
26 = 88
23 = 76
19 = 58
202 = 35
156 = 22
105 = 16
335 = 1
387 = 1
```

DB date range:

```text
min(started_at) = 2026-05-06 09:01:23
max(started_at) = 2026-05-07 15:51:18
```

## Idempotency result

Second run against the same disposable DB:

```text
validation_ok = true
ingest_processed = 297
ingest_inserted = 0
ingest_skipped = 297
db_call_records = 297
blocked = 0
warnings = 0
```

## Compatibility findings

The current `call_records` model can ingest the package cleanly with:

- `source_file`
- `source_filename`
- `source_call_id`
- `phone`
- `manager_name`
- `direction`
- `started_at`
- audio probe fields: `audio_codec`, `sample_rate`, `channels`,
  `duration_sec`

Model gaps for SaaS/productization:

- `recording_id` is available in `metadata.csv` but is not stored in
  `call_records`;
- `event_key` is available in `metadata.csv` but is not stored in
  `call_records`;
- `checksum_sha256` is available in `metadata.csv` but is not stored in
  `call_records`;
- `tenant_id/provider` are available in `metadata.csv` but are not stored in
  `call_records`;
- `source_size_bytes` is available in `metadata.csv` but is not stored in
  `call_records`;
- Mango manager values are extension refs, not human manager names. The
  product layer needs an extension-to-user mapping before this becomes
  customer-facing analytics.

## Test gate

Focused gate:

```zsh
PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_test_ingest.py \
  tests/test_productization_quarantine_test_ingest_script.py
```

Result:

```text
7 passed, 1 warning
```

Full productization gate:

```zsh
PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Result:

```text
64 passed, 1 warning
```

Warning: local Python LibreSSL/urllib3 warning. Not blocking for disposable
test ingest.

## Next recommended step

Do not import into runtime DB yet. The next productization step should define a
small provider metadata sidecar contract/table for the SaaS layer:

- `tenant_id`
- `provider`
- `provider_call_id`
- `recording_id`
- `event_key`
- `checksum_sha256`
- `manager_extension`
- `raw_payload_ref`

This can be tested against the disposable DB first. It avoids overloading
`call_records` while preserving Mango provenance needed for dedupe, audit, and
multi-tenant operation.
