# Mango raw payload archive audit

Date: 2026-05-07

Goal: archive raw Mango shadow-poll stats rows and fill
`provider_call_metadata.raw_payload_ref` for the 297 disposable quarantine
records.

## Safety boundaries

This step writes only under:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import
```

It updates only the disposable SQLite DB:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/quarantine_test_ingest.sqlite
```

It does not:

- write to `mango_mvp.db`;
- write to `stable_runtime`;
- copy files into `2026-03-09--26`;
- download audio;
- start ASR/R+A;
- write to AMO/Tallanto;
- change batch/start/run-ui scripts.

## Code added

- `src/mango_mvp/productization/payload_archive.py`
- `scripts/mango_office_payload_archive.py`
- `tests/test_productization_payload_archive.py`
- `tests/test_productization_payload_archive_script.py`

Also updated:

- `scripts/mango_office_shadow_poll.py` now supports `--raw-payload-jsonl`;
- `src/mango_mvp/productization/provider_metadata.py` now preserves existing
  `raw_payload_ref` on idempotent sidecar sync.

## Shadow poll archive

Command:

```zsh
PYTHONPATH=src python3 scripts/mango_office_shadow_poll.py \
  --tenant foton \
  --since 2026-05-06T05:50:00+00:00 \
  --until 2026-05-07T13:10:00+00:00 \
  --allow-metadata-only \
  --raw-payload-jsonl _local_archive_mango_api_downloads_20260507/quarantine_import/raw_payload_archive/shadow_poll_raw_rows_20260506_20260507.jsonl \
  --out _local_archive_mango_api_downloads_20260507/quarantine_import/raw_payload_archive/shadow_poll_report_20260506_20260507.json
```

Result:

```text
source_rows = 665
normalized_events = 665
normalization_errors = 0
raw_payload_rows = 665
metadata_provider_call_ids_matched = 297
metadata_provider_call_ids_missing = 0
```

Raw poll rows:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/raw_payload_archive/shadow_poll_raw_rows_20260506_20260507.jsonl
```

Shadow poll report:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/raw_payload_archive/shadow_poll_report_20260506_20260507.json
```

## Per-call archive

Command:

```zsh
PYTHONPATH=src python3 scripts/mango_office_payload_archive.py \
  --db _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/quarantine_test_ingest.sqlite \
  --metadata-csv _local_archive_mango_api_downloads_20260507/quarantine_import/metadata.csv \
  --source-payload _local_archive_mango_api_downloads_20260507/quarantine_import/raw_payload_archive/shadow_poll_raw_rows_20260506_20260507.jsonl \
  --archive-root _local_archive_mango_api_downloads_20260507/quarantine_import/raw_payload_archive/by_call \
  --out-root _local_archive_mango_api_downloads_20260507/quarantine_import \
  --out _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/payload_archive_audit.json \
  --replace
```

Result:

```text
validation_ok = true
source_payload_rows = 665
metadata_rows = 297
archived_entries = 297
archive_files = 2
archive_file_rows = 297
sidecar_rows = 297
sidecar_refs_updated = 297
sidecar_refs_present = 297
blocked = 0
warnings = 0
```

Archive files:

```text
raw_payload_archive/by_call/tenant=foton/provider=mango/date=2026-05-06/payloads.jsonl = 176
raw_payload_archive/by_call/tenant=foton/provider=mango/date=2026-05-07/payloads.jsonl = 121
```

Sidecar refs:

```text
provider_call_metadata rows = 297
raw_payload_ref present = 297
raw_payload_ref unique = 297
```

## Idempotency

Second payload archive run:

```text
validation_ok = true
archived_entries = 297
sidecar_refs_updated = 297
sidecar_refs_present = 297
blocked = 0
warnings = 0
```

Provider sidecar sync after payload archive:

```text
validation_ok = true
raw_payload_ref_missing = 0
blocked = 0
warnings = 0
known_gaps = manager_extension is not mapped to a human CRM/telephony user yet
```

## Test gate

Focused gate:

```zsh
PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_payload_archive.py \
  tests/test_productization_payload_archive_script.py \
  tests/test_productization_mango_shadow_poll_script.py \
  tests/test_productization_provider_metadata.py
```

Result:

```text
12 passed, 1 warning
```

Full productization gate:

```zsh
PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Result:

```text
76 passed, 1 warning
```

Warning: local Python LibreSSL/urllib3 warning. Not blocking for raw payload
archive.

## Remaining productization gap

Raw Mango poll provenance is now linked per call. The remaining immediate
SaaS gap is manager identity mapping:

- Mango extension refs: `26`, `23`, `19`, `202`, `156`, `105`, `335`, `387`;
- customer-facing analytics needs a table/contract that maps these refs to
  human users and CRM owners.
