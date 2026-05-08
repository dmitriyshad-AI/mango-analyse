# Mango provider metadata sidecar audit

Date: 2026-05-07

Goal: add a SaaS/productization sidecar table for Mango provenance in the
disposable quarantine test DB, without changing `call_records` and without
touching runtime data.

## Safety boundaries

The command writes only to the disposable DB:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/quarantine_test_ingest.sqlite
```

The command does not:

- write to `mango_mvp.db`;
- write to `stable_runtime`;
- copy files into `2026-03-09--26`;
- start ASR/R+A;
- write to AMO/Tallanto;
- change batch/start/run-ui scripts.

The runner refuses runtime-looking DB names, paths under `stable_runtime`, and
paths outside the allowed test-ingest root.

## Code added

- `src/mango_mvp/productization/provider_metadata.py`
- `scripts/mango_office_provider_metadata_sidecar.py`
- `tests/test_productization_provider_metadata.py`
- `tests/test_productization_provider_metadata_script.py`

## Table

Table name:

```text
provider_call_metadata
```

Core columns:

```text
call_record_id
source_filename
source_file
tenant_id
provider
provider_call_id
recording_id
event_key
checksum_sha256
source_size_bytes
manager_extension
raw_payload_ref
target_audio_path
source_audio_path
created_at
updated_at
```

Uniqueness constraints:

```text
UNIQUE(source_filename)
UNIQUE(tenant_id, provider, provider_call_id)
UNIQUE(tenant_id, provider, recording_id)
UNIQUE(tenant_id, provider, event_key)
```

These are the SaaS-relevant dedupe/audit keys that `call_records` does not
currently preserve.

## Command

Clean sidecar install:

```zsh
PYTHONPATH=src python3 scripts/mango_office_provider_metadata_sidecar.py \
  --db _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/quarantine_test_ingest.sqlite \
  --metadata-csv _local_archive_mango_api_downloads_20260507/quarantine_import/metadata.csv \
  --out-root _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest \
  --out _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/provider_metadata_audit.json \
  --replace
```

Idempotency run:

```zsh
PYTHONPATH=src python3 scripts/mango_office_provider_metadata_sidecar.py \
  --db _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/quarantine_test_ingest.sqlite \
  --metadata-csv _local_archive_mango_api_downloads_20260507/quarantine_import/metadata.csv \
  --out-root _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest \
  --out _local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/provider_metadata_idempotency_audit.json
```

## Outputs

Clean audit:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/provider_metadata_audit.json
```

Idempotency audit:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/test_ingest/provider_metadata_idempotency_audit.json
```

## Result

Clean install:

```text
validation_ok = true
metadata_rows = 297
call_records = 297
sidecar_rows = 297
inserted = 297
updated = 0
blocked = 0
warnings = 297
```

Idempotency run:

```text
validation_ok = true
sidecar_rows = 297
inserted = 0
updated = 297
blocked = 0
warnings = 297
```

Unique key audit:

```text
provider_call_keys = 297
recording_keys = 297
event_keys = 297
```

Tenant/provider:

```text
foton|mango = 297
```

Manager extension distribution:

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

## Known gaps

Warnings are expected at this stage:

```text
raw_payload_ref_missing = 297
```

This means the sidecar has normalized Mango identifiers, recording IDs,
checksums, source sizes and manager extensions, but it does not yet link each
row to an archived raw Mango stats payload. The next productization slice should
persist raw poll payloads under a tenant/provider/date path and store
`raw_payload_ref` in this sidecar.

Second known gap: `manager_extension` is still a Mango extension ref, not a
human user mapping. Customer-facing analytics needs an extension-to-user mapping
layer.

## Test gate

Focused gate:

```zsh
PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_provider_metadata.py \
  tests/test_productization_provider_metadata_script.py
```

Result:

```text
5 passed, 1 warning
```

Full productization gate:

```zsh
PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Result:

```text
69 passed, 1 warning
```

Warning: local Python LibreSSL/urllib3 warning. Not blocking for provider
metadata sidecar.

## Next recommended step

Add raw payload archival for Mango shadow poll:

- store normalized/raw stats rows as JSONL under an isolated archive path;
- reference each row from `provider_call_metadata.raw_payload_ref`;
- keep downloads, DB ingest and ASR as separate explicit gates.

That closes the current provenance gap needed for SaaS auditability.
