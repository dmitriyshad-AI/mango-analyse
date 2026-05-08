# Mango quarantine materialization audit

Date: 2026-05-07

Goal: materialize the validated Mango quarantine import package by copying audio
from the local Mango API archive into the isolated quarantine package directory.

This is still not a runtime ingest. It prepares an inspectable package for the
next gated step.

## Safety boundaries

The materialization command writes only under:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/audio
```

The command does not:

- copy files into `2026-03-09--26`;
- write to `mango_mvp.db`;
- write to `stable_runtime`;
- start ASR/R+A;
- write to AMO/Tallanto;
- change batch/start/run-ui scripts.

Targets outside the quarantine directory are blocked by code.

## Code added

- `materialize_quarantine_package(...)` in `src/mango_mvp/productization/quarantine_import.py`
- `scripts/mango_office_quarantine_materialize.py`
- `tests/test_productization_quarantine_materialize_script.py`

The command is idempotent:

- first run copies missing target files;
- later runs classify valid files as `already_present`;
- existing target checksum mismatches are blocked unless `--overwrite` is
  explicitly passed.

## Command

```zsh
PYTHONPATH=src python3 scripts/mango_office_quarantine_materialize.py \
  --plan _local_archive_mango_api_downloads_20260507/quarantine_import/quarantine_import_plan.json \
  --out _local_archive_mango_api_downloads_20260507/quarantine_import/materialization_audit.json \
  --mode copy
```

## Outputs

Materialized audio:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/audio
```

Primary audit:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/materialization_audit.json
```

Idempotency audit:

```text
_local_archive_mango_api_downloads_20260507/quarantine_import/materialization_idempotency_audit.json
```

## Materialization result

```text
materialize_mode = copy
total_plan_items = 297
ready_plan_items = 297
copied = 297
hardlinked = 0
already_present = 0
blocked = 0
target_audio_files = 297
target_total_mb = 180.74
expected_ready_files = 297
missing_expected_files = 0
checksum_mismatch_files = 0
checksum_verified_files = 297
zero_size_files = 0
unreferenced_audio_files = 0
```

## Idempotency result

Second run:

```text
already_present = 297
copied = 0
blocked = 0
target_audio_files = 297
checksum_verified_files = 297
missing_expected_files = 0
checksum_mismatch_files = 0
zero_size_files = 0
unreferenced_audio_files = 0
```

## Test gate

Focused tests before real materialization:

```zsh
PYTHONPATH=src python3 -m pytest -q \
  tests/test_productization_quarantine_import.py \
  tests/test_productization_quarantine_materialize_script.py
```

Result:

```text
11 passed
```

Full productization gate after materialization:

```zsh
PYTHONPATH=src python3 -m pytest -q tests/test_productization_*.py
```

Result:

```text
57 passed, 1 warning
```

Warning: local Python LibreSSL/urllib3 warning. Not blocking for quarantine
package materialization.

## Next recommended step

The next safe step is a separate test ingest plan against a disposable/test
SQLite DB, not the runtime DB. That step should map `metadata.csv` plus the
materialized audio folder into the expected current pipeline shape, run import
validation, and stop before ASR/R+A.
