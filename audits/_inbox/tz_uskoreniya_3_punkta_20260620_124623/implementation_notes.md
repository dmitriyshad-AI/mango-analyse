# Implementation Notes

Branch: `codex/tz-uskoreniya-3-punkta`

- Added `pytest-xdist` to dev extras and `uv.lock`.
- Wrapped canonical readonly timeline write loop in `CustomerTimelineSQLiteStore.bulk_write()`.
- Added logical table hash tests for canonical import determinism.
- Added top-level analysis payload migration function and CLI `--workers`.
- Kept DB writes and file exports single-process in `migrate-analysis-schema`.

