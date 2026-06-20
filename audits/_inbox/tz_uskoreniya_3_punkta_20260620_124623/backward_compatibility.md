# Backward Compatibility

- Existing `AnalyzeService.migrate_analysis_payload(call, payload)` remains available.
- `migrate-analysis-schema` default `--workers 1` preserves sequential behavior.
- Canonical import source order is unchanged.
- Runtime dependencies are unchanged; `pytest-xdist` is dev-only.

