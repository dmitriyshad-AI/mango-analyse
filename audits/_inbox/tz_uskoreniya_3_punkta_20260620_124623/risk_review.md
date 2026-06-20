# Risk Review

- No AMO/CRM/Tallanto writes were run.
- No repository `stable_runtime` writes were made.
- No ASR, Resolve+Analyze real-data run, LLM, or network API was invoked.
- `pytest-xdist` is not enabled in `addopts`; parallel pytest remains explicit.
- `bulk_write` intentionally rolls back domain writes on failure while preserving failed ingestion run outside the batch.
- `migrate-analysis-schema --workers N` parallelizes compute only; DB and export writes remain single-process.

