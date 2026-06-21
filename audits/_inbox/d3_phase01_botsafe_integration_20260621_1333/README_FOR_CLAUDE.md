# README For Claude Regread

Scope:

- Verify D3 integration of D8 next-step extractor into bot-safe summaries.
- Verify name scrubbing in `interest/title`.
- Verify stale `unknown` bot-safe chunks are retired after brand re-resolution.
- Verify metrics were run on a fresh test copy, not on production DB.

Primary code diff:

- `src/mango_mvp/customer_timeline/bot_safe_summary.py`
- `tests/test_customer_timeline_bot_safe_summary.py`

Important artifacts:

- `copy_manifest.json` - source/test copy proof.
- `prod_db_unchanged_check.json` - production DB unchanged proof.
- `after_build_report.json` - first final apply on fresh test copy.
- `idempotency_report.json` - second apply on same copy.
- `after_metrics.json` - final content metrics and examples.
- `visibility_checks.json` - runtime active-brand/unknown visibility checks.
- `runtime_pii_scan.json` - runtime PII scanner result.

Expected verdict:

- PASS if no live write path was introduced, production DB unchanged, visible bot-safe chunks have no contact PII/name leaks, and foreign explicit brand chunks are not visible to the active brand.
