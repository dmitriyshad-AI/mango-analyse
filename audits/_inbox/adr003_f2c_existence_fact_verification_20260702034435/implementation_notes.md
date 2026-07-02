# Implementation Notes

## What Changed

- Added `scripts/report_adr003_existence_fact_verification.py`.
- Added `tests/test_report_adr003_existence_fact_verification.py`.
- Added TZ `tasks/_running/2026-07-02_TZ_ADR003_F2c_existence_fact_verification_shadow_dlya_D1.md`.
- Generated F2c report from M1 `36ea110` ON transcripts, gold calibration report, and KB snapshot `kb_release_20260612_v6_7_staging_r4_1`.

## Runtime Impact

None. This is an offline report-only scorer.

No direct-path prompt, route logic, profile flag, P0 floor/preblock, Telegram, Wappi, AMO, Tallanto, CRM, or live process was changed.

## Main Result

F2b showed no route-only active candidates. F2c checks whether the real lever is fact verification for "does this course/format exist?" questions.

Current report result:

- existence/format rows: 10;
- current handoff rows: 2;
- handoff rows with exact KB evidence: 2;
- handoff rows without exact KB evidence: 0;
- already-self rows with exact KB evidence: 6;
- already-self rows without exact KB evidence: 1;
- excluded danger/money/P0 rows: 1.

The two current handoff rows with KB evidence are both about UNPK summer school / age suitability for a child who finished grade 5. They are diagnostic candidates for the next proof layer, not permission to enable active demotion.

## Important Design Constraint

The scorer uses conservative offline text/axis matching only to classify evidence in a report. This logic must not be copied into runtime as another regex understanding layer.

Before any active behavior, runtime needs a first-class deterministic proof contract, for example:

- `proof_kind`;
- `requested_product_normalized`;
- `supporting_fact_ids`;
- `supporting_fact_keys`;
- `all_supporting_facts_fresh_client_safe`;
- `scope_match`;
- `excludes_live_availability`;
- `blocked_reason`.
