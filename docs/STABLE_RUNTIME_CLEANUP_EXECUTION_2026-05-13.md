# Stable Runtime Cleanup Execution, 2026-05-13

Status: executed after explicit approval.

## Kept Anchors

The current runtime anchors were left in place:

- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict`
- `stable_runtime/canonical_master_20260510_after_quality_backfill_v1`
- `stable_runtime/CURRENT_RUNTIME.json`

The full rollback canonical layer was also kept by decision:

- `stable_runtime/canonical_master_20260509_v1`

## Deleted Superseded Layers

Deleted intermediate post-backfill sales exports:

- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v1`
- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v2_crm_text_quality`
- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v3_crm_text_quality`
- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v4_crm_text_quality_strict`

Deleted canonical dry-runs:

- `stable_runtime/canonical_master_20260509_dry_run_v1`
- `stable_runtime/canonical_master_20260509_dry_run_v2`

Approximate space released: `941M`.

## Safety Fix

Updated `scripts/build_post_backfill_amo_ready_export.py` so a future run without `--out-root` no longer writes into the old `sales_master_export_20260510_after_quality_backfill_v1` folder.

New default behavior: create a fresh timestamped folder under:

- `stable_runtime/sales_master_export_post_backfill_*`

## Verification

Safe runtime tests passed after deletion:

- `tests/test_productization_current_runtime_operator_status.py`
- `tests/test_productization_call_processing_readiness.py`

Result: `8 passed`.

No ASR, Resolve+Analyze, CRM write, Tallanto write, or heavy batch scripts were run.
