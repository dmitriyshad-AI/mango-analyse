# Runtime cleanup delete manifest — 2026-05-23

Удаление выполняется переносом в корзину macOS, не через `rm`.

## Protected / keep

- Current canonical DB: `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260521_after_mango_update_v1/canonical_calls_master.db`.
- Main audio folder `2026-03-09--26` is explicitly kept.
- `product_data/canonical_audio_store_20260516_v1` is kept: current canonical DB has `264` `source_file` references into this audio store; total current DB refs under the store: `264`. Size: `24.96 GiB`, files: `65161`.
- `product_data/mango_update_after_20260512_20260521_v1/asr_ui_batch/audio` is kept: current canonical DB points to 839 files there.

## Delete: old ASR/R+A waves fully covered by current canonical filenames

| Path | Size GiB | Files | Audio files | Unique audio names | Covered names | Reason |
|---|---:|---:|---:|---:|---:|---|
| `stable_runtime/apr_may_2025_asr_only_20260502` | 1.13 | 20123 | 4024 | 4024 | 4024 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/benchmarks` | 0.20 | 5121 | 3000 | 1000 | 1000 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/final_asr_tail_1526_20260506` | 0.28 | 7609 | 1526 | 1526 | 1526 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/history_cohort_20260211_20260326` | 0.00 | 17440 | 17439 | 17439 | 17439 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/history_cohort_20260319_20260326` | 0.76 | 16006 | 5327 | 5327 | 5327 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/jun_jul_aug_2025_asr_only_20260503` | 2.90 | 43868 | 8793 | 8793 | 8793 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/mar2026_client_asr_tail_129_20260507` | 0.01 | 660 | 129 | 129 | 129 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/messages28_full_20260407` | 0.38 | 7108 | 1631 | 1631 | 1631 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/messages28_phone_history_asr_20260408` | 0.73 | 19203 | 8851 | 8851 | 8851 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/messages28_phone_history_gap_wave_20260410` | 1.11 | 1922 | 1912 | 1912 | 1912 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/messages28_phone_history_gigaam_useful_20260409` | 0.02 | 1475 | 491 | 491 | 491 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/messages28_phone_history_llm_wave_20260409` | 1.20 | 2717 | 2706 | 2706 | 2706 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/messages29_phone_history_full_20260410` | 0.08 | 1592 | 456 | 456 | 456 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/messages30_phone_history_full_20260412` | 0.30 | 5740 | 1146 | 1146 | 1146 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/messages31_phone_history_full_20260414` | 0.12 | 2967 | 633 | 633 | 633 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/messages32_33_phone_history_full_20260422` | 0.52 | 6328 | 2071 | 2071 | 2071 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/messages34_phone_history_full_20260501` | 0.58 | 12944 | 2588 | 2588 | 2588 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/messages35_asr_only_20260506` | 0.07 | 1994 | 397 | 397 | 397 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/night_asr_3000_20260328` | 0.81 | 12749 | 3000 | 3000 | 3000 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/oct_nov_2025_asr_only_remaining_all_20260505` | 0.24 | 27118 | 9037 | 9037 | 9037 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/overnight_full_asr_priority_2000_20260413` | 0.39 | 10002 | 2000 | 2000 | 2000 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/overnight_full_asr_priority_2000_20260415` | 0.39 | 10002 | 2000 | 2000 | 2000 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/overnight_full_asr_priority_2000_20260416` | 0.42 | 14658 | 2000 | 2000 | 2000 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/overnight_history_gap_safe_1892_20260413` | 0.19 | 1905 | 1892 | 1892 | 1892 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/recent_window_20260319_20260326_mini` | 0.34 | 2034 | 675 | 675 | 675 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/sep2025_asr_only_3000_20260504` | 0.11 | 9007 | 3000 | 3000 | 3000 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/sep2025_asr_only_remaining_all_20260504` | 0.20 | 22804 | 7599 | 7599 | 7599 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/top100_history_wave1_20260331` | 0.58 | 1131 | 1115 | 1115 | 1115 | current DB has all audio filenames; no current `source_file` refs into this dir |
| `stable_runtime/top20_core_wave1_20260331` | 0.25 | 1277 | 459 | 459 | 459 | current DB has all audio filenames; no current `source_file` refs into this dir |

## Delete: explicitly approved old/intermediate roots

| Path | Size GiB | Files | Current source refs | Reason |
|---|---:|---:|---:|---|
| `stable_runtime/ra_missing_all_20260506` | 4.19 | 47264 | 0 | approved old ra_missing layer |
| `product_data/mango_missing_1372_asr_only_20260517_v1` | 0.87 | 4144 | 0 | intermediate Mango ASR-only package; not current accepted runtime and no current source refs |

## Delete: partial intermediate Mango update cleanup

| Path | Size GiB | Files | Current source refs | Reason |
|---|---:|---:|---:|---|
| `product_data/mango_update_after_20260512_20260521_v1/recordings` | 0.46 | 848 | 0 | duplicate download folder; current DB uses `asr_ui_batch/audio`, not `recordings` |

## Delete: old backup directories not inside deleted roots

| Path | Size GiB | Files | Reason |
|---|---:|---:|---|
| `stable_runtime/backups` | 0.26 | 14 | backup dir; no current source refs |
| `stable_runtime/history_remaining_excl_done_20260407/backups` | 0.58 | 7 | backup dir; no current source refs |
| `stable_runtime/manual_tail_analyze_fallback_20260507/backups` | 0.03 | 6 | backup dir; no current source refs |

## Summary

- Targets: `35`
- Estimated size: `20.69 GiB`
- Generated at: `2026-05-23T00:52:15`

## Execution result

Moved to Trash folder:

`/Users/dmitrijfabarisov/.Trash/mango_runtime_cleanup_20260523_005235`

Post-checks:

- All listed targets are absent from the working tree.
- Current runtime check passed: `validation_ok=true`, `blocked=0`, `warnings=0`, missing ASR/R+A = `0/0`.
- Targeted tests passed: `8 passed`.
- `product_data/canonical_audio_store_20260516_v1` was not deleted because the current canonical DB has `264` audio references into it.
- `product_data/mango_update_after_20260512_20260521_v1/asr_ui_batch/audio` was not deleted because the current canonical DB has `839` audio references into it; only the duplicate `recordings/` folder was moved.

Disk impact in working tree after move:

- `stable_runtime`: about `13G`.
- `product_data`: about `28G`.
- Trash now contains this cleanup batch: about `21G`.
- Earlier canonical DB cleanup batch in Trash: about `5.8G`.

Note: disk space is not actually reclaimed until macOS Trash is emptied.
