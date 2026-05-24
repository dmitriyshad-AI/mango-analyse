# Archive + deal-aware cleanup batch 4 - 2026-05-23

Способ: перенос в macOS Trash, не безвозвратное удаление.

Trash root: `/Users/dmitrijfabarisov/.Trash/MangoAnalyse_archive_dealaware_cleanup_batch4_20260523T001335Z`
Moved: `10` paths
Total moved: `1.215` GiB

## Важное решение по `_local_archive_20260424`

`messages(1).zip` не удалён: аудит SHA-256 показал `231` уникальную mp3 относительно текущего `audio_working_store`.
Удалены только производные legacy outputs / old DB / html exports внутри `_local_archive_20260424`.

## Moved paths

| Путь | Размер, байт | Файлов | Причина |
|---|---:|---:|---|
| `_local_archive_20260424/legacy_outputs` | 132437753 | 43266 | derived legacy outputs/test transcripts; unique source zip is kept |
| `_local_archive_20260424/old_db_backups` | 4636672 | 1 | old DB backups not used by current runtime; unique source zip is kept |
| `_local_archive_20260424/old_test_dbs` | 4980736 | 4 | old test DBs not used by current runtime; unique source zip is kept |
| `_local_archive_20260424/processed_message_exports` | 2594162 | 6 | derived html exports; source zip with unique audio is kept |
| `stable_runtime/deal_aware_stage2_attribution_20260514_selector_fix_phase1` | 330832837 | 8 | heavy intermediate selector-fix Phase1 artifact; test now uses small frozen fixture |
| `stable_runtime/deal_aware_stage2_attribution_20260514_selector_fix_phase2` | 330818444 | 8 | heavy intermediate selector-fix Phase2 artifact; confidence fixture preserved |
| `stable_runtime/deal_aware_stage3_deal_state_20260514_selector_fix_phase2` | 426843692 | 10 | heavy intermediate selector-fix Stage3 artifact; no current runtime dependency |
| `stable_runtime/deal_aware_stage4_preview_20260514_selector_fix_phase2` | 21643174 | 9 | intermediate selector-fix Stage4 preview artifact; superseded by later review/writeback layers |
| `stable_runtime/deal_aware_stage5_quality_gate_20260514_selector_fix_phase2` | 25308240 | 8 | intermediate selector-fix Stage5 quality artifact; superseded by later/current gates |
| `stable_runtime/deal_aware_stage6_writeback_preflight_20260514_selector_fix_phase2` | 24869552 | 8 | intermediate selector-fix Stage6 preflight artifact; not live evidence/current runtime |

## Protected / kept

- `_local_archive_20260424/source_archives/messages(1).zip`
- `product_data/audio_working_store_20260523_v1`
- `stable_runtime/canonical_master_20260523_audio_working_store_v1`
- `stable_runtime/sales_master_export_20260523_audio_working_store_v1`
- `stable_runtime/amo_writeback_queue_20260523_audio_working_store_v1`
- `stable_runtime/crm_writeback_quality_gate_20260523_audio_working_store_v1`
- `stable_runtime/deal_aware_stage100_rop_final_20260514_v1`
- `stable_runtime/deal_aware_stage709_all_batches_20260514_v1`
- `stable_runtime/deal_aware_stage709_review_20260514_selector_fix_phase2`
