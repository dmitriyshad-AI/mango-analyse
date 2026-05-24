# Runtime Cleanup Batch 2 Manifest
Date: 2026-05-23 01:12:02
Scope: old stable_runtime exports/deal-aware/transcript-quality artifacts, old audit inbox packs, old KB releases, old customer timeline versions. Mail archive is explicitly excluded.
Trash folder: `/Users/dmitrijfabarisov/.Trash/mango_cleanup_batch2_intermediate_artifacts_20260523_011201`
CSV manifest: `/Users/dmitrijfabarisov/Projects/Mango analyse/docs/RUNTIME_CLEANUP_BATCH2_MOVED_2026-05-23.csv`
Moved objects: `270`
Failed objects: `0`
Skipped protected objects: `0`
Approx moved size: `9.33 GiB`
## Protected / Not Touched
- `stable_runtime/sales_master_export_20260521_after_mango_update_v4_runtime_acceptance`
- `stable_runtime/sales_master_export_20260521_after_mango_update_v4_runtime_acceptance/summary.json`
- `stable_runtime/sales_master_export_20260521_after_mango_update_v4_runtime_acceptance/amo_export_ready_ru.csv`
- `stable_runtime/amo_writeback_queue_20260521_after_mango_update_v4_runtime_acceptance/summary.json`
- `stable_runtime/canonical_master_20260521_after_mango_update_v1/canonical_calls_master.db`
- `stable_runtime/CANONICAL_EXPORT.txt`
- `stable_runtime/canonical_master_20260521_after_mango_update_v1/summary.json`
- `stable_runtime/crm_writeback_quality_gate_20260521_after_mango_update_v4_runtime_acceptance/summary.json`
- `_local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite`
- `_local_archive_mango_api_downloads_20260507/product_appliance`
- `stable_runtime/transcript_quality_stage15_export_gate_20260510_v11_frozen_gate/summary.json`
- `stable_runtime/sales_master_export_20260521_after_mango_update_v4_runtime_acceptance`
- `stable_runtime/canonical_master_20260521_after_mango_update_v1`
- `product_data/customer_timeline/canonical_readonly_20260521_v5`
- `product_data/canonical_audio_store_20260516_v1`
- `product_data/mango_update_after_20260512_20260521_v1/asr_ui_batch`
- `_external_handoffs/mail_archive_2026-05-12`

## Moved Summary By Reason
| Count | Size GiB | Reason |
|---:|---:|---|
| 27 | 2.76 | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 7 | 2.05 | old human-history export; v8_normalized is the only referenced old layer |
| 13 | 1.81 | old research/audit/intermediate transcript quality artifact; rules are now in code/tests and current gates |
| 15 | 1.43 | superseded runtime/export/gate layer; current runtime points to 20260521 v4 |
| 40 | 0.43 | old KB release/build; current v6_3 family retained |
| 9 | 0.41 | old customer timeline build/sample; canonical_readonly_20260521_v5 retained |
| 151 | 0.29 | old audit inbox input pack; audit results/docs retained |
| 4 | 0.09 | old canonical shell after canonical DB cleanup; current canonical is 20260521 |
| 4 | 0.07 | old root KB output; latest v6_3 retained |

## Failed
None.

## Full List
| Size MiB | Source | Reason |
|---:|---|---|
| 234.6 | `stable_runtime/sales_master_export_20260521_after_mango_update_v1` | superseded runtime/export/gate layer; current runtime points to 20260521 v4 |
| 234.6 | `stable_runtime/sales_master_export_20260521_after_mango_update_v2_generic_next_step_gate` | superseded runtime/export/gate layer; current runtime points to 20260521 v4 |
| 234.6 | `stable_runtime/sales_master_export_20260521_after_mango_update_v3_runtime_acceptance` | superseded runtime/export/gate layer; current runtime points to 20260521 v4 |
| 230.9 | `stable_runtime/sales_master_export_20260516_after_mango_update_v1` | superseded runtime/export/gate layer; current runtime points to 20260521 v4 |
| 213.6 | `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict` | superseded runtime/export/gate layer; current runtime points to 20260521 v4 |
| 0.1 | `stable_runtime/amo_writeback_queue_20260521_after_mango_update_v1` | superseded runtime/export/gate layer; current runtime points to 20260521 v4 |
| 0.1 | `stable_runtime/amo_writeback_queue_20260521_after_mango_update_v2_generic_next_step_gate` | superseded runtime/export/gate layer; current runtime points to 20260521 v4 |
| 0.1 | `stable_runtime/amo_writeback_queue_20260521_after_mango_update_v3_runtime_acceptance` | superseded runtime/export/gate layer; current runtime points to 20260521 v4 |
| 0.1 | `stable_runtime/crm_writeback_quality_gate_20260521_after_mango_update_v1` | superseded runtime/export/gate layer; current runtime points to 20260521 v4 |
| 0.1 | `stable_runtime/crm_writeback_quality_gate_20260521_after_mango_update_v2_generic_next_step_gate` | superseded runtime/export/gate layer; current runtime points to 20260521 v4 |
| 0.1 | `stable_runtime/crm_writeback_quality_gate_20260521_after_mango_update_v3_runtime_acceptance` | superseded runtime/export/gate layer; current runtime points to 20260521 v4 |
| 0.1 | `stable_runtime/crm_writeback_quality_gate_20260516_after_mango_update_v1` | superseded runtime/export/gate layer; current runtime points to 20260521 v4 |
| 102.4 | `stable_runtime/insight_readiness_report_after_mango_update_20260516_v1` | superseded runtime/export/gate layer; current runtime points to 20260521 v4 |
| 108.7 | `stable_runtime/insight_readiness_report_after_quality_backfill_20260510_v1` | superseded runtime/export/gate layer; current runtime points to 20260521 v4 |
| 109.1 | `stable_runtime/insight_readiness_report_20260507` | superseded runtime/export/gate layer; current runtime points to 20260521 v4 |
| 321.7 | `stable_runtime/sales_master_export_20260513_human_history_v1` | old human-history export; v8_normalized is the only referenced old layer |
| 302.4 | `stable_runtime/sales_master_export_20260513_human_history_v2` | old human-history export; v8_normalized is the only referenced old layer |
| 295.6 | `stable_runtime/sales_master_export_20260513_human_history_v3` | old human-history export; v8_normalized is the only referenced old layer |
| 293.6 | `stable_runtime/sales_master_export_20260513_human_history_v4` | old human-history export; v8_normalized is the only referenced old layer |
| 293.6 | `stable_runtime/sales_master_export_20260513_human_history_v5` | old human-history export; v8_normalized is the only referenced old layer |
| 293.6 | `stable_runtime/sales_master_export_20260513_human_history_v6` | old human-history export; v8_normalized is the only referenced old layer |
| 293.6 | `stable_runtime/sales_master_export_20260513_human_history_v7` | old human-history export; v8_normalized is the only referenced old layer |
| 45.0 | `stable_runtime/canonical_master_20260509_v1` | old canonical shell after canonical DB cleanup; current canonical is 20260521 |
| 45.0 | `stable_runtime/canonical_master_20260510_after_quality_backfill_v1` | old canonical shell after canonical DB cleanup; current canonical is 20260521 |
| 0.0 | `stable_runtime/canonical_master_20260516_after_mango_update_v1` | old canonical shell after canonical DB cleanup; current canonical is 20260521 |
| 0.0 | `stable_runtime/canonical_master_20260517_after_mango_asr_only_v1` | old canonical shell after canonical DB cleanup; current canonical is 20260521 |
| 699.7 | `stable_runtime/ab_tests` | old research/audit/intermediate transcript quality artifact; rules are now in code/tests and current gates |
| 242.7 | `stable_runtime/non_conversation_hard_gate_owner_dry_run_20260509_v4_live_safeguards` | old research/audit/intermediate transcript quality artifact; rules are now in code/tests and current gates |
| 257.2 | `stable_runtime/non_conversation_hard_gate_owner_dry_run_20260509_v5_gpt_policy_preview` | old research/audit/intermediate transcript quality artifact; rules are now in code/tests and current gates |
| 104.2 | `stable_runtime/non_conversation_hard_gate_backup_manifest_20260509_phase7` | old research/audit/intermediate transcript quality artifact; rules are now in code/tests and current gates |
| 124.4 | `stable_runtime/transcript_quality_pipeline_v2_risky_3298_m4_20260509_0445` | old research/audit/intermediate transcript quality artifact; rules are now in code/tests and current gates |
| 79.2 | `stable_runtime/transcript_quality_adversarial_audit_20260509` | old research/audit/intermediate transcript quality artifact; rules are now in code/tests and current gates |
| 70.5 | `stable_runtime/transcript_quality_guardrails_v2_all_20260509_043621` | old research/audit/intermediate transcript quality artifact; rules are now in code/tests and current gates |
| 67.6 | `stable_runtime/transcript_quality_guardrails_dry_run_20260509` | old research/audit/intermediate transcript quality artifact; rules are now in code/tests and current gates |
| 63.6 | `stable_runtime/transcript_quality_guardrails_after_backfill_20260509` | old research/audit/intermediate transcript quality artifact; rules are now in code/tests and current gates |
| 50.4 | `stable_runtime/transcript_quality_disputed_review_v2_all_20260509_0440` | old research/audit/intermediate transcript quality artifact; rules are now in code/tests and current gates |
| 36.4 | `stable_runtime/transcript_quality_disputed_review_20260509` | old research/audit/intermediate transcript quality artifact; rules are now in code/tests and current gates |
| 30.2 | `stable_runtime/transcript_quality_pipeline_pilot_1000_20260509` | old research/audit/intermediate transcript quality artifact; rules are now in code/tests and current gates |
| 28.8 | `stable_runtime/transcript_quality_pipeline_live_1000_20260509` | old research/audit/intermediate transcript quality artifact; rules are now in code/tests and current gates |
| 378.1 | `stable_runtime/deal_aware_stage1_snapshot_20260513_v1` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 378.1 | `stable_runtime/deal_aware_stage1_snapshot_20260513_v2` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 323.7 | `stable_runtime/deal_aware_stage2_attribution_20260513_v1` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 318.4 | `stable_runtime/deal_aware_stage2_attribution_20260513_v2` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 406.9 | `stable_runtime/deal_aware_stage3_deal_state_20260513_v1` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 410.1 | `stable_runtime/deal_aware_stage3_deal_state_20260514_selector_fix_phase1` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 23.0 | `stable_runtime/deal_aware_stage4_preview_20260513_v1` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 22.8 | `stable_runtime/deal_aware_stage4_preview_20260513_rop_iter01` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 22.9 | `stable_runtime/deal_aware_stage4_preview_20260513_rop_iter02` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 22.5 | `stable_runtime/deal_aware_stage4_preview_20260513_rop_iter03` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 22.2 | `stable_runtime/deal_aware_stage4_preview_20260514_iter01` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 26.6 | `stable_runtime/deal_aware_stage4_preview_20260514_selector_fix_phase1` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 28.3 | `stable_runtime/deal_aware_stage5_quality_gate_20260513_v1` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 28.0 | `stable_runtime/deal_aware_stage5_quality_gate_20260513_rop_iter01` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 28.1 | `stable_runtime/deal_aware_stage5_quality_gate_20260513_rop_iter02` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 27.7 | `stable_runtime/deal_aware_stage5_quality_gate_20260513_rop_iter03` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 27.3 | `stable_runtime/deal_aware_stage5_quality_gate_20260514_iter01` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 32.8 | `stable_runtime/deal_aware_stage5_quality_gate_20260514_selector_fix_phase1` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 28.2 | `stable_runtime/deal_aware_stage6_writeback_preflight_20260513_v1` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 26.6 | `stable_runtime/deal_aware_stage6_writeback_preflight_20260513_rop_iter01` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 26.8 | `stable_runtime/deal_aware_stage6_writeback_preflight_20260513_rop_iter02` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 26.3 | `stable_runtime/deal_aware_stage6_writeback_preflight_20260513_rop_iter03` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 26.0 | `stable_runtime/deal_aware_stage6_writeback_preflight_20260514_iter01` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 30.4 | `stable_runtime/deal_aware_stage6_writeback_preflight_20260514_selector_fix_phase1` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 42.9 | `stable_runtime/deal_aware_stage709_review_20260514_v1` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 42.2 | `stable_runtime/deal_aware_stage709_review_20260514_iter01` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 48.7 | `stable_runtime/deal_aware_stage709_review_20260514_selector_fix_phase1` | old deal-aware intermediate; selector_fix_phase2/final accepted layers retained |
| 80.4 | `product_data/customer_timeline/canonical_readonly_20260521_v1` | old customer timeline build/sample; canonical_readonly_20260521_v5 retained |
| 80.7 | `product_data/customer_timeline/canonical_readonly_20260521_v2` | old customer timeline build/sample; canonical_readonly_20260521_v5 retained |
| 80.5 | `product_data/customer_timeline/canonical_readonly_20260521_v3` | old customer timeline build/sample; canonical_readonly_20260521_v5 retained |
| 80.3 | `product_data/customer_timeline/canonical_readonly_20260521_v4` | old customer timeline build/sample; canonical_readonly_20260521_v5 retained |
| 19.3 | `product_data/customer_timeline/contact_control_sample_20260516` | old customer timeline build/sample; canonical_readonly_20260521_v5 retained |
| 11.6 | `product_data/customer_timeline/contact_control_sample_20260516_v2` | old customer timeline build/sample; canonical_readonly_20260521_v5 retained |
| 11.6 | `product_data/customer_timeline/contact_control_sample_20260516_v3` | old customer timeline build/sample; canonical_readonly_20260521_v5 retained |
| 20.2 | `product_data/customer_timeline/contact_control_sample_hard_20260516_032348` | old customer timeline build/sample; canonical_readonly_20260521_v5 retained |
| 32.8 | `product_data/customer_timeline/deal_aware_sample_20260515` | old customer timeline build/sample; canonical_readonly_20260521_v5 retained |
| 16.4 | `product_data/knowledge_base/full_kb_20260517_v1` | old KB release/build; current v6_3 family retained |
| 3.3 | `product_data/knowledge_base/kb_night_20260517_v1` | old KB release/build; current v6_3 family retained |
| 86.6 | `product_data/knowledge_base/kb_release_20260517_v1` | old KB release/build; current v6_3 family retained |
| 86.6 | `product_data/knowledge_base/kb_release_20260517_v1_agent_pack` | old KB release/build; current v6_3 family retained |
| 11.2 | `product_data/knowledge_base/kb_release_20260517_v2` | old KB release/build; current v6_3 family retained |
| 8.8 | `product_data/knowledge_base/kb_release_20260517_v2_agent_pack` | old KB release/build; current v6_3 family retained |
| 19.3 | `product_data/knowledge_base/kb_release_20260517_v2_handoff_for_claude_and_team` | old KB release/build; current v6_3 family retained |
| 20.0 | `product_data/knowledge_base/kb_release_20260518_v3` | old KB release/build; current v6_3 family retained |
| 11.5 | `product_data/knowledge_base/kb_release_20260518_v3_2` | old KB release/build; current v6_3 family retained |
| 13.1 | `product_data/knowledge_base/kb_release_20260518_v3_2_bot_pack` | old KB release/build; current v6_3 family retained |
| 0.4 | `product_data/knowledge_base/kb_release_20260518_v3_2_employee_pack` | old KB release/build; current v6_3 family retained |
| 11.1 | `product_data/knowledge_base/kb_release_20260518_v3_2_handoff_for_claude_and_team` | old KB release/build; current v6_3 family retained |
| 0.6 | `product_data/knowledge_base/kb_release_20260518_v3_2_smoke50_codex` | old KB release/build; current v6_3 family retained |
| 0.2 | `product_data/knowledge_base/kb_release_20260518_v3_2_smoke50_fake` | old KB release/build; current v6_3 family retained |
| 0.1 | `product_data/knowledge_base/kb_release_20260518_v3_2_smoke50_input` | old KB release/build; current v6_3 family retained |
| 12.4 | `product_data/knowledge_base/kb_release_20260518_v3_3` | old KB release/build; current v6_3 family retained |
| 7.1 | `product_data/knowledge_base/kb_release_20260518_v3_3_bot_pack` | old KB release/build; current v6_3 family retained |
| 0.5 | `product_data/knowledge_base/kb_release_20260518_v3_3_employee_pack` | old KB release/build; current v6_3 family retained |
| 11.9 | `product_data/knowledge_base/kb_release_20260518_v3_3_handoff_for_claude_and_team` | old KB release/build; current v6_3 family retained |
| 0.5 | `product_data/knowledge_base/kb_release_20260518_v3_3_smoke20_codex` | old KB release/build; current v6_3 family retained |
| 0.3 | `product_data/knowledge_base/kb_release_20260518_v3_3_smoke20_fake` | old KB release/build; current v6_3 family retained |
| 11.1 | `product_data/knowledge_base/kb_release_20260518_v3_handoff_for_claude_and_team` | old KB release/build; current v6_3 family retained |
| 13.2 | `product_data/knowledge_base/kb_release_20260520_v4` | old KB release/build; current v6_3 family retained |
| 7.5 | `product_data/knowledge_base/kb_release_20260520_v4_bot_pack` | old KB release/build; current v6_3 family retained |
| 0.5 | `product_data/knowledge_base/kb_release_20260520_v4_employee_pack` | old KB release/build; current v6_3 family retained |
| 12.6 | `product_data/knowledge_base/kb_release_20260520_v4_handoff_for_claude_and_team` | old KB release/build; current v6_3 family retained |
| 0.8 | `product_data/knowledge_base/kb_release_20260520_v4_smoke_small_codex` | old KB release/build; current v6_3 family retained |
| 0.0 | `product_data/knowledge_base/kb_release_20260520_v4_smoke_small_input` | old KB release/build; current v6_3 family retained |
| 13.2 | `product_data/knowledge_base/kb_release_20260520_v6_1_team_answers` | old KB release/build; current v6_3 family retained |
| 7.6 | `product_data/knowledge_base/kb_release_20260520_v6_1_team_answers_bot_pack` | old KB release/build; current v6_3 family retained |
| 0.5 | `product_data/knowledge_base/kb_release_20260520_v6_1_team_answers_employee_pack` | old KB release/build; current v6_3 family retained |
| 12.6 | `product_data/knowledge_base/kb_release_20260520_v6_1_team_answers_handoff_for_claude_and_team` | old KB release/build; current v6_3 family retained |
| 0.0 | `product_data/knowledge_base/kb_release_20260520_v6_1_team_answers_smoke_not_run` | old KB release/build; current v6_3 family retained |
| 0.3 | `product_data/knowledge_base/kb_release_20260520_v6_1_team_answers_sources` | old KB release/build; current v6_3 family retained |
| 13.2 | `product_data/knowledge_base/kb_release_20260520_v6_2_team_answers` | old KB release/build; current v6_3 family retained |
| 7.6 | `product_data/knowledge_base/kb_release_20260520_v6_2_team_answers_bot_pack` | old KB release/build; current v6_3 family retained |
| 0.5 | `product_data/knowledge_base/kb_release_20260520_v6_2_team_answers_employee_pack` | old KB release/build; current v6_3 family retained |
| 12.6 | `product_data/knowledge_base/kb_release_20260520_v6_2_team_answers_handoff_for_claude_and_team` | old KB release/build; current v6_3 family retained |
| 0.0 | `product_data/knowledge_base/kb_release_20260520_v6_2_team_answers_smoke_not_run` | old KB release/build; current v6_3 family retained |
| 0.3 | `product_data/knowledge_base/kb_release_20260520_v6_2_team_answers_sources` | old KB release/build; current v6_3 family retained |
| 32.4 | `Mango_Bot_KB_FINAL_v3_3_2026-05-19` | old root KB output; latest v6_3 retained |
| 18.3 | `Mango_Bot_KB_FINAL_v6_1_2026-05-20` | old root KB output; latest v6_3 retained |
| 18.3 | `Mango_Bot_KB_FINAL_v6_2_2026-05-20` | old root KB output; latest v6_3 retained |
| 6.3 | `Claude Mango_Bot_Knowledge_Base_FINAL_2026-05-17` | old root KB output; latest v6_3 retained |
| 0.2 | `audits/_inbox/amo_duplicate_after_staff_done_20260511_v1` | old audit inbox input pack; audit results/docs retained |
| 0.2 | `audits/_inbox/amo_duplicate_post_merge_recheck_gate_20260511_v1` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/amo_duplicate_resolution_20260511_v1` | old audit inbox input pack; audit results/docs retained |
| 0.2 | `audits/_inbox/amo_duplicate_staff_tasks_20260511_v1` | old audit inbox input pack; audit results/docs retained |
| 0.2 | `audits/_inbox/amo_manual_resolution_after_xlsx_20260511_v1` | old audit inbox input pack; audit results/docs retained |
| 0.8 | `audits/_inbox/amo_manual_resolution_operator_status_20260511_v1` | old audit inbox input pack; audit results/docs retained |
| 3.4 | `audits/_inbox/amo_post_backfill_writeback_20260510_v5_product_gate` | old audit inbox input pack; audit results/docs retained |
| 4.2 | `audits/_inbox/amo_stage100_batch2_orphan_resolved_preflight_20260512_v1` | old audit inbox input pack; audit results/docs retained |
| 3.8 | `audits/_inbox/amo_stage100_orphan_resolved_preflight_20260512_v1` | old audit inbox input pack; audit results/docs retained |
| 2.6 | `audits/_inbox/amo_stage200_batch3_orphan_resolved_preflight_20260512_v1` | old audit inbox input pack; audit results/docs retained |
| 0.5 | `audits/_inbox/amo_stage40_live20_readback_remaining20_20260510_v1` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/amo_stage50_stage86_preflight_blocked_20260511_v1` | old audit inbox input pack; audit results/docs retained |
| 0.9 | `audits/_inbox/amo_stage51_live_script_preflight_20260512_v1` | old audit inbox input pack; audit results/docs retained |
| 0.5 | `audits/_inbox/amo_stage51_summary_repair_preflight_20260512_v1` | old audit inbox input pack; audit results/docs retained |
| 0.6 | `audits/_inbox/amo_stage51_textarea_repair_preflight_20260512_v2` | old audit inbox input pack; audit results/docs retained |
| 1.0 | `audits/_inbox/amo_stage54_post_oauth_dryrun_20260511_v1` | old audit inbox input pack; audit results/docs retained |
| 0.9 | `audits/_inbox/amo_stage55_post_oauth_dryrun_20260511_v1` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/amo_tallanto_live_recheck_10rows_20260516_043415` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/amo_tallanto_live_recheck_10rows_fast_20260516_043801` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/amo_tallanto_snapshot_recheck_10rows_20260516_044344` | old audit inbox input pack; audit results/docs retained |
| 1.5 | `audits/_inbox/amo_waiting_autonomous_work_20260511_v1` | old audit inbox input pack; audit results/docs retained |
| 0.3 | `audits/_inbox/amo_waiting_network_dryrun_20260511_v1` | old audit inbox input pack; audit results/docs retained |
| 0.5 | `audits/_inbox/amo_waiting_network_dryrun_20260511_v2` | old audit inbox input pack; audit results/docs retained |
| 1.4 | `audits/_inbox/amo_writeback_f008_closure_20260510` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/amo_writeback_production_loop_20260510_v1` | old audit inbox input pack; audit results/docs retained |
| 0.4 | `audits/_inbox/amo_writeback_stage1_completion_20260510_v1` | old audit inbox input pack; audit results/docs retained |
| 0.6 | `audits/_inbox/audio_store_downstream_switch_20260516_v1` | old audit inbox input pack; audit results/docs retained |
| 19.3 | `audits/_inbox/canonical_audio_store_20260516_v1` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/channel_product_storage_20260511` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/claude_cli_kb_review_20260518T195004Z` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/claude_cli_kb_review_20260518T210642Z` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/claude_cli_kb_review_20260518T2148Z` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/claude_cli_kb_review_20260518T224634` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/claude_cli_kb_review_20260518T2343` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/claude_cli_kb_review_20260518T2350` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/claude_cli_kb_review_20260518_2058_v3_3_bot_pack` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/claude_cli_kb_review_20260519_0008_v3_3_final` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/claude_cli_kb_review_20260519_002912` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/claude_cli_kb_review_20260519_0032_v3_3_after_notes` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/claude_cli_kb_review_20260519_003617` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/claude_cli_kb_review_20260519_0050_v3_3_final_after_all_fixes` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/claude_cli_review_setup_20260518_225242` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/claude_smoke_test_20260513` | old audit inbox input pack; audit results/docs retained |
| 0.4 | `audits/_inbox/crm_text_quality_stage20_20260510_v1` | old audit inbox input pack; audit results/docs retained |
| 37.4 | `audits/_inbox/crm_text_quality_stage69_preflight_20260510_v1` | old audit inbox input pack; audit results/docs retained |
| 7.7 | `audits/_inbox/crm_writeback_defect_classes_20260510_v1` | old audit inbox input pack; audit results/docs retained |
| 0.5 | `audits/_inbox/customer_timeline_100_phone_coverage_20260515_175607` | old audit inbox input pack; audit results/docs retained |
| 0.6 | `audits/_inbox/customer_timeline_contact_control_sample_20260516_030000` | old audit inbox input pack; audit results/docs retained |
| 1.1 | `audits/_inbox/customer_timeline_hard_control_sample_20260516_033500` | old audit inbox input pack; audit results/docs retained |
| 0.2 | `audits/_inbox/customer_timeline_local_import_20260515_191000` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/deal_aware_iterative_improvement_plan_20260514_v1` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/deal_aware_next_plan_review_20260513_v1` | old audit inbox input pack; audit results/docs retained |
| 0.7 | `audits/_inbox/deal_aware_preview_50_20260512_v1` | old audit inbox input pack; audit results/docs retained |
| 0.7 | `audits/_inbox/deal_aware_preview_50_20260512_v2` | old audit inbox input pack; audit results/docs retained |
| 0.7 | `audits/_inbox/deal_aware_preview_50_20260512_v3` | old audit inbox input pack; audit results/docs retained |
| 0.7 | `audits/_inbox/deal_aware_preview_50_20260512_v4` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/deal_aware_preview_active_50_20260513_v1` | old audit inbox input pack; audit results/docs retained |
| 0.3 | `audits/_inbox/deal_aware_selector_fix_phase1_20260514` | old audit inbox input pack; audit results/docs retained |
| 1.0 | `audits/_inbox/deal_aware_selector_fix_phase2_20260514` | old audit inbox input pack; audit results/docs retained |
| 27.3 | `audits/_inbox/deal_aware_stage100_iter01_20260514` | old audit inbox input pack; audit results/docs retained |
| 0.6 | `audits/_inbox/deal_aware_stage100_rowlevel_audit_20260514_v1` | old audit inbox input pack; audit results/docs retained |
| 0.8 | `audits/_inbox/deal_aware_stage100_rowlevel_iter01_20260514` | old audit inbox input pack; audit results/docs retained |
| 27.9 | `audits/_inbox/deal_aware_stage100_stratified_preview_20260514_v1` | old audit inbox input pack; audit results/docs retained |
| 1.7 | `audits/_inbox/deal_aware_stage100_stratified_preview_20260514_v2_slim` | old audit inbox input pack; audit results/docs retained |
| 1.7 | `audits/_inbox/deal_aware_stage100_stratified_preview_20260514_v3_after_claude_fixes` | old audit inbox input pack; audit results/docs retained |
| 0.8 | `audits/_inbox/deal_aware_stage1_5_chain_micro_20260513_v1` | old audit inbox input pack; audit results/docs retained |
| 0.3 | `audits/_inbox/deal_aware_stage20_post_live_20260513_v1` | old audit inbox input pack; audit results/docs retained |
| 0.2 | `audits/_inbox/deal_aware_stage20_rop_iter01_20260513` | old audit inbox input pack; audit results/docs retained |
| 0.2 | `audits/_inbox/deal_aware_stage20_rop_iter02_20260513` | old audit inbox input pack; audit results/docs retained |
| 0.2 | `audits/_inbox/deal_aware_stage20_rop_iter03_20260513` | old audit inbox input pack; audit results/docs retained |
| 0.2 | `audits/_inbox/deal_aware_stage20_rop_precheck_20260513_v1` | old audit inbox input pack; audit results/docs retained |
| 13.1 | `audits/_inbox/deal_aware_stage4_preview_20260513_v1` | old audit inbox input pack; audit results/docs retained |
| 0.6 | `audits/_inbox/deal_aware_stage4_preview_micro_20260513_v1` | old audit inbox input pack; audit results/docs retained |
| 3.0 | `audits/_inbox/deal_aware_stage4_preview_slim_20260513_v1` | old audit inbox input pack; audit results/docs retained |
| 15.1 | `audits/_inbox/deal_aware_stage5_quality_gate_20260513_v1` | old audit inbox input pack; audit results/docs retained |
| 2.2 | `audits/_inbox/deal_aware_stage5_quality_gate_slim_20260513_v1` | old audit inbox input pack; audit results/docs retained |
| 2.1 | `audits/_inbox/deal_aware_stage6_writeback_preflight_micro_20260513_v1` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/deal_aware_stage6_writeback_preflight_nano_20260513_v1` | old audit inbox input pack; audit results/docs retained |
| 32.1 | `audits/_inbox/deal_aware_stage709_20260514_selector_fix_phase1` | old audit inbox input pack; audit results/docs retained |
| 25.0 | `audits/_inbox/deal_aware_stage709_20260514_selector_fix_phase2` | old audit inbox input pack; audit results/docs retained |
| 8.4 | `audits/_inbox/deal_aware_stage709_all_batches_20260514_v1` | old audit inbox input pack; audit results/docs retained |
| 2.1 | `audits/_inbox/deal_aware_writeback_batch1_20260514_v1` | old audit inbox input pack; audit results/docs retained |
| 1.8 | `audits/_inbox/deal_aware_writeback_batch1_20260514_v2` | old audit inbox input pack; audit results/docs retained |
| 1.7 | `audits/_inbox/deal_aware_writeback_batch1_allow60_20260514_v1` | old audit inbox input pack; audit results/docs retained |
| 11.3 | `audits/_inbox/for_claude_kb_release_v32_semantic_smoke_20260518_220609` | old audit inbox input pack; audit results/docs retained |
| 8.0 | `audits/_inbox/for_claude_kb_semantic_final_review_20260518_145156` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/full_kc_knowledge_base_20260517_v1` | old audit inbox input pack; audit results/docs retained |
| 0.2 | `audits/_inbox/gold_dialogues_paid_after_calls_20260521_v1` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/kc_final_release_20260517_v1` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/kc_knowledge_facts_pilot_20260516_161003` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mango_asr_integration_20260517_v1` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/mango_audio_update_20260516_v1` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mango_calls_update_20260516_v1` | old audit inbox input pack; audit results/docs retained |
| 0.7 | `audits/_inbox/mango_canonical_rebuild_plan_20260516_v1` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/mango_runtime_rebuild_20260516_v1` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mango_update_after_20260512_20260521_v1` | old audit inbox input pack; audit results/docs retained |
| 4.0 | `audits/_inbox/mega_smoke_v1_20260519_codex_run` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_full_20260519_220558` | old audit inbox input pack; audit results/docs retained |
| 0.4 | `audits/_inbox/mega_smoke_v3_codex_full_all_20260520_112720` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_p0_full_20260519_224502` | old audit inbox input pack; audit results/docs retained |
| 0.2 | `audits/_inbox/mega_smoke_v3_codex_p0_full_after_patches_20260520_025330` | old audit inbox input pack; audit results/docs retained |
| 0.2 | `audits/_inbox/mega_smoke_v3_codex_p0_full_clean_candidate_20260520_073717` | old audit inbox input pack; audit results/docs retained |
| 0.2 | `audits/_inbox/mega_smoke_v3_codex_p0_full_clean_candidate_2_20260520_085723` | old audit inbox input pack; audit results/docs retained |
| 0.2 | `audits/_inbox/mega_smoke_v3_codex_p0_full_clean_candidate_3_20260520_101139` | old audit inbox input pack; audit results/docs retained |
| 0.2 | `audits/_inbox/mega_smoke_v3_codex_p0_full_final_20260520_041938` | old audit inbox input pack; audit results/docs retained |
| 0.2 | `audits/_inbox/mega_smoke_v3_codex_p0_full_final_after_priority_fix_20260520_055440` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_20260519_214628` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_admission_after_proydet_fix_20260520_073531` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_adv_pii_self_02_after_ceny_fix_20260520_100928` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_after_discount_templates_20260519_223157` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_after_legal_order_20260519_215831` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_after_p0_templates_20260519_215050` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_after_result_guarantee_20260519_221738` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_contact_pii_fix_20260520_010337` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_cross_identity_after_final_fix_20260520_040416` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_guarantee_legal_after_priority_fix_20260520_053211` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_identity_after_wording_20260520_041701` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_last_pii_doc_after_specific_fix_20260520_085409` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_last_two_after_specific_fix_20260520_085150` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_offset112_20260520_011535` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_offset112_after_final_patches_20260520_021221` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_offset112_after_refund_wording_20260520_023932` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_offset112_after_terminal_v2_20260520_014028` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_offset25_20260519_230127` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_offset25_after_cross_docs_teachers_20260519_234249` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_offset25_after_pricing_20260519_230703` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_offset25_after_tax_camp_20260519_232042` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_offset36_after_exact_terms_20260520_000242` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_offset70_20260520_001935` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_offset70_after_adversarial_terminal_20260520_004246` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/mega_smoke_v3_codex_p0_probe_tax_pii_identity_after_rules_20260520_070113` | old audit inbox input pack; audit results/docs retained |
| 0.9 | `audits/_inbox/mega_smoke_v3_final_v6_3_fast_20260520_205833` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/mega_smoke_v3_final_v6_3_p0_decisions_20260521_010350` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_final_v6_3_reputation_check_20260521_011101` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/mega_smoke_v3_harness_sanity_20260519_214618` | old audit inbox input pack; audit results/docs retained |
| 0.3 | `audits/_inbox/night_cycle_2_operator_status_20260511_v1` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/operator_runtime_status_20260511_v1` | old audit inbox input pack; audit results/docs retained |
| 0.5 | `audits/_inbox/operator_status_after_night_20260511_v1` | old audit inbox input pack; audit results/docs retained |
| 0.2 | `audits/_inbox/project_cleanup_manifest_20260511_v1` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/qa_answer_quality_iter1_20260514` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/qa_answer_quality_iter2_20260514` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/qa_answer_quality_plan_20260514_v1` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/question_catalog_llm_calibration_20260516_1745` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/question_catalog_parallel_checks_20260516_1815` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/question_catalog_rop_policy_20260515` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/question_catalog_sample200_20260516_1759` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/rop_blocker_markup_review_20260514_v1` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/rop_blocker_markup_review_20260514_v2` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/saas_productization_review_20260510` | old audit inbox input pack; audit results/docs retained |
| 0.1 | `audits/_inbox/stage2_cleanup_pipeline_20260510_v1` | old audit inbox input pack; audit results/docs retained |
| 0.0 | `audits/_inbox/super_resolve_strategy_20260513_v1` | old audit inbox input pack; audit results/docs retained |
