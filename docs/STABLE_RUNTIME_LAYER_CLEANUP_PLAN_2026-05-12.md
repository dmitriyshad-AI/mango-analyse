# Stable Runtime Layer Cleanup Plan, 2026-05-12

Status: planning only. No `stable_runtime` cleanup from this plan has been executed yet.

Goal: clean `stable_runtime` by meaning and dependency layer, not by folder size. The safe order is:

1. Pin current production anchors.
2. Mark recent operational evidence.
3. Archive or delete clearly superseded waves.
4. Only then clean large historical processing waves.

## Current State

- `stable_runtime` size: about `29G`.
- Top-level directories: `300`.
- Regular files: `255479`.
- Symlinks: `95945`.
- Regular audio inside `stable_runtime`: `0` in the major ASR/history batch folders checked; the remaining audio references are symlinks to source audio.

## Current Runtime Anchors

These paths are pinned by `stable_runtime/CURRENT_RUNTIME.json` and must not be deleted or moved until a new runtime contract replaces them:

| Role | Path | Size |
|---|---|---:|
| Active export | `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict` | `214M` |
| Canonical DB | `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/canonical_calls_master.db` | `1.4G` |
| Canonical summary | `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/summary.json` | `4K` |
| Stage15 quality summary | `stable_runtime/transcript_quality_stage15_export_gate_20260510_v11_frozen_gate/summary.json` | `16K` |
| CRM quality summary | `stable_runtime/crm_writeback_quality_gate_20260510_v10_crm_text_quality_strict/summary.json` | `8K` |
| AMO queue summary | `stable_runtime/amo_writeback_queue_20260512_after_stage51_repair_v1/summary.json` | `40K` |
| Product appliance | `_local_archive_mango_api_downloads_20260507/product_appliance` | `223M` |

Also keep as operational evidence unless a later explicit decision says otherwise:

- `stable_runtime/amocrm_runtime`
- `stable_runtime/amo_live_stage*_20260512_*`
- `stable_runtime/amo_orphan_lookup*_20260512_*`
- `stable_runtime/amo_writeback_queue_20260512_after_stage51_repair_v1`
- `stable_runtime/tallanto_write_off_visits_history_20260512`
- `stable_runtime/kc_knowledge_base_20260512`
- `stable_runtime/operator_status_20260511_v4_waiting_work`
- `stable_runtime/deal_aware_preview_50_20260512_v4`

## Size By Layer

| Layer | Dirs | Size | Meaning |
|---|---:|---:|---|
| Message/history waves | `23` | `9357.1 MB` | Old customer-history and message-history build waves. |
| ASR batches | `12` | `5876.5 MB` | Historical ASR-only batches; mostly symlink-based audio references plus outputs. |
| Resolve/Analyze | `9` | `4360.5 MB` | R+A output waves, dominated by `ra_missing_all_20260506`. |
| Canonical masters | `4` | `3108.1 MB` | Current and previous canonical call master builds. |
| Quality audits | `114` | `2428.4 MB` | Non-conversation, transcript-quality, Claude/GPT audit waves. |
| Sales exports | `26` | `2314.6 MB` | Old and current AMO-ready/sales export layers. |
| Reports/knowledge | `45` | `987.2 MB` | ROP validation packs, insight reports, sales knowledge outputs. |
| Infra/misc | `16` | `523.1 MB` | Backups, benchmarks, migrations, profiles, runs. |
| CRM/AMO live evidence | `29` | `149.4 MB` | Small but important live/dry-run/readback traces. |
| Tallanto | `5` | `166.5 MB` | Tallanto snapshots and write-off visit history. |

## Cleanup Order

### Layer 0: Freeze Anchors

Before deleting more runtime data:

- Keep every path in `CURRENT_RUNTIME.json`.
- Keep `stable_runtime/CANONICAL_EXPORT.txt`.
- Keep current AMO/Tallanto evidence from 2026-05-11 and 2026-05-12.
- Keep the latest operator status and deal-aware preview `v4`.

Decision needed: whether `amo_writeback_queue_20260512_after_stage51_repair_v1` should fully replace `amo_writeback_queue_20260510_v2_production` as the named current AMO queue layer.

### Layer 1: Clearly Superseded Same-Day Versions

These are the best first candidates because they have a clear newer version:

- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v1`
- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v2_crm_text_quality`
- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v3_crm_text_quality`
- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v4_crm_text_quality_strict`

Keep:

- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict`

Recommended action: archive or delete v1-v4 after confirming no docs/scripts reference them as active inputs.

### Layer 2: Superseded Canonical Masters

Current:

- `stable_runtime/canonical_master_20260510_after_quality_backfill_v1`

Likely superseded:

- `stable_runtime/canonical_master_20260509_v1`
- `stable_runtime/canonical_master_20260509_dry_run_v1`
- `stable_runtime/canonical_master_20260509_dry_run_v2`

Recommended action: keep current canonical, archive/delete previous canonical builds only after recording the current canonical summary in docs and confirming `CURRENT_RUNTIME.json` stays green.

### Layer 3: Historical ASR Batches

Major ASR batch folders currently use symlinked audio references, not physical audio copies:

- `stable_runtime/jun_jul_aug_2025_asr_only_20260503` (`2.9G`, `8793` audio symlinks)
- `stable_runtime/apr_may_2025_asr_only_20260502` (`1.1G`, `4024` audio symlinks)
- `stable_runtime/night_asr_3000_20260328` (`830M`, `3000` audio symlinks)
- `stable_runtime/oct_nov_2025_asr_only_remaining_all_20260505` (`256M`, `9037` audio symlinks)
- `stable_runtime/sep2025_asr_only_remaining_all_20260504` (`221M`, `7599` audio symlinks)

The current readiness gate says missing actionable ASR is `0`, so these are likely historical processing waves. They should be archived or deleted only after confirming the current canonical DB contains their useful output.

Recommended action: create a per-folder summary of DB/JSON/CSV outputs and references, then move old ASR waves to an external archive or delete them in monthly groups.

### Layer 4: Resolve/Analyze Waves

Largest folder:

- `stable_runtime/ra_missing_all_20260506` (`4.2G`)

This is too large to delete by size alone. It may be the source wave that closed R+A gaps before the current canonical master.

Recommended action: verify that every useful result from this wave is represented in `canonical_master_20260510_after_quality_backfill_v1`. If yes, archive or delete `ra_missing_all_20260506` and keep only a compact summary/report.

### Layer 5: Message/Customer-History Waves

Large historical waves:

- `stable_runtime/messages28_phone_history_llm_wave_20260409` (`1.2G`)
- `stable_runtime/messages28_phone_history_gap_wave_20260410` (`1.1G`)
- `stable_runtime/messages28_phone_history_asr_20260408` (`758M`)
- `stable_runtime/history_cohort_20260319_20260326` (`790M`)
- `stable_runtime/history_remaining_excl_done_20260407` (`682M`)

These should not be deleted until the new unified customer history store has imported the needed phone history.

Recommended action: after the current customer timeline SQLite store is populated and checked, keep only the final imported dataset plus import manifest, then archive/delete old history waves.

### Layer 6: Quality Audit Waves

Examples:

- `stable_runtime/non_conversation_hard_gate_owner_dry_run_20260509_v1` to `v5`
- `stable_runtime/non_conversation_hard_gate_gpt_review_20260509_priority*`
- `stable_runtime/transcript_quality_pipeline_*`
- `stable_runtime/claude_stage15_*`

These are important for traceability but not all need to stay in full form. Keep final reports and frozen gates; archive/delete intermediate reviews after the final gate is documented.

Recommended action: keep final/frozen artifacts, summarize intermediate waves, then delete old bulky review folders.

## Practical Next Steps

1. Create a protected-path list from `CURRENT_RUNTIME.json` plus the 2026-05-12 operational evidence.
2. Run a read-only reference scan for Layer 1 and Layer 2 candidates.
3. Delete or archive same-day superseded exports v1-v4.
4. Delete or archive superseded canonical 20260509 builds.
5. Build a compact inventory for `ra_missing_all_20260506` and the large ASR batches.
6. Only after current customer timeline import is verified, clean the old message-history waves.

## Decisions Needed

1. Delete same-day sales export v1-v4 or move them to an external archive?
2. Delete previous canonical masters from 2026-05-09 or keep one rollback copy?
3. For old ASR/R+A waves: delete locally after proving canonical coverage, or archive outside the project?
4. For message-history waves: wait until the unified customer timeline is fully populated, or preserve all old waves for audit history?
