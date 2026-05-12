# Stable Runtime Reference Check, 2026-05-12

Status: read-only reference check. Nothing was deleted by this check.

Checked cleanup candidates:

- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v1`
- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v2_crm_text_quality`
- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v3_crm_text_quality`
- `stable_runtime/sales_master_export_20260510_after_quality_backfill_v4_crm_text_quality_strict`
- `stable_runtime/canonical_master_20260509_v1`
- `stable_runtime/canonical_master_20260509_dry_run_v1`
- `stable_runtime/canonical_master_20260509_dry_run_v2`

## Findings

These candidates are not pinned by `stable_runtime/CURRENT_RUNTIME.json`.

No symlinks were found that point to these candidate folders.

Current pinned runtime uses:

- current export: `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict`
- current canonical: `stable_runtime/canonical_master_20260510_after_quality_backfill_v1`

## Active Reference Risk

One script still has an outdated default output root:

- `scripts/build_post_backfill_amo_ready_export.py`
- current default: `stable_runtime/sales_master_export_20260510_after_quality_backfill_v1`

This is not a live dependency on old data, but it is a cleanup hazard: a future run without `--out-root` could write into or recreate the old `v1` folder name. Update this default before deleting `v1` to `v4`.

Other matches are historical docs or tests:

- docs mention `v1` to `v4` as previous build stages.
- docs mention `canonical_master_20260509_*` as provenance of the first canonical build.
- tests use old path names as fixtures, not runtime dependencies.

## Sales Export Chain

All `sales_master_export_20260510_after_quality_backfill_v1` to `v5` point to the same current canonical DB and Stage15 quality summary. They differ by filtering/quality rules:

| Folder | Generated | AMO-ready rows | Meaning |
|---|---|---:|---|
| `v1` | `2026-05-10T13:32:15+00:00` | `103` | First post-backfill export. |
| `v2_crm_text_quality` | `2026-05-10T16:16:43+00:00` | `93` | Stricter CRM text filtering. |
| `v3_crm_text_quality` | `2026-05-10T16:19:54+00:00` | `85` | Stricter follow-up filtering. |
| `v4_crm_text_quality_strict` | `2026-05-10T16:22:41+00:00` | `75` | Strict version before final tightening. |
| `v5_crm_text_quality_strict` | `2026-05-10T16:25:02+00:00` | `69` | Current pinned export. |

Conclusion: `v1` to `v4` are superseded intermediate exports. They can be deleted or archived after the script default is fixed.

Approximate space:

- `v1` to `v4` together: about `851M`.

## Canonical Master Chain

Current:

- `stable_runtime/canonical_master_20260510_after_quality_backfill_v1`
- size: about `1.5G`
- validation: passed
- actionable calls: `64832`
- missing ASR: `0`
- missing R+A: `0`

Older:

- `stable_runtime/canonical_master_20260509_v1`
- `stable_runtime/canonical_master_20260509_dry_run_v1`
- `stable_runtime/canonical_master_20260509_dry_run_v2`

The old canonical layers also passed validation and have the same high-level counts. They are provenance/rollback artifacts, not current runtime inputs.

Approximate space:

- `canonical_master_20260509_v1`: about `1.5G`
- two dry-runs: about `90M`

Recommended cleanup:

1. Delete or archive the two dry-run folders first.
2. Decide whether to keep `canonical_master_20260509_v1` as one rollback copy. If rollback is not needed, archive/delete it after keeping its report docs.

## History/Message Layer

The `history/message` estimate of about `9.3G` is not Codex conversation history. It is old call/customer-history processing output inside `stable_runtime`.

It contains SQLite DBs, DB backups, selection manifests, selected call lists, phone lists, and symlinked audio references from older waves of building customer history by phone number.

Largest folders:

| Folder | Size | Meaning |
|---|---:|---|
| `messages28_phone_history_llm_wave_20260409` | `1.2G` | LLM wave for phone-history processing. |
| `messages28_phone_history_gap_wave_20260410` | `1.1G` | Gap-filling wave for phone history. |
| `messages28_phone_history_asr_20260408` | `758M` | ASR wave for message/history batch. |
| `history_cohort_20260319_20260326` | `790M` | History cohort processing DBs/backups. |
| `history_remaining_excl_done_20260407` | `682M` | Remaining history processing excluding completed items. |
| `messages34_phone_history_full_20260501` | `606M` | Later full phone-history batch. |
| `top100_history_wave1_20260331` | `597M` | Top-priority contact history wave. |
| `messages32_33_phone_history_full_20260422` | `535M` | Full phone-history batch for messages 32/33. |

Most of these folders contain DBs and backup DBs. The audio references inside the checked message/history and ASR batch folders are symlinks, not physical audio copies.

Do not delete this layer until the new unified customer timeline store has imported and verified the needed phone history.

## Recommendation

Next safe cleanup sequence:

1. Update `scripts/build_post_backfill_amo_ready_export.py` so its default output path no longer targets `v1`.
2. Delete/archive `sales_master_export_20260510_after_quality_backfill_v1` to `v4`.
3. Delete/archive `canonical_master_20260509_dry_run_v1` and `canonical_master_20260509_dry_run_v2`.
4. Decide whether `canonical_master_20260509_v1` stays as rollback or is archived outside the project.
5. Leave history/message waves untouched until the new customer timeline store is populated and checked.
