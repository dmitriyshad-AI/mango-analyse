# Next safe cleanup candidates, 2026-05-12

Status: cleanup in progress. Approved items have been deleted; remaining items require separate decision.

Already done:
- Deleted `6370` exact duplicate ASR audio files from old external M1 batch folders.
- Freed `2162964384` bytes (`2.0 GiB`).
- Full local manifest: `audits/_results/EXACT_AUDIO_DUPLICATES_DELETE_CANDIDATES_2026-05-12.csv`.
- Deleted old ASR batch remnants after owner approval:
  - `stable_runtime/external_m1_jan_mar_2025_asr_only_20260504` (`81` files, `5352986` bytes).
  - `stable_runtime/external_m1_jan2025_test300_20260503` (`1008` files, `58112651` bytes).
- Deleted additional approved cleanup targets:
  - `stable_runtime/venv_stable.broken_20260407` (`30966` files, `959912011` bytes).
  - `.codex_workers` (`8382` files, `80200599` bytes).
  - `_cleanup_quarantine_20260510_stage2` (`12380` files, `1150510505` bytes), including quarantined `external_m1_jan_mar_2025_asr_only_20260504_result_20260506_103021`.
  - Root-level `external_m1_jan_mar_2025_asr_only_20260504_result_20260506_103021` was already missing.
  - Root-level `external_m1_jan_mar_2025_asr_only_20260504_result_20260506_103021.zip` still exists (`10M`); it was not deleted because it is a separate zip file.
- Checked `2026-03-05-21-06-49-ч1` and `2026-03-05-21-06-49-ч2` against canonical `2026-03-09--26` by filename, size, and SHA-256:
  - exact duplicates found: `0`.
  - files left untouched: `4975`.
  - report: `docs/EXACT_AUDIO_DUPLICATES_2026-03-05_DELETE_2026-05-12.md`.
- Deleted `.cache/llm_responses` after owner approval:
  - files: `79476`.
  - payload bytes: `153671245`.
  - `.cache` now only contains small non-production leftovers.
- Deleted intermediate deal-aware preview packs after owner approval:
  - `stable_runtime/deal_aware_preview_50_20260512_v1`.
  - `stable_runtime/deal_aware_preview_50_20260512_v2`.
  - `stable_runtime/deal_aware_preview_50_20260512_v3`.
  - `stable_runtime/deal_aware_preview_50_20260512_v4` was kept.
- Repacked `_local_archive_20260424/source_archives/messages(1).zip` after owner approval:
  - removed duplicate audio entries matching `2026-03-09--26`: `1880`.
  - kept unique audio entries: `231`.
  - kept non-audio entries: `1`.
  - zip size changed from `901236473` bytes to `96833095` bytes.
  - report: `docs/MESSAGES1_ZIP_DUPLICATE_REMOVAL_2026-05-12.md`.

## Highest-confidence delete candidates

These items are ignored by Git and do not contain canonical project source code.

Update 2026-05-21:

- `.venv-asrbench` is no longer a safe cleanup candidate. It is the current preferred ASR runtime for fresh Mango calls and must be kept until a replacement ASR runtime passes `scripts/check_asr_runtime_contract.py`.
- Do not delete `.venv-asrbench` while any ASR UI batch is active.
- `stable_runtime/venv_stable.broken_20260407` was deleted, but legacy launchers still reference it. Those launchers are not safe entrypoints until updated.

| Path | Size | Files | Why it is safe | Tradeoff |
|---|---:|---:|---|---|
| `.cache/llm_responses` | `311M` | `79478` | Local LLM response cache. | Deleted after owner approval. |
| `.venv-asrbench` | `931M` | `28387` | Reclassified 2026-05-21: current ASR runtime, keep. | Delete only after a replacement runtime is verified. |
| `stable_runtime/venv_stable.broken_20260407` | `997M` | `30963` | Old broken virtual environment, explicitly ignored by Git. | Deleted after owner approval. |
| `.codex_workers` | `96M` | `8373` | Local Codex worker homes from previous parallel work. | Deleted after owner approval. |
| `_cleanup_quarantine_20260510_stage2` | `1.1G` | `12380` | Prior cleanup quarantine; manifest marks all 30 moved items as `SAFE`. | Deleted after owner approval. |

Remaining expected impact if `.venv-asrbench` is deleted after replacement: about `931M` and about `28387` files removed.

Decision notes:
- `.cache/llm_responses` mostly contained cached `analyze` responses (`277M`, `70801` files) and transcript-quality review responses (`30M`, `7731` files). It has been deleted. This did not delete final exports, transcripts, DBs, or source audio.
- `.venv-asrbench` is referenced by the GUI and is currently the preferred ASR backend Python for fresh calls. Keep it until a replacement passes the ASR runtime preflight.
- `stable_runtime/venv_stable.broken_20260407` was a broken old virtual environment and has been deleted.
- `.codex_workers` contained only local worker homes (`ra1`, `ra2`, `ra3`) with Codex config/auth/cache files. No project source or audit result was found there. It has been deleted.
- `_cleanup_quarantine_20260510_stage2` contained 30 previously moved items, all marked `SAFE` in its manifest. It has been deleted.
- `_local_archive_20260424/source_archives/messages(1).zip` was not fully redundant: `1880` audio entries matched canonical `2026-03-09--26` by content, but `231` audio entries were not present in canonical by content. The zip was repacked to keep only the unique audio plus `index.html`.
- `stable_runtime/deal_aware_preview_50_20260512_v1` to `v4` are small ROP/deal-aware preview packs. `v4` is the latest and strictest; `v1` to `v3` are likely intermediate review artifacts, not production anchors.

## Old ASR batch remnants

The two old external M1 ASR batch folders were removed after explicit owner approval. They had `1089` files total and no regular audio files under `batch_asr_only` after the exact duplicate cleanup.

## Not recommended for immediate deletion

| Path | Size | Reason |
|---|---:|---|
| `2026-03-09--26` | `24G` | Canonical call audio folder. Owner explicitly said it is needed. |
| `telegram_exports (2)` | `1.2G` | Owner explicitly said it is needed. |
| `_local_archive_mango_api_downloads_20260507/product_appliance` | `223M` | Referenced by current runtime state. |
| `stable_runtime/canonical_master_*`, `sales_master_export_*`, `crm_writeback_quality_gate_*`, `amocrm_runtime`, `tallanto_*` | varies | These are product/runtime evidence, CRM/Tallanto state, or current processing outputs. Delete only after a separate data-retention decision. |
| `.git` | `2.4G` | Repository history; do not manually delete. |
