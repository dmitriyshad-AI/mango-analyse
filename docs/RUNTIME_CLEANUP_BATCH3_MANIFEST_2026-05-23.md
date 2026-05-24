# Runtime cleanup batch 3 - 2026-05-23

Способ: перенос в macOS Trash, не безвозвратное удаление.

Trash root: `/Users/dmitrijfabarisov/.Trash/MangoAnalyse_runtime_cleanup_batch3_20260522T233839Z`
Moved: `30` paths
Skipped missing: `0` paths
Total moved: `2.403` GiB

## Moved paths

| Путь | Размер, байт | Файлов | Причина |
|---|---:|---:|---|
| `stable_runtime/canonical_master_20260521_after_mango_update_v1` | 1572388517 | 2 | superseded by canonical_master_20260523_audio_working_store_v1; DB compare matched except audio source path |
| `stable_runtime/sales_master_export_20260521_after_mango_update_v4_runtime_acceptance` | 245948270 | 5 | superseded by sales_master_export_20260523_audio_working_store_v1; key CSVs have identical content hashes |
| `product_data/canonical_audio_store_20260516_v1` | 366267929 | 23 | old audio-store metadata/projection; actual audio already moved to the 20260523 working store and old audio dir was moved to Trash |
| `stable_runtime/ra_pending_mango_api_20260517_v1` | 44873225 | 2730 | old pending R+A layer; current runtime has missing ASR/R+A = 0 |
| `stable_runtime/history_remaining_excl_done_20260407` | 89433276 | 14 | historical remaining-history layer; no current runtime dependency |
| `stable_runtime/start_remaining_history_resolve4.sh` | 1952 | 1 | legacy launcher that points to removed history_remaining_excl_done_20260407 |
| `АКТУАЛЬНО_AI_review.xlsx` | 4792 | 1 | stale root Excel; empty technical sheet, replaced by current runtime artifacts |
| `АКТУАЛЬНО_AMO_ready.xlsx` | 2675024 | 1 | stale root AMO Excel; write_amo_ready_contacts now defaults to active CANONICAL_EXPORT CSV |
| `АКТУАЛЬНО_Tallanto_match_issues.xlsx` | 1577317 | 1 | stale root Tallanto issue export; current contact/review data is in active 20260523 export |
| `АКТУАЛЬНО_Звонки_общая_таблица.xlsx` | 9671394 | 1 | stale root calls Excel; replaced by active master_calls_ru.csv with 65,939 actionable rows |
| `АКТУАЛЬНО_История_еще_не_добита.xlsx` | 4793 | 1 | stale root Excel; empty technical sheet |
| `АКТУАЛЬНО_Контакты_для_продаж.xlsx` | 4470706 | 1 | stale root contacts Excel; replaced by active master_contacts_ru.csv |
| `АКТУАЛЬНО_Полный_пакет_экспорта.xlsx` | 18279661 | 1 | stale root all-in-one export; replaced by active 20260523 CSV export folder |
| `АКТУАЛЬНО_РОП_очередь_сделок_30д_live.xlsx` | 548522 | 1 | stale root ROP deal queue; current AMO/deal-aware layers use current runtime/audit packs |
| `stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v2_hybrid_reuse` | 25968527 | 11 | intermediate Stage12 KB export superseded by v11 frozen gate and current runtime |
| `stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v4_stage15_hardened` | 34776193 | 11 | intermediate Stage15 KB export superseded by later safety/frozen-gate layers |
| `stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v5_claude_safety_fix` | 32261755 | 11 | intermediate sanitizer fix export superseded by later frozen-gate layers |
| `stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v6_claude_safety_fix` | 32261949 | 11 | intermediate sanitizer fix export superseded by later frozen-gate layers |
| `stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v7_location_fix` | 32263198 | 11 | intermediate sanitizer fix export superseded by later frozen-gate layers |
| `stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v8_orphan_surname_fix` | 32268033 | 11 | intermediate sanitizer fix export superseded by v10/v11 frozen-gate layers; docs keep historical report |
| `stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v10_fixpoint` | 32316726 | 11 | intermediate fixpoint export superseded by v11 frozen-gate layer |
| `stable_runtime/amo_live_stage51_summary_repair_20260512_v1` | 352385 | 7 | superseded dry-run repair pack; no approval/readback history, replaced by stage51_textarea_repair_v2 |
| `stable_runtime/amo_live_stage55_20260511_v2_wrong_person_gate_check` | 183940 | 11 | failed wrong-person gate check pack; no downstream refs |
| `stable_runtime/crm_writeback_quality_gate_20260510_v5_product_gate` | 268730 | 11 | early product gate superseded by later/current CRM gates; no refs |
| `stable_runtime/amo_writeback_queue_20260516_after_mango_update_v1` | 83000 | 7 | old AMO queue superseded by current 20260523 AMO queue |
| `stable_runtime/amo_writeback_queue_20260521_after_mango_update_v4_runtime_acceptance` | 54717 | 7 | old AMO queue superseded by current 20260523 AMO queue |
| `stable_runtime/crm_writeback_quality_gate_20260513_human_history_v1` | 301383 | 11 | old human-history CRM quality gate superseded by current 20260523 gate |
| `stable_runtime/crm_writeback_quality_gate_20260513_human_history_v2` | 301387 | 11 | old human-history CRM quality gate superseded by current 20260523 gate |
| `stable_runtime/crm_writeback_quality_gate_20260513_human_history_v3` | 301399 | 11 | old human-history CRM quality gate superseded by current 20260523 gate |
| `stable_runtime/crm_writeback_quality_gate_20260521_after_mango_update_v4_runtime_acceptance` | 52880 | 11 | old CRM quality gate superseded by current 20260523 gate |

## Protected current layers

Не трогались: текущий `CURRENT_RUNTIME.json`, активный export/canonical DB 20260523, текущий audio working store, mail archive, Telegram exports, customer timeline, текущий KB v11/ROP pack v11.

## Post-check corrections

После дополнительной проверки восстановлены из корзины:

- `stable_runtime/history_remaining_excl_done_20260407` - текущая canonical DB и active export сохраняют provenance-ссылки на этот source DB слой;
- `stable_runtime/start_remaining_history_resolve4.sh` - восстановлен вместе с указанным source DB как связанный legacy launcher.

Дополнительно исправлен текущий active export:

- `stable_runtime/sales_master_export_20260523_audio_working_store_v1/master_calls_ru.csv` был обновлён из текущей canonical DB;
- все `65 939` строк теперь указывают на `product_data/audio_working_store_20260523_v1`;
- старых audio-path ссылок и отсутствующих файлов после ремонта: `0`;
- отчёт: `docs/ACTIVE_EXPORT_AUDIO_PATH_REPAIR_2026-05-23.json`.

Финальная чистая экономия после восстановления: `2.32` GiB.
