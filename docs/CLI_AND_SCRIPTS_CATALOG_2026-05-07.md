# CLI And Scripts Catalog

Дата: 2026-05-07
Обновлено: 2026-05-09

Цель: зафиксировать полный текущий каталог `scripts/`, владельца потока,
production/legacy статус и класс безопасности. Подробные правила запуска
смотреть в `docs/SCRIPT_SAFETY_MATRIX.md`.

## Summary

Текущий inventory:

```zsh
find scripts -maxdepth 1 -type f -print | sort
```

На 2026-05-09 в SaaS/productization baseline-каталоге закреплено `111` файлов.

| Group | Count | Meaning |
|---|---:|---|
| productization | 53 | Mango/SaaS capture, scheduling, product DB/API, quarantine, worker dry-run |
| processing | 32 | ASR/R+A, transcript quality, batch prep, runtime processing |
| crm | 15 | amoCRM/Tallanto/Telegram packs, matching, writeback |
| insights | 8 | sales insight knowledge/readiness/ROP validation |
| ops | 7 | audit, git/devops, smoke, token/usage summaries |

## Status rules

| Status | Meaning |
|---|---|
| `canonical` | Основной поддерживаемый путь для текущего направления |
| `supported` | Полезный рабочий инструмент, но не обязательно основной product path |
| `processing_owned` | Не менять и не запускать из SaaS/productization диалога |
| `guarded_live` | Может писать во внешнюю систему только с explicit confirmation |
| `legacy` | Исторический путь, не использовать как default |
| `review_required` | Перед запуском прочитать код и уточнить side effects |

## Full Catalog

| Script | Owner | Status | Safety class | Notes |
|---|---|---|---|---|
| `scripts/autocommit_push_loop.sh` | ops | legacy | `DANGEROUS_LEGACY` | Auto commit/push loop, не normal workflow |
| `scripts/benchmark_asr_compare.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | ASR benchmark |
| `scripts/benchmark_codex_merge.py` | processing | supported | `SAFE_REPORT_WRITES` | Merge benchmark |
| `scripts/benchmark_codex_merge_models.py` | processing | supported | `SAFE_REPORT_WRITES` | Merge model benchmark |
| `scripts/build_amocrm_delivery_pack.py` | crm | supported | `SAFE_REPORT_WRITES` | Delivery pack, проверять sensitive output |
| `scripts/build_final_processing_coverage_report.py` | processing | processing_owned | `SAFE_REPORT_WRITES` | Coverage report, не ASR/R+A |
| `scripts/build_insight_readiness_report.py` | insights | canonical | `SAFE_REPORT_WRITES` | Insight readiness |
| `scripts/build_messages28_master_exports.py` | processing | processing_owned | `SAFE_REPORT_WRITES` | Message export |
| `scripts/build_outcome_linkage_report.py` | insights | canonical | `SAFE_REPORT_WRITES` | Outcome linkage |
| `scripts/build_pilot_sales_moments.py` | insights | canonical | `SAFE_REPORT_WRITES` | Pilot sales moments |
| `scripts/build_rop_deal_pack.py` | crm | supported | `SAFE_REPORT_WRITES` | ROP deal pack |
| `scripts/build_rop_validation_pack.py` | insights | canonical | `SAFE_REPORT_WRITES` | ROP validation |
| `scripts/build_sales_insight_knowledge_base.py` | insights | canonical | `SAFE_REPORT_WRITES` | Knowledge base |
| `scripts/build_telegram_high_utility_drafts.py` | crm | supported | `SAFE_REPORT_WRITES` | Telegram drafts, not live send |
| `scripts/build_telegram_openclaw_final.py` | crm | supported | `SAFE_REPORT_WRITES` | Telegram final pack |
| `scripts/build_telegram_outreach_pack.py` | crm | supported | `SAFE_REPORT_WRITES` | Outreach pack |
| `scripts/build_transcript_quality_auto_fix_review.py` | processing | processing_owned | `SAFE_REPORT_WRITES` | Transcript quality auto-fix review |
| `scripts/build_transcript_quality_baseline.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Transcript quality branch |
| `scripts/build_transcript_quality_guardrails_dry_run.py` | processing | processing_owned | `SAFE_REPORT_WRITES` | Transcript guardrails dry-run |
| `scripts/enrich_telegram_phones_live.py` | crm | supported | `NETWORK_READ_ONLY` | Live enrichment, no sends |
| `scripts/estimate_token_budget.py` | ops | supported | `SAFE_READ_ONLY` | Token estimation |
| `scripts/evaluate_dialogue_quality.py` | processing | processing_owned | `SAFE_REPORT_WRITES` | Dialogue quality report |
| `scripts/export_tallanto_schema.py` | crm | canonical | `NETWORK_READ_ONLY` | Tallanto schema discovery |
| `scripts/finalize_manual_non_conversation_tail.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Processing finalization |
| `scripts/finalize_messages30_tail.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Processing finalization |
| `scripts/git_bootstrap.sh` | ops | legacy | `DANGEROUS_LEGACY` | Git/bootstrap side effects |
| `scripts/mango_office_amo_snapshot_export.py` | productization | canonical | `NETWORK_READ_ONLY` | Read-only amoCRM snapshot under product root |
| `scripts/mango_office_appliance.py` | productization | canonical | `SAFE_REPORT_WRITES` | Client-hosted appliance command surface |
| `scripts/mango_office_appliance_config_wizard.py` | productization | canonical | `SAFE_REPORT_WRITES` | Appliance install/config readiness checks |
| `scripts/mango_office_appliance_loop_dry_run.py` | productization | canonical | `SAFE_REPORT_WRITES` | Appliance loop dry-run |
| `scripts/mango_office_appliance_service_pack.py` | productization | canonical | `SAFE_REPORT_WRITES` | Generates launchd/systemd templates without installing services |
| `scripts/mango_office_asr_approval_record.py` | productization | supported | `SAFE_REPORT_WRITES` | ASR approval record only |
| `scripts/mango_office_asr_execution_approval_gate.py` | productization | supported | `SAFE_REPORT_WRITES` | Approval gate |
| `scripts/mango_office_asr_execution_plan.py` | productization | supported | `SAFE_REPORT_WRITES` | Execution plan only |
| `scripts/mango_office_asr_scheduler_dry_run.py` | productization | supported | `SAFE_REPORT_WRITES` | Scheduler dry-run |
| `scripts/mango_office_asr_worker_dry_run.py` | productization | supported | `SAFE_REPORT_WRITES` | Worker dry-run |
| `scripts/mango_office_asr_worker_pack.py` | productization | supported | `SAFE_REPORT_WRITES` | Worker pack |
| `scripts/mango_office_asr_worker_pack_verify.py` | productization | supported | `SAFE_READ_ONLY` | Worker pack verify |
| `scripts/mango_office_asr_worker_sandbox_approval_packet.py` | productization | supported | `SAFE_REPORT_WRITES` | Approval packet |
| `scripts/mango_office_asr_worker_sandbox_contract.py` | productization | supported | `SAFE_REPORT_WRITES` | Sandbox contract |
| `scripts/mango_office_asr_worker_sandbox_execution_request.py` | productization | supported | `SAFE_REPORT_WRITES` | Execution request, not execution |
| `scripts/mango_office_asr_worker_sandbox_human_approval.py` | productization | supported | `SAFE_REPORT_WRITES` | Human approval record |
| `scripts/mango_office_asr_worker_sandbox_preflight.py` | productization | supported | `SAFE_REPORT_WRITES` | Preflight report |
| `scripts/mango_office_asr_worker_sandbox_readiness.py` | productization | supported | `SAFE_REPORT_WRITES` | Readiness report |
| `scripts/mango_office_capture_audit.py` | productization | canonical | `SAFE_REPORT_WRITES` | Capture audit |
| `scripts/mango_office_capture_inbox.py` | productization | canonical | `SAFE_REPORT_WRITES` | Capture inbox metadata |
| `scripts/mango_office_capture_stage.py` | productization | canonical | `SAFE_REPORT_WRITES` | Capture staging |
| `scripts/mango_office_controlled_capture_ingest.py` | productization | canonical | `SAFE_REPORT_WRITES` | Controlled shadow-to-inbox ingest plan/apply |
| `scripts/mango_office_crm_entity_resolver.py` | productization | canonical | `SAFE_READ_ONLY` | CRM snapshot -> product call entity candidates |
| `scripts/mango_office_crm_tallanto_mapping_preview.py` | productization | canonical | `SAFE_REPORT_WRITES` | Local AMO/Tallanto mapping preview, no live calls |
| `scripts/mango_office_crm_writeback_preview.py` | productization | canonical | `SAFE_REPORT_WRITES` | CRM writeback preview only |
| `scripts/mango_office_demo_tenant.py` | productization | canonical | `SAFE_REPORT_WRITES` | Builds anonymized demo product root |
| `scripts/mango_office_demo_pilot_playbook.py` | productization | canonical | `SAFE_REPORT_WRITES` | Builds demo/pilot playbook from product-safe data |
| `scripts/mango_office_download_recordings.py` | productization | legacy | `DANGEROUS_LEGACY` | Prefer guarded downloader |
| `scripts/mango_office_manager_identity_map.py` | productization | canonical | `SAFE_REPORT_WRITES` | Manager identity map |
| `scripts/mango_office_payload_archive.py` | productization | canonical | `SAFE_REPORT_WRITES` | Raw payload archive |
| `scripts/mango_office_pipeline_bridge_dry_run.py` | productization | canonical | `SAFE_REPORT_WRITES` | Bridge dry-run |
| `scripts/mango_office_processing_handoff.py` | productization | canonical | `SAFE_REPORT_WRITES` | Handoff only |
| `scripts/mango_office_processing_lifecycle.py` | productization | canonical | `SAFE_REPORT_WRITES` | Capture-to-handoff lifecycle report |
| `scripts/mango_office_processing_acceptance_gates.py` | productization | canonical | `SAFE_REPORT_WRITES` | Read-only gates before processing integration |
| `scripts/mango_office_product_api_http.py` | productization | canonical | `SAFE_READ_ONLY` | Product API HTTP |
| `scripts/mango_office_product_api_readiness.py` | productization | canonical | `SAFE_REPORT_WRITES` | Product API readiness |
| `scripts/mango_office_product_db_admin.py` | productization | review_required | `REVIEW_REQUIRED` | Product DB admin |
| `scripts/mango_office_product_db_bootstrap.py` | productization | canonical | `SAFE_REPORT_WRITES` | Product DB bootstrap |
| `scripts/mango_office_product_ops.py` | productization | canonical | `SAFE_REPORT_WRITES` | Healthcheck, backup, verify, restore dry-run |
| `scripts/mango_office_product_owner_config.py` | productization | canonical | `SAFE_REPORT_WRITES` | Owner config |
| `scripts/mango_office_provider_metadata_sidecar.py` | productization | canonical | `SAFE_REPORT_WRITES` | Provider metadata sidecar |
| `scripts/mango_office_quarantine_import_plan.py` | productization | canonical | `SAFE_REPORT_WRITES` | Import plan |
| `scripts/mango_office_quarantine_materialize.py` | productization | supported | `CONTROLLED_DOWNLOAD` | Quarantine materialization |
| `scripts/mango_office_quarantine_test_ingest.py` | productization | supported | `SAFE_REPORT_WRITES` | Test ingest |
| `scripts/mango_office_recording_asset_ingest.py` | productization | canonical | `SAFE_REPORT_WRITES` | Asset ingest metadata |
| `scripts/mango_office_recording_bridge_dry_run.py` | productization | canonical | `SAFE_REPORT_WRITES` | Recording bridge dry-run |
| `scripts/mango_office_recording_capture_download.py` | productization | canonical | `CONTROLLED_DOWNLOAD` | Guarded recording download |
| `scripts/mango_office_recording_capture_plan.py` | productization | canonical | `SAFE_REPORT_WRITES` | Recording capture plan |
| `scripts/mango_office_recording_quarantine_package.py` | productization | supported | `SAFE_REPORT_WRITES` | Quarantine package |
| `scripts/mango_office_saas_productization_audit.py` | productization | canonical | `SAFE_REPORT_WRITES` | SaaS audit |
| `scripts/mango_office_saas_stage_gates.py` | productization | canonical | `SAFE_REPORT_WRITES` | Stage gates |
| `scripts/mango_office_sanitized_real_demo.py` | productization | canonical | `SAFE_REPORT_WRITES` | Builds masked real-data demo product root |
| `scripts/mango_office_scheduler_control_plane.py` | productization | canonical | `SAFE_READ_ONLY` | Scheduler/supervisor recommended actions, no job execution |
| `scripts/mango_office_scheduler_health.py` | productization | canonical | `SAFE_READ_ONLY` | Scheduler due/failed/locked/stale readiness |
| `scripts/mango_office_scheduler_runtime.py` | productization | canonical | `SAFE_REPORT_WRITES` | Scheduler controlled/dry |
| `scripts/mango_office_shadow_poll.py` | productization | canonical | `NETWORK_READ_ONLY` | Mango shadow poll |
| `scripts/mango_office_tenant_isolation.py` | productization | canonical | `SAFE_REPORT_WRITES` | Tenant isolation report and optional scaffold under product root |
| `scripts/mango_office_tallanto_snapshot_export.py` | productization | canonical | `NETWORK_READ_ONLY` | Read-only Tallanto phone snapshot under product root |
| `scripts/match_priority_contacts_with_tallanto.py` | crm | canonical | `NETWORK_READ_ONLY` | Priority contact matching |
| `scripts/merge_pilot_sales_moment_llm_reviews.py` | insights | supported | `SAFE_REPORT_WRITES` | Merge LLM reviews |
| `scripts/merge_telegram_live_enrichment_chunks.py` | crm | supported | `SAFE_REPORT_WRITES` | Merge enrichment chunks |
| `scripts/monitor_subset_progress.py` | processing | processing_owned | `SAFE_READ_ONLY` | Progress monitor |
| `scripts/normalize_tallanto_contacts.py` | crm | canonical | `SAFE_REPORT_WRITES` | Normalize Tallanto contacts export |
| `scripts/prefill_asr_from_dbs.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | ASR prefill |
| `scripts/prepare_asr_only_date_window.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | ASR-only batch prep |
| `scripts/prepare_contact_history_batch.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Batch prep |
| `scripts/prepare_date_window_subset.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Subset DB prep |
| `scripts/prepare_dual_asr_new_llm_wave.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | ASR/LLM wave prep |
| `scripts/prepare_gigaam_useful_subset.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | GigaAM subset prep |
| `scripts/prepare_history_gap_wave.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | History gap wave |
| `scripts/prepare_llm_wave_from_recommendations.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | LLM wave prep |
| `scripts/prepare_manual_tail_analyze_fallback.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Manual tail fallback |
| `scripts/prepare_message_archive_history_full_cycle.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Archive full cycle |
| `scripts/prepare_message_archive_wave.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Archive wave |
| `scripts/prepare_message_archives_history_full_cycle.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Archive full cycle |
| `scripts/prepare_overnight_full_asr_priority.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Overnight ASR priority |
| `scripts/prepare_phone_history_batch.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Phone history batch |
| `scripts/prepare_priority_history_wave.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Priority history wave |
| `scripts/prepare_remaining_asr_batch.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Remaining ASR batch |
| `scripts/prepare_resolve_analyze_missing_batch.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Missing R+A batch |
| `scripts/prepare_untranscribed_merge_batches.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Untranscribed merge batches |
| `scripts/project_audit.py` | ops | canonical | `SAFE_REPORT_WRITES` | Writes audit artifacts |
| `scripts/promote_ai_review_to_amo_ready.py` | crm | canonical | `SAFE_REPORT_WRITES` | AMO-ready export, not live write |
| `scripts/repair_and_move_message_archives.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Repair/move archive files |
| `scripts/requeue_secondary_backfill.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Requeue/backfill |
| `scripts/run_analyze_ab_test.py` | processing | processing_owned | `PROCESSING_MUTATES_DB` | Analyze A/B workflow |
| `scripts/run_pilot_sales_moment_llm_review.py` | insights | supported | `NETWORK_READ_ONLY` | LLM review, no CRM writes |
| `scripts/smoke_test_tallanto.py` | crm | supported | `NETWORK_READ_ONLY` | Tallanto smoke-check |
| `scripts/start_autocommit_push.sh` | ops | legacy | `DANGEROUS_LEGACY` | Starts auto commit/push loop |
| `scripts/stop_autocommit_push.sh` | ops | supported | `SAFE_READ_ONLY` | Stops auto commit/push loop |
| `scripts/summarize_merge_usage.py` | ops | supported | `SAFE_READ_ONLY` | Merge usage summary |
| `scripts/write_amo_ready_contacts.py` | crm | guarded_live | `CRM_LIVE_GUARDED` | Default dry-run; live requires confirmation |
| `scripts/write_recent_actionable_deals.py` | crm | guarded_live | `CRM_LIVE_GUARDED` | Default dry-run; live requires confirmation |

## Canonical productization commands

Use these first for SaaS/productization work:

```zsh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_shadow_poll.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_amo_snapshot_export.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_appliance.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_controlled_capture_ingest.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_recording_capture_plan.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_processing_lifecycle.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_crm_entity_resolver.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_crm_tallanto_mapping_preview.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_crm_writeback_preview.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_demo_tenant.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_appliance_config_wizard.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_product_ops.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_appliance_service_pack.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_tenant_isolation.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_tallanto_snapshot_export.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_demo_pilot_playbook.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_processing_acceptance_gates.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_scheduler_control_plane.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_scheduler_health.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_sanitized_real_demo.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_appliance_loop_dry_run.py --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/mango_office_saas_stage_gates.py --help
```

## Commands requiring extra attention

- `scripts/write_amo_ready_contacts.py`
- `scripts/write_recent_actionable_deals.py`
- `scripts/mango_office_product_db_admin.py`
- `scripts/mango_office_download_recordings.py`
- all `prepare_*`, `finalize_*`, `prefill_*`, `requeue_*`, `repair_*` processing scripts
- `scripts/start_autocommit_push.sh`
- `scripts/autocommit_push_loop.sh`

## Next catalog maintenance

1. Keep this file and `docs/SCRIPT_SAFETY_MATRIX.md` synchronized when new scripts are added.
2. New product scripts should start in `SAFE_REPORT_WRITES` or `NETWORK_READ_ONLY`.
3. Any live external write must be `guarded_live` and have focused tests.
4. Any processing script remains owned by the processing dialog unless explicitly reassigned.
