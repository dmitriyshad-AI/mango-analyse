# Script Safety Matrix

Дата: 2026-05-09

Назначение: дать один понятный справочник по запуску скриптов проекта. Матрица
не удаляет legacy-скрипты и не запрещает live-доступ, но разделяет команды на
безопасные, отчетные, сетевые, runtime-mutating и live-write.

## Правила использования

1. Если скрипта нет в этой матрице, считать его `REVIEW_REQUIRED` до проверки.
2. `SAFE_READ_ONLY` можно запускать для диагностики.
3. `SAFE_REPORT_WRITES` может создавать файлы отчетов, но не должен писать в
   runtime-БД, CRM или запускать ASR/R+A.
4. `NETWORK_READ_ONLY` может читать внешние API, но не должен менять внешние
   системы.
5. `CONTROLLED_DOWNLOAD` может скачивать файлы только в явно заданные staging,
   inbox или quarantine папки.
6. `PROCESSING_MUTATES_DB` принадлежит processing-диалогу и не запускается в
   SaaS/productization ветке без отдельного решения.
7. `CRM_LIVE_GUARDED` имеет live-доступ, но live-запись разрешается только через
   явный флаг и контрольную строку.
8. `DANGEROUS_LEGACY` не запускать как обычную команду. Сначала читать код,
   делать dry-run/backup/approval.

## Safety classes

| Class | Meaning | Typical approval |
|---|---|---|
| `SAFE_READ_ONLY` | Только чтение/печать, без файловых и внешних write effects. | Не требуется |
| `SAFE_REPORT_WRITES` | Пишет локальные отчеты/JSON/CSV/docs, не меняет runtime-БД и внешние системы. | Не требуется, если output path понятен |
| `NETWORK_READ_ONLY` | Читает Mango/AMO/Tallanto/Telegram/LLM API, но не пишет туда. | Нужны credentials, но не отдельный live-write approval |
| `CONTROLLED_DOWNLOAD` | Скачивает записи или payload в staging/quarantine. | Нужен явный output path |
| `PROCESSING_MUTATES_DB` | Создает batch, меняет processing/runtime DB или запускает ASR/R+A workflow. | Только processing-диалог |
| `CRM_LIVE_GUARDED` | Может писать в AMO/Tallanto/CRM при явном подтверждении. | Нужен explicit live confirmation |
| `DANGEROUS_LEGACY` | Исторический или потенциально опасный путь. | Ручной разбор перед запуском |
| `REVIEW_REQUIRED` | Назначение не закреплено или требует отдельного аудита. | Ручной разбор перед запуском |

## 2026-05-11 AMO waiting / duplicate workflow additions

| Script / generated command | Owner | Safety class | Side effects / risk | Recommended use |
|---|---|---|---|---|
| `scripts/run_amo_waiting_autonomous_work.py` | crm/productization | `SAFE_REPORT_WRITES` | Строит waiting-work pack, generated commands, no AMO write. | Запускать пока сотрудники объединяют дубли. |
| `scripts/build_amo_duplicate_staff_tasks.py` | crm/productization | `SAFE_REPORT_WRITES` | Строит задачи сотрудникам по дублям, no AMO write. | Передать сотрудникам для ручной склейки. |
| `scripts/check_amo_duplicate_post_merge_recheck.py` | crm/productization | `SAFE_REPORT_WRITES` | Проверяет уже созданный dry-run report, no network/write. | После сообщения сотрудников `готово`. |
| `scripts/run_amo_duplicate_after_staff_done.py` | crm/productization | `SAFE_REPORT_WRITES` | Строит bounded candidates после recheck, no AMO write. | После post-merge recheck. |
| `generated next_*_quality_gate_command.sh` | crm/productization | `SAFE_REPORT_WRITES` | Local CRM quality gates, пишет только отчеты. | Перед dry-run/live stage. |
| `generated next_*real_tunnel_dry_run_command.sh` | crm/productization | `NETWORK_READ_ONLY` | AMO lookup dry-run, no `--execute-live-write`. | Только при поднятом tunnel; не live-write. |
| `generated next_readback_missing_commands.sh` | crm/productization | `NETWORK_READ_ONLY` | AMO readback only, no write. | Перед refresh already-written строк. |

## Full script inventory

| Script | Owner | Safety class | Side effects / risk | Recommended use |
|---|---|---|---|---|
| `autocommit_push_loop.sh` | ops | `DANGEROUS_LEGACY` | Автоматический git commit/push loop. | Не использовать как нормальный workflow. |
| `benchmark_asr_compare.py` | processing | `PROCESSING_MUTATES_DB` | ASR/benchmark workflow может быть тяжелым. | Только processing-диалог. |
| `benchmark_codex_merge.py` | processing | `SAFE_REPORT_WRITES` | Benchmark/report artifacts. | Запускать только на тестовом input. |
| `benchmark_codex_merge_models.py` | processing | `SAFE_REPORT_WRITES` | Benchmark/report artifacts. | Запускать только на тестовом input. |
| `build_amocrm_delivery_pack.py` | crm | `SAFE_REPORT_WRITES` | Собирает delivery pack, может включать sensitive data. | Проверять output перед передачей. |
| `build_final_processing_coverage_report.py` | processing | `SAFE_REPORT_WRITES` | Пишет coverage report в `stable_runtime`. | Только read/report, без ASR/R+A. |
| `build_insight_readiness_report.py` | insights | `SAFE_REPORT_WRITES` | Пишет readiness report. | Безопасно для insight-аудита. |
| `build_messages28_master_exports.py` | processing | `SAFE_REPORT_WRITES` | Экспортные файлы. | Проверять output path. |
| `build_outcome_linkage_report.py` | insights | `SAFE_REPORT_WRITES` | Пишет linkage report. | Безопасно на копиях/exports. |
| `build_pilot_sales_moments.py` | insights | `SAFE_REPORT_WRITES` | Пишет pilot moments. | Безопасно для sales insight. |
| `build_rop_deal_pack.py` | crm | `SAFE_REPORT_WRITES` | Формирует ROP pack, не должен писать в CRM. | Проверять входные данные. |
| `build_rop_validation_pack.py` | insights | `SAFE_REPORT_WRITES` | Пишет validation pack. | Безопасно для ROP-review. |
| `build_sales_insight_knowledge_base.py` | insights | `SAFE_REPORT_WRITES` | Пишет knowledge base artifacts. | Безопасно для insight layer. |
| `build_telegram_high_utility_drafts.py` | crm | `SAFE_REPORT_WRITES` | Генерирует drafts, не должен отправлять. | Проверять перед live-отправкой. |
| `build_telegram_openclaw_final.py` | crm | `SAFE_REPORT_WRITES` | Генерирует final pack. | Проверять secrets/contacts в output. |
| `build_telegram_outreach_pack.py` | crm | `SAFE_REPORT_WRITES` | Генерирует outreach pack. | Не считать live-send. |
| `build_transcript_quality_auto_fix_review.py` | processing | `SAFE_REPORT_WRITES` | Принадлежит transcript quality ветке, auto-fix review artifacts. | Не трогать в этом диалоге. |
| `build_transcript_quality_baseline.py` | processing | `PROCESSING_MUTATES_DB` | Принадлежит transcript quality ветке. | Не трогать в этом диалоге. |
| `build_transcript_quality_guardrails_dry_run.py` | processing | `SAFE_REPORT_WRITES` | Принадлежит transcript quality ветке, dry-run guardrail report. | Не трогать в этом диалоге. |
| `build_transcript_quality_stage14_comparison.py` | processing | `SAFE_REPORT_WRITES` | Сравнивает качество v2/v3, пишет Stage14 acceptance/audit package. | Запускать перед Stage15 export gate. |
| `run_transcript_quality_stage15_gate.py` | processing | `SAFE_REPORT_WRITES` | Проверяет Stage14/baseline/allowlist перед ROP/CRM/bot export, пишет safe bot allowlist. | Обязательный gate перед production export; не пишет CRM и DB. |
| `enrich_telegram_phones_live.py` | crm | `NETWORK_READ_ONLY` | Live enrichment через внешние данные, риск credentials. | Запускать малыми batch, без отправки сообщений. |
| `estimate_token_budget.py` | ops | `SAFE_READ_ONLY` | Считает budget. | Безопасно. |
| `evaluate_dialogue_quality.py` | processing | `SAFE_REPORT_WRITES` | Оценка качества, может читать transcripts. | Не менять processing-логику здесь. |
| `export_tallanto_schema.py` | crm | `NETWORK_READ_ONLY` | Читает Tallanto schema. | Можно для field mapping. |
| `finalize_manual_non_conversation_tail.py` | processing | `PROCESSING_MUTATES_DB` | Финализирует хвосты обработки. | Только processing-диалог. |
| `finalize_messages30_tail.py` | processing | `PROCESSING_MUTATES_DB` | Финализирует batch/tail. | Только processing-диалог. |
| `git_bootstrap.sh` | ops | `DANGEROUS_LEGACY` | Git/bootstrap side effects. | Не запускать без чтения. |
| `mango_office_appliance_loop_dry_run.py` | productization | `SAFE_REPORT_WRITES` | Dry-run appliance loop. | Безопасный SaaS smoke. |
| `mango_office_asr_approval_record.py` | productization | `SAFE_REPORT_WRITES` | Записывает approval record, не запускает ASR. | Безопасно. |
| `mango_office_asr_execution_approval_gate.py` | productization | `SAFE_REPORT_WRITES` | Проверяет approval gate. | Безопасно. |
| `mango_office_asr_execution_plan.py` | productization | `SAFE_REPORT_WRITES` | Строит plan, не исполняет ASR. | Безопасно. |
| `mango_office_asr_scheduler_dry_run.py` | productization | `SAFE_REPORT_WRITES` | Dry-run scheduler. | Безопасно. |
| `mango_office_asr_worker_dry_run.py` | productization | `SAFE_REPORT_WRITES` | Dry-run worker. | Безопасно. |
| `mango_office_asr_worker_pack.py` | productization | `SAFE_REPORT_WRITES` | Собирает worker pack. | Не запускать ASR из этого шага. |
| `mango_office_asr_worker_pack_verify.py` | productization | `SAFE_READ_ONLY` | Проверяет worker pack. | Безопасно. |
| `mango_office_asr_worker_sandbox_approval_packet.py` | productization | `SAFE_REPORT_WRITES` | Approval packet. | Безопасно. |
| `mango_office_asr_worker_sandbox_contract.py` | productization | `SAFE_REPORT_WRITES` | Contract docs/artifacts. | Безопасно. |
| `mango_office_asr_worker_sandbox_execution_request.py` | productization | `SAFE_REPORT_WRITES` | Execution request, не ASR execution. | Требует human approval дальше. |
| `mango_office_asr_worker_sandbox_human_approval.py` | productization | `SAFE_REPORT_WRITES` | Human approval record. | Безопасно. |
| `mango_office_asr_worker_sandbox_preflight.py` | productization | `SAFE_REPORT_WRITES` | Preflight report. | Безопасно. |
| `mango_office_asr_worker_sandbox_readiness.py` | productization | `SAFE_REPORT_WRITES` | Readiness report. | Безопасно. |
| `mango_office_amo_snapshot_export.py` | productization | `NETWORK_READ_ONLY` | Читает amoCRM contacts/leads и пишет локальный snapshot under product root. | Использовать для CRM candidates без live write. |
| `mango_office_appliance.py` | productization | `SAFE_REPORT_WRITES` | Пишет command-surface/runbook report для client-hosted appliance. | Не исполняет команды из отчета. |
| `mango_office_appliance_config_wizard.py` | productization | `SAFE_REPORT_WRITES` | Проверяет product root, DB, Mango env, CRM snapshot, retention, backups. | Безопасно для client-hosted setup. |
| `mango_office_appliance_service_pack.py` | productization | `SAFE_REPORT_WRITES` | Генерирует launchd/systemd templates под product root. | Не устанавливает и не запускает services. |
| `mango_office_capture_audit.py` | productization | `SAFE_REPORT_WRITES` | Capture audit report. | Безопасно. |
| `mango_office_capture_inbox.py` | productization | `SAFE_REPORT_WRITES` | Пишет capture inbox metadata. | Безопасно для productization staging. |
| `mango_office_capture_stage.py` | productization | `SAFE_REPORT_WRITES` | Stage report/metadata. | Не пишет runtime DB. |
| `mango_office_controlled_capture_ingest.py` | productization | `SAFE_REPORT_WRITES` | Shadow poll -> controlled ingest plan/apply. | Apply пишет только product DB capture inbox. |
| `mango_office_crm_entity_resolver.py` | productization | `SAFE_READ_ONLY` | Матчит product calls с локальным CRM snapshot. | Не делает live CRM calls. |
| `mango_office_crm_tallanto_mapping_preview.py` | productization | `SAFE_REPORT_WRITES` | Сверяет product capture rows с локальными AMO/Tallanto snapshots. | Без live CRM calls и без writeback. |
| `mango_office_crm_writeback_preview.py` | productization | `SAFE_REPORT_WRITES` | CRM preview diff, gates, rollback plan. | Live CRM write выключен. |
| `mango_office_demo_tenant.py` | productization | `SAFE_REPORT_WRITES` | Создает обезличенный demo product root. | Безопасно для демо и UI smoke. |
| `mango_office_demo_pilot_playbook.py` | productization | `SAFE_REPORT_WRITES` | Пишет demo/pilot playbook из product-safe данных. | Не читает runtime DB и не пишет CRM. |
| `mango_office_download_recordings.py` | productization | `DANGEROUS_LEGACY` | Старый путь скачивания записей. | Предпочитать guarded downloader ниже. |
| `mango_office_manager_identity_map.py` | productization | `SAFE_REPORT_WRITES` | Пишет manager identity map. | Безопасно. |
| `mango_office_payload_archive.py` | productization | `SAFE_REPORT_WRITES` | Архивирует raw payload локально. | Не включать secrets в bundle. |
| `mango_office_pipeline_bridge_dry_run.py` | productization | `SAFE_REPORT_WRITES` | Dry-run bridge. | Безопасно. |
| `mango_office_processing_handoff.py` | productization | `SAFE_REPORT_WRITES` | Готовит handoff, не запускает processing. | Безопасно. |
| `mango_office_processing_lifecycle.py` | productization | `SAFE_REPORT_WRITES` | Capture-to-handoff lifecycle report. | No ASR/R+A auto-trigger. |
| `mango_office_processing_acceptance_gates.py` | productization | `SAFE_REPORT_WRITES` | Проверяет read-only gates перед подключением processing. | Processing quality остается внешним blocker до явного evidence. |
| `mango_office_product_api_http.py` | productization | `SAFE_READ_ONLY` | Поднимает/проверяет HTTP API. | Проверять порт и env. |
| `mango_office_product_api_readiness.py` | productization | `SAFE_REPORT_WRITES` | Readiness report. | Безопасно. |
| `mango_office_product_db_admin.py` | productization | `REVIEW_REQUIRED` | Admin операции с product DB. | Только после чтения `--help`. |
| `mango_office_product_db_bootstrap.py` | productization | `SAFE_REPORT_WRITES` | Создает/инициализирует product DB, не runtime DB. | Использовать только отдельный product DB path. |
| `mango_office_product_ops.py` | productization | `SAFE_REPORT_WRITES` | Healthcheck, backup, verify backup, restore dry-run. | Безопасно для product DB under product root. |
| `mango_office_product_owner_config.py` | productization | `SAFE_REPORT_WRITES` | Создает owner config. | Проверять secrets. |
| `mango_office_provider_metadata_sidecar.py` | productization | `SAFE_REPORT_WRITES` | Пишет sidecar metadata. | Безопасно. |
| `mango_office_quarantine_import_plan.py` | productization | `SAFE_REPORT_WRITES` | План импорта quarantine. | Безопасно. |
| `mango_office_quarantine_materialize.py` | productization | `CONTROLLED_DOWNLOAD` | Материализует quarantine assets. | Только в отдельный quarantine path. |
| `mango_office_quarantine_test_ingest.py` | productization | `SAFE_REPORT_WRITES` | Test ingest artifacts. | Безопасно. |
| `mango_office_recording_asset_ingest.py` | productization | `SAFE_REPORT_WRITES` | Индексирует assets в staging/product context. | Не runtime DB. |
| `mango_office_recording_bridge_dry_run.py` | productization | `SAFE_REPORT_WRITES` | Dry-run bridge. | Безопасно. |
| `mango_office_recording_capture_download.py` | productization | `CONTROLLED_DOWNLOAD` | Guarded recording download. | Использовать вместо legacy downloader. |
| `mango_office_recording_capture_plan.py` | productization | `SAFE_REPORT_WRITES` | План скачивания. | Безопасно. |
| `mango_office_recording_quarantine_package.py` | productization | `SAFE_REPORT_WRITES` | Quarantine package. | Безопасно. |
| `mango_office_saas_productization_audit.py` | productization | `SAFE_REPORT_WRITES` | SaaS audit report. | Безопасно. |
| `mango_office_saas_stage_gates.py` | productization | `SAFE_REPORT_WRITES` | Stage gates report. | Безопасно. |
| `mango_office_sanitized_real_demo.py` | productization | `SAFE_REPORT_WRITES` | Создает обезличенный demo root из реального product DB. | Не читает runtime DB, не копирует audio, не пишет CRM. |
| `mango_office_scheduler_control_plane.py` | productization | `SAFE_READ_ONLY` | Показывает recommended scheduler/supervisor actions. | Не исполняет jobs. |
| `mango_office_scheduler_health.py` | productization | `SAFE_READ_ONLY` | Показывает due/failed/locked/stale scheduler jobs. | Безопасно для readiness панели. |
| `mango_office_scheduler_runtime.py` | productization | `SAFE_REPORT_WRITES` | Scheduler dry/controlled runtime. | Запускать сначала dry-run. |
| `mango_office_shadow_poll.py` | productization | `NETWORK_READ_ONLY` | Читает Mango API, не скачивает аудио. | Безопасный shadow poll. |
| `mango_office_tenant_isolation.py` | productization | `SAFE_REPORT_WRITES` | Проверяет tenant-scoped rows и опционально создает пустой tenant scaffold. | Не меняет product DB, только product-root reports/scaffold. |
| `mango_office_tallanto_snapshot_export.py` | productization | `NETWORK_READ_ONLY` | Читает Tallanto contacts по телефонам из product DB и пишет локальный snapshot. | Не пишет Tallanto/CRM, не меняет product DB. |
| `match_priority_contacts_with_tallanto.py` | crm | `NETWORK_READ_ONLY` | Читает/матчит Tallanto. | Проверять output. |
| `merge_pilot_sales_moment_llm_reviews.py` | insights | `SAFE_REPORT_WRITES` | Merge local LLM reviews. | Безопасно. |
| `merge_telegram_live_enrichment_chunks.py` | crm | `SAFE_REPORT_WRITES` | Merge enrichment chunks. | Безопасно. |
| `monitor_subset_progress.py` | processing | `SAFE_READ_ONLY` | Мониторит progress. | Безопасно. |
| `normalize_tallanto_contacts.py` | crm | `SAFE_REPORT_WRITES` | Нормализует contacts export. | Безопасно. |
| `prefill_asr_from_dbs.py` | processing | `PROCESSING_MUTATES_DB` | Может префиллить ASR из DB. | Только processing-диалог. |
| `prepare_asr_only_date_window.py` | processing | `PROCESSING_MUTATES_DB` | Готовит ASR-only batch. | Только processing-диалог. |
| `prepare_contact_history_batch.py` | processing | `PROCESSING_MUTATES_DB` | Готовит batch/history. | Только processing-диалог. |
| `prepare_date_window_subset.py` | processing | `PROCESSING_MUTATES_DB` | Готовит subset. | Только processing-диалог. |
| `prepare_dual_asr_new_llm_wave.py` | processing | `PROCESSING_MUTATES_DB` | Готовит ASR/LLM wave. | Только processing-диалог. |
| `prepare_gigaam_useful_subset.py` | processing | `PROCESSING_MUTATES_DB` | Готовит ASR subset. | Только processing-диалог. |
| `prepare_history_gap_wave.py` | processing | `PROCESSING_MUTATES_DB` | Готовит history gap wave. | Только processing-диалог. |
| `prepare_llm_wave_from_recommendations.py` | processing | `PROCESSING_MUTATES_DB` | Готовит LLM wave. | Только processing-диалог. |
| `prepare_manual_tail_analyze_fallback.py` | processing | `PROCESSING_MUTATES_DB` | Готовит manual R+A fallback. | Только processing-диалог. |
| `prepare_message_archive_history_full_cycle.py` | processing | `PROCESSING_MUTATES_DB` | Full-cycle archive/history. | Только processing-диалог. |
| `prepare_message_archive_wave.py` | processing | `PROCESSING_MUTATES_DB` | Message archive wave. | Только processing-диалог. |
| `prepare_message_archives_history_full_cycle.py` | processing | `PROCESSING_MUTATES_DB` | Full-cycle archive/history. | Только processing-диалог. |
| `prepare_overnight_full_asr_priority.py` | processing | `PROCESSING_MUTATES_DB` | Overnight ASR priority. | Только processing-диалог. |
| `prepare_phone_history_batch.py` | processing | `PROCESSING_MUTATES_DB` | Phone history batch. | Только processing-диалог. |
| `prepare_priority_history_wave.py` | processing | `PROCESSING_MUTATES_DB` | Priority history wave. | Только processing-диалог. |
| `prepare_remaining_asr_batch.py` | processing | `PROCESSING_MUTATES_DB` | Remaining ASR batch. | Только processing-диалог. |
| `prepare_resolve_analyze_missing_batch.py` | processing | `PROCESSING_MUTATES_DB` | Missing R+A batch. | Только processing-диалог. |
| `prepare_untranscribed_merge_batches.py` | processing | `PROCESSING_MUTATES_DB` | Merge batches. | Только processing-диалог. |
| `project_audit.py` | ops | `SAFE_REPORT_WRITES` | Пишет audit report. | Безопасно, output документировать. |
| `promote_ai_review_to_amo_ready.py` | crm | `SAFE_REPORT_WRITES` | Готовит AMO-ready export, не live write. | Проверять перед writeback. |
| `repair_and_move_message_archives.py` | processing | `PROCESSING_MUTATES_DB` | Repair/move archive files. | Только processing-диалог. |
| `requeue_secondary_backfill.py` | processing | `PROCESSING_MUTATES_DB` | Requeue/backfill. | Только processing-диалог. |
| `run_analyze_ab_test.py` | processing | `PROCESSING_MUTATES_DB` | Analyze A/B workflow. | Только processing-диалог. |
| `run_pilot_sales_moment_llm_review.py` | insights | `NETWORK_READ_ONLY` | Может обращаться к LLM API, пишет review artifacts. | Малые batch, без CRM writes. |
| `smoke_test_tallanto.py` | crm | `NETWORK_READ_ONLY` | Читает Tallanto API. | Безопасно при credentials. |
| `start_autocommit_push.sh` | ops | `DANGEROUS_LEGACY` | Включает auto commit/push. | Не использовать как нормальный workflow. |
| `stop_autocommit_push.sh` | ops | `SAFE_READ_ONLY` | Останавливает auto loop. | Можно для остановки legacy loop. |
| `summarize_merge_usage.py` | ops | `SAFE_READ_ONLY` | Summaries/statistics. | Безопасно. |
| `write_amo_ready_contacts.py` | crm | `CRM_LIVE_GUARDED` | По умолчанию dry-run report; live contact write только с confirmation. | Live: `--execute-live-write --live-confirmation WRITE_AMO_LIVE`. |
| `write_recent_actionable_deals.py` | crm | `CRM_LIVE_GUARDED` | По умолчанию dry-run report; live deal writeback только с confirmation. | Live: `--execute-live-write --live-confirmation WRITE_AMO_LIVE`. |

## Canonical recommendations

### Mango capture

- Для read-only проверки новых звонков: `mango_office_shadow_poll.py`.
- Для controlled shadow-to-inbox плана: `mango_office_controlled_capture_ingest.py plan`.
- Для плана скачивания: `mango_office_recording_capture_plan.py`.
- Для guarded download: `mango_office_recording_capture_download.py`.
- Для bridge readiness без ASR: `mango_office_processing_lifecycle.py`.
- Для CRM entity candidates из локального snapshot: `mango_office_crm_entity_resolver.py`.
- Для read-only amoCRM snapshot: `mango_office_amo_snapshot_export.py`.
- Для единого appliance command surface: `mango_office_appliance.py`.
- Для demo product root: `mango_office_demo_tenant.py`.
- Для продающего demo на реальной структуре данных: `mango_office_sanitized_real_demo.py`.
- Для проверки client-hosted установки: `mango_office_appliance_config_wizard.py`.
- Для AMO/Tallanto mapping preview: `mango_office_crm_tallanto_mapping_preview.py`.
- Для backup/restore readiness: `mango_office_product_ops.py`.
- Для scheduler/supervisor next actions: `mango_office_scheduler_control_plane.py`.
- Для scheduler readiness: `mango_office_scheduler_health.py`.
- Старый `mango_office_download_recordings.py` оставить как legacy-reference, не как
  основной путь.

### AMO writeback

- Для контактов: `write_amo_ready_contacts.py` теперь по умолчанию делает dry-run
  отчет.
- Для сделок: `write_recent_actionable_deals.py` теперь по умолчанию делает
  dry-run отчет.
- Для productization preview без live write: `mango_office_crm_writeback_preview.py`.
- Live-запись в amoCRM требует оба параметра:

```zsh
--execute-live-write --live-confirmation WRITE_AMO_LIVE
```

### Processing

Все `prepare_*`, `finalize_*`, `prefill_asr_from_dbs.py`,
`run_analyze_ab_test.py`, `repair_and_move_message_archives.py` и похожие
скрипты считаются владением processing-диалога. В SaaS/productization ветке их
не запускать и не менять без отдельного согласования.

### Productization

Скрипты `mango_office_*_dry_run.py`, `mango_office_*_audit.py`,
`mango_office_*_readiness.py`, `mango_office_*_plan.py` являются основным
безопасным путем для SaaS-разработки: сначала plan/dry-run/readiness, затем
отдельное решение на controlled download или live write.
