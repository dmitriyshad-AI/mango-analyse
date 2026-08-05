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

## Full script inventory

| Script | Owner | Safety class | Side effects / risk | Recommended use |
|---|---|---|---|---|
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
| `build_transcript_quality_baseline.py` | processing | `PROCESSING_MUTATES_DB` | Принадлежит transcript quality ветке. | Не трогать в этом диалоге. |
| `build_transcript_quality_stage14_comparison.py` | processing | `SAFE_REPORT_WRITES` | Сравнивает качество v2/v3, пишет Stage14 acceptance/audit package. | Запускать перед Stage15 export gate. |
| `run_transcript_quality_stage15_gate.py` | processing | `SAFE_REPORT_WRITES` | Проверяет Stage14/baseline/allowlist перед ROP/CRM/bot export, пишет safe bot allowlist. | Обязательный gate перед production export; не пишет CRM и DB. |
| `estimate_token_budget.py` | ops | `SAFE_READ_ONLY` | Считает budget. | Безопасно. |
| `evaluate_dialogue_quality.py` | processing | `SAFE_REPORT_WRITES` | Оценка качества, может читать transcripts. | Не менять processing-логику здесь. |
| `export_tallanto_schema.py` | crm | `NETWORK_READ_ONLY` | Читает Tallanto schema. | Можно для field mapping. |
| `finalize_messages30_tail.py` | processing | `PROCESSING_MUTATES_DB` | Финализирует batch/tail. | Только processing-диалог. |
| `mango_office_mail_archive.py` | productization | `NETWORK_READ_ONLY` | Read-only IMAP ingest через `BODY.PEEK[]`; пишет локальный mail archive/matching artifacts, не отправляет, не удаляет, не двигает письма, не пишет CRM/Tallanto. | Запускать малыми pilot batch; output держать в ignored `_external_handoffs/`. |
| `mango_office_tallanto_snapshot_export.py` | productization | `NETWORK_READ_ONLY` | Читает Tallanto contacts по телефонам из product DB и пишет локальный snapshot. | Не пишет Tallanto/CRM, не меняет product DB. |
| `match_priority_contacts_with_tallanto.py` | crm | `NETWORK_READ_ONLY` | Читает/матчит Tallanto. | Проверять output. |
| `merge_pilot_sales_moment_llm_reviews.py` | insights | `SAFE_REPORT_WRITES` | Merge local LLM reviews. | Безопасно. |
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
| `summarize_merge_usage.py` | ops | `SAFE_READ_ONLY` | Summaries/statistics. | Безопасно. |
| `write_recent_actionable_deals.py` | crm | `CRM_LIVE_GUARDED` | По умолчанию dry-run report; live deal writeback только с confirmation. | Live: `--execute-live-write --live-confirmation WRITE_AMO_LIVE`. |

## Canonical recommendations

### Calls, mail, and Tallanto

- Звонки обслуживают только `run_mango_calls_process.sh` и
  `mango_mvp.customer_timeline.calls_two_processes`; старые отдельные
  `mango_office_*` capture-команды удалены.
- Почтовый архив обслуживает `mango_office_mail_archive.py` через штатную
  цепочку `run_customer_timeline_codex_task.py`.
- Read-only снимок Tallanto запускается только через
  `mango_office_tallanto_snapshot_export.py`.
- Контракт текущего runtime строится через `mango_office_current_runtime.py`.

### AMO writeback

- Для сделок: `write_recent_actionable_deals.py` теперь по умолчанию делает
  dry-run отчет.
- Live-запись в amoCRM требует оба параметра:

```zsh
--execute-live-write --live-confirmation WRITE_AMO_LIVE
```

### Processing

Все `prepare_*`, `finalize_*`, `prefill_asr_from_dbs.py`,
`run_analyze_ab_test.py`, `repair_and_move_message_archives.py` и похожие
скрипты считаются отдельным контуром обработки. Их не запускать и не менять без
отдельного согласования.
