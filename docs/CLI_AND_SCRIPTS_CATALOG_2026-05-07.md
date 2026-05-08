# Каталог CLI и scripts

Дата: 2026-05-07

Цель: зафиксировать, какие скрипты есть в проекте, какие можно считать production-кандидатами, какие являются maintenance/research/legacy, и где нужна ручная ревизия перед SaaS-этапом.

Источник первичной инвентаризации: `stable_runtime/project_audit_20260507_125704/script_catalog.tsv`.

## Сводка

- Всего файлов в каталоге `scripts`: 49
- `devops_legacy`: 4
- `maintenance`: 21
- `needs_review`: 4
- `one_off_or_research`: 2
- `production_candidate`: 12
- `research`: 6

## Правило использования

- `production_candidate`: можно доводить до поддерживаемого интерфейса и документировать в README/runbook.
- `maintenance`: рабочие утилиты для batch/runtime; оставлять, но описать входы/выходы и опасные режимы.
- `research`: benchmark/AB/eval; не удалять, но отделить от production flow.
- `one_off_or_research`: вероятно разовые Telegram/outreach задачи; после завершения кампаний архивировать или переписать как поддерживаемый модуль.
- `devops_legacy`: использовать только после ревизии; автокоммит/автопуш не должен быть частью production SaaS.
- `needs_review`: нет достаточного описания, нужна ручная классификация.

## Каталог

| Скрипт | Статус | Интерфейс | Подсказка описания |
|---|---|---|---|
| `scripts/autocommit_push_loop.sh` | `devops_legacy` | `shell` |  |
| `scripts/benchmark_asr_compare.py` | `research` | `manual` |  |
| `scripts/benchmark_codex_merge.py` | `research` | `manual` |  |
| `scripts/benchmark_codex_merge_models.py` | `research` | `manual` | Force real merge call path (avoid skip_high_similarity fast path). |
| `scripts/build_amocrm_delivery_pack.py` | `production_candidate` | `argparse` | parser = argparse.ArgumentParser(description="Build merged AI+Tallanto review pack and amo-ready CSV.") |
| `scripts/build_messages28_master_exports.py` | `production_candidate` | `manual` |  |
| `scripts/build_rop_deal_pack.py` | `production_candidate` | `manual` |  |
| `scripts/build_telegram_high_utility_drafts.py` | `production_candidate` | `manual` |  |
| `scripts/build_telegram_openclaw_final.py` | `production_candidate` | `manual` |  |
| `scripts/build_telegram_outreach_pack.py` | `production_candidate` | `manual` |  |
| `scripts/enrich_telegram_phones_live.py` | `one_off_or_research` | `manual` |  |
| `scripts/estimate_token_budget.py` | `needs_review` | `argparse` | description="Estimate LLM token budget for dual-ASR merge + structured analysis" |
| `scripts/evaluate_dialogue_quality.py` | `research` | `argparse` | Evaluate dialogue transcript quality signals from exported files. |
| `scripts/export_tallanto_schema.py` | `production_candidate` | `argparse` | parser = argparse.ArgumentParser(description="Export Tallanto schema metadata for Mango analyse.") |
| `scripts/finalize_manual_non_conversation_tail.py` | `maintenance` | `manual` |  |
| `scripts/finalize_messages30_tail.py` | `maintenance` | `manual` |  |
| `scripts/git_bootstrap.sh` | `devops_legacy` | `shell` |  |
| `scripts/match_priority_contacts_with_tallanto.py` | `production_candidate` | `argparse` | parser = argparse.ArgumentParser(description="Build exact-phone Tallanto match tables for priority AI contacts.") |
| `scripts/merge_telegram_live_enrichment_chunks.py` | `one_off_or_research` | `argparse` |  |
| `scripts/monitor_subset_progress.py` | `maintenance` | `argparse` | description="Poll subset DB and print progress snapshots with ETA." |
| `scripts/normalize_tallanto_contacts.py` | `production_candidate` | `argparse` | parser = argparse.ArgumentParser(description="Normalize Tallanto Contacts.xls into a clean CSV snapshot.") |
| `scripts/prefill_asr_from_dbs.py` | `maintenance` | `argparse` | description="Reuse already completed ASR results from existing DBs into a new target DB." |
| `scripts/prepare_asr_only_date_window.py` | `maintenance` | `manual` |  |
| `scripts/prepare_contact_history_batch.py` | `maintenance` | `argparse` | description=( |
| `scripts/prepare_date_window_subset.py` | `maintenance` | `argparse` | description="Create a date-window subset DB from an existing ASR-ready SQLite DB." |
| `scripts/prepare_dual_asr_new_llm_wave.py` | `maintenance` | `argparse` | description=( |
| `scripts/prepare_gigaam_useful_subset.py` | `maintenance` | `manual` |  |
| `scripts/prepare_history_gap_wave.py` | `maintenance` | `argparse` | description=( |
| `scripts/prepare_llm_wave_from_recommendations.py` | `maintenance` | `argparse` | description=( |
| `scripts/prepare_message_archive_history_full_cycle.py` | `maintenance` | `manual` |  |
| `scripts/prepare_message_archive_wave.py` | `maintenance` | `argparse` | description="Create a symlink batch for one messages(N) archive using its index.html and the normalized target folder." |
| `scripts/prepare_message_archives_history_full_cycle.py` | `maintenance` | `manual` |  |
| `scripts/prepare_overnight_full_asr_priority.py` | `maintenance` | `argparse` | description="Prepare exactly N newest unique raw audio calls that were never seen in any local DB." |
| `scripts/prepare_phone_history_batch.py` | `maintenance` | `argparse` | description=( |
| `scripts/prepare_priority_history_wave.py` | `maintenance` | `argparse` | description=( |
| `scripts/prepare_remaining_asr_batch.py` | `maintenance` | `argparse` |  |
| `scripts/prepare_resolve_analyze_missing_batch.py` | `maintenance` | `manual` |  |
| `scripts/prepare_untranscribed_merge_batches.py` | `maintenance` | `argparse` | parser = argparse.ArgumentParser(description="Prepare 500+500 newest untranscribed audio batches") |
| `scripts/project_audit.py` | `needs_review` | `manual` | Generate a local project audit report without touching runtime data. |
| `scripts/promote_ai_review_to_amo_ready.py` | `production_candidate` | `argparse` | parser = argparse.ArgumentParser(description="Promote AI-review contacts into AMO-ready export.") |
| `scripts/repair_and_move_message_archives.py` | `needs_review` | `manual` |  |
| `scripts/requeue_secondary_backfill.py` | `maintenance` | `argparse` |  |
| `scripts/run_analyze_ab_test.py` | `research` | `argparse` | parser = argparse.ArgumentParser(description="Run A/B Analyze test on the same sample of calls.") |
| `scripts/smoke_test_tallanto.py` | `needs_review` | `argparse` | parser = argparse.ArgumentParser(description="Tallanto smoke-check for Mango analyse runtime.") |
| `scripts/start_autocommit_push.sh` | `devops_legacy` | `shell` |  |
| `scripts/stop_autocommit_push.sh` | `devops_legacy` | `shell` |  |
| `scripts/summarize_merge_usage.py` | `research` | `argparse` | parser = argparse.ArgumentParser(description="Summarize transcribe merge token usage from DB") |
| `scripts/write_amo_ready_contacts.py` | `production_candidate` | `manual` |  |
| `scripts/write_recent_actionable_deals.py` | `production_candidate` | `argparse` | parser = argparse.ArgumentParser(description='Build fresh recent closed queue and write only new actionable deals to amoCRM.') |

## Следующие действия по каталогу

1. Для всех `production_candidate` добавить нормальный `argparse description`, `--dry-run` там, где есть запись в CRM/БД, и пример запуска в runbook.
2. Для `maintenance` явно отметить, какие скрипты безопасны, а какие меняют статусы/БД.
3. Для `needs_review` принять решение: production, maintenance, research или archive.
4. Для Telegram one-off скриптов выделить отдельный `outreach`/`experiments` слой или архивировать после фиксации результатов.
5. Перед SaaS-этапом не держать критичную бизнес-логику только в `scripts`; переносить повторяемые функции в `src/mango_mvp/services` и покрывать тестами.
