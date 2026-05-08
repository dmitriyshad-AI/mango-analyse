# Runtime/Data Manifest

Дата: 2026-05-07

Цель: отделить рабочие данные и runtime-артефакты от кода, чтобы проект можно было безопасно чистить и переводить к SaaS-архитектуре.

Источник machine-readable отчета: `stable_runtime/project_audit_20260507_125704`.

## Top-level размеры

| Путь | Размер | Решение |
|---|---:|---|
| `stable_runtime` | 59.0G | `keep_until_coverage_v4` |
| `2026-03-09--26` | 24.4G | `keep_until_coverage_v4` |
| `telegram_exports (2)` | 1.2G | `archive_or_clean_after_confirmation` |
| `_local_archive_20260424` | 997.5M | `archive_or_clean_after_confirmation` |
| `2026-03-05-21-06-49-ч1` | 980.0M | `review` |
| `2026-03-05-21-06-49-ч2` | 979.7M | `review` |
| `.venv-asrbench` | 854.3M | `archive_or_clean_after_confirmation` |
| `test_sets` | 256.1M | `review` |
| `external_m1_jan_mar_2025_asr_only_20260504_result_20260506_103021` | 215.5M | `archive_or_clean_after_confirmation` |
| `.cache` | 138.6M | `archive_or_clean_after_confirmation` |
| `mango_mvp.db` | 89.0M | `keep_until_coverage_v4` |
| `.local` | 47.1M | `archive_or_clean_after_confirmation` |
| `.codex_workers` | 45.3M | `archive_or_clean_after_confirmation` |
| `Contacts.xls` | 20.6M | `review` |
| `prod_runtime_transfer` | 18.3M | `review` |
| `АКТУАЛЬНО_Полный_пакет_экспорта.xlsx` | 17.4M | `export_archive_candidate` |
| `.codex_local` | 16.9M | `archive_or_clean_after_confirmation` |
| `external_m1_jan_mar_2025_asr_only_20260504_result_20260506_103021.zip` | 10.4M | `review` |
| `АКТУАЛЬНО_Звонки_общая_таблица.xlsx` | 9.2M | `export_archive_candidate` |
| `АКТУАЛЬНО_Контакты_для_продаж.xlsx` | 4.3M | `export_archive_candidate` |

## DB категории

- `backup_candidate`: 141
- `final_ra_source`: 3
- `research_archive_candidate`: 39
- `root_runtime_db`: 4
- `runtime_db`: 137

## Крупнейшие DB

| Путь | Размер | Категория |
|---|---:|---|
| `stable_runtime/ra_missing_all_20260506/ra_missing_all_20260506.db` | 451.7M | `final_ra_source` |
| `stable_runtime/jun_jul_aug_2025_asr_only_20260503/jun_jul_aug_2025_asr_only_20260503.db` | 192.9M | `runtime_db` |
| `stable_runtime/jun_jul_aug_2025_asr_only_20260503/jun_jul_aug_2025_asr_only_20260503.before_requeue_ra_status_20260504_150846.db` | 192.9M | `backup_candidate` |
| `stable_runtime/jun_jul_aug_2025_asr_only_20260503/jun_jul_aug_2025_asr_only_20260503.before_requeue_ra_6an_20260504_125221.db` | 191.2M | `backup_candidate` |
| `stable_runtime/jun_jul_aug_2025_asr_only_20260503/jun_jul_aug_2025_asr_only_20260503.before_requeue_ra_20260504_124501.db` | 191.1M | `backup_candidate` |
| `stable_runtime/jun_jul_aug_2025_asr_only_20260503/jun_jul_aug_2025_asr_only_20260503.before_strict_dedupe_20260504_123429.db` | 190.4M | `backup_candidate` |
| `stable_runtime/jun_jul_aug_2025_asr_only_20260503/jun_jul_aug_2025_asr_only_20260503.before_requeue_after_strict_stop_20260504_123001.db` | 189.9M | `backup_candidate` |
| `stable_runtime/messages28_phone_history_asr_20260408/messages28_phone_history_asr_20260408.before_requeue_after_strict_stop_20260504_123001.db` | 162.8M | `backup_candidate` |
| `stable_runtime/messages28_phone_history_asr_20260408/messages28_phone_history_asr_20260408.before_attempt_shield_20260504_044847.db` | 162.8M | `backup_candidate` |
| `stable_runtime/messages28_phone_history_asr_20260408/messages28_phone_history_asr_20260408.db` | 162.8M | `runtime_db` |
| `stable_runtime/messages28_phone_history_asr_20260408/messages28_phone_history_asr_20260408.before_pause_bad_worker_20260504_044616.db` | 162.4M | `backup_candidate` |
| `stable_runtime/messages28_phone_history_gap_wave_20260410/messages28_phone_history_gap_wave_20260410.db` | 162.4M | `runtime_db` |
| `stable_runtime/messages28_phone_history_llm_wave_20260409/messages28_phone_history_llm_wave_20260409.db` | 153.4M | `runtime_db` |
| `stable_runtime/jun_jul_aug_2025_asr_only_20260503/jun_jul_aug_2025_asr_only_20260503.before_attempt_shield_20260504_044847.db` | 149.4M | `backup_candidate` |
| `stable_runtime/jun_jul_aug_2025_asr_only_20260503/jun_jul_aug_2025_asr_only_20260503.before_pause_bad_worker_20260504_044616.db` | 149.1M | `backup_candidate` |
| `stable_runtime/jun_jul_aug_2025_asr_only_20260503/jun_jul_aug_2025_asr_only_20260503.before_requeue_killed_workers_20260504_044114.db` | 145.7M | `backup_candidate` |
| `external_m1_jan_mar_2025_asr_only_20260504_result_20260506_103021/external_m1_jan_mar_2025_asr_only_20260504.db` | 115.7M | `runtime_db` |
| `stable_runtime/oct_nov_2025_asr_only_remaining_all_20260505/oct_nov_2025_asr_only_remaining_all_20260505.db` | 113.3M | `runtime_db` |
| `stable_runtime/sep2025_asr_only_remaining_all_20260504/sep2025_asr_only_remaining_all_20260504.db` | 98.2M | `runtime_db` |
| `stable_runtime/ab_tests/20260326_g54_vs_g54mini_50/smoke1/gpt-5.4/test.db` | 89.2M | `research_archive_candidate` |
| `stable_runtime/ab_tests/20260326_g54_vs_g54mini_50/smoke1/gpt-5.4-mini/test.db` | 89.2M | `research_archive_candidate` |
| `stable_runtime/ab_tests/20260326_g54_vs_g54mini_50/results/gpt-5.4/test.db` | 89.2M | `research_archive_candidate` |
| `stable_runtime/ab_tests/20260326_g54_vs_g54mini_50/results/gpt-5.4-mini/test.db` | 89.2M | `research_archive_candidate` |
| `mango_mvp.db` | 89.0M | `root_runtime_db` |
| `stable_runtime/backups/mango_mvp_before_reapply_first_1000_enhanced_20260322.db` | 88.9M | `backup_candidate` |
| `stable_runtime/backups/mango_mvp_before_reapply_rules_700_20260322_205454.db` | 88.9M | `backup_candidate` |
| `stable_runtime/benchmarks/20260322_problem_reanalyze_100/reanalyze_sample.db` | 88.8M | `research_archive_candidate` |
| `stable_runtime/backups/mango_mvp_before_reapply_rules_300_20260322_204925.db` | 88.8M | `backup_candidate` |
| `stable_runtime/history_cohort_20260319_20260326/history_cohort_20260319_20260326_asr.before_requeue_ra_6an_20260504_125221.db` | 85.6M | `backup_candidate` |
| `stable_runtime/history_cohort_20260319_20260326/history_cohort_20260319_20260326_asr.before_requeue_ra_20260504_124501.db` | 85.6M | `backup_candidate` |

## Крупные служебные cleanup-кандидаты

Особо заметны `stable_runtime/**/codex_home/tmp/arg0/**` бинарники `applypatch/apply_patch/codex-execve-wrapper` примерно по 189M. Это служебные артефакты Codex runtime, не бизнес-данные. Их можно удалить отдельным безопасным шагом после остановки активных Codex/Analyze процессов и проверки, что текущие batch уже завершены.

## Retention policy

1. `final_ra_source`, текущие batch DB и DB из coverage report держать до coverage v4 и пересборки contact-layer.
2. `.before_*` backup DB архивировать после coverage v4; локально оставлять максимум 1-2 последних backup на важный batch.
3. Research/benchmark DB переносить во внешний архив или сжимать.
4. Telegram exports, external M1 results и старые raw exports держать вне корня проекта после фиксации import/merge report.
5. Сырые аудио `2026-03-09--26` пока не трогать.
