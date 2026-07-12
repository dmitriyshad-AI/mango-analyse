# Реестр worktree

Обновлено: 2026-07-12.

Источник факта: `git worktree list --porcelain` из общего gitdir Mango.

Правила:

- Этот файл фиксирует только существующие worktree и защитные границы.
- Удаление worktree, веток, тегов, runtime-данных и `git clean/gc/prune` требует отдельного явного решения Дмитрия.
- Живой Telegram-бот, Wappi-наблюдение, резерв отката и папка-хост `.git` не трогаются в задачах ADR-003.
- Для preflight важен сам факт регистрации worktree; статус веток и ценность кода проверяются отдельным аудитом перед любыми cleanup-действиями.

## Активные worktree

| Worktree | HEAD | Ветка / состояние | Назначение | Решение |
|---|---:|---|---|---|
| `/Users/dmitrijfabarisov/Projects/Mango analyse` | `9e8fb3b` | `codex/tz135-direct-wow-tone` | Папка-хост общего `.git`; историческая tz135-ветка. | Не переключать и не удалять без отдельного ТЗ. |
| `/private/tmp/mango_phase1_base_7da` | `7da115e` | detached | Временный base-worktree Phase1. | Удалять только отдельным cleanup-решением. |
| `/Users/dmitrijfabarisov/Projects/Mango_britva_main` | `ca3626b` | `codex/faza1-test-speed-faza3-monolith-cleanup` | Отдельный worktree монолитной чистки. | Не трогать в Timeline/Phase1b консолидации без отдельного ТЗ. |
| `/Users/dmitrijfabarisov/Projects/Mango_britva_wave0_merge` | `3cf3f66` | `codex/britva-wave0-tz135-merge` | Параллельный merge-трек Wave 0. | Не трогать в ТЗ-В. |
| `/Users/dmitrijfabarisov/Projects/Mango_timeline_main_consolidation` | `cca8aeb+` | `main` | Консолидированный main: код `cca8aeb4` плюс документы deploy; опубликован в `origin` и `yandex`. | Канон для следующих интеграций; текущий live должен оставаться его предком. |
| `/Users/dmitrijfabarisov/Projects/Mango_calls_two_processes` | `aa6f08c` | detached | Исторический worktree calls/two-processes; branch ref удалён после archive-tag в обоих remote. | Папка сохранена; удалять только отдельным cleanup-решением. |
| `/Users/dmitrijfabarisov/Projects/Mango_email_pipeline_restore` | `631030a` | `codex/email-pipeline-restore` | Параллельный worktree памяти/Timeline staging с незакоммиченными изменениями другого диалога. | Ветка намеренно не удалена; чужие tracked/untracked файлы не трогать. |
| `/Users/dmitrijfabarisov/Projects/Mango_frame_gate_3_defects` | `e1fbbba` | `codex/frame-gate-3-defects` | Параллельный трек frame-gate. | Не трогать в ТЗ-В. |
| `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff` | `15accd2` | `codex/adr003-semanticframe-migration` | Каноническая рабочая ветка ADR-003 / SemanticFrame. | Базовый worktree D1; сейчас содержит чужие untracked-артефакты, для Phase1 используется чистый отдельный worktree. |
| `/Users/dmitrijfabarisov/Projects/Mango_next_step_proactivity` | `495b6fd` | detached | Исторический Phase1-port worktree; branch ref удалён после archive-tag в обоих remote. | Папка сохранена; удалять только отдельным cleanup-решением. |
| `/Users/dmitrijfabarisov/Projects/Mango_live_4caa5eb_release_venue_autonomy` | `4caa5eb` | detached | Резерв отката старого live-кода. | Не трогать без отдельного катовер/cleanup ТЗ. |
| `/Users/dmitrijfabarisov/Projects/Mango_live_deploy` | `c262b95` | detached | Live deploy worktree / исторический deploy-снимок. | Не трогать без live-чеклиста и явного решения Дмитрия. |
| `/Users/dmitrijfabarisov/Projects/Mango_live_next_step_phase1b` | `2cc82b1` | detached | Предыдущий live-код Phase1b; точка полного rollback после deploy `cca8aeb4`. | Не трогать до завершения смыслового триажа нового live. |
| `/Users/dmitrijfabarisov/Projects/Mango_live_cca8aeb4_consolidated` | `cca8aeb` | detached, live | Фактический Wappi→AMO draft-loop после deploy 2026-07-10; память и strong-звонки ON. | Текущий `live_truth`; следующий deploy обязан содержать этот commit. |
| `/Users/dmitrijfabarisov/Projects/Mango_m1_memory_exam_build` | `796c0b8` | `codex/m1-memory-exam-build-20260711` | Параллельная сборка экзамена памяти M1. | Не трогать в ТЗ-В. |
| `/Users/dmitrijfabarisov/Projects/Mango_overclaim_part_a` | `a083174` | `codex/overclaim-part-a-focus-m1` | Параллельный трек проверки выдумок. | Не трогать в ТЗ-В. |
| `/Users/dmitrijfabarisov/Projects/Mango_tallanto_freshness_pilot_20260712` | `7126f6d` | `codex/tallanto-freshness-publish-amo-pilot-20260712` | Параллельный пилот свежести Tallanto. | Не трогать в ТЗ-В. |
| `/Users/dmitrijfabarisov/Projects/Mango_live_rollback_eb6fa0b` | `eb6fa0b` | detached | Live rollback worktree. | Не трогать без live-чеклиста и явного решения Дмитрия. |
| `/Users/dmitrijfabarisov/Projects/Mango_skills_top5_tools` | `49720dc` | `codex/skills-top5-tools` | Отдельный трек tools/skills. | Не трогать в ADR-003 задачах; разбирать отдельным аудитом. |
| `/Users/dmitrijfabarisov/Projects/Mango_wave0_refresh` | `636c52c` | `codex/wave0-refresh-20260710` | Внешний параллельный трек wave0 refresh; назначение в этой задаче не аудировалось. | Не трогать в Phase1/live-swap задаче. |
| `/Users/dmitrijfabarisov/Projects/Mango_calls_tzv_20260712` | `7126f6d` | `codex/tzv-calls-schedule-brand` | ТЗ-В: раздельное расписание Process A/B, диагностика свежести и brand_evidence. | Единственный worktree текущей задачи; runtime и чужие worktree не менять. |
| `/Users/dmitrijfabarisov/Projects/Mango_z0_z1_latch_seats_20260712` | `e983a77` | `codex/z0-z1-latch-seats-20260712` | Параллельный трек Z0/Z1. | Не трогать в ТЗ-В. |

## Текущая задача

Консолидация завершена: `main`, `origin/main` и `yandex/main` содержат live
code commit `cca8aeb4`, а draft-loop запущен из
`/Users/dmitrijfabarisov/Projects/Mango_live_cca8aeb4_consolidated`.
Первый смысловой триаж 10–20 новых черновиков остаётся обязательным; до него
live имеет операционный `PASS_WITH_NOTES`, но не semantic-pass.
