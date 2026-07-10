# Реестр worktree

Обновлено: 2026-07-10.

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
| `/Users/dmitrijfabarisov/Projects/Mango_timeline_main_consolidation` | `a209a0d+` | `main` | Локальная сборка main для консолидации Timeline/email/calls + Phase1b. | Не пушить `main` без отдельного решения владельца; использовать только для локальной проверки и audit pack. |
| `/Users/dmitrijfabarisov/Projects/Mango_calls_two_processes` | `aa6f08c` | `codex/calls-two-processes` | Отдельный трек calls/two-processes; ветка сохранена в `origin`/`yandex` и archive-tag. | Не трогать в ADR-003 задачах; разбирать отдельным аудитом. |
| `/Users/dmitrijfabarisov/Projects/Mango_email_pipeline_restore` | `631030a` | `codex/email-pipeline-restore` | Отдельная рабочая ветка восстановления email pipeline / Timeline staging. | Не вливать целиком вне consolidation; грязь/параллельные задачи не трогать из других worktree. |
| `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff` | `15accd2` | `codex/adr003-semanticframe-migration` | Каноническая рабочая ветка ADR-003 / SemanticFrame. | Базовый worktree D1; сейчас содержит чужие untracked-артефакты, для Phase1 используется чистый отдельный worktree. |
| `/Users/dmitrijfabarisov/Projects/Mango_next_step_proactivity` | `495b6fd` | `codex/next-step-proactivity-port` | Чистый worktree для точечного Phase1-port next-step safety. | Работать только в зонах ТЗ `2026-07-10_TZ_next_step_proaktivnost_phase1_port.md`. |
| `/Users/dmitrijfabarisov/Projects/Mango_live_4caa5eb_release_venue_autonomy` | `4caa5eb` | detached | Резерв отката старого live-кода. | Не трогать без отдельного катовер/cleanup ТЗ. |
| `/Users/dmitrijfabarisov/Projects/Mango_live_deploy` | `c262b95` | detached | Live deploy worktree / исторический deploy-снимок. | Не трогать без live-чеклиста и явного решения Дмитрия. |
| `/Users/dmitrijfabarisov/Projects/Mango_live_next_step_phase1b` | `2cc82b1` | detached | Фактический live-код Phase1b после SWAP трека «Понимание» 2026-07-10 14:35 МСК. | Не трогать без live-чеклиста; следующий deploy обязан содержать этот коммит или проверенный эквивалент. |
| `/Users/dmitrijfabarisov/Projects/Mango_live_rollback_eb6fa0b` | `eb6fa0b` | detached | Live rollback worktree. | Не трогать без live-чеклиста и явного решения Дмитрия. |
| `/Users/dmitrijfabarisov/Projects/Mango_skills_top5_tools` | `49720dc` | `codex/skills-top5-tools` | Отдельный трек tools/skills. | Не трогать в ADR-003 задачах; разбирать отдельным аудитом. |
| `/Users/dmitrijfabarisov/Projects/Mango_wave0_refresh` | `636c52c` | `codex/wave0-refresh-20260710` | Внешний параллельный трек wave0 refresh; назначение в этой задаче не аудировалось. | Не трогать в Phase1/live-swap задаче. |

## Текущая задача

Текущая консолидация Timeline/email/calls + Phase1b выполняется только в `/Users/dmitrijfabarisov/Projects/Mango_timeline_main_consolidation`. Цель: локальный `main` должен содержать Timeline/email/calls и фактический live-коммит Phase1b `2cc82b13`. Пуш `main` запрещён без отдельного решения владельца.
