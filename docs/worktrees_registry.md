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
| `/Users/dmitrijfabarisov/Projects/Mango_britva_main` | `8e8492b` | `main` | Отдельный main-worktree. | Не трогать в ADR-003 задачах без отдельного ТЗ. |
| `/Users/dmitrijfabarisov/Projects/Mango_calls_two_processes` | `410cb8d` | `codex/calls-two-processes` | Отдельный трек calls/two-processes. | Не трогать в ADR-003 задачах; разбирать отдельным аудитом. |
| `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff` | `15accd2` | `codex/adr003-semanticframe-migration` | Каноническая рабочая ветка ADR-003 / SemanticFrame. | Базовый worktree D1; сейчас содержит чужие untracked-артефакты, для Phase1 используется чистый отдельный worktree. |
| `/Users/dmitrijfabarisov/Projects/Mango_next_step_proactivity` | `57d6837` | `codex/next-step-proactivity-port` | Чистый worktree для точечного Phase1-port next-step safety. | Работать только в зонах ТЗ `2026-07-10_TZ_next_step_proaktivnost_phase1_port.md`. |
| `/Users/dmitrijfabarisov/Projects/Mango_live_4caa5eb_release_venue_autonomy` | `4caa5eb` | detached | Резерв отката старого live-кода. | Не трогать без отдельного катовер/cleanup ТЗ. |
| `/Users/dmitrijfabarisov/Projects/Mango_live_deploy` | `c262b95` | detached | Live deploy worktree / исторический deploy-снимок. | Не трогать без live-чеклиста и явного решения Дмитрия. |
| `/Users/dmitrijfabarisov/Projects/Mango_live_rollback_eb6fa0b` | `eb6fa0b` | detached | Live rollback worktree. | Не трогать без live-чеклиста и явного решения Дмитрия. |
| `/Users/dmitrijfabarisov/Projects/Mango_wappi_controlled_watch_observe` | `4c90081` | `codex/wappi-controlled-watch-observe` | Wappi observe / controlled watch. | Не трогать в ADR-003 задачах. |
| `/Users/dmitrijfabarisov/Projects/Mango_botsafe_slot_builder_port` | `b92bf4c` | `codex/port-botsafe-slot-builder` | Отдельная рабочая ветка botsafe slot builder port. | Не трогать в ADR-003 Ф2b; разбирать отдельным аудитом. |
| `/Users/dmitrijfabarisov/Projects/Mango_email_pipeline_restore` | `b7648ac` | `codex/email-pipeline-restore` | Отдельная рабочая ветка восстановления email pipeline; источник next-step фикса только для точечного порта. | Не вливать целиком; читать/портировать только по отдельному ТЗ. |
| `/Users/dmitrijfabarisov/Projects/Mango_replay_m1_pack_03eca184` | `03eca18` | detached | Замороженный replay/M1 пакет. | Не трогать и не перетирать. |
| `/Users/dmitrijfabarisov/Projects/Mango_replay_m1_pack_5d631930` | `5d63193` | detached | Замороженный replay/M1 пакет. | Не трогать и не перетирать. |
| `/Users/dmitrijfabarisov/Projects/Mango_skills_top5_tools` | `49720dc` | `codex/skills-top5-tools` | Отдельный трек tools/skills. | Не трогать в ADR-003 задачах; разбирать отдельным аудитом. |
| `/Users/dmitrijfabarisov/Projects/Mango_wave0_refresh` | `49720dc` | `codex/wave0-refresh-20260710` | Внешний параллельный трек wave0 refresh; назначение в этой задаче не аудировалось. | Не трогать в Phase1/live-swap задаче. |

## Текущая задача

ADR-003 Ф2b работает только в `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff` и ограничена отчётным скриптом, тестом и audit pack. Остальные worktree являются внешними по отношению к задаче.
