# Реестр worktree

Обновлено: 2026-07-02.

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
| `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff` | `36ea110` | `codex/adr003-semanticframe-migration` | Текущая рабочая ветка ADR-003 / SemanticFrame. | Рабочее дерево D1 для текущих ADR-003 задач. |
| `/Users/dmitrijfabarisov/Projects/Mango_live_4caa5eb_release_venue_autonomy` | `4caa5eb` | detached | Резерв отката старого live-кода. | Не трогать без отдельного катовер/cleanup ТЗ. |
| `/Users/dmitrijfabarisov/Projects/Mango_wappi_controlled_watch_observe` | `4c90081` | `codex/wappi-controlled-watch-observe` | Wappi observe / controlled watch. | Не трогать в ADR-003 задачах. |
| `/Users/dmitrijfabarisov/Projects/Mango_botsafe_slot_builder_port` | `b92bf4c` | `codex/port-botsafe-slot-builder` | Отдельная рабочая ветка botsafe slot builder port. | Не трогать в ADR-003 Ф2b; разбирать отдельным аудитом. |
| `/Users/dmitrijfabarisov/Projects/Mango_email_pipeline_restore` | `7425faf` | `codex/email-pipeline-restore` | Отдельная рабочая ветка восстановления email pipeline. | Не трогать в ADR-003 Ф2b; разбирать отдельным аудитом. |

## Текущая задача

ADR-003 Ф2b работает только в `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff` и ограничена отчётным скриптом, тестом и audit pack. Остальные worktree являются внешними по отношению к задаче.
