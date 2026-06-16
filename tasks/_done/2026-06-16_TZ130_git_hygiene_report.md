# ТЗ-130: git-гигиена, этап 0-pre

Дата: 2026-06-16.

## Бэкап

Перед изменениями создан полный bundle всех refs:

`/Users/dmitrijfabarisov/Backups/mango_pre_tz130_20260616.bundle`

Размер: 28 MB.

## Сохранение грязи `codex/tz119-assumed-scope-guard-main`

Грязь была сохранена в самой ветке `codex/tz119-assumed-scope-guard-main`, до переключения главной папки:

- `a51444c` `tz119: preserve assumed-scope source and tests`
- `364322f` `tz119: preserve task cards and reports`

После коммитов ветка чистая. В коммит отчётов попал маленький маркер `tasks/_inbox_codex/.write_test_claude` (`ok\n`), потому что ТЗ запрещало удалять незакоммиченное, а цель была `0 грязных`.

## Канон

`Mango_main_tz121_merge` был чистым worktree на `main`. После сохранения грязи:

- `git worktree prune` выполнен;
- `/Users/dmitrijfabarisov/Projects/Mango_main_tz121_merge` удалён штатным `git worktree remove` без `--force`;
- главная папка `/Users/dmitrijfabarisov/Projects/Mango analyse` переключена на `main`;
- `git pull --ff-only origin main` выполнен;
- текущий `main = origin/main = ee45e45b`.

## Удалённые worktree/ветки

Для каждой ветки сначала проверялось `git cherry origin/main <branch>`; удаление ветки выполнялось только через `git branch -d`.

Удалены:

- `codex/answerability-shadow-neutrality`
- `codex/tz22-d7-crm`
- `codex/tz25-graphify-structural`
- `codex/postlayers-kb-prices`
- `codex/tz115-judge-date-meta-leak`
- `codex/tz121-group4-remaining`
- `codex/tz125-finalize-group4`
- `codex/tz126-overhandoff-metric`
- `codex/tz24-dubli-deti`

Worktree для этих веток также удалены штатно, без `--force`.

## Не удалены из-за предохранителей

- `codex/tz120-child-identity-off-on`: `git cherry` пустой, но worktree содержит untracked `tasks/_inbox_codex/2026-06-15_TZ120_nabor_child_identity_OFF_ON.md`. Worktree и ветка оставлены, чтобы не потерять файл.
- `codex/tz118-d-primary-clean`: `git cherry` вернул только `- b00b3bd...`, worktree был чист и удалён. Но `git branch -d` отказал, потому что ветка не считается ancestry-merged. `-D` запрещён ТЗ, поэтому ветка оставлена без worktree.

## Detached/locked worktree

- `/tmp/mango_audit_wt`: locked detached на `6b93b77`, commit содержится в `origin/main`; worktree разблокирован/prune, запись удалена.
- `/Users/dmitrijfabarisov/Projects/Mango_tz122_main_compare`: detached на `dd00d65`, clean, commit содержится в `origin/main`; worktree удалён штатно.

## Реестр

Создан `docs/worktrees_registry.md`. В нём зафиксированы активные несведённые worktree/ветки, `codex/tz20-autoresolver` как живая ветка, и два исключения после ТЗ-130 (`tz120`, `tz118-d-primary-clean`).

## Финальное состояние

- Главная папка на `main`.
- `stable_runtime` и живой пилот не трогались.
- `codex/tz20-autoresolver` не трогался.
- `branch -D`, `git reset`, `git clean`, `stash drop`, `worktree remove --force` не использовались.
