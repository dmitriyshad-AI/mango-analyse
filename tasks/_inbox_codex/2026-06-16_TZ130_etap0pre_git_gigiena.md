# ТЗ-130 (после аудитора) — Этап 0-pre прокачки Кодекса: git-гигиена

- **Дата:** 2026-06-16. Заказчик: Дмитрий. Постановщик: Claude #1 (+ фокус-аудитор). Исполнитель: **D1** (владеет каноном). Это ПЕРВЫЙ этап прокачки Кодекса (предпосылка ко всему).
- **Цель:** вернуть инвариант «main = главная папка», разгрести 27 worktree в реестр, НЕ потеряв ни строки работы. ТОЛЬКО git-гигиена. Поведение бота не меняем, `stable_runtime`/живой пилот не трогаем.

## Факты на старте (проверено)
- Главная папка `Projects/Mango analyse` — на `codex/tz119-assumed-scope-guard-main` (aea3674) + **41 грязный** (4 tracked src/tests tz119 + 37 untracked: карточки ТЗ в `_inbox_codex/` и 5 отчётов в `_done/`). `git stash list` пуст → бэкапа нет.
- `main` (ee45e45, tz125+tz126 влиты, запушен в origin) checked out в worktree `Mango_main_tz121_merge`.
- 27 worktree, 26 веток не влиты в origin/main. `git log origin/main..main` пуст (main без своих коммитов).

## ЖЁСТКИЕ ЗАПРЕТЫ (необратимая потеря)
1. **НИКАКОГО `git checkout`/`stash drop`/`reset` до коммита грязи tz119** — иначе 4 правки src затрутся без следа.
2. **Только `git branch -d` (строчная)**, НИКОГДА `-D`. `-d` сама откажет, если есть несведённые коммиты.
3. **`worktree remove --force` запрещён** до cherry-проверки. Чистый путь — `git worktree prune`.
4. Критерий «влита» — НЕ `branch --merged`, а **`git cherry origin/main <branch> | grep '^+'` ПУСТО** (ловит cherry-pick/rebase/squash).
5. Не удалять ничего НЕвлитого. `stable_runtime/`, живой пилот, `codex/tz20-autoresolver` — не трогать.

## Порядок (строго)
1. **Бэкап-точка.** `git tag pre-tz130/<branch>` на все держимые ветки (список ниже) ИЛИ `git bundle create ~/mango_pre_tz130.bundle --all`. Без бэкапа не начинать.
2. **Закоммитить грязь tz119** в ветку `codex/tz119-assumed-scope-guard-main` (НЕ в main): отдельно (1) src/tests (`dialogue_contract_pipeline.py`, `direct_path.py`, `post_layers.py` + тест), отдельно (2) untracked карточки/отчёты. Цель: 0 грязных. Untracked карточки ТЗ — это постановки, тоже сохранить.
3. **Освободить main:** `git worktree prune` (снесёт prunable-запись `Mango_main_tz121_merge`; main запушен в origin, копия одноразовая — потерь нет). Если запись осталась — `git worktree remove`.
4. **Вернуть канон:** в главной папке `git checkout main && git pull` → главная папка на main=ee45e45. (ТОЛЬКО после шага 2.)
5. **Снос ВЛИТЫХ веток** (cherry-чисто, проверено аудитором — 11 шт): для каждой `git cherry origin/main <b> | grep '^+'` пусто → `git worktree prune`/`remove` → `git branch -d <b>`:
   `answerability-shadow-neutrality`, `tz22-d7-crm`, `tz25-graphify-structural`, `postlayers-kb-prices`, `tz115-judge-date-meta-leak`, `tz120-child-identity-off-on`, `tz121-group4-remaining`, `tz125-finalize-group4`, `tz126-overhandoff-metric`, `tz24-dubli-deti`, `tz118-d-primary-clean` (ahead=1, но содержимое уже в main → cherry пусто, сносить можно).
6. **Реестр `docs/worktrees_registry.md`** — НЕсведённые (есть уникальные коммиты), worktree НЕ сносить, записать: ветка | тема | uniq-коммитов | СТОП-дата | решение (влить/бросить):
   `measure-flags-honest`(+9, бандл-источник, жив до флагового регрейда), `measure-tz122-tz123-tz124`(+9), `tz118-group4-primary-d`(+10, грязная сестра — бросить), `tz116-offline-understanding`(+7), `tz123-tz124-remeasure`(+5), `tz106-real006-model-p0-on`(+2), `tz123-question-instead-of-handoff`(+2), `tz122-wrong-intent-fact`(+3), `tz124-slot-anchor`(+2), `tz20-blacklist57`(+2), `block-a-deal-gold-expanded`(+1), `tz119-assumed-scope-guard-main`(после шага 2).
7. **`codex/tz20-autoresolver` (+1, живая AMO) — НЕ трогать**, строкой в реестр «живая, отдельным ТЗ под контролем Дмитрия».
8. **Хвосты-worktree аудитора:** `/tmp/mango_audit_wt` (detached 6b93b77, **locked**) и detached `tz122_main_compare` (dd00d65, без ветки). Для detached — проверить `git cherry origin/main <sha>`; если пусто → prune; иначе в реестр. `/tmp/...` разлочить и снести, если временный.

## Гейт выхода
- Главная папка на `main` (ee45e45), 0 грязных вне реестра.
- `git worktree list` = только учтённые записи (всё прочее — в `docs/worktrees_registry.md` с решением).
- Ни одна несведённая работа не удалена; бэкап-теги/bundle на месте.
- Отчёт в `tasks/_done`: что снесено, что в реестре, бэкап-точка. Дальше — мой регрейд (diff-список снесённого vs cherry-пусто) перед закрытием.
