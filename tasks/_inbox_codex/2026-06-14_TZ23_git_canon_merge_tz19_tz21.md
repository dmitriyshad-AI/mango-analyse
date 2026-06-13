# ТЗ-23: вернуть главную папку на main и влить принятую работу TZ-19/TZ-21 (git-гигиена)

Дата: 2026-06-14. Постановка: Claude (архитектор). Исполнитель: диалог, открытый в `/Users/dmitrijfabarisov/Projects/Mango analyse` (сейчас это **D6**). Характер: одноразовая git-уборка канона. Регрейд TZ-19/TZ-21 пройден (PASS, `D1_audit_backlog/existing_clients/REGRADE_tz19_tz21_ingest_2026-06-14.md`).

## Контекст (проверено по git)

- Главная папка стоит на `codex/d6-autonomy-sim` (работа D6 уже закоммичена, `4f6893a`). Принятая работа TZ-19/TZ-21 — 8 коммитов на ветке `codex/tz19-calls-review-table` (вершина `435e55e`).
- `main` — предок `codex/tz19-calls-review-table` (0 встречных коммитов, 8 поверх) → возможен **fast-forward без слияния-коммита**.
- Незакоммичено в дереве: правка `CLAUDE.md` (легитимная, решение 13.06 про авто-резолвер) + неотслеживаемые .md-документы. `CLAUDE.md` 8 коммитами НЕ тронут → стэшить/коммитить отдельно безопасно.

## Предусловия

1. В этой папке сейчас НЕ должны параллельно писать другие диалоги (D4/D7 — read-only/остановлены; убедиться).
2. Никаких git-процессов на фоне (если есть `.git/index.lock` без живого процесса — убрать только его).

## Шаги (только безопасные операции; НЕЛЬЗЯ reset --hard / checkout с потерей / clean / push --force)

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"

# 1. Проверка исходного состояния
git status --short            # ожидаем: ' M CLAUDE.md' + неотслеживаемые .md, больше ничего застейдженного
git rev-parse --abbrev-ref HEAD   # codex/d6-autonomy-sim

# 2. Временно убрать правку CLAUDE.md, чтобы дерево было чистым для переключения
git stash push -- CLAUDE.md   # неотслеживаемые доки НЕ трогаем, они не мешают

# 3. Перейти на канон
git switch main

# 4. Влить TZ-19/TZ-21 строго fast-forward (если не получится — СТОП, доложить, ничего не форсить)
git merge --ff-only codex/tz19-calls-review-table
git log --oneline -9          # верхние 8 должны быть TZ19/TZ21 поверх main

# 5. Вернуть и закоммитить правку CLAUDE.md отдельной строкой
git stash pop
git add CLAUDE.md
git commit -m "policy(CLAUDE.md): решение 13.06 — черновик на все опознанные сделки (авто-резолвер), отмена списка пар"

# 6. Финальная проверка
git rev-parse --abbrev-ref HEAD   # main
git log --oneline -6
git status --short                # должно остаться только неотслеживаемые .md-доки (их коммитим опц. шагом 7)
```

```bash
# 7. (ОПЦИОНАЛЬНО, после ручной проверки git status) зафиксировать документы трека в каноне
#    Сначала посмотреть, что именно добавится — НЕ коммитить, если попадает что-то кроме .md-документов.
git add D1_audit_backlog/ audits/_inbox/crm_layer_audit_2026-06-13/ tasks/_inbox_codex/ tasks/_done/
git status --short
git commit -m "docs: аудит CRM-слоя + регрейды TZ-19/21 + ТЗ-22(D7)/ТЗ-23 + решения трека"
```

## Границы

- Никаких записей в AMO/Tallanto/CRM, никаких прогонов analyze/ASR. Это чисто git-операции над документами и историей.
- НЕ мерджить `codex/d6-autonomy-sim`, `codex/tz20-autoresolver`, `codex/tz21-tail-ingest-profiles` — они на отдельный разбор позже.
- Если `--ff-only` не проходит или `stash pop` даёт конфликт — остановиться и доложить, ничего не форсить.
- Устаревший worktree `Mango_tz20_autoresolver` (prunable) не трогать в этом ТЗ.

## Приёмка

- `git rev-parse --abbrev-ref HEAD` = `main`; в `git log` сверху видны 8 коммитов TZ-19/TZ-21 + коммит CLAUDE.md.
- `scripts/import_tz19_analyze_tail_results.py` и правка `builder.py` присутствуют в main (нужны для TZ-20 и совместимости с D7).
- Рабочее дерево чистое (или только осознанно не закоммиченное).
- Доложить короткий итог: новый `git log --oneline -6` и `git status`.

После этого ТЗ: D7 (ТЗ-22) и TZ-20 (blacklist-57) разворачиваются в отдельных worktree от обновлённого main — это следующий шаг, отдельными промптами.
