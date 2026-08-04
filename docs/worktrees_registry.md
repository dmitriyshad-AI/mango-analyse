# Реестр worktree

Обновлено: 2026-08-04. Источник факта: `git worktree list --porcelain`,
активные процессы и `stable_runtime/CURRENT_RUNTIME.json`.

## Правила

- Один worktree принадлежит одному исполнителю или live-службе.
- Активный live-worktree нельзя переключать или удалять до отдельного cutover.
- Отсутствующая в `git worktree list` папка не считается действующей только
  потому, что упоминалась в старом отчёте.
- История удалённых worktree живёт в Git; второй архивный список здесь не нужен.

## Текущие worktree

| Путь | HEAD / ветка | Назначение | Условие удаления |
|---|---|---|---|
| `/Users/dmitrijfabarisov/Projects/Mango analyse` | `e917db33`, `claude/timeline-final-20260803` | Канонический путь данных и процессов calls A/B и Customer Timeline nightly. Кодовая ветка временно отстаёт от `main`; дерево чистое. | Только после отдельного cutover служб на проверенный `main` и проверки PID/HEAD/env. |
| `/Users/dmitrijfabarisov/Projects/Mango_noncontentful_call_memory_integration_20260804` | `main` после коммита этого блока | Последовательная приёмка донорских веток. | Удалить сам worktree только когда закончена очередь доноров. |
| `/Users/dmitrijfabarisov/Projects/Mango_rollback_wappi_ca1779bc` | detached `ca1779bc` | Проверенный rollback старого Wappi runtime. | После M1 PASS, безопасного редеплоя и отдельного решения владельца. |

## Runtime

- Wappi остановлен владельцем.
- Calls A/B и Customer Timeline используют путь `Mango analyse`; эту папку
  нельзя переключать или удалять в ходе интеграции Git.
- Последний известный код выхода процесса A/nightly `1` требует отдельного
  операционного разбора и не является следствием дисковой уборки.
- Текущая Timeline-БД:
  `product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`.
- Истина по runtime-указателям: `stable_runtime/CURRENT_RUNTIME.json`.

## Убрано

Все остальные записи предыдущей версии реестра удалены как устаревшие: этих
worktree нет в Git и на диске. Их коммиты, решения и audit packs остаются в
истории репозитория; создавать новые архивные копии не требуется.
