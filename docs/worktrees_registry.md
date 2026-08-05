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
| `/Users/dmitrijfabarisov/Projects/Mango analyse` | `e917db33`, `claude/timeline-final-20260803` | Канонический путь данных и процессов calls A/B и Customer Timeline nightly. Все 15 уникальных коммитов разобраны: полезное уже в `main`, остаток отклонён; ветка не содержит очереди на слияние. Дерево чистое. | Только после отдельного cutover служб на проверенный `main` и проверки PID/HEAD/env; после этого ветку можно удалить. |
| `/Users/dmitrijfabarisov/Projects/Mango_noncontentful_call_memory_integration_20260804` | `main`; SHA через `git rev-parse main` | Чистый канонический код: все донорские ветки разобраны; отсюда готовится следующий рефакторинг и будущий cutover. | После перевода служб на проверенный `main` в основной папке. |
| `/Users/dmitrijfabarisov/Projects/Mango_global_code_cut_wave10_20260805` | `codex/global-code-cut-wave10`; SHA через `git rev-parse HEAD` | Уборка доказанно мёртвого старого AMO CRM writeback-контура; live/runtime не используется. | После вливания проверенного коммита в `main` и сверки отсутствия незабранных изменений. |
| `/Users/dmitrijfabarisov/Projects/Mango_rollback_wappi_ca1779bc` | detached `ca1779bc` | Проверенный rollback старого Wappi runtime. | После M1 PASS, безопасного редеплоя и отдельного решения владельца. |

## Runtime

- Wappi остановлен владельцем.
- Calls A/B и Customer Timeline используют путь `Mango analyse`; эту папку
  нельзя переключать или удалять в ходе интеграции Git.
- Calls A/B 4 августа запущены из `Mango analyse`; `live_truth.py` их не видит,
  потому что проверяет клиентские каналы, поэтому cutover требует отдельной
  сверки PID, командной строки и конфигурации calls.
- Последний зарегистрированный выход calls A и Customer Timeline nightly был с
  кодом `1`; calls A сейчас снова работает, nightly сейчас не запущен. До cutover
  нужен отдельный разбор последнего отчёта каждой службы, а не только проверка PID.
- Текущая Timeline-БД:
  `product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`.
- Истина по runtime-указателям: `stable_runtime/CURRENT_RUNTIME.json`.

## Убрано

Все остальные записи предыдущей версии реестра удалены как устаревшие: этих
worktree нет в Git и на диске. Их коммиты, решения и audit packs остаются в
истории репозитория; создавать новые архивные копии не требуется.
