# Реестр worktree

Обновлено: 2026-07-17.

Источник факта: `git worktree list --porcelain`, активные `launchd`-конфиги и
рабочие каталоги процессов на хосте.

## Правила

- Один активный worktree закрепляется за одной задачей или службой.
- Реальное состояние службы подтверждается по `launchd` и процессу, а не по
  старой записи в документе.
- Активный live-worktree нельзя переключать или удалять до штатного перевода
  службы на другой путь.
- Финансовые данные и скрипты живут отдельно в
  `/Users/dmitrijfabarisov/Projects/Foton_Finance` и не отслеживаются Git
  бот-проекта.

## Текущие worktree

| Worktree | Состояние | Назначение | Решение |
|---|---|---|---|
| `/Users/dmitrijfabarisov/Projects/Mango analyse` | `codex/tz135-direct-wow-tone` (`9e8fb3b`) | Историческая F1-папка; в ней остаётся используемая Wappi база customer timeline. | Не переключать до отдельного регрейда Этапа 3 и решения зависимости runtime-БД. |
| `/Users/dmitrijfabarisov/Projects/Mango_integrate_d3_d4_20260712` | `main` | Канонический dev-worktree; отсюда работают calls и customer-timeline nightly. | Сохранять чистым, использовать для изменений `main`. |
| `/Users/dmitrijfabarisov/Projects/Mango_live_5d109c38_wappi` | detached `5d109c38` | Фактический Wappi draft-loop (`com.mango.wappi-draft-loop`). | Не трогать до отдельного redeploy. |

## Runtime-истина

- Wappi-код: `/Users/dmitrijfabarisov/Projects/Mango_live_5d109c38_wappi`.
- Calls и customer-timeline nightly:
  `/Users/dmitrijfabarisov/Projects/Mango_integrate_d3_d4_20260712`.
- Wappi пока читает customer timeline из
  `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`.
  Это единственная известная runtime-зависимость от F1-папки и блокер её
  окончательного переключения или удаления.

## Удалено 2026-07-17

- `Mango_overclaim_part_a`;
- `Mango_z0_metric_pair2_20260713`;
- `Mango_live_cca8aeb4_consolidated`;
- `Mango_live_next_step_phase1b`.

Экспериментальные ветки `overclaim`, `z0` и `tallanto-freshness` удалены после
проверки существующих archive-тегов. Активный Wappi-worktree сохранён.
