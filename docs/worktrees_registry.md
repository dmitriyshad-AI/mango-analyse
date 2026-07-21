# Реестр worktree

Обновлено: 2026-07-21.

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
| `/Users/dmitrijfabarisov/Projects/Mango analyse` | `main` (текущий HEAD) | Каноническая папка бота, всех четырёх launchd-служб и локальных runtime-данных. | Основной worktree для последовательной разработки и runtime. |
| `/Users/dmitrijfabarisov/Projects/Mango_payment_subject_guards` | `exam/payment-subject-guards` | Неизменяемый приватный M1-экзамен двух защит; production-код уже в `main`, флаг OFF. | Удалить после завершения и смысловой приёмки M1. |
| `/Users/dmitrijfabarisov/Projects/Mango_rollback_wappi_ca1779bc` | detached `ca1779bc` | Проверенный rollback Wappi до текущего live-поколения. | Удалить только после M1 PASS, включения защит и live-приёмки. |

## Runtime-истина

- Wappi, calls A/B и customer-timeline nightly запускаются из
  `/Users/dmitrijfabarisov/Projects/Mango analyse`.
- Точную загруженную ревизию Wappi подтверждают startup manifest, PID/env и
  `scripts/skills/live_truth.py`; один путь в plist сам по себе её не доказывает.
- Wappi пока читает customer timeline из
  `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`.
  Поэтому папку `Mango analyse` нельзя удалять, но переключение её на `main`
  путь к базе не меняет.
- Старый launchd job `com.mango.calls-two-processes` и его plist удалены;
  рабочими остаются только отдельные процессы A/B.

## Завершено 2026-07-21

- `codex/payment-subject-guards` влита fast-forward в `main`; две защиты
  остаются default-OFF до результата M1 и отдельного решения владельца;
- feature-ветка удаляется после публикации `main`; её коммит остаётся в истории
  `main`, поэтому отдельная копия исходников не нужна.

## Удалено 2026-07-19

- `Mango_integrate_d3_d4_20260712`;
- `Mango_live_5d109c38_wappi`;
- `Mango_refactoring_v2`;
- локальная ветка `codex/refactoring-v2-package0` после влития в `main`.

Старые логи внутри worktree признаны расходными и удалены вместе с папками без
создания новых копий. Нужные правила Ponytail из единственного уникального
коммита `e6bfc2d0` уже содержатся в актуальном `AGENTS.md`.

## Удалено 2026-07-17

- `Mango_overclaim_part_a`;
- `Mango_z0_metric_pair2_20260713`;
- `Mango_live_cca8aeb4_consolidated`;
- `Mango_live_next_step_phase1b`.

Экспериментальные ветки `overclaim`, `z0` и `tallanto-freshness` удалены после
проверки существующих archive-тегов. Активный Wappi-worktree сохранён.
Локальная ветка `codex/tz135-direct-wow-tone` удалена после переноса
актуальных правил и проверки archive-тега на её вершине.
