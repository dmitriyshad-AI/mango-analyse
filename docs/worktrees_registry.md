# Реестр worktree

Обновлено: 2026-07-24.

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
| `/Users/dmitrijfabarisov/Projects/Mango_customer_timeline_arch_audit_20260724` | `codex/customer-timeline-architecture-audit` | Read-only аудит Customer Timeline и подготовка понятного владельцу документа. | Удалить после приёмки документа и переноса нужных выводов в следующие ТЗ. |
| `/Users/dmitrijfabarisov/Projects/Mango_ai_employee_timeline` | `codex/ai-employee-timeline`, активная незавершённая работа | Отдельная задача связки Wappi/AMO с customer timeline. | Не трогать и не удалять до завершения задачи владельцем ветки. |
| `/Users/dmitrijfabarisov/Projects/Mango_m1_measurement_fixes` | `codex/m1-measurement-fixes` | Изолированные исправления измерителей M1. | Не трогать и не удалять до приёмки владельцем ветки. |
| `/Users/dmitrijfabarisov/Projects/Mango_payment_subject_guards` | `exam/payment-subject-guards` | Неизменяемый приватный M1-экзамен двух защит; production-код уже в `main`, флаг OFF. | Удалить после завершения и смысловой приёмки M1. |
| `/Users/dmitrijfabarisov/Projects/Mango_rollback_wappi_ca1779bc` | detached `ca1779bc` | Проверенный rollback Wappi до текущего live-поколения. | Удалить только после M1 PASS, включения защит и live-приёмки. |
| `/Users/dmitrijfabarisov/Projects/Mango_ai_employee_final` | `codex/ai-employee-final` | Единственная интеграционная линия финального ИИ-сотрудника. | Перенести подтверждённые коммиты, проверить, затем влить в `main`. |
| `/Users/dmitrijfabarisov/Projects/Mango_amo_note_idempotency` | `codex/amo-note-idempotency-timeline` | Идемпотентность AMO note поверх Timeline spine. | Не трогать до решения commit-matrix и переноса принятого коммита. |
| `/Users/dmitrijfabarisov/Projects/Mango_customer_timeline_mail_source_contract` | `codex/customer-timeline-mail-source-contract` | Канонический корень почтового источника. | Не трогать до переноса и тестов. |
| `/Users/dmitrijfabarisov/Projects/Mango_customer_timeline_nightly_contracts` | `codex/customer-timeline-nightly-contracts` | Проверки изменяющей nightly-цепочки. | Не трогать до переноса и тестов. |
| `/Users/dmitrijfabarisov/Projects/Mango_family_root_v1` | `codex/family-root-v1` | Устойчивый корень семьи. | Не трогать до переноса и тестов. |
| `/Users/dmitrijfabarisov/Projects/Mango_mail_relink_strong_revalidation` | `codex/mail-relink-strong-revalidation` | Сохранение strong mail links при повторной проверке. | Не трогать до переноса и тестов. |
| `/Users/dmitrijfabarisov/Projects/Mango_owner50_family_xlsx` | `codex/owner50-family-xlsx`, незакоммиченный diff | Owner50, смысловой регрейд `BLOCKED`. | Не переносить и не чистить до исправления и реальной приёмки XLSX. |
| `/Users/dmitrijfabarisov/Projects/Mango_p0_identity_conflicts` | `codex/p0-identity-conflicts` | Блокировка небезопасного объединения семей. | Не переносить snapshot/stash; принять только проверенный функциональный коммит. |
| `/Users/dmitrijfabarisov/Projects/Mango_tallanto_attendance_fix` | `codex/tallanto-attendance-audit-fixes` | Базовое усиление импорта посещений Tallanto. | Источник последовательной цепочки Tallanto; не удалять до переноса. |
| `/Users/dmitrijfabarisov/Projects/Mango_tallanto_attendance_api_increment` | `codex/tallanto-attendance-api-increment` | API-инкремент посещений и partial-import контракт. | Не трогать до переноса и тестов. |
| `/Users/dmitrijfabarisov/Projects/Mango_tallanto_attendance_bot_safe` | `codex/fix-tallanto-attendance-bot-safe-api` | Bot-safe гейт посещений поверх Tallanto chain. | Не трогать до переноса и тестов. |
| `/Users/dmitrijfabarisov/Projects/Mango_wappi_widget_coverage` | `codex/wappi-widget-coverage` | Полный режим покрытия Wappi widget. | Источник последовательной Wappi-цепочки; не удалять до переноса. |
| `/Users/dmitrijfabarisov/Projects/Mango_wappi_amo_talk_authoritative` | `codex/wappi-amo-talk-authoritative` | Точная проверка AMO talk links. | Не трогать до переноса и тестов. |
| `/Users/dmitrijfabarisov/Projects/Mango_wappi_history_attribution_fix` | `codex/wappi-history-attribution-fix` | Сохранение точной атрибуции Wappi history. | Не трогать до переноса и тестов. |

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
- feature-ветка удалена после публикации `main`; её коммит остаётся в истории
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
