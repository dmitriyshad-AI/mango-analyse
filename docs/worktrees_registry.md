# Реестр worktree

Обновлено: 2026-07-25.

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
| `/Users/dmitrijfabarisov/Projects/Mango_ai_employee_final` | `codex/ai-employee-final` | Единственная интеграционная линия текущего цикла ИИ-сотрудника. | Влить в `main` только после staging E2E и смысловой приёмки. |
| `/Users/dmitrijfabarisov/Projects/Mango_ai_employee_timeline` | `codex/ai-employee-timeline`, активная незавершённая работа | Отдельная задача связки Wappi/AMO с customer timeline. | Не трогать и не удалять до завершения задачи владельцем ветки. |
| `/Users/dmitrijfabarisov/Projects/Mango_owner50_family_xlsx` | `codex/owner50-family-xlsx`, незакоммиченная уникальная работа | Донор исследования Owner50; целиком не переносится из-за избыточного diff. | Сохранить до точечного переноса полезных правил и проверки реальной витрины. |
| `/Users/dmitrijfabarisov/Projects/Mango_rollback_wappi_ca1779bc` | detached `ca1779bc` | Проверенный rollback Wappi до текущего live-поколения. | Удалить только после M1 PASS, включения защит и live-приёмки. |

## Удалено 2026-07-25

После проверки чистоты, отсутствия процессов и смыслового поглощения веткой
`codex/ai-employee-final` удалены 14 старых worktree и их локальные ветки:

- `Mango_amo_note_idempotency`;
- `Mango_customer_timeline_arch_audit_20260724`;
- `Mango_customer_timeline_mail_source_contract`;
- `Mango_customer_timeline_nightly_contracts`;
- `Mango_family_root_v1`;
- `Mango_m1_measurement_fixes`;
- `Mango_mail_relink_strong_revalidation`;
- `Mango_p0_identity_conflicts`;
- `Mango_tallanto_attendance_api_increment`;
- `Mango_tallanto_attendance_bot_safe`;
- `Mango_tallanto_attendance_fix`;
- `Mango_wappi_amo_talk_authoritative`;
- `Mango_wappi_history_attribution_fix`;
- `Mango_wappi_widget_coverage`.

Также удалены две пустые локальные ветки без worktree, обе указывали точно на
`main`: `codex/amo-note-idempotency` и
`codex/fix-tallanto-attendance-bot-safe`.

После завершения M1-приёмки штатно снят
`Mango_payment_subject_guards` и удалена ветка
`exam/payment-subject-guards`. Полный пакет остаётся достижим по точному тегу
`exam-payment-subject-guards-v5`; новая архивная копия не создавалась.

## Runtime-истина

- Wappi сейчас остановлен; активные calls A/B запускаются из
  `/Users/dmitrijfabarisov/Projects/Mango analyse`.
- Текущий процесс customer-timeline nightly запущен из
  `/Users/dmitrijfabarisov/Projects/Mango_ai_employee_timeline`; этот worktree
  нельзя снимать до штатного перевода ночной службы.
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
