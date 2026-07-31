# Реестр worktree

Обновлено: 2026-07-26.

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
| `/Users/dmitrijfabarisov/Projects/Mango analyse` | `main`, после проверенной интеграции A-F главным Codex | Каноническая папка бота, всех четырёх launchd-служб и локальных runtime-данных; Wappi остановлен владельцем. | Оставить на `main`; runtime/live включать только отдельным этапом после реального staging и смысловой приёмки. |
| `/Users/dmitrijfabarisov/Projects/Mango_calls_dialogue_m1_20260730` | `codex/calls-dialogue-m1-20260730`, временный | Изолированная доработка новых звонков: сохранение дорожек и времени, честное разделение реплик, суточная публикация и подготовка переноса службы на M1. | После поэтапных тестов, независимого аудита, вливания в `main` и отдельного переключения службы запросить снятие worktree; до этого не удалять. |
| `/Users/dmitrijfabarisov/Projects/Mango_critical_gates_20260731` | `codex/critical-gates-integration-20260731`, активный, чистый при старте | Узкая интеграция публичного startup-гейта, карты ADR-003 и теневого P0-замера без live-запуска. | После тестов, audit pack, вливания в `main` и проверки отсутствия процессов запросить снятие worktree; ветку удалить только после поглощения. |
| `/Users/dmitrijfabarisov/Projects/Mango_customer_timeline_junk_map_20260731` | `codex/customer-timeline-junk-map-20260731`, активный | ТЗ №201: read-only перепись пустых звонков и обратимая разметка только на staging; занятые зоны памяти не трогает. | После приёмки, вливания в `main` и проверки отсутствия процессов запросить отдельное разрешение на снятие worktree и удаление ветки. |
| `/Users/dmitrijfabarisov/Projects/Mango_regex_map_d1_20260731` | `codex/regex-to-understanding-map-20260731` от `ca1c9ce5` | Д1: карта ADR-003, только ведро 2 (пол безопасности) и ведро 3 (формат/гигиена); без правок `src/**`. | Снять после сведения карты архитектором, вливания принятого генератора/разметки и проверки отсутствия процессов. |
| `/Users/dmitrijfabarisov/Projects/Mango_name_regex_case_scope_20260730` | `codex/name-regex-case-scope-20260730`, временный | Изолированное исправление ложных имён из-за глобального `re.I`; создан после внешней очистки незакоммиченных правок в канонической папке. | После тестов, audit pack и вливания в `main` снять worktree и удалить ветку. |
| `/Users/dmitrijfabarisov/Projects/Mango_rollback_wappi_ca1779bc` | detached `ca1779bc` | Проверенный rollback Wappi до текущего live-поколения. | Удалить только после M1 PASS, включения защит и live-приёмки. |
| `/Users/dmitrijfabarisov/Projects/Mango_p0_model_led` | `codex/p0-model-led` | Временная разработка перевода P0 с регулярных выражений на решение модели; рабочие системы не использует. | После двух PASS на M1, вливания в `main` и проверки отсутствия процессов снять worktree и удалить ветку. |
| `/Users/dmitrijfabarisov/Projects/Mango_tallanto_linkage` | `codex/tallanto-linkage-root` | Корневое исправление порядка Tallanto, связи оплат с учениками и разделения детей одной семьи; только код и синтетические тесты. | После staging-проходов 2-3, смысловой приёмки и вливания в `main` снять worktree и удалить ветку. |
| `/Users/dmitrijfabarisov/Projects/Mango_customer_timeline_mail_relink` | `codex/customer-timeline-mail-relink` | Изолированная доработка повторной привязки старых писем после обновления Tallanto; исполнитель не пишет в runtime. | После проверки и переноса полезного патча в `main` запросить отдельное разрешение на снятие worktree и удаление ветки. |
| `/Users/dmitrijfabarisov/Projects/Mango_owner50_manager_dossier` | `codex/owner50-manager-dossier` | Изолированная доработка доказательства реального человека в Owner50 через общие identity links; исполнитель не пишет в runtime. | После проверки и переноса полезного патча в `main` запросить отдельное разрешение на снятие worktree и удаление ветки. |
| `/Users/dmitrijfabarisov/Projects/Mango_call_dialogue_timing_20260728` | `codex/call-dialogue-timing-20260728`, чистый | Изолированный анализ длительности реплик звонков; текущая задача Timeline его не использует. | Проверить поглощение перед отдельным решением о снятии; до этого не переключать и не удалять. |
| `/Users/dmitrijfabarisov/Projects/Mango_nightly_source_gate_20260729` | detached `d9dd3cb4`, временный | Изолированное исправление честного подтверждения источников ночной Timeline; только `nightly_service.py` и узкие тесты. | После переноса принятого патча в `main` и проверки отсутствия процессов запросить снятие worktree; ветки нет. |

## Удалено 2026-07-26

После интеграции A-F в `main@74ce3778` удалены два поглощённых донорских
worktree и их локальные ветки: `Mango_ai_employee_implementation` /
`codex/ai-employee-implementation` и `Mango_owner50_family_xlsx` /
`codex/owner50-family-xlsx`. Первый донор целиком покрыт более строгой версией
кода и тестов в `main`. Во втором не переносился одноразовый 840-строчный
скрипт августовской рассылки без тестов, который обходил общий Owner50-контур.
Новых архивных копий и тегов для этого мусора не создавалось.

Ветка `codex/ai-employee-timeline@8210eb87` удалена после проверки, что её
полезный код поглощён более новыми реализациями в `main`. Три уникальных audit
pack перемещены в канонический `audits/_inbox`; вершина сохранена тегом
`archive/ai-employee-timeline-8210eb87`. Ночная служба уже была настроена на
`Mango analyse`, поэтому перевод runtime не требовался.

Интеграционная ветка `codex/ai-employee-final@fcf62571` влита в `main` чистым
fast-forward после независимой проверки девяти дублирующих изменений и полного
`pytest` (`4309 passed, 2 skipped`). Worktree `Mango_ai_employee_final` снят,
локальная ветка удалена через `-d`; audit pack перенесён в каноническую папку.

На зеркале Yandex удалена расходная remote-ветка
`exam/payment-subject-guards`; её точная вершина `6295d550` остаётся достижима
по существующему тегу `exam-payment-subject-guards-v4`.

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
- Customer-timeline nightly настроен на `/Users/dmitrijfabarisov/Projects/Mango analyse`.
  Сейчас процесс не запущен; последний код выхода `1` требует отдельного разбора
  в текущей задаче AI employee.
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
