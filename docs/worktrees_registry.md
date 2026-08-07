# Реестр worktree

Обновлено: 2026-08-05. Источник факта: `git worktree list --porcelain`,
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
| `/Users/dmitrijfabarisov/Projects/Mango analyse` | `676cc772`, `claude/timeline-final-20260803` | Канонический путь данных и процессов calls A/B и Customer Timeline nightly. Все уникальные коммиты разобраны: полезное уже в `main`, остаток отклонён; ветка не содержит очереди на слияние. Дерево чистое. | Только после отдельного cutover служб на проверенный `main` и проверки PID/HEAD/env; после этого ветку можно удалить. |
| `/Users/dmitrijfabarisov/Projects/Mango_noncontentful_call_memory_integration_20260804` | `main`; SHA через `git rev-parse main` | Чистый канонический код: все донорские ветки разобраны; отсюда готовится следующий рефакторинг и будущий cutover. | После перевода служб на проверенный `main` в основной папке. |
| `/Users/dmitrijfabarisov/Projects/Mango_project_state_cleanup_audit_20260805` | `codex/project-state-cleanup-audit-20260805`; SHA через `git rev-parse HEAD` | Read-only переаудит состояния проекта, карты D1 и остатка безопасной уборки; код и runtime не изменяются. | После сохранения отчёта, audit pack, слияния документации в `main` и отдельного подтверждения владельца. |
| `/Users/dmitrijfabarisov/Projects/Mango_m1_minimal_p0_exam_20260805` | `codex/m1-minimal-p0-exam-20260805`; SHA через `git rev-parse HEAD` | Подготовка минимального read-only P0-экзамена и точечного класса `child_safety` из D-103; runtime не изменяется. | После получения M1-результата, приёмки главным Codex, слияния в `main` и отдельного подтверждения владельца. |
| `/Users/dmitrijfabarisov/Projects/Mango_global_code_cut_wave10_20260805` | `codex/global-code-cut-wave12`; SHA через `git rev-parse HEAD` | Уборка невызванного CRM profile summary, сломанного readiness-скрипта и закрытие устаревшего PROFILE_PHONE_INDEX флага. | После audit pack, слияния в `main` и отдельного подтверждения владельца. |
| `/Users/dmitrijfabarisov/Projects/Mango_p0_model_led_finish_20260805` | `codex/p0-model-led-finish`; SHA через `git rev-parse HEAD` | D-113: модель является единственным владельцем смысла P0 в черновиковом direct path; live не запускался. | После слияния D-113 в `main`, сверки `p0_model_only_cutover_*` audit pack и отдельного подтверждения владельца. |
| `/Users/dmitrijfabarisov/Projects/Mango_timeline_family_mail_gold_fix_20260805` | `codex/timeline-family-mail-gold-fix`; SHA через `git rev-parse HEAD` | Строгая семейная связь exact Tallanto cards и эталонной почты на копии staging. | После PASS эталона, слияния в `main` и удаления только временной копии БД. |
| `/Users/dmitrijfabarisov/Projects/Mango_timeline_owner_relink_fix_20260805` | `codex/timeline-exact-family-anchor-20260806`; SHA через `git rev-parse HEAD` | Exact Tallanto anchor для семейного графа и bot-safe отбора; только существующие функции, тесты и APFS-клон staging. | После лестницы 1→10→всё, audit pack, слияния в `main` и отдельного подтверждения владельца. |
| `/Users/dmitrijfabarisov/Projects/Mango_question_catalog_cleanup_20260806` | `codex/global-code-cut-sales-insights`; SHA через `git rev-parse HEAD` | Удаление замкнутого старого Sales Insights reporting island с сохранением живых sanitizers/tone_score и их защитных тестов. | После тестов, audit pack, слияния в `main` и отдельного подтверждения владельца. |
| `/Users/dmitrijfabarisov/Projects/Mango_m1_calls_final_handoff_20260807` | `codex/m1-calls-final-handoff-20260807`; SHA через `git rev-parse HEAD` | Финализация единого пакета переноса конвейера звонков на M1: инструкции, план, основной скрипт, ТЗ и безопасные тесты. | После независимого аудита, коммита и вливания в `main`; запуск службы на M1 остаётся отдельным решением владельца. |
| `/Users/dmitrijfabarisov/Projects/Mango_rollback_wappi_ca1779bc` | detached `ca1779bc` | Проверенный rollback старого Wappi runtime. | После M1 PASS, безопасного редеплоя и отдельного решения владельца. |

## Runtime

- Wappi остановлен владельцем.
- Read-only сверка 7 августа не нашла процессов Calls A/B и загруженных меток
  `com.mango.calls-two-processes`, `com.mango.calls-process-a` или
  `com.mango.calls-process-b`. Файлы plist для A/B существуют, но не загружены.
- `live_truth.py` не проверяет calls, поэтому перед каждым merge/cutover всё
  равно обязательна новая сверка PID, launchd-меток, командной строки и config;
  исторический статус от 4 августа не использовать как текущий.
- Customer Timeline может использовать путь `Mango analyse`; эту папку нельзя
  переключать или удалять в ходе интеграции Git без отдельной runtime-сверки.
- Текущая Timeline-БД:
  `product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`.
- Истина по runtime-указателям: `stable_runtime/CURRENT_RUNTIME.json`.

## Убрано

Все остальные записи предыдущей версии реестра удалены как устаревшие: этих
worktree нет в Git и на диске. Их коммиты, решения и audit packs остаются в
истории репозитория; создавать новые архивные копии не требуется.
