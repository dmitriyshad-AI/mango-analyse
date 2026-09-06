# Реестр worktree

Обновлено: 2026-08-30. Базовая сверка: 2026-08-13.
Локальный источник факта: `git worktree list --porcelain`.
Состояние M1 указано отдельно: главный Mac не видит его PID и рабочую папку.

## Правила

- Один worktree принадлежит одному исполнителю или live-службе.
- Активный live-worktree нельзя переключать или удалять до отдельного cutover.
- Отсутствующая в `git worktree list` папка не считается действующей только
  потому, что упоминалась в старом отчёте.
- История удалённых worktree живёт в Git; архивный список здесь не нужен.

## Этот Mac

| Путь | HEAD / ветка | Назначение | Условие удаления |
|---|---|---|---|
| `/Users/dmitrijfabarisov/Projects/Mango analyse/.codex_workers/ai-passport-group-quality-20260906` | база `e177f0ea`, `codex/ai-passport-group-quality-20260906` | Выделенный владелец: исполнитель локального экспорта AI-паспортов одной группы, ТЗ от 06.09.2026. Код, тесты и локальные 13 паспортов; конечная Google-проверка у main. Без runtime/external write. | После успешной проверки листа, финального review и merge; удаление только с отдельного разрешения Дмитрия. |
| `/Users/dmitrijfabarisov/Projects/Mango analyse` | `15ab6d4c`, `main` | Канонический путь репозитория; текущая интеграция здесь не ведётся. | Не удалять. |
| `/Users/dmitrijfabarisov/Projects/Mango analyse/.codex_workers/customer-timeline-single-writer-20260830` | `15ab6d4c`, `codex/customer-timeline-single-writer-20260830` | Текущая интеграция единственного Customer Timeline writer и трёх релизных гейтов. | После коммита, слияния в `main`, runtime-cutover и отдельной проверки чистоты. |
| `/Users/dmitrijfabarisov/Projects/Mango analyse/.codex_workers/customer-timeline-runtime-20260830` | `227609ca`, `codex/customer-timeline-runtime-20260830` | Зафиксированный код первого M4 runtime-цикла; сохраняется как донор и доказательство до успешного cutover на исправленную ревизию. | После переноса исправлений, двух зелёных циклов, canary и отдельной проверки отсутствия runtime-ссылок. |
| `/Users/dmitrijfabarisov/Projects/Mango analyse/.codex_workers/customer-timeline-source-gates-20260902` | `227609ca`, `codex/customer-timeline-source-gates-20260902` | Изолированный фикс двух source-gate сбоев первого M4-цикла: Mail timestamp proof и неизменяемые Wappi pair-входы. | После тестов, аудита, пуша и успешного runtime-cutover на эту ревизию. |
| `/Users/dmitrijfabarisov/Projects/Mango analyse/.codex_workers/tallanto-group-reconcile-skill-20260903` | `cd7985b4`, `codex/tallanto-group-reconcile-skill-20260903` (проверено 05.09) | Независимый Tallanto-трек; текущая задача только фиксирует существование, не меняет его код и файлы. | Только после отдельной приёмки владельцем этого трека; здесь не удалять. |
| `/Users/dmitrijfabarisov/Projects/Mango analyse/.codex_workers/context-reuse-gate-20260823` | `8237c7de`, `codex/context-reuse-gate-20260823` | Отдельный context-reuse gate; текущая задача его не меняет. | После завершения, коммита и отдельной приёмки. |
| `/Users/dmitrijfabarisov/Projects/Mango_noncontentful_call_memory_integration_20260804` | detached `d9f3df73` | Отдельная интеграционная копия Calls; текущая задача её не меняет. | После отдельной приёмки владельцем. |
| `/Users/dmitrijfabarisov/Projects/Mango analyse/.codex_workers/finance-plan-20260829` | `ba496455`, `codex/finance-plan-20260829` | Отдельный финансовый трек. | После завершения, слияния и отдельной проверки чистоты. |
| `/Users/dmitrijfabarisov/Projects/Mango analyse/.codex_workers/kb-owner-update-20260813` | `2438ec20`, `codex/kb-owner-update-20260813` | Отдельный релиз базы знаний. | После слияния релиза в `main` и проверки рабочего указателя. |
| `/Users/dmitrijfabarisov/Projects/Mango analyse/.codex_workers/unpk-partnership-playbook-20260816` | `279dc0ba`, `codex/unpk-partnership-playbook-20260816` | Документация партнёрской модели УНПК; чистый отдельный worktree. | После приёмки и слияния партнёрского playbook владельцем. |
| `/Users/dmitrijfabarisov/Projects/Mango analyse/.codex_workers/calls-neutral-analysis-fix-20260824` | `806d724e`, `codex/calls-neutral-analysis-fix-20260824` | Отдельный незавершённый Calls-трек; дерево грязное. Не использовать и не менять из Wappi-задачи. | После завершения, коммита и отдельной приёмки владельцем Calls-трека. |
| `/Users/dmitrijfabarisov/Projects/Mango analyse/.codex_workers/development-process-token-budget-20260825` | `63157021`, `codex/development-process-token-budget-20260825` | Чистый отдельный трек правил и бюджета процесса разработки. Не использовать из Wappi-задачи. | После отдельной приёмки и слияния владельцем. |
| `/Users/dmitrijfabarisov/Projects/Mango_asr_model_benchmark_20260808` | `0bd51462`, `codex/asr-model-benchmark-20260808` | Завершённый ASR-бенчмарк; относится к активному Calls-треку. | После приёмки владельцем Calls-трека. |
| `/Users/dmitrijfabarisov/Projects/Mango_calls_quality_v3_m4_20260816` | `23757645`, `codex/mango-calls-quality-v3-m4-20260816` | Документация качества звонков и выборочного Resolve; чистый отдельный worktree. | После приёмки и слияния документации Calls владельцем. |
| `/Users/dmitrijfabarisov/Projects/Mango_calls_utm_google_20260822` | `db63ee2b`, `codex/calls-utm-google-20260822` | Чужой незавершённый Calls-трек UTM/Google; 22.08 дерево грязное. Не использовать и не менять из других задач. | Только после завершения, коммита и отдельной приёмки владельцем Calls-трека. |
| `/Users/dmitrijfabarisov/Projects/Mango_calls_utm_google_v3_20260822` | `89efc14d`, `codex/calls-provider-selective-pilot-20260822` | Отдельный Calls UTM/Google pilot. | После завершения и отдельной приёмки владельцем Calls-трека. |
| `/Users/dmitrijfabarisov/Projects/Mango_m1_calls_final_handoff_20260807` | `f8faabf1`, `codex/m1-calls-final-handoff-20260807` | Пакет передачи Calls на M1. | После завершения и приёмки Calls-cutover. |
| `/Users/dmitrijfabarisov/Projects/Mango_m1_calls_service_integration_20260811` | `868bacd6`, `codex/m1-calls-service-integration-20260811` | Интеграция службы Calls; параллельный трек M1. | После завершения и приёмки Calls-cutover. |
| `/Users/dmitrijfabarisov/Projects/Mango_rollback_wappi_ca1779bc` | detached `ca1779bc` | Проверенный rollback старого Wappi runtime. | После безопасного редеплоя и отдельного решения владельца. |

Поглощённый worktree `model-owned-semantics-cleanup-20260811` удалён
12 августа после проверки чистоты и `merge-base --is-ancestor ... main`.

### Зарегистрированные изолированные ветки M1

Сверка `git worktree list --porcelain` от 2026-08-28. Эти worktree относятся
к отдельным Calls, ASR и прежним Timeline-трекам; текущая задача их не меняет:

- `codex/asr-single-call-progress-20260818`
- `codex/gigaam-batch-20260814`
- `codex/gigaam-parallel-20260814`
- `codex/gigaam-v3-batch-20260814`
- `codex/google-publisher-20260815`
- `codex/mango-calls-m4-handoff-20260816`
- `codex/pipeline-interactive-20260814`
- `codex/pipeline-stall-watchdog-20260818`
- `codex/selective-gigaam-v2-20260814`
- `codex/m1-autonomous-bootstrap-20260807`
- `codex/m1-calls-cutover-20260807`
- `codex/m1-calls-provider-selective-integration-20260823`
- `codex/m1-calls-provider-selective-integration-20260824`
- `codex/m1-calls-real-service-final-20260812`
- `codex/m1-calls-service-fast-value-20260811`
- `codex/m1-customer-timeline-exam-regrade-20260812`
- `codex/m1-customer-timeline-finalization-20260807`
- `codex/m1-customer-timeline-followup-20260811`
- `codex/m1-customer-timeline-release-evidence-20260811`
- `codex/m1-customer-timeline-working-final-20260812`
- `codex/memory-selection-quality-20260731`
- `codex/p0-live-exam-20260731`
- `codex/p0-output-floor-20260730`
- `codex/timeline-quality-20260731`
- `codex/unblock-memory-filters-20260730`

## M1

- По сообщению владельца на M1 активен только трек Calls. Последняя видимая
  ветка: `yandex/codex/m1-calls-service-integration-20260811`, `868bacd6`.
- Customer Timeline на M1 завершён как код-кандидат; его старые worktree не
  перечисляются здесь как активные. Артефакты и ветки остаются в Git/Yandex.
- Точный путь, PID, HEAD и env Calls-worktree проверяются на M1 перед cutover;
  главный Mac не может подтвердить их этим реестром.

## Runtime

- Wappi остановлен владельцем.
- Calls сейчас обрабатывается параллельно на M1; этот файл не является
  доказательством live-PID. Перед merge/cutover обязателен `live_truth.py` на M1.
- Customer Timeline может использовать путь `Mango analyse`; эту папку нельзя
  переключать или удалять без отдельной runtime-сверки.
- Текущая Timeline-БД на главном Mac:
  `product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`.
- Истина по runtime-указателям: `stable_runtime/CURRENT_RUNTIME.json`, если файл
  присутствует в конкретном runtime-пакете.
