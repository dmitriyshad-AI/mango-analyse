# Реестр worktree

Обновлено: 2026-08-23. Базовая сверка: 2026-08-13.
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
| `/Users/dmitrijfabarisov/Projects/Mango analyse` | `676cc772`, `claude/timeline-final-20260803` | Канонический путь данных и старого Customer Timeline. Ветку не использовать для нового кода. | Только после отдельного cutover на проверенный `main` и новой проверки PID/HEAD/env. |
| `/Users/dmitrijfabarisov/Projects/Mango analyse/.codex_workers/context-reuse-gate-20260823` | `950eef0b`, `codex/context-reuse-gate-20260823` | Незавершённый грязный worktree отдельного context-reuse gate. Не использовать и не менять из Wappi-задачи. | После завершения, коммита и отдельной приёмки context-reuse gate. |
| `/Users/dmitrijfabarisov/Projects/Mango_noncontentful_call_memory_integration_20260804` | `main`, `a3593d74` на момент сверки | Чистый канонический код. | После переноса канонического `main` в основной путь. |
| `/Users/dmitrijfabarisov/Projects/Mango analyse/.codex_workers/final-cleanup-regex-20260812` | `codex/final-cleanup-regex-20260812`, поглощена `main` в `a3593d74` | Завершённая перепись смысловых regex и удаление мёртвого кода. | Можно удалить после проверки чистоты по ранее данному разрешению владельца. |
| `/Users/dmitrijfabarisov/Projects/Mango analyse/.codex_workers/kb-owner-update-20260813` | `codex/kb-owner-update-20260813`, база `a3593d74` | Новый неизменяемый релиз базы знаний по подтверждённым владельцем фактам и переключение рабочего указателя. | После слияния релиза в `main` и проверки рабочего указателя. |
| `/Users/dmitrijfabarisov/Projects/Mango analyse/.codex_workers/unpk-partnership-playbook-20260816` | `279dc0ba`, `codex/unpk-partnership-playbook-20260816` | Документация партнёрской модели УНПК; чистый отдельный worktree. | После приёмки и слияния партнёрского playbook владельцем. |
| `/Users/dmitrijfabarisov/Projects/Mango analyse/.codex_workers/context-reuse-gate-20260823` | `dbabf787`, `codex/context-reuse-gate-20260823` | Проектирование обязательного поиска существующей реализации и воспроизводимого контекста Claude CLI. | После приёмки и слияния ТЗ владельцем. |
| `/Users/dmitrijfabarisov/Projects/Mango_asr_model_benchmark_20260808` | `0bd51462`, `codex/asr-model-benchmark-20260808` | Завершённый ASR-бенчмарк; относится к активному Calls-треку. | После приёмки владельцем Calls-трека. |
| `/Users/dmitrijfabarisov/Projects/Mango_calls_quality_v3_m4_20260816` | `23757645`, `codex/mango-calls-quality-v3-m4-20260816` | Документация качества звонков и выборочного Resolve; чистый отдельный worktree. | После приёмки и слияния документации Calls владельцем. |
| `/Users/dmitrijfabarisov/Projects/Mango_calls_utm_google_20260822` | `db63ee2b`, `codex/calls-utm-google-20260822` | Чужой незавершённый Calls-трек UTM/Google; 22.08 дерево грязное. Не использовать и не менять из других задач. | Только после завершения, коммита и отдельной приёмки владельцем Calls-трека. |
| `/Users/dmitrijfabarisov/Projects/Mango_calls_utm_google_v3_20260822` | `ecdbbf23`, `codex/calls-utm-google-v3-20260822` | Чистый отдельный worktree Calls UTM/Google v3. Не пересекается с текущей Wappi-задачей. | После завершения и отдельной приёмки владельцем Calls-трека. |
| `/Users/dmitrijfabarisov/Projects/Mango_m1_calls_final_handoff_20260807` | `f8faabf1`, `codex/m1-calls-final-handoff-20260807` | Пакет передачи Calls на M1. | После завершения и приёмки Calls-cutover. |
| `/Users/dmitrijfabarisov/Projects/Mango_m1_calls_service_integration_20260811` | `868bacd6`, `codex/m1-calls-service-integration-20260811` | Интеграция службы Calls; параллельный трек M1. | После завершения и приёмки Calls-cutover. |
| `/Users/dmitrijfabarisov/Projects/Mango_rollback_wappi_ca1779bc` | detached `ca1779bc` | Проверенный rollback старого Wappi runtime. | После безопасного редеплоя и отдельного решения владельца. |

Поглощённый worktree `model-owned-semantics-cleanup-20260811` удалён
12 августа после проверки чистоты и `merge-base --is-ancestor ... main`.

## M1

- Факт сверки 2026-08-24: `git worktree list --porcelain` выполнен локально на
  M1. Ни одно дерево при регистрации не удалялось и не переключалось.
- Два ранее грязных Calls-дерева сохранены собственными WIP-коммитами:
  `codex/selective-gigaam-v2-20260814` и `codex/gigaam-batch-20260814`.
- Текущая интеграция ведётся отдельно в
  `codex/m1-calls-provider-selective-integration-20260824`; preflight не требует
  регистрировать собственный текущий worktree.

| Путь M1 | Ветка | Назначение / статус |
|---|---|---|
| `/Users/dmitriy/Codex projects/Mango Analyse m1 pro/.asr-single-call-progress.20260818` | `codex/asr-single-call-progress-20260818` | Отдельная попытка прогресса одного ASR-звонка; не менять из интеграции. |
| `/Users/dmitriy/Codex projects/Mango Analyse m1 pro/.gigaam-batch.2Wg1KM` | `codex/gigaam-batch-20260814` | Сохранённый WIP пакетного GigaAM; чистое дерево. |
| `/Users/dmitriy/Codex projects/Mango Analyse m1 pro/.gigaam-parallel.JGmNSc` | `codex/gigaam-parallel-20260814` | Отдельная попытка параллельного GigaAM. |
| `/Users/dmitriy/Codex projects/Mango Analyse m1 pro/.gigaam-v3-batch.20260814` | `codex/gigaam-v3-batch-20260814` | Отдельная попытка GigaAM v3 batch. |
| `/Users/dmitriy/Codex projects/Mango Analyse m1 pro/.google-publisher.20260815` | `codex/google-publisher-20260815` | Отдельная ветка Google publisher. |
| `/Users/dmitriy/Codex projects/Mango Analyse m1 pro/.m4-handoff.20260816` | `codex/mango-calls-m4-handoff-20260816` | Старый пакет передачи Calls с M4. |
| `/Users/dmitriy/Codex projects/Mango Analyse m1 pro/.pipeline-interactive.20260814` | `codex/pipeline-interactive-20260814` | Отдельная попытка интерактивного pipeline. |
| `/Users/dmitriy/Codex projects/Mango Analyse m1 pro/.pipeline-stall-watchdog.20260818` | `codex/pipeline-stall-watchdog-20260818` | Донор watchdog; не менять напрямую. |
| `/Users/dmitriy/Codex projects/Mango Analyse m1 pro/.selective-gigaam-v2.A77tnJ` | `codex/selective-gigaam-v2-20260814` | Сохранённый WIP selective GigaAM; чистое дерево. |
| `/Users/dmitriy/Projects/Mango_m1_autonomous_bootstrap_20260807` | `codex/m1-autonomous-bootstrap-20260807` | Старый bootstrap M1. |
| `/Users/dmitriy/Projects/Mango_m1_calls_cutover_20260807` | `codex/m1-calls-cutover-20260807` | Старый Calls cutover. |
| `/Users/dmitriy/Projects/Mango_m1_calls_provider_selective_integration_20260823` | `codex/m1-calls-provider-selective-integration-20260823` | Заблокированная попытка с отчётом STOP; не сливать. |
| `/Users/dmitriy/Projects/Mango_m1_calls_real_service_final_20260812` | `codex/m1-calls-real-service-final-20260812` | Канонический Calls-код до новой интеграции. |
| `/Users/dmitriy/Projects/Mango_m1_calls_service_fast_value_20260811` | `codex/m1-calls-service-fast-value-20260811` | Предыдущая ветка службы Calls. |
| `/Users/dmitriy/Projects/Mango_m1_customer_timeline_exam_regrade_20260812` | `codex/m1-customer-timeline-exam-regrade-20260812` | Старый Customer Timeline exam. |
| `/Users/dmitriy/Projects/Mango_m1_customer_timeline_finalization_20260807` | `codex/m1-customer-timeline-finalization-20260807` | Старый Customer Timeline finalization. |
| `/Users/dmitriy/Projects/Mango_m1_customer_timeline_followup_20260811` | `codex/m1-customer-timeline-followup-20260811` | Старый Customer Timeline follow-up. |
| `/Users/dmitriy/Projects/Mango_m1_customer_timeline_release_evidence_20260811` | `codex/m1-customer-timeline-release-evidence-20260811` | Старые release evidence Customer Timeline. |
| `/Users/dmitriy/Projects/Mango_m1_customer_timeline_working_final_20260812` | `codex/m1-customer-timeline-working-final-20260812` | Старый рабочий финал Customer Timeline. |
| `/Users/dmitriy/Projects/Mango_memory_selection_20260731` | `codex/memory-selection-quality-20260731` | Отдельная ветка качества памяти. |
| `/Users/dmitriy/Projects/Mango_p0_live_exam_20260731` | `codex/p0-live-exam-20260731` | Отдельный P0 live exam. |
| `/Users/dmitriy/Projects/Mango_p0_output_floor_20260730` | `codex/p0-output-floor-20260730` | Отдельный P0 output floor. |
| `/Users/dmitriy/Projects/Mango_timeline_quality_20260731` | `codex/timeline-quality-20260731` | Отдельная ветка качества timeline. |
| `/Users/dmitriy/Projects/Mango_unblock_memory_filters_20260730` | `codex/unblock-memory-filters-20260730` | Отдельная ветка фильтров памяти. |

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
