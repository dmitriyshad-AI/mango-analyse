# Реестр worktree

Обновлено: 2026-08-12. Локальный источник факта: `git worktree list --porcelain`.
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
| `/Users/dmitrijfabarisov/Projects/Mango_noncontentful_call_memory_integration_20260804` | `main`, `0ba2d723` на момент сверки | Чистый канонический код. | После переноса канонического `main` в основной путь. |
| `/Users/dmitrijfabarisov/Projects/Mango analyse/.codex_workers/final-cleanup-regex-20260812` | `codex/final-cleanup-regex-20260812`, база `0ba2d723` | Финальная перепись смысловых regex и удаление мёртвого кода; проверки завершены, ожидает слияния. | После слияния в `main`; удалить можно по ранее данному разрешению владельца на безопасную уборку поглощённых worktree. |
| `/Users/dmitrijfabarisov/Projects/Mango_asr_model_benchmark_20260808` | `0bd51462`, `codex/asr-model-benchmark-20260808` | Завершённый ASR-бенчмарк; относится к активному Calls-треку. | После приёмки владельцем Calls-трека. |
| `/Users/dmitrijfabarisov/Projects/Mango_m1_calls_final_handoff_20260807` | `f8faabf1`, `codex/m1-calls-final-handoff-20260807` | Пакет передачи Calls на M1. | После завершения и приёмки Calls-cutover. |
| `/Users/dmitrijfabarisov/Projects/Mango_m1_calls_service_integration_20260811` | `868bacd6`, `codex/m1-calls-service-integration-20260811` | Интеграция службы Calls; параллельный трек M1. | После завершения и приёмки Calls-cutover. |
| `/Users/dmitrijfabarisov/Projects/Mango_rollback_wappi_ca1779bc` | detached `ca1779bc` | Проверенный rollback старого Wappi runtime. | После безопасного редеплоя и отдельного решения владельца. |

Поглощённый worktree `model-owned-semantics-cleanup-20260811` удалён
12 августа после проверки чистоты и `merge-base --is-ancestor ... main`.

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
