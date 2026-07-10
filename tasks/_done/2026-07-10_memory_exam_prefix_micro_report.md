# D4: экзамен памяти, предусловия и микро-пилот

Дата: 2026-07-10
Ветка: `codex/email-pipeline-restore`
Базовый HEAD перед сохранением стенда: `631030ae`
Режим: read-only replay, клиентам 0, AMO/Tallanto/CRM/prod 0 write.

## Что сделано

1. Проверено ТЗ `2026-07-10_TZ_B_ekzamen_pamyati_epoha2.md`.
2. Подключены аудиторы-архитекторы:
   - аудитор подтвердил: полный 100x2 нельзя запускать до prefix replay + микро-пилота;
   - аудитор по данным подтвердил корпус: `telegram_history` 83, `wappi_telegram` 49, `wappi_max` 5 пригодных диалогов.
3. Исправлена явность ON/OFF флагов памяти:
   - ON: `TELEGRAM_BOT_SAFE_CRM_CONTEXT=1`, `TELEGRAM_TIMELINE_MEMORY_IN_PROMPT=1`, `_DB=<staging db>`;
   - OFF: `TELEGRAM_BOT_SAFE_CRM_CONTEXT=0`, `TELEGRAM_TIMELINE_MEMORY_IN_PROMPT=0`.
4. Добавлен живой prefix replay:
   - текущий ход получает клиентское сообщение из старой переписки;
   - следующий ход видит реальный предыдущий ответ менеджера, а не новый ответ бота;
   - судья получает реальный ответ менеджера как эталон сравнения.
5. Собран corpus 100 диалогов из staging.
6. Проведён микро-пилот 5 диалогов OFF/ON.

## Артефакты

Raw/PДн только локально:

- `.codex_local/staging/memory_exam_live_prefix_20260710/memory_exam_live_prefix_report.json`
- `.codex_local/staging/memory_exam_live_prefix_20260710/memory_exam_live_prefix_scenarios.jsonl`
- `.codex_local/staging/memory_exam_live_prefix_20260710/memory_exam_live_prefix_replay.jsonl`
- `.codex_local/staging/memory_exam_live_prefix_20260710/micro_off/`
- `.codex_local/staging/memory_exam_live_prefix_20260710/micro_on/`

DB и KB:

- timeline DB: `.codex_local/staging/customer_timeline_staging.sqlite`, sha256 prefix `9947b9ceddeeb84e`
- KB snapshot: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`, sha256 prefix `f2f211a56cafc213`

## Corpus 100

- eligible: `telegram_history=83`, `wappi_telegram=49`, `wappi_max=5`
- selected: `telegram_history=46`, `wappi_telegram=49`, `wappi_max=5`
- selected dialogs: `100`
- selected turns: `338`
- runtime-visible memory dialogs: `52`
- without visible memory: `48`
- PII scan по replay text: `{}`

Ограничение: для части `telegram_history` staging содержит уже обрезанный текст; метрики трактовать как проверку на доступном staging-тексте, не на полном сыром Telegram export.

## Preconditions

- `probe_memory_measure_context.py` на staging: passed.
- live micro-5 expected-hit: `off_clear=true`, `on_visible_personas=5/5`, `other_brand_clear=true`.
- fake smoke replay: прошёл, 1 диалог, 4 хода.
- next-step guard tests: зелёные в составе точечных тестов.

## Микро-пилот 5

OFF:

- dialogs: `5`
- turns: `16`
- PASS_WITH_NOTES: `4`
- FAIL: `1`
- hard_gate_failures: `1`
- llm_calls_total: `21` (`bot_draft=16`, `judge=5`)

ON:

- dialogs: `5`
- turns: `16`
- PASS_WITH_NOTES: `4`
- FAIL: `1`
- hard_gate_failures: `1`
- llm_calls_total: `21` (`bot_draft=16`, `judge=5`)

Сравнение:

- memory_helped: `1/5`
- memory_hurt: `0/5`
- FAIL не ухудшился: `1 -> 1`
- manager_reference_alignment: без улучшения (`major_edit=4`, `unsafe=1` в обоих плечах)
- send_unedited_proxy: `0.5 -> 0.5`
- brand leak: не зафиксирован
- P0 ухудшения: не зафиксировано

Наблюдение: единственный FAIL в обоих плечах связан с неподтверждённой фразой про промокоды, это не вред памяти.

## Бюджет полного 100x2

Микро: `42` LLM-вызова на оба плеча, `16` ходов на плечо.
Полный corpus: `338` ходов на плечо.

Оценка полного 100x2: около `887` LLM-вызовов:

- bot_draft: около `676`
- judge: около `211`

Полный 100x2 не запускался: по ТЗ после микро-пилота нужен отдельный budget review.

## Тесты

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_memory_measure_apparatus.py \
  tests/test_memory_exam_live_prefix_set.py \
  tests/test_telegram_dynamic_client_sim.py::test_replay_prefix_uses_real_manager_reply_instead_of_generated_bot_reply \
  tests/test_bot_safe_memory_step_guard.py

24 passed
```

## Изменённые файлы этой задачи

- `scripts/probe_memory_measure_context.py`
- `scripts/run_memory_measure_off_on.py`
- `scripts/run_telegram_dynamic_client_sim.py`
- `scripts/build_memory_exam_live_prefix_set.py`
- `tests/test_memory_measure_apparatus.py`
- `tests/test_telegram_dynamic_client_sim.py`
- `tests/test_memory_exam_live_prefix_set.py`

В worktree есть чужие незакоммиченные изменения вне этой задачи; они не трогались и не включались.

## Решение

Полный 100x2 сейчас не запущен. Дальше безопасный следующий шаг: Claude #1 смотрит сырьё микро-пилота и бюджет; после отдельной отмашки можно запускать полный 100x2 тем же harness.
