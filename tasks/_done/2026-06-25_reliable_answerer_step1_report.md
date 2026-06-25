Ветка: codex/reliable-answerer-step1
База: codex/release-venue-autonomy @ 4caa5eb
ТЗ: /Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-25_TZ_FINAL_reliable_answerer_step1_merged.md
Live/AMO/Tallanto/CRM: не трогались
Push/merge: не выполнялись

# Reliable Answerer Step1 — отчёт для регрейда

## Предстартовая проверка базы

- Worktree: `/Users/dmitrijfabarisov/Projects/Mango_reliable_answerer_step1`
- HEAD базы: `4caa5eb`
- `src/mango_mvp/channels/fact_venue_scope.py` есть.
- `TELEGRAM_FACT_VENUE_SCOPE` есть в `fact_venue_scope.py`.
- `TELEGRAM_AUTONOMY_SCOPE_PRECISION` есть в `dialogue_contract_pipeline.py`.
- Worktree был clean перед началом.

## Что реализовано

1. Добавлен флаг `TELEGRAM_RELIABLE_ANSWERER_STEP1`, default OFF.
   - Не добавлен в `pilot_gold_v1`.
   - В `dynamic_summary.json` виден как `run_config.key_flags.reliable_answerer_step1`.

2. Добавлен слой `answer_coverage_plan` в прямом пути.
   - Файл: `src/mango_mvp/channels/subscription_llm_parts/reliable_answerer.py`.
   - Facets: price, schedule, address, dates, format, documents, platform, trial, enrollment, availability, other.
   - Covered считается только по exact-фактам из текущего fact_pack.
   - Venue-sensitive facets (`price`, `schedule`, `address`, `dates`) не покрываются фактом без конкретной venue-метки.
   - Чужой venue не считается покрытием.

3. Под Step1 включается структурный venue-слой ретривера без включения `TELEGRAM_FACT_VENUE_SCOPE` в профиль.
   - Это нужно, чтобы Step1 видел `requested_scope` и venue/program_kind метаданные.
   - Venue/autonomy код не переносился в задачу, база взята из `4caa5eb`.

4. В prompt прямого пути добавлен блок «Надёжный ответчик» только под флагом.
   - Требует отвечать на подтверждённую часть вопроса.
   - Недостающую live-часть оставлять менеджеру.
   - Запрещает обещать места/бронь/запись/наличие группы без отдельного проверенного факта.
   - Запрещает брать цены/расписание/адреса/даты/места из CRM/Tallanto/customer memory.

5. Добавлен детерминированный выходной guard на обещания availability.
   - Если draft обещает место/бронь/запись/наличие без покрытого availability-факта, маршрут становится `draft_for_manager`.
   - Добавляется `reliable_answerer_availability_promise_blocked`.
   - Гейт не зависит от LLM-верификатора.

6. Исправлен узел `apply_autonomy_matrix_guard` для live-status missing.
   - Если Step1 ON и draft уже ответил хотя бы на один covered facet, draft сохраняется.
   - Маршрут остаётся `draft_for_manager`.
   - Добавляется детерминированная оговорка: менеджер отдельно проверит актуальное наличие места/группы.
   - Если покрытой части нет или draft обещает availability — старое безопасное понижение сохраняется.

7. Расширена трассировка.
   - В ход добавляется `bot_reliable_answerer`.
   - В `dynamic_summary.json` добавляется блок `reliable_answerer`:
     - `denominator`
     - `answerable_but_handoff`
     - `partial_answer_success`
     - `hard_safety_failures`
     - `evidence_cases`

8. Добавлен целевой набор:
   - `product_data/telegram_dynamic_test_sets/reliable_answerer_step1_20260625.jsonl`
   - 12 персон + simulator/judge spec.
   - Сюжеты: составные вопросы, цена+места, расписание+места, адрес+пробное, бронь, P0, кросс-бренд.

## Проверки

- Точечные тесты:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_reliable_answerer_step1.py tests/test_telegram_dynamic_client_sim.py::test_summary_includes_reliable_answerer_metrics tests/test_telegram_dynamic_client_sim.py::test_summary_dumps_key_run_flags tests/test_subscription_llm_draft_provider.py::test_live_availability_missing_fact_blocks_autonomy_even_with_verified_program_fact tests/test_subscription_llm_draft_provider.py::test_live_availability_fixation_question_answers_process_not_repeated_handoff tests/test_subscription_llm_draft_provider.py::test_live_availability_data_needed_question_uses_known_slots tests/test_subscription_llm_draft_provider.py::test_llm_missing_facts_do_not_block_autonomy_when_context_fact_is_verified`
  - Результат: `13 passed`.

- Полный pytest:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q`
  - Результат: `3619 passed, 5 skipped, 1 warning`.

- Фасадный импорт:
  `from mango_mvp.channels.subscription_llm import RELIABLE_ANSWERER_STEP1_ENV, build_answer_coverage_plan`
  - Результат: успешно.

## Смысловая самопроверка

`formal_pass`: да, тесты зелёные.

`semantic_pass`: частичный, только локальная проверка логики. Поведенческий M1-прогон и независимый регрейд Claude #1 ещё нужны.

Проверенные смысловые риски:

- Бот не должен обещать место/бронь/запись без live-факта: закрыто детерминированным guard и тестом.
- Бот не должен целиком сдавать составной вопрос менеджеру, если есть проверенная часть: закрыто prompt-блоком, coverage plan и тестом live-status preserve.
- Бот не должен брать live-данные из CRM/Tallanto/customer memory: добавлено в prompt-блок; runtime-гейт на memory-числа уже отдельный слой, не менялся.
- Бренд/P0/числа/scope-гейты не ослаблялись.

Что не проверено без M1:

- Реальная модель под флагом может всё ещё не ответить на covered facet; это измеряется `answerable_but_handoff` и требует OFF/ON прогона.
- Простые regex-фасеты могут быть грубыми; они не должны открывать self-send, но могут завысить/занизить denominator. Это метрика для регрейда, не runtime-разрешение.
- Набор сценариев создан, но не прогонялся на живой модели в этой задаче.

## Рекомендация для следующего шага

Запустить OFF/ON замер на `reliable_answerer_step1_20260625.jsonl` с текущей боевой связкой `4caa5eb` и флагом `TELEGRAM_RELIABLE_ANSWERER_STEP1=1` только в ON-плече. Проверять:

- `answerable_but_handoff` должен снижаться.
- `partial_answer_success` должен расти.
- `availability_promise_detected` и hard-safety failures должны быть 0.
- P0/бренд/числа/scope не должны ухудшиться.

