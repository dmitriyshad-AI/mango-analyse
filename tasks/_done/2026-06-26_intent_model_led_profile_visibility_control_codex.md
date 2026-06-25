# Intent model-led: profile activation, visibility, control60

Дата: 2026-06-26  
Ветка: `codex/intent-model-led`  
Исходный HEAD: `d35f662`  
Режим: кодовый worktree, без live-write, без Telegram polling, без AMO/Tallanto/stable_runtime write.

## Самопроверка промпта Claude

Вывод: направление верное, но перед закреплением в профиль я нашёл и закрыл 3 обязательных уточнения.

1. `TELEGRAM_INTENT_MODEL_LED` действительно был default OFF и не входил в `pilot_gold_v1`; закрепил через `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`.
2. Visibility-gap подтвердился: `model_intent` не попадал в `dynamic_dialog_transcripts.jsonl`/`dynamic_turns.csv`; добавил `bot_model_intent`, `bot_intent_model_led` и CSV-колонки `bot_model_intent_primary_intent/scope/sense/confidence`.
3. Дополнительный риск по смыслу: один mixed availability-кейс `есть ли места и где` после handoff поднимался авто-матрицей обратно в `bot_answer_self_for_pilot`. Добавил запрет promotion для `conversation_intent_plan_live_availability`.

## Изменения

- `src/mango_mvp/channels/subscription_llm_parts/support.py`: `INTENT_MODEL_LED_ENV` добавлен в `pilot_gold_v1`; `_intent_model_led_enabled()` теперь учитывает профиль, но явный `TELEGRAM_INTENT_MODEL_LED=0` остаётся override для отключения.
- `src/mango_mvp/channels/pilot_profile_runtime.py`: heartbeat/selfcheck показывает эффективный `intent_model_led`, а не только явный env.
- `scripts/run_telegram_dynamic_client_sim.py`: сериализация `model_intent` в JSONL/CSV; `run_config.key_flags.intent_model_led.effective=true` при `pilot_gold_v1` даже без env override.
- `src/mango_mvp/channels/subscription_llm_parts/policy_routing.py`: confidence floor `0.72`; прямые вопросы про наличие/бронь/остаток мест не демоутятся model-led; live-availability draft не повышается авто-матрицей до self-route.
- Тесты: добавлены регрессии на профиль, visibility, summary-флаг, явный availability floor, low-confidence floor, запрет promotion live-availability.

## Control60 ON

Вход:

- `runs/intent_model_led_control60_20260626_input/control60_scenarios.jsonl`
- `runs/intent_model_led_control60_20260626_input/control60_replay.jsonl`

Финальный прогон:

- output: `runs/intent_model_led_control60_ON_20260626_r2`
- `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`
- `TELEGRAM_INTENT_MODEL_LED` не задан явно; включение через профиль
- `TELEGRAM_P0_MODEL_LED=1`, `TELEGRAM_PROSE_MODEL_LED=1`, `TELEGRAM_FACT_VENUE_SCOPE=1`, `TELEGRAM_AUTONOMY_SCOPE_PRECISION=1`
- `TELEGRAM_BOT_SAFE_CRM_CONTEXT=0`
- `--parallel 4`, `--model gpt-5.5`, `--bot-reasoning high`, replay/fake client

Итог:

- dialogs/turns: `60/60`
- verdicts: `PASS=38`, `PASS_WITH_NOTES=22`, `FAIL=0`
- hard_gate_failures: `0`
- violated_gates: `{}`
- routes: `draft_for_manager=37`, `bot_answer_self_for_pilot=17`, `manager_only=6`
- direct_path_error/llm_fallback markers: `0`
- provider_errors/fallback_reasons: пусто по всем 60
- `run_config.key_flags.intent_model_led = {"env": "", "effective": true}`
- `bot_model_intent_*` CSV-колонки есть; `model_intent` заполнен в `57/60` ходах

Контроль availability:

- `intent_true_seats_foton` -> `draft_for_manager`
- `intent_true_booking_unpk` -> `draft_for_manager`
- `intent_true_seat_count_foton` -> `draft_for_manager`
- `wappi_019d873a59e57427` -> `draft_for_manager`
- `wappi_02f4c1269b997c7b` -> `draft_for_manager`
- `wappi_5076bc3c8237a600` -> `draft_for_manager`
- `wappi_421e9f2a6acc69f2` -> `draft_for_manager`

P0/brand controls:

- refund/complaint synthetic controls: PASS, manager-only/preblock path.
- cross-brand synthetic controls: PASS; brand-leak attempts blocked/manager-routed.

## Тесты

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m py_compile ...` -> OK
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py -k 'intent_model_led or live_availability_draft_is_not_promoted or safe_green_draft_is_promoted or live_availability_missing_fact'` -> `10 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_telegram_dynamic_client_sim.py -k 'key_run_flags or model_intent'` -> `6 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_telegram_public_pilot_bots.py -k 'selfcheck_reports_intent_model_led_from_profile or selfcheck_reports_release_flag_guards or env_file_sync_precedes'` -> `3 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q` -> `3624 passed, 5 skipped, 1 warning`

## Safety / semantic review

Formal pass: да.  
Внутренний semantic pass для критерия этого захода: да, после фикса promotion все явные `есть места?/бронь/сколько мест` в control60 остались менеджерскими.  
Остаточный риск: `22 PASS_WITH_NOTES` требуют регрейда Claude #1 по сырью; verdict "в прод" не выносился.

Live boundaries:

- клиентам ничего не отправлялось;
- AMO/Tallanto не писались;
- live Telegram bot не трогался;
- stable_runtime не трогался;
- записи только в worktree: код/тесты/этот отчёт и локальные `runs/` артефакты.

