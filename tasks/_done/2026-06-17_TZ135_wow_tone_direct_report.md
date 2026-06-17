# TZ135 wow-tone direct path

Дата: 2026-06-17
Ветка: `codex/tz135-direct-wow-tone`

## Что изменено

- Добавлен флаг `TELEGRAM_DIRECT_WOW_TONE`, default OFF.
- Флаг работает только на живом direct-path: дополняет `_direct_path_mission_text` и добавляет style-only примеры с placeholder-ами `[... из фактов]`.
- `humanity_x2`/legacy tone-переписчики не включались и не трогались.
- `pilot_gold_v1` не менялся: `TELEGRAM_DIRECT_WOW_TONE` не входит в default-on флаги.
- Для нечисловых продуктовых приписок добавлены NEG-калибровки в существующий LLM semantic verifier, без нового широкого детерминированного гарда.
- Добавлен микронабор `product_data/telegram_dynamic_test_sets/tz135_wow_tone_micro_20260617.jsonl`.

## Проверки

- Узкие тесты direct/semantic: `78 passed, 408 deselected`.
- Полный pytest: `3331 passed, 2 skipped, 1 warning`.

## OFF -> ON микроизмерение

Команда: `run_telegram_dynamic_client_sim`, `pilot_gold_v1`, судья `v9.1`, модель `gpt-5.5`, временный `CODEX_HOME` с `service_tier=fast`.

- OFF: `runs/20260617_tz135_wow_tone_micro_OFF_fast`
  - `PASS=4`, `PASS_WITH_NOTES=2`, `FAIL=0`
  - hard gates: `0`
  - semantic verifier: `checked_turns=11`, `downgraded_turns=0`, `unavailable_turns=0`
- ON: `runs/20260617_tz135_wow_tone_micro_ON_fast_retry`
  - `PASS=3`, `PASS_WITH_NOTES=3`, `FAIL=0`
  - hard gates: `0`
  - semantic verifier: `checked_turns=11`, `downgraded_turns=2`, `regen_attempts=1`, `unavailable_turns=0`

Невалидные попытки:

- `runs/20260617_tz135_wow_tone_micro_OFF` — старый пользовательский `service_tier=default`.
- `runs/20260617_tz135_wow_tone_micro_ON_fast` — сетевой обрыв Codex backend, `turns=0`.

## Смысловой статус

Это `formal_pass` и первичный micro semantic check: hard-gate регрессов на микронаборе нет.
Флаг не включён в профиль и требует регрейда Claude по сырью перед любым включением.
