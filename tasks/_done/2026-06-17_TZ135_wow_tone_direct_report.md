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
- По регрейду добавлен большой тон-набор `product_data/telegram_dynamic_test_sets/tz135_wow_tone_coverage_20260617.jsonl` на 24 диалога: разговорные вопросы, закрытия, брендовые ловушки, P0 и неподкреплённые claims.
- Для закрывающих реплик в wow-режиме добавлен отдельный prompt-mode: короткое тёплое закрытие без CTA, новых фактов, цены, расписания, группы и повторного подбора.
- Для closing-mode direct prompt намеренно не подаёт fact blocks; иначе поздний autonomy fact-template мог заменить тёплое закрытие факт-дампом.
- В `apply_autonomy_matrix_guard` добавлен узкий guard: если `direct_path.wow_closing_mode=true`, `autonomy_verified_fact_answer_template_applied` не подменяет модельное закрытие фактами.
- В wow-инструкцию добавлен запрет упоминать старые платформы/технические названия (`MTS-Link`, `МТС Линк`, `Webinar`) в клиентском ответе; в платформенных вопросах называем только текущую платформу из фактов.

## Проверки

- Узкие тесты direct/semantic первичного этапа: `78 passed, 408 deselected`.
- Узкие тесты после closing-фикса: `5 passed, 483 deselected`.
- Полный pytest первичного этапа: `3331 passed, 2 skipped, 1 warning`.
- Финальный полный pytest после большого набора: `3334 passed, 2 skipped, 1 warning`.

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

## OFF -> ON большой тон-набор

Команда: `run_telegram_dynamic_client_sim`, `pilot_gold_v1`, судья `v9.1`, модель `gpt-5.5`, `--parallel 4`, временный `CODEX_HOME` с `service_tier=fast`.

Валидная OFF-база:

- `runs/20260617_tz135_wow_tone_coverage_TZ135_OFF`
  - `dialogs=24`, `turns=49`
  - `PASS=10`, `PASS_WITH_NOTES=13`, `FAIL=1`
  - `hard_gate_failures=1`, `violated_gates={"brand_leak": 1}`
  - `closing_fact_dump=11`, `bad_format=5`, `dry=3`, `wrong_scope=2`, `unwanted_cta=1`

Финальный ON после closing/platform правок:

- `runs/20260617_tz135_wow_tone_coverage_TZ135_ON_v6`
  - `dialogs=24`, `turns=48`
  - `PASS=19`, `PASS_WITH_NOTES=5`, `FAIL=0`
  - `hard_gate_failures=0`, `violated_gates={}`
  - `closing_fact_dump=0`, `bad_format=1`, `dry=2`, `wrong_scope=0`, `unwanted_cta=0`
  - `bot_faithfulness=0`, `bot_direct_draft=40`, `bot_semantic_output_verifier=42`, `bot_retriever=40`

Невалидные/промежуточные попытки большого набора:

- `runs/20260617_tz135_wow_tone_coverage_OFF` и `runs/20260617_tz135_wow_tone_coverage_ON` — невалидны для TZ135: во время долгого запуска рабочее дерево было переключено другим треком на `codex/tz137-adr002-direct-slots-fallback`.
- `runs/20260617_tz135_wow_tone_coverage_TZ135_ON_v2` — hard 0, но `closing_fact_dump=14`: prompt-only closing не перебил поздний fact-template.
- `runs/20260617_tz135_wow_tone_coverage_TZ135_ON_v3` — hard 0, но `closing_fact_dump=13`: причина та же.
- `runs/20260617_tz135_wow_tone_coverage_TZ135_ON_v4` — `closing_fact_dump=1`, но hard `internal_leak=1` на фразе «другие значения перед ответом нужно проверить...».
- `runs/20260617_tz135_wow_tone_coverage_TZ135_ON_v5` — `closing_fact_dump=2`, hard `brand_leak=1` на упоминании `MTS-Link`.

## Смысловой статус

Это `formal_pass` и расширенный semantic smoke на фиксированном тон-наборе: на финальном ON v6 hard-gate регрессов нет, closing fact dump снят на наборе `24` диалога.
Флаг не включён в профиль и требует регрейда Claude по сырью перед любым включением.

## Независимый semantic review

Вердикт: `PASS_WITH_NOTES`.

Что подтверждено:

- `ON_v6`: `FAIL=0`, `hard_gate_failures=0`, `violated_gates={}`.
- P0 остаётся на `manager_only`; брендовые ловушки не дали смешения.
- Закрывающие реплики короткие и без новых фактов.
- Старые платформы (`MTS-Link`, `МТС Линк`, `Webinar`) не всплыли в клиентском тексте.
- `TELEGRAM_DIRECT_WOW_TONE` default OFF и не входит в `pilot_gold_v1`.

Риск/заметка:

- Автоматическая tone-метрика не доказывает улучшение: `tone_metric` OFF `62.1` → ON `58.8`, `tone_metric_non_p0_self` OFF `67.7` → ON `61.4`.
- Набор 24 диалога не является holdout; нужен регрейд Claude по сырью, особенно по tone quality и спорным `no_match/risky_claim`.
