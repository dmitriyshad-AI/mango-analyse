# TZ-142 Implementation Notes

Дата: 2026-06-18
Ветка: `codex/tz142-flagb-closing-fix`
База: `codex/tz137-adr002-direct-slots-fallback`

## Что сделано

1. Прямой путь теперь применяет существующий слой закрывающих сообщений после слоя действий и переспроса по пустому подбору фактов.
2. Профиль `pilot_gold_v1` включает `TELEGRAM_TONE_CLOSE_DETECT` по умолчанию, но явное `TELEGRAM_TONE_CLOSE_DETECT=0` в контексте или окружении по-прежнему выключает слой.
3. Закрывающие сообщения в прямом пути не создают новый призыв оставить телефон или обещание звонка, если клиент просто завершает разговор.
4. Подтвержденные лагерные факты не режутся на реальных вопросах про лагерь: слой закрытия срабатывает только на сообщения-благодарности/завершения, а не на содержательные вопросы.
5. Флаг B (`TELEGRAM_DIRECT_KEYWORD_FALLBACK_RELEVANCE`) измерен на расширенном наборе, но не включен в профиль из-за P0-регрессии.

## Сырые артефакты прогонов

- Большой OFF-прогон: `runs/tz142_flagb_closing_20260618/B_OFF_profile_close/`
- Большой ON-прогон флага B: `runs/tz142_flagb_closing_20260618/B_ON_profile_close/`
- Дымовой прогон закрытия после no-CTA фикса: `runs/tz142_flagb_closing_20260618/closing_smoke_no_cta_v2/`
- Короткая перепроверка регрессий ON после no-CTA фикса: `runs/tz142_flagb_closing_20260618/B_ON_profile_close_regression_recheck_v2/`
- Сценарий большого набора: `runs/tz142_flagb_closing_20260618/scenarios/tz142_flagb_large_v1.jsonl`
- Сценарий закрывающего дыма: `runs/tz142_flagb_closing_20260618/scenarios/tz142_closing_smoke_v1.jsonl`

## Ключевые метрики

### Большой набор, B OFF

- Диалогов: 36
- Ходов: 76
- PASS: 22
- PASS_WITH_NOTES: 14
- FAIL: 0
- Hard-gate failures: 0
- Причины понижения: `wrong_intent_fact=9`, `fact_grounding=5`, `hard_p0=1`, `brand_leak=1`
- Закрывающий слой: 26 ходов, 15 fired, 8 suppressed_handoff, 2 suppressed_p0, 1 suppressed_pending

### Большой набор, B ON

- Диалогов: 36
- Ходов: 76
- PASS: 22
- PASS_WITH_NOTES: 12
- FAIL: 2
- Hard-gate failures: 2
- Нарушения: `p0_mishandled=1`, `made_a_promise=1`
- Причины понижения: `wrong_intent_fact=8`, `fact_grounding=3`, `hard_p0=1`, `brand_leak=2`, `unsupported_promise=1`
- Вывод: флаг B направленчески снижает часть понижений, но небезопасен для включения из-за P0.

### Закрывающий дым после фикса

- Диалогов: 4
- Ходов: 11
- PASS: 3
- PASS_WITH_NOTES: 1
- FAIL: 0
- Hard-gate failures: 0
- Закрывающий слой: 7 ходов, 6 return, 0 contact_requested
- Замечание: один soft `wrong_scope` относится к первому адресу УНПК, а не к закрывающему ходу.

### ON-перепроверка регрессий после фикса закрытия

- Диалогов: 4
- Ходов: 9
- PASS: 2
- PASS_WITH_NOTES: 1
- FAIL: 1
- Hard-gate failures: 1
- Остался `p0_mishandled` на возврате.
- `tz142_pos_open_literature_unpk` больше не FAIL: стало `PASS_WITH_NOTES` с over-handoff, без обещания звонка.

## Вывод

Часть 2 ТЗ-142 закрыта: дефект закрывающего хода исправлен без вырезания подтвержденных лагерных фактов.

Часть 1 дала честный измерительный результат: флаг B нельзя включать в пилот до отдельной доводки P0-ветки и проверки over-handoff.
