# Risk Review

## Verdict

Риск live-регрессии: низкий, потому что изменения только измерительные.

Риск неверного продуктового вывода: высокий, если принять текущие числа за разрешение на автономию. Этого делать нельзя.

## Checked Risks

- Runtime route/text не менялись.
- Новые regex/keyword-понимания не добавлялись.
- `tests/fixtures/adr003_*snapshot.json` не менялись.
- `docs/ADR003_REGEX_UNDERSTANDING_MORATORIUM.md` не менялся.
- P0-preblock/floor не трогались.
- Live bot/Wappi/Telegram/AMO/Tallanto/CRM не трогались.

## Measurement Risks

- Gold покрывает 75 строк mismatch/manager-action очереди, а не все 241 хода full131.
- Две строки помечены `unclear` и исключены из `must_handoff` accuracy.
- `too_confident=0` доказано только на этой gold-очереди, не на полном будущем трафике.
- `answerability` показывает schema mismatch: сохранённые frames используют `yes/no/uncertain`, хотя схема требует `answer_self/manager_only/uncertain`.

## Stop Conditions For Next Phases

- Не переходить к Ф2/Ф3 без улучшения frame-инструкции/схемы и повторной Ф1-калибровки.
- Не добавлять active self-answer gate, пока `must_handoff` accuracy и per-field accuracy не станут приемлемыми для выбранного класса.
- Не считать high confidence достаточной: bucket `0.90-1.00` всё ещё имеет 8 too-cautious rows.
