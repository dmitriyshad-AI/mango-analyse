# Semantic Review

## Verdict

PASS_WITH_NOTES.

## What Passed

- Не предложено включать активное понижение маршрута.
- Не предложено ослаблять P0, деньги, запись, живые места или manager-only.
- Вывод основан на сыром runtime subset после F2j, а не только на гипотезе.

## Blocking Issues

Активное включение всё ещё заблокировано:

- нет strict F3 candidates;
- часть безопасных справочных вопросов не имеет runtime exact-proof;
- prompt-only исправление `risk_class/answerability` без proof создаст риск
  выдумки.

## Non-Blocking Risks

- F2j subset маленький; финальное решение требует полного paired shadow eval.
- Gold может быть оптимистичнее runtime: некоторые expected self rows требуют
  факта, которого бот реально не получил.

## Required Regression/Gate

Перед любым future active:

- проверить `too_confident=0`;
- проверить `P0/money/manager_only_lowered=0`;
- требовать fresh client-safe exact proof в runtime metadata;
- отдельно считать existence/format vs live availability.

## Recommended Next Action

Строить не новый regex/prompt-bypass, а shadow-слой доставки проверенного
existence/format proof в runtime metadata.
