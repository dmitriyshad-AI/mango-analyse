# Semantic Review

Verdict: `PASS_WITH_NOTES`

## What Passed

- Новая инструкция соответствует бизнес-смыслу: “существует ли курс/формат” не равно “есть ли места”.
- Prompt не разрешает отвечать про live availability, бронь, запись, конкретную группу или личный статус.
- Local posthoc measurement показал улучшение `requested_action`: 6 ошибок -> 1.
- `too_confident=0` на subset.

## Blocking Issues

Нет блокеров для shadow/prompt-calibration.

## Non-Blocking Risks

- `risk_class` и `answerability` не улучшились на локальном subset: safe reference всё ещё часто классифицируется как `missing_facts/manager_only`.
- Это posthoc measurement, не полный M1 paired run.
- Prompt-тесты доказывают наличие инструкции, но не доказывают стабильное поведение модели.

## Required Regression/Gates

- Перед любым active: новый paired eval на полном ADR003 наборе.
- Active только если `too_confident=0`, `P0 lowered=0`, brand/fabrication=0.
- `must_handoff` и `answerability` должны улучшиться, а не только `requested_action`.

## Recommended Next Action

Следующий шаг — калибровать `risk_class/answerability`: safe reference с точным фактом не должен становиться `missing_facts/manager_only`, если вопрос не про места/бронь/запись.
