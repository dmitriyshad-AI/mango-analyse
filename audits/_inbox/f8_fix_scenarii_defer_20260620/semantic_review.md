# Semantic review — F8 clean defer

Verdict: PASS_WITH_NOTES

## What passed

- The two wrong scenarios now match the confirmed business rule: no УНПК weekday online product for 10th grade.
- The selector now treats weekday/weekend as a real axis when it is explicitly present in the client question.
- The clean-defer behavior is default-off and only activates when both price-axis selector and clean-defer flags are on.
- After explicit approval, `pilot_gold_v1` now activates both F8 flags by default.
- Valid products are not suppressed: the УНПК 5th-grade weekend semester case still returns `37 000 ₽`.
- The KB was not edited, so no source fact was invented.

## Non-blocking risks

- The M1 judge still needs raw transcript review for `f8_018` and `f8_030`; aggregate PASS may hide verbose but irrelevant handoff wording.
- The rule currently returns an empty fact pack on price-selector dead-end. This is intentional for clean defer, but first live drafts must be reviewed by raw transcript.

## Required regression checks

- `10 класс онлайн по будням за семестр сколько стоит?` must not use weekend price.
- `УНПК онлайн, математика 5 класс по выходным, семестр сколько?` must still return `37 000 ₽`.
- Manager draft / clarification for dead-end price query must not include olympiad, curator or unrelated product facts.
- `TELEGRAM_PRICE_AXES_SELECTOR=0` and `TELEGRAM_PRICE_AXES_CLEAN_DEFER=0` must disable profile defaults for rollback.
