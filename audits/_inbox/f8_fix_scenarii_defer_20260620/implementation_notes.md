# Implementation notes — F8 scenario fix and clean defer

Implemented the safe part of `2026-06-21_TZ_F8_fix_scenarii_i_defer.md`.

The confirmed business rule is that УНПК weekday online products exist for 9th and 11th grade, not for 10th grade. The code now respects the explicit weekday/weekend axis when selecting price facts.

The new flag `TELEGRAM_PRICE_AXES_CLEAN_DEFER` is default-off. When enabled together with `TELEGRAM_PRICE_AXES_SELECTOR`, a dead-end price query returns no selected facts, preventing unrelated facts from leaking into a manager draft / clarification.

The pilot profile was intentionally not changed in this step.
