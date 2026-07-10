# Semantic Review

Verdict: `PASS_WITH_NOTES` for measurement artifact, `BLOCKED` for active autonomy.

## What Passed

- Gold labels encode the current business boundary without client texts:
  - safe reference questions can be self-answer candidates;
  - money/payment/check/enrollment/booking/live availability remain manager-only;
  - unclear cases stay unresolved rather than forced.
- Calibration report makes the business risk visible instead of hiding behind formal shadow pass.
- `too_confident=0` on this queue supports the hypothesis that current SemanticFrame errs on the cautious side.

## Blocking Issues For Active Use

- `must_handoff` accuracy is only 0.6027.
- `too_cautious=29`, so frame would keep many safe questions with manager.
- `answerability` schema is not followed in saved frames (`yes/no` vs expected enum).
- Gold does not yet cover all 241 full131 turns.

## Required Next Action

Improve SemanticFrame prompt/schema, rerun paired enrichment, rerun this calibration, then ask Claude #1/Dmitry for regрейд before any Ф2/Ф3 behavior change.
