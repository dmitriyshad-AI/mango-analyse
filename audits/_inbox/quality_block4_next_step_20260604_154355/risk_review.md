# Risk Review

## Safety Invariants

- Brand separation: unchanged; candidate still passes `_hard_check`.
- P0/high-risk: helper skips P0, refund, complaint, payment dispute, and legal risks.
- No fabrication: helper adds no product numbers/dates/prices; final candidate is verified.
- No pressure: explicit pressure wording is rejected before use.
- Default behavior: `TELEGRAM_Q_NEXT_STEP` is default OFF.

## Residual Risks

- If the underlying answer already contains a weak but detectable next step, the helper will not replace it. This is deliberate to keep one owner and avoid duplication.
- If semantic faithfulness is unavailable, no next step is added; this preserves safety but may reduce the observed effect in some runs.
