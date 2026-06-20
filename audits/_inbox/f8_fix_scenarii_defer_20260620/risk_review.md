# Risk review — F8 clean defer

## Reduced risks

- Wrong price risk: explicit weekday requests no longer silently fall back to weekend prices.
- Irrelevant-fact risk: dead-end price queries can produce a clean empty fact pack under `TELEGRAM_PRICE_AXES_CLEAN_DEFER`.
- False-measurement risk: `f8_018` and `f8_030` expectations no longer demand a nonexistent 10th-grade weekday product price.

## Remaining risks

- The new clean defer flag is not yet live in `pilot_gold_v1`; M1 must confirm it before activation.
- Empty fact pack behavior depends on downstream route logic producing a useful clarification / manager draft.
- Wider phrasings of weekday/weekend were not exhaustively tested; current tests cover direct Russian wording.

## Safety boundary

No live write paths were used. No KB, CRM, AMO, Tallanto, ASR, Resolve or Analyze data was changed.
