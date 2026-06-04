# Semantic Review

Verdict: `PASS_WITH_NOTES`

Audience: Telegram lead/client answer path.

## What Passed

- The new behavior is disabled by default behind `TELEGRAM_Q_PARTIAL_YIELD`.
- Partial-yield only uses retrieved client-safe facts and explicitly names the missing part as something the manager will verify.
- P0/refund/high-risk paths are not overridden by partial-yield.
- Foreign-brand fact candidates are blocked by full-text verification before reaching the client.
- Travel/route estimate requires an uncertainty marker and still passes the number gate.

## Blocking Issues

No blocking semantic issue found in unit-level samples.

## Non-Blocking Risks

- The partial-yield handoff wording is intentionally conservative and may still sound mechanical. It should be judged on raw transcripts before enabling the flag.
- The travel/logistics detector is deterministic and narrow. It prevents known price/product leakage, but M1/regression data should confirm it does not miss common phrasing.
- This review did not inspect live Telegram transcripts; it only reviewed code paths and deterministic tests.

## Missing Checks

- No live/M1 regrade of customer-facing outputs for this flag yet.
- No broad tone score for the new partial-yield phrasing yet.
- No production-like run with all roadmap flags enabled beyond the invariant unit suite.

## Regression / Gate Coverage Added

- Travel estimate reaches the estimate composer before manager-only when the planner mislabels route/logistics as pricing.
- Product price with travel words remains blocked.
- Flag OFF preserves old cite-only behavior.
- Grounded subquestion + missing subquestion yields partial answer.
- P0 and refund policy do not get replaced by neighboring facts.
- Foreign-brand fact candidate is blocked.
- No grounded fact keeps the existing no-fabrication fallback.
- Invariant suite with quality flags ON keeps P0, brand, and product-price safety.

## Recommended Next Action

Run Claude/M1 semantic regrade on Block 1 raw outputs before enabling `TELEGRAM_Q_PARTIAL_YIELD`.
