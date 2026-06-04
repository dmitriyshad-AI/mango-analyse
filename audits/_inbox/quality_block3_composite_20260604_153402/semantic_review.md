# Semantic Review

Verdict: `PASS_WITH_NOTES`

Audience: customer-facing Telegram bot answers.

## What Passed

- The composite answer is built only from retrieved client-safe facts.
- Missing parts are explicitly deferred to a manager instead of being invented.
- P0/high-risk in any part of a composite question keeps the whole turn on the manager route.
- Cross-brand fact contamination is blocked by the existing verifier before the composite candidate is returned.
- The flag is default OFF, so current production behavior is unchanged until explicit enablement.

## Non-Blocking Risks

- The feature depends on the existing `subquestions` produced by understanding. If understanding fails to split a compound turn, this block will not fire.
- The wording of the missing-tail handoff is safe but still template-like; tone polish belongs to later roadmap blocks.

## Required Regression Coverage

Added tests for:

- flag OFF behavior;
- all-grounded composite answer;
- grounded plus missing/manager tail;
- P0 in one part forcing whole-turn manager route;
- foreign-brand fact blocked;
- no grounded facts preserving no-fabrication fallback;
- all new quality flags ON together with P0, brand, and product-number controls.
