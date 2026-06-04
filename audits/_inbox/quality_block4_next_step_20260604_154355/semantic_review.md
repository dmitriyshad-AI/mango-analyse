# Semantic Review

Verdict: `PASS_WITH_NOTES`

Audience: customer-facing Telegram bot answers.

## What Passed

- The added next step is a generic action request, not a new business fact.
- The helper rejects explicit next steps containing PII requests, numbers/dates/prices, or pressure.
- P0, refund, complaint, payment dispute, legal, and manager-only routes are not changed.
- The whole final candidate is re-verified through existing output checks before replacement.
- The feature is default OFF.

## Non-Blocking Risks

- The deterministic step wording is intentionally conservative and may feel generic. Tone/playbook refinement belongs to Block 6.
- The helper does not make a business decision about enrollment; it only points to the next conversational action.

## Regression Coverage

Added tests for:

- flag OFF old behavior;
- safe next step on autonomous fact answer;
- explicit safe `subquestions[].next_step`;
- no duplicate if the answer already contains a step;
- P0 and manager-only not touched;
- pressure/concrete explicit steps not exposed;
- all current quality flags ON together with P0 and brand controls.
