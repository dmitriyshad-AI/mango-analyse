# Semantic review

Verdict: PASS_WITH_NOTES.

Audience: Telegram client draft / manager-reviewed pilot draft.

What passed:
- Direct promises of admission now produce the approved refusal + statistic text in v2.
- The approved `97%` statistic inside the safe template is not blocked by the v2 re-verifier.
- A question about admission statistics can still pass when grounded by retrieved facts.
- Enrollment wording "как поступить на курс" is not treated as a university admission guarantee.

Non-blocking risks:
- The template uses a fixed statistic. It remains approved by current source-of-truth, but should be rechecked if admission outcome data changes.
- Full semantic validation remains pending on M1 after Phase 12 batch.

Regression created:
- admission guarantee positive.
- exact "точно поступит" branch.
- admission statistic negative.
- enrollment question negative.

