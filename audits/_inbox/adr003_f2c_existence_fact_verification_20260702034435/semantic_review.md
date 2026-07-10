# Semantic Review

## Verdict

PASS_WITH_NOTES.

## What Passed

- The report answers the business question better than F2b alone: route-only autonomy is still not enough; the real lever is fact-backed existence/format proof.
- The result is cautious: it does not claim the bot can be enabled now.
- Brand and freshness are considered in the diagnostic matcher.
- P0/money/danger-adjacent rows are separated from autonomy candidates.
- The report identifies only 2 current handoff rows with plausible exact KB evidence, not a broad active rollout opportunity.

## Blocking Issues

No blocker for the report-only phase.

Blocking issue for any future active phase: current runtime telemetry does not yet provide a strict product-existence proof field. `bot_confirmed_facts`, `missing_facts`, and `freshness.ok` are useful signals, but they are not a machine contract that a specific requested product/format/class exists.

## Non-Blocking Risks

- Offline string/axis matching can still misclassify edge cases. It is intentionally diagnostic only.
- Some facts prove nearby business context but not live availability. A future active layer must keep "exists/format" separate from "free places/enroll".
- The 2 current candidates are `manager_only`, so a future active step would require an explicit policy decision, not just route-only demotion.

## Required Regression/Gate

For any future active work, add a runtime gate that refuses self-answer unless all are true:

- product-existence proof is exact;
- all supporting facts are fresh and client-safe;
- active brand matches;
- no P0/money/legal/complaint/payment/refund signal;
- live availability / booking / enrollment remains excluded;
- no unsupported prices, dates, places, or guarantees.

## Recommended Next Action

Build a shadow-only `product_existence_axes_catalog` / proof extractor modeled after the existing price axes catalog, then rerun F2c with runtime proof fields instead of offline matching.
