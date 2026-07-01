# Risk Review

## Residual risks

- The report can only prove route/text no-op and model-call delta when both OFF and ON runs are provided.
- `must_handoff` alignment uses runtime route/P0 signals as a proxy; true SemanticFrame accuracy still requires manual `expected_frame` gold.
- Fake bot smoke has no SemanticFrame and correctly reports `needs_review`; it is a format smoke, not a quality measurement.

## Mitigations

- Missing OFF artifacts produce explicit `needs_review` notes instead of a false PASS.
- Missing frames make acceptance fail.
- The report preserves mismatch examples for Claude/manual regrey by raw transcripts.
