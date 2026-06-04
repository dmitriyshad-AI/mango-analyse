# Risk Review

## Safety Invariants

- Brand split: protected by full-text `verify_output` on partial-yield candidate.
- P0/refund/high-risk: protected by `_cite_only_recover_blocked` and existing early P0/refund branches.
- Fabrication: candidate is built only from retrieved facts and then checked by output verification and faithfulness.
- Estimate numbers: travel estimate still goes through `_hard_check` and the free number gate.
- Downgrade-only principle: partial-yield does not invent new facts; it only replaces a handoff with a verified grounded answer plus a manager handoff for the missing part.

## Main Residual Risks

1. Deterministic travel detector can be under-inclusive or over-inclusive on unseen phrasing.
2. Partial-yield wording may still not be humane enough for final client experience.
3. If semantic faithfulness is unavailable, partial-yield does not fire; this is safer but may preserve over-handoff.
4. Broad interaction with future roadmap flags is covered only by an invariant unit test, not by a full dynamic run.

## Risk Classification

- Expected failures from missing KB release artifact in this worktree: `infrastructure_bug`.
- Potential wording/tone weakness in generated partial-yield text: `object_bug`, non-blocking while flag is OFF.
- Future judge disagreement on whether partial-yield is helpful enough: possible `measurement_bug`, to be resolved by raw transcript review.
