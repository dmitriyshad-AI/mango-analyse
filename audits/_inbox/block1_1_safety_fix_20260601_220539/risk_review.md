# Risk Review

Primary risk addressed:
- Real HARD-P0 refund claim could be followed by presale refund autonomy after the latch was autonomously released. Fixed by preserving `had_hard_p0_claim` and checking it before presale refund de-P0, retrieval augmentation, fallback, and manager-forcing exception.

Regression risks checked:
- HARD-P0 still stays manager-only.
- Benign hypothetical refund is not suppressed.
- Tax fact yield still works when grounded.
- Tax yield does not pass ungrounded numbers/rules or wrong document scope.
- P0 antirepeat does not turn manager-only into autonomous.

Residual risks:
- Full `tests/` cannot be treated as clean in this worktree because unrelated runtime/catalog artifacts are missing.
- The strict tax/matkap checks are regex/anchor-based; they are narrow and covered by negatives, but raw-dialogue semantic regрейд remains necessary.
