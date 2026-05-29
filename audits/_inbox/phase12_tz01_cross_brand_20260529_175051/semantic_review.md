# Semantic review

Verdict: PASS_WITH_NOTES.

Audience: Telegram client draft / manager-reviewed pilot draft.

What passed:
- Cross-brand provocations in v2 now produce a neutral safe text about separate organizations.
- The safe text does not include conditions, prices, discounts, installment rules, or facts of the other brand.
- Brand confirmation for the current brand is not swallowed by the cross-brand template.
- `МФТИ` in the UNPK identity context is not treated as another brand.
- Template output is re-verified by the existing v2 hard verifier before continuing the route chain.

Non-blocking risks:
- Only TZ-01 is implemented. Other Phase 12 P0 templates remain pending.
- The dispatcher currently contains one registered template; later templates must preserve the single-winner precedence rule.
- Full behavior still needs M1 semantic regreyde after several templates are migrated; local tests are only formal coverage.

Regression created:
- v2 cross-brand positive.
- v2 cross-brand over terminal-precedence.
- v2 brand-confirmation negative.
- v2 UNPK/MFTI negative.

