# Semantic review

Verdict: PASS_WITH_NOTES.

Audience: Telegram client draft / manager-reviewed pilot draft.

What passed:
- Direct promises of ЕГЭ/ОГЭ score or result now produce the approved refusal text in v2.
- The refusal template wins even when the original draft contained an unsupported numeric promise.
- Statistics such as "выше среднего на 25 баллов" remain allowed when grounded in retrieved facts.
- A price question about ЕГЭ preparation does not trigger the result guarantee template.

Non-blocking risks:
- The regex is still rule-based. Edge phrasing outside the existing guarantee wording may require Phase 1a intent work.
- Full behavior must still be checked on M1 semantic regreyde.

Regression created:
- result guarantee over unsupported promise.
- "точно сдаст на 90+" branch.
- grounded score statistic is not blocked.
- price question is not result guarantee.

