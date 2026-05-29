# Semantic review

Verdict: PASS_WITH_NOTES.

Audience: Telegram client draft / manager-reviewed pilot draft.

What passed:
- Prompt/identity probes in v2 now route through a safe terminal response.
- Output-side identity leaks in v2 are blocked even if the draft model produced them.
- Cross-brand keeps precedence over terminal when both could match.
- Ordinary brand confirmation and UNPK/MFTI wording are not caught by `cross_brand`.
- Identity detector no longer uses raw substring matching and does not flag benign text like "интенсивы" or "России".

Non-blocking risks:
- Address/contact terminal templates in live v2 still rely on retrieved facts being present for hard re-verification when they introduce phone/address numbers.
- Full semantic validation remains pending on M1 after several Phase 12 templates are migrated.

Regression created:
- v2 terminal prompt-probe safe answer.
- v2 contact safe answer with retrieved fact.
- identity detector boundary check.
- v2 output identity leak safety-net.

