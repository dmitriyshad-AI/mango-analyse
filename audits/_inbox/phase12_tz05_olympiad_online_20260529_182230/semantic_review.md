# Semantic review

Verdict: PASS_WITH_NOTES.

Audience: Telegram client draft / manager-reviewed pilot draft.

What passed:
- A regular online price question is not answered with olympiad Физтех facts.
- Explicit olympiad-online questions still use the approved olympiad safe templates.
- 10th grade olympiad question is not falsely confirmed; it gets the "9 и 11" handoff.
- 9th/11th grade explicit olympiad-online question can use the approved fact.
- Offline olympiad question is not caught by the online-only template.

Non-blocking risks:
- Full useful answer for regular UNPK online price still depends on Phase 2 KB v6.4 client-safe price facts.
- The regular-online fallback is safe but not maximally useful until those facts are available.

Regression created:
- regular online vs olympiad wrong-scope.
- explicit 10th grade olympiad handoff.
- explicit 11th grade olympiad allowed.
- offline question negative.

