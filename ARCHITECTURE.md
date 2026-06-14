# Mango Bot Architecture Control Map

1. Telegram/Wappi input is assembled into pilot context, then the subscription draft provider chooses direct-path, deterministic-rule, or manager-only routing.
2. The authoritative structural rules live in `src/mango_mvp/channels/rules_engine.py`; Graphify is only a navigation map, not a source of truth.
3. `rules_engine.apply_rule` first checks cross-brand current-center leakage before dispatching individual migrated rules.
4. Brand isolation depends on active brand plus `_mentions_other_brand`; price, schedule, discount, trial, and camp answers must stay scoped to Foton or UNPK.
5. Real refund/payment dispute paths are P0-like manager-only routes; benign hypothetical refund wording is explicitly carved out by `p0_recall_spec`.
6. `camp_lvsh` and `enrollment_process` contain hard manager handoffs for real refund claims and live availability ambiguity.
7. Direct-path post layers add output guards after drafting: authoritative gate, unsupported-promise guard, PII/name echo guard, semantic verifier, and diagnosis guard.
8. X2/humanity rewrites are post-processing only and must not touch manager-only or P0-locked routes.
9. Structured KB facts are read from release files and must preserve client-safe versus manager-only/internal markings.
10. `MANAGER_ONLY`, `internal_only`, `manager_only_route`, and `forbidden_for_client` are labels for navigation and must not become client-safe facts.
11. Public draft text may use prices, dates, and addresses only after source-file confirmation, never from the graph alone.
12. Negative conclusions from Graphify are invalid when the graph revision differs from HEAD; absence must be checked with raw-file search.
