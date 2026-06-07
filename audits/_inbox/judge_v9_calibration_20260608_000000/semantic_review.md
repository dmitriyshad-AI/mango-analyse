# Semantic review

Scope: measurement layer only. This block does not alter customer-facing bot responses.

Semantic checks performed:

- v9 distinguishes final autonomous answers from manager drafts when evaluating semantic verifier findings.
- `derived_claim_draft` is treated as review-priority soft signal, not a hard gate.
- Hard classes remain hard in any route: fabricated product numbers/prices/dates/schedule/address, brand leakage, P0 mishandling, promises, internal leakage, and AI/vendor disclosure.
- Broader fact-claim matching is judge-only, so it can reduce false fabrication labels without weakening runtime safety.

Semantic status: formal measurement implementation is complete; live semantic calibration of v9 labels remains pending until Dmitry/Claude run the requested re-judge passes.
