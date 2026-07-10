# Risk Review

Primary risk if misused: treating partial KB support as permission to answer the client.

Mitigations in this change:

- `active_behavior_allowed` is always `False`.
- Acceptance is always `active_readiness=no_go`.
- The report separates `proven_parts`, `missing_slots`, and `uncovered_categories`.
- Product existence only covers existence/class/program direction axes; it does not cover price, dates, location, boarding, payment access, or live availability.
- Danger-adjacent rows remain explicitly excluded.

Residual risks:

- KB support matching is still a diagnostic heuristic, not a production verifier.
- The report does not generate client-safe text and does not prove a renderer would be safe.
- Exact policy for partial answers remains a product/semantic decision, not resolved here.
