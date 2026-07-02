# Risk Review

Primary risk if misused: treating "partial KB support" as permission to answer the client.

Mitigations:

- `active_readiness=no_go`.
- `active_behavior_allowed=false` for all partial cases.
- `generated_text_exported=false`.
- `manager_only` remains blocked.
- live availability, payment access, enroll, refund, handoff and P0-like rows are excluded.
- No raw candidate text is generated.

Residual risks:

- The support scan is heuristic and useful for triage, not a production verifier.
- Partial answer policy is not defined here.
- Exact wording and semantic safety of any future partial answer still require a separate owner-approved text policy and semantic review.
