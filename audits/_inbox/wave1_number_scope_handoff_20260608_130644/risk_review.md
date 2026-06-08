# Risk Review

## Safety Impact

The changes are feature-gated and default OFF:

- `TELEGRAM_NUMBER_GATE_SCOPE_AWARE`
- `TELEGRAM_VERIFIER_HANDOFF_CLAIMS`

Hard lower layers were not edited:

- P0 latch and high-risk routing.
- Brand token checks.
- Meta, identity, and PII scans.
- Authoritative output gate action mapping.

## Main Risks

1. Scope false positive: a fact with incomplete scope metadata may be treated as allowed because the helper intentionally does not block facts with no detectable scope.
2. Scope false negative: a real same-scope fact may downgrade if query text contains a conflicting format/grade signal.
3. Verifier load: enabling B5 can increase semantic-verifier calls for handoff-shaped texts with substantive content.
4. Live model variance: unit tests use deterministic verifier stubs and do not measure model availability or false positives.

## Mitigations

- Both changes are behind separate OFF-by-default flags.
- Wrong-scope uses existing `wrong_scope` downgrade action, not a new route policy.
- Canonical handoff whitelist is exact normalized text, not a broad heuristic.
- Added NEG tests around P0, brand, wrong-scope, and ungrounded numbers.

## Residual Risk

Transcript regrade on M1 remains required before enabling either flag in acceptance runs.

