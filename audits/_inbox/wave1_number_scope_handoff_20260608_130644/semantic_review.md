# Semantic Review

Verdict: `PASS_WITH_NOTES`

## Artifact And Audience

Artifact: bot safety and quality code paths for direct-path output verification.
Audience: client-facing Telegram drafts, but this review covers code-level guard behavior only.

## What Passed

- Brand/P0/hard safety layers were not edited.
- B1 keeps product numbers grounded to facts of the matching scope when enabled.
- B1 preserves legacy flat behavior when disabled.
- B1 downgrades a wrong-scope price instead of allowing a plausible but misleading numeric answer.
- B5 keeps canonical pure handoff templates skipped.
- B5 sends a handoff-shaped answer with substantive unsupported claim to the semantic output verifier.
- Existing authoritative gate remains the final routing layer for verifier findings and wrong-scope findings.

## Blocking Issues

None found in code-level review and unit tests.

## Non-blocking Risks

- Scope extraction depends on current `fact_scope_spec` coverage plus format/grade heuristics. The new tests cover the target price scope class, not every product family.
- Arithmetic support was not expanded. The implementation allows normalized same-scope surfaces but does not try to prove broader derived arithmetic beyond existing number-surface machinery.
- B5 increases verifier coverage only when enabled; live model behavior must be measured on M1.

## Missing Checks

- No live simulator run was executed, per TZ.
- No transcript-level semantic regrade was performed in this local block.
- OFF parity is covered by default-off flags and unit tests; no byte-for-byte dynamic replay was run locally.

## Regression Tests Added

- Wrong-scope numeric price.
- Same-scope normalized price.
- New ungrounded product number.
- Direct-path wrong-scope downgrade.
- Pure handoff OFF parity.
- Canonical handoff whitelist.
- Handoff with substantive claim.
- P0 and brand hard controls around the new handoff verifier path.

## Recommended Next Action

Run the M1 acceptance/regression batch with both flags explicitly ON after Дмитрий approves branch review.

