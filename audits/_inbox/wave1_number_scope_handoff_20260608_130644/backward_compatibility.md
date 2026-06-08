# Backward Compatibility

## Default Behavior

Both new features are OFF by default:

- `TELEGRAM_NUMBER_GATE_SCOPE_AWARE=0`
- `TELEGRAM_VERIFIER_HANDOFF_CLAIMS=0`

With both flags OFF, existing behavior remains:

- The free-number gate uses the flat union of all retrieved fact number surfaces.
- Pure handoff text without extracted factual claim skips the semantic output verifier.

## Compatibility Tests

- `test_wave1_number_scope_aware_flag_off_keeps_flat_fact_surfaces`
- `test_wave1_verifier_handoff_claims_off_keeps_current_pure_handoff_skip`
- Full `pytest tests`: `2817 passed, 2 skipped`.

## Git Boundary

Work is on branch `codex/wave1-number-scope-handoff`.
No merge into `main` was performed.

