# Risk Review

## Residual risks

- The guard freezes structural patterns; it does not semantically classify every existing regex as good or bad.
- A new keyword could still be hidden in an unusual non-uppercase local variable; the snapshot covers the main runtime pattern classes but is not a full taint analyzer.
- Legitimate output scrub changes will now require intentional snapshot refresh and review.

## Mitigations

- The direct-path snapshot covers inline regex calls, marker tables, and text-like string contains checks, not only `re.compile` counts.
- Docs require eval-case + SemanticFrame calibration for new client-meaning failures.
- Full pytest passed with the guard enabled.
