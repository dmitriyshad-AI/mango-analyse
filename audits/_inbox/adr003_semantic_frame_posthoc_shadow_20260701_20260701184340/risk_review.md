# Risk Review

## Controlled risks

- Default OFF; not added to pilot profile.
- Post-hoc frame cannot modify route/text by construction: tests compare ON with OFF after the normal direct-path chain.
- Provider error is fail-soft and only writes metadata status.
- Report does not hide route/text diffs; expected frame-only call delta is not enough for pass.

## Residual risks

- Full M1 route/text no-op cannot be proven by two independent LLM draft runs because the draft itself is nondeterministic. Need paired transcript/enrichment runner or equivalent measurement.
- Post-hoc frame is a new model call per turn when enabled; keep OFF outside controlled eval.
- True SemanticFrame quality still needs manual expected_frame gold.
