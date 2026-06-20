# Risk Review

## Safety Boundaries

- No live AMO/CRM/Tallanto writes.
- No ASR.
- No Resolve+Analyze.
- No stable_runtime writes.
- No source DB mutation.
- No destructive git commands.

## Main Technical Risks

- Existing customer_id generation remains legacy-compatible; Work A adds mapping rather than changing `stable_customer_id()` globally.
- Cross-run resolver reads existing native store identity links only when applying to the local timeline DB.
- Family phone with multiple Tallanto students remains split and conflict-marked; calls are duplicated per split customer with ambiguous status instead of being silently assigned to one child.
- Split mapping allows one old phone-level id to map to many new ids only when `mapping_kind="split"`.

## Residual Risks

- Real snapshot distribution may expose additional source labels not covered by fixtures.
- Existing scrub policy is key-based and does not prove semantic PII removal from every `record` value.
- Work B-F still require separate staged design/review.
