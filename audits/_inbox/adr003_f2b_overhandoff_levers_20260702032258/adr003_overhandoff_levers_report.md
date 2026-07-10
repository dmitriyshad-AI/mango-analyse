# ADR-003 Ф2b Over-Handoff Levers

- Status: `pass`
- Source rev: `36ea110`
- Total turns: `241`
- Gold compared rows: `79`
- Gold safe/self rows: `32`
- Safe already self: `21`
- Safe handoff total: `11`
- Safe manager_only: `8`
- Safe draft_for_manager: `3`
- Existence/format blocked before fact verification: `1`
- Danger-adjacent blocked: `2`
- Frame too cautious safe/self rows: `14`
- Frame too cautious existence/format rows: `8`
- Harmless context/status candidates: `0`
- Draft candidates for possible future route-only active: `0`
- Manager-only candidates needing policy decision: `0`

## Группы

- `existence_format_needs_fact_verification_blocked`: `1`
- `danger_adjacent_blocked`: `2`
- `harmless_context_ack_status_candidate`: `0`
- `safe_reference_without_exact_facts`: `1`
- `low_confidence_or_missing_facts_blocked`: `0`
- `p0_or_money_or_operational_blocked`: `7`
- `unclear_review_required`: `0`
- `already_self`: `21`

## Доминанта по requested_action

- `answer_question`: `22`
- `check_availability`: `7`
- `handoff_manager`: `2`
- `enroll`: `1`

## Frame Too Cautious по requested_action

- `check_availability`: `7`
- `answer_question`: `4`
- `handoff_manager`: `2`
- `enroll`: `1`

## Кандидаты


## Acceptance Notes

- No harmless context/status candidates found; this may still be a valid negative result.
- Frame too-cautious existence/format rows need fact verification, not route-only demotion.
- Some safe-label handoffs are danger-adjacent and must stay out of clean active candidates.
- No draft_for_manager candidates are ready even for a future route-only active discussion.
