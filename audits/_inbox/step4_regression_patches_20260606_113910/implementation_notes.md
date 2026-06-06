# Step 4 regression patches

Scope:
- Fail-soft handling for Codex `understand()` subprocess timeout.
- Output sanitizer backstop for internal `client-safe` jargon.
- Refund latch calibration after benign presale/hypothetical refund questions.
- Canonical first-block cross-brand text when brand separation guard catches a mixed-brand draft.

Implementation:
- `AnswerContract.runtime_error` carries provider runtime failures from understanding into the pipeline.
- Runtime errors return `draft_for_manager` with `fallback_reason=semantic_check_unavailable`, `reason_class=provider_runtime`, and do not call draft generation.
- Benign refund latch no longer promotes a harmless next turn to hard P0 unless `had_hard_p0_claim` is true or the current text itself contains hard P0.
- Rules-engine no-fact enrollment payloads no longer contain `client-safe` wording.
- `strip_internal_service_markers()` removes `client-safe` jargon if it appears in final text.
- Brand separation guard now uses the canonical cross-brand separation phrase instead of generic fallback when the draft mixes brands.

Out of scope:
- No changes to Step 4 keep-answer logic.
- No changes to number grounding, KB, live bot, or main dirty worktree.

