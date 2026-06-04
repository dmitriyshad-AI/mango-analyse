# Semantic Review

Verdict: `PASS_WITH_NOTES`

Audience: Telegram lead/client answer path.

## What Passed

- The new behavior is behind `TELEGRAM_Q_THREAD_MEMORY`, default OFF.
- The implementation does not ask the model again; it only enriches deterministic retrieval from existing memory.
- Explicit current-turn format/grade overrides stale memory.
- P0 is not masked by previous topic memory.
- Explicit subject switch is not glued to the old topic.
- Camp/product-family memory does not collapse into regular course.
- Active brand remains the channel brand even if memory says another brand.

## Blocking Issues

No blocking semantic issue found in deterministic tests.

## Non-Blocking Risks

- Existing `topic_focus` behavior remains always available because it was already accepted in the base. This block only gates the new known-slots enrichment.
- Slot source handling allows `memory_llm`; this is useful for the intended memory chain, but should be watched in raw multi-turn regressions.
- The implementation improves retrieval on known-slot ellipses, but it does not itself improve stale memory cleanup; explicit topic-change guards cover the tested cases.

## Missing Checks

- No M1/raw transcript regrade yet for real multi-turn ellipses.
- No broad dynamic run with several roadmap flags beyond unit invariant.

## Regression / Gate Coverage Added

- Flag OFF preserves current known-slots-only behavior.
- Known slots recover `а очно?` into the current subject/class under flag ON.
- `bot_inferred` known slots alone are not trusted.
- P0/refund claim stays manager-only.
- Explicit subject switch does not glue old subject.
- Camp-family stays camp.
- Channel active brand wins over memory brand.
- All-flags invariant keeps P0 and product-price safety.

## Recommended Next Action

Let Claude/M1 regregrade Block 2 on multi-turn raw transcripts before enabling `TELEGRAM_Q_THREAD_MEMORY`.
