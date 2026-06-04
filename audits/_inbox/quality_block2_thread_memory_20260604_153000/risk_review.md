# Risk Review

## Safety Invariants

- P0: `_augment_contract_with_memory_topic` returns unchanged when `contract.is_p0` is true; regression verifies refund/payment complaint is not masked.
- Brand: active brand is not read from memory and remains the `run_pipeline(active_brand=...)` argument.
- Topic switch: explicit subject/product markers prevent deterministic memory glue.
- Product family: `camp` aliases keep ЛВШ/camp keys separate from regular course keys.
- Fabrication: memory enrichment only changes retrieval keys; final answer still passes existing draft verification and gates.

## Main Residual Risks

1. If memory contains a plausible but stale `topic_focus`, old baseline behavior may still reuse it. This block does not rewrite the memory lifecycle.
2. `memory_llm` is an allowed slot source for known-slot enrichment. This matches the accepted memory chain, but raw review should watch for hallucinated slots.
3. The deterministic alias scorer can miss uncommon fact-key naming; that causes safe over-handoff, not unsafe answer.

## Attempted Break Cases

- P0 after previous course topic.
- Explicit new subject with old subject in memory.
- Camp follow-up with regular-course neighbor fact.
- Memory active_brand conflict.
- Product-price estimate with all quality flags ON.

## Risk Classification

- Missing v6.4 artifact in this worktree without `KB_RELEASE_V3_DIR`: `infrastructure_bug`.
- Any future stale-memory raw failure: likely `object_bug` in memory lifecycle, not in this narrow flag.
