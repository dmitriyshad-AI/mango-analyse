# Risk Review

## P0 risks

No P0 introduced in this block.

The implementation keeps phase 1 constraints:

- no client auto-send;
- no AMO/CRM/Tallanto write;
- no stable_runtime write;
- no ASR/R+A;
- LLM preview disabled unless explicitly enabled by env.

## P1 risks

1. Real AMO/Tallanto/timeline context builders are still incomplete for live pilot quality.
   Mitigation: Stage 6 historical dialog pack before live traffic.

2. High-risk theme list may be too conservative.
   Mitigation: current behavior routes to manager-only, which is safe for phase 1.

3. `codex exec` latency may be high in live pilot.
   Mitigation: debounce, cache outside stable_runtime, timeout fallback.

4. Context quality can be present but semantically wrong if upstream matching is wrong.
   Mitigation: context warnings, manager-only for family/multiple/conflict conditions, manual review.

## P2 risks

1. Manager message may become verbose.
   Mitigation: keep first pilot simple and collect Nastya feedback.

2. Contextual 9 969 analysis still needs a separate sampling stage.
   Mitigation: do not run full 9 969 until 100-300 contextual sample has acceptable context_found_rate.
