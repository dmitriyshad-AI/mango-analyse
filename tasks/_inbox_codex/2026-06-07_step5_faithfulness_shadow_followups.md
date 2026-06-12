# Step 5 Faithfulness Shadow Follow-ups

Status: deferred.

Execute only after the Step 5 shadow scaffold is implemented. Do not wait for M1 shadow runs. Do not implement ABSORB from this task.

## Scope

1. Add deterministic boundary NEG tests that do not depend on ABSORB:
   - fabricated product number in a draft is caught by the number gate without the faithfulness critic;
   - recovery candidate with a foreign anchor, such as a deadline or number from a neighboring fact, is rejected by `new_concrete_anchors`;
   - P0 and brand provocation with `TELEGRAM_FAITHFULNESS_SHADOW=1` do not change route or canonical text;
   - `partial_yield` and `composite` work under shadow, with trace not containing `faithfulness_fn_missing`.

2. If not already done in the main Step 5 shadow block:
   - split runner role `bot_faithfulness` from `bot_critic`;
   - split reason `understanding_runtime_error` from `semantic_check_unavailable`.

3. Add `scripts/summarize_faithfulness_shadow.py`:
   - read-only script;
   - input: `dynamic_dialog_transcripts.jsonl`;
   - output: table of shadow verdicts by site: `site / verdicts / available / dialog_id / turn`;
   - include counters by site and verdict;
   - intended user: Claude architect reviewing shadow delta.

## Explicit Non-Scope

- Do not implement `TELEGRAM_FAITHFULNESS_ABSORB`.
- Do not run M1 shadow batches from this task.
- Do not weaken P0, brand, number gate, leak gate, or authoritative output gate.
