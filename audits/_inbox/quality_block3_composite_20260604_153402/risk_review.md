# Risk Review

## Safety Invariants

- Brand separation: covered by `verify_output` and explicit foreign-brand negative tests.
- P0: covered by `_cite_only_recover_blocked`, `p0_pre_gate`, and tests where P0 appears as one part of the composite turn.
- Fabrication: composite candidates pass `_partial_yield_full_check`, including numeric/product checks and semantic faithfulness.
- Downgrade-only gate: this block does not raise route authority; it only returns `bot_answer_self` when a verified, cite-only composite candidate exists.
- Default behavior: `TELEGRAM_Q_COMPOSITE` is default OFF.

## Residual Risks

- A badly formed contract with an unsafe self subquestion and no P0 marker can still reach the existing verifier. Current mitigation is the existing full output check; no new bypass was added.
- If semantic faithfulness is unavailable while the toggle requires it, the composite branch fails closed and does not answer.
