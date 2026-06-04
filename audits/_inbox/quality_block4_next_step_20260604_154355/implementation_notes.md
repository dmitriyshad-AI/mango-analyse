# Quality Roadmap Block 4 - Answer Plus Next Step

## Scope

Implemented `TELEGRAM_Q_NEXT_STEP` behind a default-OFF flag.

The block adds one safe next step to verified autonomous answers when the answer does not already contain a step.

## Code Changes

- `src/mango_mvp/channels/dialogue_contract_pipeline.py`
  - Added `QUALITY_NEXT_STEP_ENV`.
  - Added `quality_next_step_enabled`.
  - Added `next_step_applied` and `next_step_text` fields to `DialogueContractPipelineResult`.
  - Added `_quality_next_step_result` as the single owner of next-step insertion.
  - Applied the helper to verified autonomous result paths: estimate success, composite result, schedule/publication answer, direct fact answer, verified fallback, cite-only recover, and normal draft finalization.
- `src/mango_mvp/channels/subscription_llm.py`
  - Exposed next-step metadata in `dialogue_contract_pipeline`.
  - Added `dialogue_contract_next_step_applied` safety flag.
- `tests/test_dialogue_contract_pipeline.py`
  - Added positive and negative tests for the flag.

## Safety Mechanism

The helper:

- skips non-autonomous routes, P0, high-risk/refund/complaint/payment/legal risk;
- skips answers that already contain a next step;
- blocks explicit model-supplied next steps with PII, numbers/dates/prices, or pressure;
- re-verifies the final candidate through `_hard_check`;
- leaves the original text unchanged if semantic faithfulness is unavailable or any finding appears.

No new LLM call was added.
