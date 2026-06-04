# Quality Roadmap Block 3 - Composite Questions

## Scope

Implemented `TELEGRAM_Q_COMPOSITE` behind a default-OFF flag.

The block handles multi-part questions before draft generation:

- grounded subquestions are answered from retrieved client-safe facts;
- missing or manager-only non-P0 parts are deferred honestly in the same answer;
- P0/high-risk in any part blocks the whole turn and keeps `manager_only`;
- the generated candidate still passes the existing output verifier and semantic faithfulness check.

## Code Changes

- `src/mango_mvp/channels/dialogue_contract_pipeline.py`
  - Added `QUALITY_COMPOSITE_ENV`.
  - Added `quality_composite_enabled`.
  - Added composite metadata fields to `DialogueContractPipelineResult`.
  - Added `_quality_composite_result_before_draft` and helper functions.
  - Inserted the composite gate after estimate handling and before draft/slot fallback.
- `src/mango_mvp/channels/subscription_llm.py`
  - Exposed composite metadata in `dialogue_contract_pipeline` metadata.
  - Added `dialogue_contract_composite_applied` safety flag.
- `tests/test_dialogue_contract_pipeline.py`
  - Added composite positive and negative regression tests.

## Mechanism

The composite helper reuses existing coverage and verification primitives:

- `_partial_yield_findings_and_missing`;
- `_coverage_cite_only_answer_from_findings`;
- `_partial_yield_full_check`;
- `_cite_only_recover_blocked`;
- `_avoid_repeating_text`.

No new LLM call was added.
