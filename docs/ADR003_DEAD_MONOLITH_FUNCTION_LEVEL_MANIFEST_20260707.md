# ADR003 Dead Monolith Function-Level Manifest, 2026-07-07

Scope: audit while M1 measures package `a246ece2`. This file is a function-level manifest, not a physical deletion patch.

Decision: no runtime code is removed in this pass. The audit found code that is dead for the current `pilot_gold_v1` direct path, but most of it is still imported by rollback, fake-provider, dialogue-contract, or legacy tests. Physical deletion before the M1 result would weaken rollback safety and would not satisfy the required byte-for-byte smoke against `a246ece2`.

## Direct-Path Boundary

The live direct path enters `SubscriptionLlmDraftProvider.build_draft()`, calls `_build_direct_path_draft()`, and returns before the legacy monolith tail. Functions defined below the early return are not automatically dead: several direct-path runners and verifier helpers live below that point and are still called by `_build_direct_path_draft()`.

## Function-Level Status

| Area | Files / functions | Direct-path status | Deletion status | Reason |
| --- | --- | --- | --- | --- |
| Provider legacy tail | `subscription_llm_parts/provider.py` legacy branch after the direct-path return | Dead for current direct path | Removed in refactoring Package 2 | Owner retired `TELEGRAM_DIRECT_PATH=0` and the DCP fallback; the direct-path branch is now unconditional. |
| Answer-quality rewrite | Former `answer_quality_rewriter.py`, `apply_answer_quality_rewriter()`, `_answer_quality_llm_rewrite_runner()` | Dead for current direct path | Removed in refactoring Package 2 | Its imports, flags, simulator hooks and dead-path tests were removed with the module. |
| Humanity guards layer | Former `post_layers.apply_humanity_guards()` | Dead for current direct path | Removed in refactoring Package 3/4 | Live meta/repeat helpers were moved to `output_verification_floor.py`; the wrapper and compatibility module were removed. |
| Humanity X2 rewrite | Former `apply_humanity_x2_rewriter()`, `_humanity_x2_rewrite_runner()`, `DRAFT_X2_*` flags | Dead for current direct path | Removed in refactoring Package 3 | Owner retired the dialogue-contract fallback. |
| Phase-2 tone | Former `apply_phase2_tone_layer()`, `TELEGRAM_PH2_TONE` | Dead for current direct path | Removed in refactoring Package 3 | No live direct-path callers remained. |
| Semantic diagnosis guard | `apply_semantic_diagnosis_guard()`, `TELEGRAM_SEMANTIC_DIAGNOSIS_GUARD*` | Dead for current direct path | Not removed | Do not confuse with the live `SEMANTIC_OUTPUT_VERIFIER` path. |
| Rules-engine dispatcher | Former `_apply_migrated_rules_engine()` through `apply_dialogue_contract_v2_template_dispatcher()` | Dead for current direct path | Removed in refactoring Package 3 | Dispatcher, rules-engine module, registry and isolated tests were removed together. |
| Redundant-question guard | `find_redundant_questions_for_known_context()`, `apply_known_context_redundant_question_guard()` | Live | Do not touch | `reask_read/final_text` calls this path under the current profile. |
| Conversation-intent/live-status traces | `apply_conversation_intent_plan_guard()`, `apply_live_status_read_plan_trace()` | Live | Do not touch | Called from `_build_direct_path_draft()`. |
| Direct-path helpers below early return | direct retriever/draft/verifier/frame helpers | Live | Do not touch | Physically below legacy return, but logically used by direct path. |
| Dialogue contract pipeline | Former `dialogue_contract_pipeline.py`, `Q_*`, `A_ESTIMATE*`, `A_TRAVEL*`, `STEP4_*` | Not current direct path | Removed in refactoring Package 4 | Owner retired the fallback; live output-floor functions and their regression tests remain under the physical owner. |

## Dead Flags For Future Cleanup

Candidates for a later physical cleanup after the M1 result and rollback decision:

- `ANSWER_QUALITY_*`
- `DRAFT_X2_*`
- `PH2_TONE`, `PH2_OBJECTION`, `PH2_ANXIETY`
- `HUMANITY_BLOCK_A_ROUTE_FIX`, `ANTIREPEAT_STRICT`
- `SEMANTIC_DIAGNOSIS_GUARD*`
- `A_SELLING_*`, `A_COVERAGE`

Do not remove in the same commit as behavior changes. Each removal needs import audit, focused tests, and a deterministic byte-for-byte smoke.

## Smoke Status

The requested 10-dialog byte-for-byte smoke is not run here because no physical deletion was made. Running a model-backed dynamic smoke would be a new measurement, not a cleanup proof, and must not overwrite the active M1 package/output. For a future deletion pass, first build a deterministic stub replay that exercises the same direct-path inputs against `a246ece2` and the deletion candidate.

## Residual Risk

Keeping the code is low behavior risk and preserves rollback. The remaining cost is maintenance noise. The safe next action is to wait for the M1 result, then delete only the areas Fable marks nonessential for rollback.
