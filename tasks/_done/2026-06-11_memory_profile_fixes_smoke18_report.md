# Memory profile fixes smoke18 — 2026-06-11

## Scope

Implemented three smoke18 memory/masking fixes behind independent flags:

1. `TELEGRAM_MEMORY_PROVENANCE_COMPACT`
   - `compact_dialogue_memory_view` preserves `slot_sources`, `client_confirmed_slots`, and `slot_provenance` after compaction.
   - `dialogue_memory_from_mapping` can rebuild `DialogueSlot.source=memory_provenance` and quote from compacted prompt-view data.

2. `TELEGRAM_PII_RELATION_STOPWORDS`
   - PII de-echo no longer treats relation words like `сын` / `дочь` / `ребёнок` / `мальчик` / `девочка` as names.
   - Unmentioned real names remain masked.

3. `TELEGRAM_MEMORY_CHILD_ELLIPSIS`
   - Provenance extractor handles ellipsis like `дочь в 4-м` after another child grade.

Commits:
- `0267e850` — guarded fixes, all flags default OFF.
- `922969b8` — enable the three flags in `pilot_gold_v1`.

## Tests

Targeted tests after enabling profile:
- `6 passed`

Full pytest before profile enable:
- `3004 passed, 5 skipped, 1 warning`

Full pytest after profile enable:
- `3004 passed, 5 skipped, 1 warning`

NEG coverage:
- Explicit `=0` over `pilot_gold_v1` preserves old compact behavior.
- Explicit `=0` over `pilot_gold_v1` preserves old child-ellipsis behavior.
- Explicit `=0` over `pilot_gold_v1` preserves old relation-word masking behavior.
- Profile ON preserves provenance source/quote across build → view → compact → build.
- Profile ON extracts `child_1_grade=7`, `child_2_grade=4` from `сын в 7 классе и дочь в 4-м`.
- Profile ON keeps relation words unmasked while still masking unmentioned `Ирина`.

## Smoke18

- Run directory: `/Users/dmitrijfabarisov/Projects/Mango analyse/runs/20260611_memory_profile_fixes_smoke18`
- Scenario set: `product_data/telegram_dynamic_test_sets/pilot_smoke18_2026-06-10.jsonl`
- Snapshot: `product_data/knowledge_base/kb_release_20260610_v6_7_staging_r3/kb_release_v3_snapshot.json`
- Profile: `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`
- Judge: `v9.1`
- Parallel: `4`
- Temp `CODEX_HOME`: `/private/tmp/mango_codex_home_memory_fixes_20260611_fast`, with `service_tier=fast`.

Result:
- `dialogs=18`
- `turns=44`
- `PASS=9`
- `PASS_WITH_NOTES=9`
- `FAIL=0`
- `hard_gate_failures=0`
- `config_validity.invalid=false`
- `infra_error_dialogs=[]`

LLM calls:
- `memory=0`
- `bot_direct_draft=42`
- `bot_retriever=42`
- `bot_semantic_output_verifier=47`
- `bot_faithfulness=0`

Key controls:
- `pilot_smoke18_13_foton_p0_payment`: PASS, both turns `manager_only`, `preblocked=true`, reason `p0_deferral`.
- `pilot_smoke18_16_foton_two_children_ambiguous`: PASS_WITH_NOTES, `slot_provenance` contains `child_1_grade=7` from `сын в 7 классе` and `child_2_grade=4` from `дочь в 4-м`.
- `pilot_smoke18_17_unpk_lead_pii_no_echo`: PASS_WITH_NOTES, hard gates passed; no hard PII failure.

## Residual Note

While reading smoke18 raw turns, I noticed an adjacent extractor issue outside this mini-TZ: in `pilot_smoke18_16`, the phrase `для обоих` can still become `child_name=Обоих`. I did not broaden scope or patch it here; this should be reviewed separately if the architect considers it material.

## Status

`formal_pass`: yes.

Semantic/regression verdict over smoke transcripts remains with the architect, per task instruction.
