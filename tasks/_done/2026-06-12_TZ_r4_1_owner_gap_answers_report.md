# r4.1 owner gap answers report — 2026-06-12

## Scope

- ТЗ: `tasks/_inbox_codex/2026-06-12_TZ_r4_1_owner_gap_answers.md`.
- Source overlay: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1_sources`.
- Release: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1`.
- Handoff: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1_handoff_for_claude_and_team`.
- Bot pack: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1_bot_pack`.
- Default snapshot was not switched; current default remains `kb_release_20260611_v6_7_staging_r4`.

## Build

- Command: `scripts/build_kb_release_v6_1_team_answers.py` with explicit r4.1 source/release paths.
- Result: `quality_passed=true`, `semantic_pass=true`.
- Facts total: `1075`.
- Client-safe facts: `699`.
- Approval queue items: `1053`.
- Control numbers missing: `[]`.

## Formal tests

- Targeted r4.1 tests: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_kb_r4_1_owner_gap_answers.py`
  - Result: `5 passed`.
- Full pytest: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`
  - Result: `3029 passed, 2 skipped, 1 warning`.

## Semantic review

- Command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_kb_semantic_review.py --release-dir product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1_handoff_for_claude_and_team`
- Result: `semantic_pass=true`, `findings_total=0`, `blocking_findings=0`.

## Manual grep / semantic checks

- `37 500` and `37500` are absent from r4.1 client-safe exports.
- `Премиум 10` and `Премиум+` are absent from r4.1 client-safe exports.
- `пробного периода` and `пробной недели` are absent from r4.1 client-safe exports.
- Foton city summer school: `49 000 ₽` is explicitly `База + половина факультативного блока`; full optional block is `59 000 ₽`.
- Trial/acquaintance wording says one-off free visit in another group only `по согласованию с менеджером`; it does not claim a permanent free trial format.
- Internal facts for matkap refund, nonpayment freeze and mock exams have empty `client_safe_text` and manager-only route.
- Updated owner section `Продукт М9/М11` is included:
  - old `new_online_oge_ege_math_product_internal` is absent;
  - old `modular_courses_m9_m11` / `team_answers.q11.*` discontinued facts are absent;
  - M9 and M11 are client-safe Foton facts with 6 modules, 35 weeks, exam goal and full tariff grid;
  - M9/M11 client-safe tariff texts contain only period prices: `18 900`, `47 250`, `59 900`, `94 500`; `8 790` / `8790` is absent;
  - tariff manager text keeps internal notes: no monthly price in client text and no teacher names to clients.

## Smoke18

- Scenario set: `product_data/telegram_dynamic_test_sets/pilot_smoke18_2026-06-10.jsonl`.
- Snapshot: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`.
- Out dir: `runs/20260612_r4_1_smoke18_m9m11_final`.
- Profile: `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`.
- Judge: `v9.1`.
- Result: `18 dialogs`, `43 turns`, `PASS=12`, `PASS_WITH_NOTES=6`, `FAIL=0`, `hard_gate_failures=0`.
- LLM calls: `bot_direct_draft=41`, `bot_retriever=41`, `bot_semantic_output_verifier=48`, `bot_semantic_output_regen=5`, `bot_faithfulness=0`, `total=196`.
- Key flags: profile/render/rubric/retriever/memory_provenance effective `true`.
- Config validity: `invalid=false`; `hard_gate_failure_dialogs=[]`.
- Note: first attempt was invalid because copied `CODEX_HOME` had stale `service_tier=default`; it was moved to `runs/20260612_r4_1_smoke18_invalid_service_tier`. A later smoke started before the final M9/M11 tariff-text correction was interrupted and moved to `runs/20260612_r4_1_smoke18_m9m11_update_invalid_interrupted`. The valid run above used a temp config with `service_tier=fast`.

## Fact-key diff r4 -> r4.1

### Added

- `foton | r4_1_owner_2026_06_12.foton.acquaintance_mechanics`
- `foton | r4_1_owner_2026_06_12.foton.author_program_not_single_textbook`
- `foton | r4_1_owner_2026_06_12.foton.cancelled_lvsh_shift_august_15_23`
- `foton | r4_1_owner_2026_06_12.foton.city_summer_school_tariffs`
- `foton | r4_1_owner_2026_06_12.foton.funds_transfer_and_makeup`
- `foton | r4_1_owner_2026_06_12.foton.individual_lessons_request`
- `foton | r4_1_owner_2026_06_12.foton.m11_online_math_ege_product`
- `foton | r4_1_owner_2026_06_12.foton.m11_online_math_ege_tariffs`
- `foton | r4_1_owner_2026_06_12.foton.m9_online_math_oge_product`
- `foton | r4_1_owner_2026_06_12.foton.m9_online_math_oge_tariffs`
- `foton | r4_1_owner_2026_06_12.foton.matkap_refund_to_sfr_internal`
- `foton | r4_1_owner_2026_06_12.foton.midyear_entry_payment_and_records`
- `foton | r4_1_owner_2026_06_12.foton.no_boarding_except_lvsh`
- `foton | r4_1_owner_2026_06_12.foton.nonpayment_second_semester_freeze_internal`
- `foton | r4_1_owner_2026_06_12.foton.oge_ege_mock_exams_internal`
- `foton | r4_1_owner_2026_06_12.foton.organization_payment_invoice`
- `foton | r4_1_owner_2026_06_12.foton.regular_group_size`
- `foton | r4_1_owner_2026_06_12.foton.schedule_timezone_msk`
- `foton | r4_1_owner_2026_06_12.foton.university_target_and_open_days`
- `unpk | r4_1_owner_2026_06_12.unpk.acquaintance_mechanics`
- `unpk | r4_1_owner_2026_06_12.unpk.author_program_not_single_textbook`
- `unpk | r4_1_owner_2026_06_12.unpk.cancelled_lvsh_shift_august_15_23`
- `unpk | r4_1_owner_2026_06_12.unpk.city_summer_school_tariffs`
- `unpk | r4_1_owner_2026_06_12.unpk.funds_transfer_and_makeup`
- `unpk | r4_1_owner_2026_06_12.unpk.individual_lessons_request`
- `unpk | r4_1_owner_2026_06_12.unpk.matkap_refund_to_sfr_internal`
- `unpk | r4_1_owner_2026_06_12.unpk.midyear_entry_payment_and_records`
- `unpk | r4_1_owner_2026_06_12.unpk.no_boarding_except_lvsh`
- `unpk | r4_1_owner_2026_06_12.unpk.nonpayment_second_semester_freeze_internal`
- `unpk | r4_1_owner_2026_06_12.unpk.oge_ege_mock_exams_internal`
- `unpk | r4_1_owner_2026_06_12.unpk.organization_payment_invoice`
- `unpk | r4_1_owner_2026_06_12.unpk.regular_group_size`
- `unpk | r4_1_owner_2026_06_12.unpk.schedule_timezone_msk`
- `unpk | r4_1_owner_2026_06_12.unpk.university_target_and_open_days`

### Removed

- `foton | individual_lessons_foton.format`
- `foton | individual_lessons_foton.note_internal`
- `foton | individual_lessons_foton.prices.lesson_45min`
- `foton | individual_lessons_foton.prices.package_5_sessions`
- `foton | individual_lessons_foton.prices.session_90min`
- `foton | kb_v6_6_client_safe_facts_2026_06_08.trial_other_group_free_trial.client_safe_text`
- `foton | modular_courses_m9_m11.bot_behavior`
- `foton | modular_courses_m9_m11.note`
- `foton | team_answers.q11.modular_courses_m9_m11.discontinued`
- `foton | team_answers.q11.modular_courses_m9_m11.old_price_range.max`
- `foton | team_answers.q11.modular_courses_m9_m11.old_price_range.min`
- `unpk | tg_unpk_verified_2026_05_21.client_facts.city_school_formats.client_safe_text`

### Changed

- `foton | ls_city_2026_foton.moscow_foton.prices.base`
- `foton | ls_city_2026_foton.moscow_foton.prices.plus_half`
- `foton | ls_city_2026_foton.moscow_foton.prices.plus_full`

### Builder compatibility fix

- `scripts/build_kb_release_v3_from_claude_handoff.py`: manifest `remove_from_release` now also removes hardcoded manual specs. This is required for the owner decision that reopens M9/M11: otherwise old `team_answers.q11.modular_courses_m9_m11.*` discontinued facts are appended after the manifest delete pass.

## Status

- `formal_pass`: yes.
- `semantic_pass`: yes for KB release script and manual grep checks.
- Architect regрейд is still required before switching default snapshot to r4.1.
