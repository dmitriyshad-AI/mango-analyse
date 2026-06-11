# TZ r4 owner decisions report — 2026-06-11

## Scope

- Task: `tasks/_inbox_codex/2026-06-11_TZ_r4_owner_decisions.md`.
- Source release: `kb_release_20260610_v6_7_staging_r3`.
- New release: `product_data/knowledge_base/kb_release_20260611_v6_7_staging_r4`.
- Builder: `scripts/build_kb_release_v6_1_team_answers.py`.
- Source mutation path: `manual_decision_fact_overrides` in `kb_release_20260610_v6_7_staging_r3_sources/release_manifest.yaml`.
- Default snapshot: not switched to r4.

## Implementation Notes

- Added manifest-driven `remove_from_release` support in the KB builder, with regression test, because r4 requires deleting stale source facts while keeping source YAML read-only.
- Preserved both active UNPK Moscow addresses after Dmitry correction: `Сретенка, 20` and `Верхняя Красносельская, 30`.
- Foton duration wording intentionally excludes `2 раза в неделю`, because the r3 Foton source did not ground that number in the same-fact quality check. UNPK keeps it, grounded by existing online-format facts.
- Foton Max channel was taken from the approved contact-page exception: `https://cdpofoton.ru/contacts/`.
- One invalid smoke attempt exists at `runs/20260611_r4_smoke18`: it had `turns=0` due to empty temp `CODEX_HOME` auth and is not used for quality assessment.

## Build And Reviews

- Build r4: `quality_passed=true`, `semantic_pass=true`, blocking findings `0`.
- External semantic review:
  - Path: `product_data/knowledge_base/kb_release_20260611_v6_7_staging_r4_semantic_review_external`.
  - Result: `semantic_pass=true`, findings `0`.
- Manual grep review:
  - Platform transition hits: `8`; stale `online_platform.name = SohoLMS only` hits: `0`.
  - Matkap new family-age wording hits: `2`; old `младшему / на которого оформлен / до 18 лет` hits: `0`.
  - `total_alumni_confirmation` / `2 026 учеников` hits: `0`; `100 000 учеников` preserved.
  - Foton online-summer-school leakage hits: `0`; UNPK online-summer alternative hits: `2`.
  - UNPK address hits: `Сретенка, 20` = `19`, `Верхняя Красносельская, 30` = `15`.
  - Foton Max hits: `1`.

## Tests

- Targeted builder tests: `5 passed`.
- Full pytest: `3010 passed, 2 skipped, 1 warning`.
- Smoke18 final:
  - Run: `runs/20260611_r4_smoke18_final`.
  - Snapshot: `product_data/knowledge_base/kb_release_20260611_v6_7_staging_r4/kb_release_v3_snapshot.json`.
  - Profile: `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`.
  - Judge: `v9.1`.
  - Result: `PASS=11`, `PASS_WITH_NOTES=7`, `FAIL=0`, hard-gate failures `0`, turns `43`.
  - Validity: `config_validity.invalid=false`; retriever/render/rubric/memory profile flags effective; `bot_direct_draft=41`, `bot_retriever=41`, `bot_semantic_output_verifier=51`, `bot_faithfulness=0`.

## Added Fact Keys

- `foton` `r4_owner_2026_06_11.foton.address_contact_source_policy_internal`
- `foton` `r4_owner_2026_06_11.foton.center_visit_by_appointment`
- `foton` `r4_owner_2026_06_11.foton.city_summer_school_availability`
- `foton` `r4_owner_2026_06_11.foton.internal_payment_split`
- `foton` `r4_owner_2026_06_11.foton.lesson_duration_by_format`
- `foton` `r4_owner_2026_06_11.foton.lvsh_availability_alternatives`
- `foton` `r4_owner_2026_06_11.foton.matkap_family_age_rule`
- `foton` `r4_owner_2026_06_11.foton.midyear_new_student_curator_support`
- `foton` `r4_owner_2026_06_11.foton.online_platform_transition`
- `foton` `r4_owner_2026_06_11.foton.small_groups_personal_attention`
- `foton` `r4_owner_2026_06_11.foton.subject_lineup_2026_27`
- `unpk` `r4_owner_2026_06_11.unpk.address_contact_source_policy_internal`
- `unpk` `r4_owner_2026_06_11.unpk.center_visit_by_appointment`
- `unpk` `r4_owner_2026_06_11.unpk.internal_payment_split`
- `unpk` `r4_owner_2026_06_11.unpk.lesson_duration_by_format`
- `unpk` `r4_owner_2026_06_11.unpk.lobnya_zhukovsky_manager_only_internal`
- `unpk` `r4_owner_2026_06_11.unpk.lvsh_availability_alternatives`
- `unpk` `r4_owner_2026_06_11.unpk.main_office_dolgoprudny`
- `unpk` `r4_owner_2026_06_11.unpk.matkap_family_age_rule`
- `unpk` `r4_owner_2026_06_11.unpk.midyear_new_student_curator_support`
- `unpk` `r4_owner_2026_06_11.unpk.moscow_regular_address`
- `unpk` `r4_owner_2026_06_11.unpk.online_platform_transition`
- `unpk` `r4_owner_2026_06_11.unpk.refer_friend_online_cashback`
- `unpk` `r4_owner_2026_06_11.unpk.small_groups_personal_attention`
- `unpk` `r4_owner_2026_06_11.unpk.subject_lineup_2026_27`

## Deleted Fact Keys

- `foton` `academic_year_2026_27.daily_hours`
- `foton` `academic_year_2026_27.weekly_lessons`
- `foton` `lvsh_mendeleevo_2026.availability_2026.client_safe_text`
- `foton` `matkap.child_age.sertificate_owner_min`
- `foton` `matkap.child_age.student_max`
- `foton` `matkap.client_safe_text.when_age_over_18`
- `foton` `matkap.client_safe_text.when_asked`
- `foton` `online_platform.name`
- `unpk` `academic_year_2026_27.daily_hours`
- `unpk` `academic_year_2026_27.weekly_lessons`
- `unpk` `lvsh_mendeleevo_2026.availability_2026.client_safe_text`
- `unpk` `matkap.child_age.sertificate_owner_min`
- `unpk` `matkap.child_age.student_max`
- `unpk` `matkap.client_safe_text.when_age_over_18`
- `unpk` `matkap.client_safe_text.when_asked`
- `unpk` `online_platform.name`
- `unpk` `results_social_proof.total_alumni_confirmation`

## Changed Client-Safe Fact Keys

- `foton` `processes_2026_06_10.foton.after_payment_access`
- `foton` `processes_2026_06_10.foton.camp_enrollment`
- `foton` `processes_2026_06_10.foton.contact_channels`
- `foton` `processes_2026_06_10.foton.installment_options`
- `foton` `processes_2026_06_10.foton.online_access_platform`
- `foton` `processes_2026_06_10.foton.personal_cabinet`
- `unpk` `processes_2026_06_10.unpk.after_payment_access`
- `unpk` `processes_2026_06_10.unpk.camp_enrollment`
- `unpk` `processes_2026_06_10.unpk.online_access_platform`
- `unpk` `processes_2026_06_10.unpk.payment_schedules_without_bank_installment`
- `unpk` `processes_2026_06_10.unpk.personal_cabinet`
- `unpk` `tg_unpk_verified_2026_05_21.client_facts.matkap_federal_only.client_safe_text`

## Residual Notes

- Smoke18 is a formal smoke plus judge pass; raw semantic review of transcripts remains with the architect.
- r4 should not be made default until architect regrade passes.
