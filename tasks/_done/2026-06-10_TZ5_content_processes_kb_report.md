# TZ5 content/processes/kb report

Date: 2026-06-10

## Scope

Executed `2026-06-10_TZ5_content_processes_kb.md` in the requested order: E -> A -> C -> B -> D.

Default KB snapshot was not repointed. `DEFAULT_KB_SNAPSHOT_PATH` still points to `kb_release_20260608_v6_6_staging/kb_release_v3_snapshot.json`.

## Commits

- E: `e466c74d` — removed unused `subscription_llm.py` helpers.
- A: `74ecd6fd` — updated client-safe process literals and added regression coverage.
- C: `31b04cfd` — built KB v6.7 staging process facts.
- B: `8f546182` — rendered terminal templates from KB behind `TELEGRAM_TEMPLATE_FROM_KB`.
- D: `383ea96e` — added flagged night-hours manager note behind `TELEGRAM_NIGHT_HOURS_NOTE`.
- Semantic follow-up: `9af0946d` — fixed v6.7 LVSH pricing punctuation found during semantic review.

## Part E

Removed dead helpers from `src/mango_mvp/channels/subscription_llm.py`.

Checks:

- Targeted: `tests/test_subscription_llm_draft_provider.py -k 'direct_path or presale or pilot_gold_v1'` -> 49 passed.
- Full: `pytest tests` -> 2856 passed, 2 skipped, 1 warning.

## Part A

Updated literals for:

- LVSH sold out/waitlist/city-school alternative for both brands.
- No online-shift mention.
- Semester/year advantage as already reflected in price, no percentage calculation.
- Alfa-Bank acquiring for business-card payment.
- SohoLMS platform wording.

NEG:

- `test_tz5_client_safe_literals_do_not_regress_process_decisions`

Checks:

- Targeted: 35 passed.
- Full: `pytest tests` -> 2857 passed, 2 skipped, 1 warning.

## Part C

Built v6.7 staging only through `scripts/build_kb_release_v6_1_team_answers.py`.

Release paths:

- `product_data/knowledge_base/kb_release_20260610_v6_7_staging`
- `product_data/knowledge_base/kb_release_20260610_v6_7_staging_bot_pack`
- `product_data/knowledge_base/kb_release_20260610_v6_7_staging_employee_pack`
- `product_data/knowledge_base/kb_release_20260610_v6_7_staging_handoff_for_claude_and_team`
- Snapshot: `product_data/knowledge_base/kb_release_20260610_v6_7_staging/kb_release_v3_snapshot.json`

Build result:

- facts_total: 1043
- client_allowed_facts: 669
- quality_passed: true
- semantic_pass: true
- blocking_findings: 0

Content summary:

- Added client-safe process facts: 60 total, 30 per brand.
- Added internal process facts: 2 total, 1 per brand, for the 4+ illness online-switch exception.
- Added Foton LVSH availability fact.
- Updated platform facts to SohoLMS for both brands.
- Updated payment/discount process facts per P1-P5.
- Generated `D1_audit_backlog/DIFF_kb_v6_7_client_texts.md`: 61 added client-safe texts, 14 changed, 0 removed.

NEG:

- `tests/test_kb_v67_staging_content.py`

Checks:

- Builder: quality_passed true, semantic_pass true.
- `scripts/run_kb_semantic_review.py --release-dir product_data/knowledge_base/kb_release_20260610_v6_7_staging` -> semantic_pass true, blocking 0.
- Targeted: `tests/test_kb_v67_staging_content.py` -> 3 passed.
- Full: `pytest tests` -> 2860 passed, 2 skipped, 1 warning.

Semantic follow-up:

- Found and fixed a double-period typo in LVSH pricing client text (`городская очная школа..`).
- Rebuilt v6.7 through the same builder.
- Re-ran semantic review: semantic_pass true, blocking 0.
- Grep checks: no `городская очная школа..`, no new `МТС Линк`/`Webinar` in v6.7 client-safe registry, no online-shift mention.
- Final full: `pytest tests` -> 2866 passed, 2 skipped, 1 warning.

## Part B

Added `TELEGRAM_TEMPLATE_FROM_KB`, default OFF, not included in pilot profile.

Rendered terminal address/contact templates from the active snapshot when enabled:

- Foton Moscow address.
- UNPK Moscow address.
- UNPK all addresses.
- Foton contacts.
- UNPK contacts.

Missing or foreign-brand fact falls back to a neutral non-numeric manager-safe text. LVSH date templates were intentionally not rendered in this pass because the v6.7 LVSH date facts are not client-safe.

NEG:

- OFF keeps literal template.
- ON renders address/contact values from v6.7 snapshot.
- Missing/foreign fact returns neutral fallback.

Checks:

- Targeted: `tests/test_subscription_llm_draft_provider.py -k template_from_kb` -> 3 passed.
- Full: `pytest tests` -> 2863 passed, 2 skipped, 1 warning.

## Part D

Added `TELEGRAM_NIGHT_HOURS_NOTE`, default OFF, not included in pilot profile.

If final outgoing text contains a manager/sотрудник contact promise and current Moscow time is outside 10:00-18:00, appends exactly once:

`Сейчас нерабочее время — менеджер ответит ежедневно с 10:00 до 18:00 по Москве.`

The hook is placed after `apply_authoritative_output_gate`, so it covers model text and deterministic P0 manager text without weakening P0/brand/number gates.

NEG:

- OFF leaves text unchanged.
- Daytime leaves text unchanged.
- Nighttime appends exactly once.
- P0 manager text also receives the note.

Checks:

- Targeted: `tests/test_subscription_llm_draft_provider.py -k 'night_hours_note or template_from_kb'` -> 6 passed.
- Full: `pytest tests` -> 2866 passed, 2 skipped, 1 warning.

## Remaining risks

- v6.7 default snapshot intentionally not switched. Needs separate gate with renderer enablement after Claude regreyde.
- `semantic_review.json` produced by the builder is formally green; one generated copy records the handoff path in its `release_dir` field, while `v6_1_build_result.json` carries the canonical release/snapshot path.
