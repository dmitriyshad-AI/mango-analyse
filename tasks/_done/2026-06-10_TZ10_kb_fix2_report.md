# TZ-10 KB fix2 report — 2026-06-10

## Scope

Implemented all 5 TZ-10 items on a new KB release directory:

- source: `product_data/knowledge_base/kb_release_20260610_v6_7_staging_r3_sources/`
- release: `product_data/knowledge_base/kb_release_20260610_v6_7_staging_r3/`
- bot pack: `product_data/knowledge_base/kb_release_20260610_v6_7_staging_r3_bot_pack/`
- employee pack: `product_data/knowledge_base/kb_release_20260610_v6_7_staging_r3_employee_pack/`

Default snapshot was switched to:

`product_data/knowledge_base/kb_release_20260610_v6_7_staging_r3/kb_release_v3_snapshot.json`

## Changed client facts

Removed from client-safe output:

- Foton `locations_foton.free_trial_offline.offer`: old promise of a free offline trial.
- Foton `prices_regular_2026_27.online_1_4_class.*`: old client scope "1-4 class, online".
- Foton `bot_policy.approved_phrases.theme_11_contract.foton`: "Договор пришлёт менеджер в ближайшие дни...".
- UNPK `tg_unpk_verified_2026_05_21.client_facts.electronic_document_flow.client_safe_text`: scan-copy / on-shift contract wording.
- UNPK `tg_unpk_verified_2026_05_21.client_facts.lvsh_contract.client_safe_text`: scans/original camp contract wording.
- UNPK `tg_unpk_verified_2026_05_21.client_facts.summer_online_schools.client_safe_text`: summer online schools mention.
- UNPK `bot_policy.approved_phrases.theme_11_contract.unpk`: "Договор пришлёт менеджер в ближайшие дни...".

Added / replaced:

- Foton contract phrase: договор-оферта arrives with receipt after enrollment; payment means acceptance; ask email.
- UNPK contract phrase: same offer contract wording.
- UNPK `electronic_document_flow`: offer contract; paper contract only by client request or matkapital.
- UNPK `lvsh_contract`: camp contract as offer; paper by request or matkapital.
- Foton free offline trial: explicit negative wording, no promise.
- Foton online prices: client-safe scope is now "3-4 class, online"; prices unchanged.
- Foton lesson load text: online junior scope is now "3-4 class", not "1-4 online".
- UNPK summer online schools: moved to internal note, not client-safe.

## Build and semantic checks

Builder:

`scripts/build_kb_release_v6_1_team_answers.py`

Result:

- `quality_passed=true`
- `semantic_pass=true`
- `facts_total=1045`
- `client_safe_facts=668`

Separate semantic review:

- command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_kb_semantic_review.py --release-dir product_data/knowledge_base/kb_release_20260610_v6_7_staging_r3 --out-dir product_data/knowledge_base/kb_release_20260610_v6_7_staging_r3`
- `semantic_pass=true`
- `blocking_findings=0`
- `findings_total=0`

Manual grep over client-safe facts:

- old contract promise: 0
- scan/original contract wording: 0
- old Foton free offline trial promise: 0
- summer online / online-shift client mention: 0
- online "1-4 class" client scope: 0

Note: old UNPK offline trial text still exists only as `allowed_for_client_answer=false` / `do_not_use` manager-only fact. TZ-10 explicitly targeted the Foton client-safe promise; this was not changed.

## Tests

Targeted:

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_kb_release_v3_import.py tests/test_kb_v67_staging_content.py tests/test_subscription_llm_draft_provider.py -q`

Result: passed.

Full:

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest tests`

Result: `2893 passed, 2 skipped, 1 warning in 68.51s`.

## Files touched

- KB r3 source/release directories under `product_data/knowledge_base/`.
- Default snapshot pointers in `src/`, `scripts/`, `tests/`, and `CLAUDE.md`.
- This report.

## Residual risk

No live bot / AMO / CRM write was run. This is a KB build and default-pointer change only. Semantic review is formal and deterministic; final business acceptance of customer wording remains Claude/Dmitry review over changed client-safe texts.
