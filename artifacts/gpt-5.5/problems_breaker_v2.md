# GPT-5.5 breaker report: TZ 200 D1 v2

Проверял текущие файлы в worktree `/Users/dmitrijfabarisov/Projects/Mango_regex_map_d1_20260731` против `main@ca1c9ce534b9f64b8d0c775df5753694cfbb101f`.

Снимок на момент записи:

- `docs/adr003_understanding_map.yaml`: 832 canonical rows, bucket 2 = 313, bucket 3 = 28, unassigned = 491.
- `marker_helper_actual_snapshot_rows = 130`, `marker_helper_budget_ceiling = 255`; 255 используется как потолок, не знаменатель.
- Безопасные проверки:
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_adr003_regex_understanding_moratorium.py` -> `11 passed`.
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_deal_action_decision.py` -> `9 passed`.
- LLM/live/ASR/AMO/Tallanto не запускал. `src/**` не менял. Ветку и коммиты не трогал.

## findings

### finding-001

severity: P1/blocker

row_id/file:line:

- `adr003:dc8c2f7b8cb9974a141d6bb3` / `src/mango_mvp/channels/subscription_llm_parts/post_layers.py:1103`
- `adr003:38b5313e178afd2becf4b8d2` + `adr003:2342f10824311bc8216ce33f` / `src/mango_mvp/channels/subscription_llm_parts/post_layers.py:1113`
- `adr003:76eeed9cf10f867e7087b4aa` + `adr003:920674cf8a960e1367b26008` / `src/mango_mvp/channels/subscription_llm_parts/post_layers.py:1121`
- `adr003:237a6d26db824c261cafccff` + `adr003:ab573ea451b866d63c5d4dbf` / `src/mango_mvp/channels/subscription_llm_parts/post_layers.py:1128`
- `adr003:82bcbb30118042019d3392cc` / `src/mango_mvp/channels/subscription_llm_parts/post_layers.py:1440`
- `adr003:84932fc088e2ae27b94e4402` / `src/mango_mvp/channels/subscription_llm_parts/post_layers.py:1452`

mechanism: 27 rows in the `deal_action` safety region `post_layers.py:1103-1452` are still unassigned. This block does not only classify intent; it gates manager approval actions, `send_payment_link`, `send_crm_data`, `send_document`, fact questions, payment preconditions, CRM identity/brand safety, and text/action sync. Current map leaves these rows under `unassigned_for_parallel_owner`, even though they protect money, CRM data, documents, contacts, and manager actions.

business-risk: a future ADR-003 cleanup can treat these rows as ordinary understanding/format debt and remove or rewrite them without bucket-2 safety review. Worst cases: unsafe payment-link recommendation, CRM/payment balance answer without strict identity, document/account action recommendation, or lead/contact action not supported by the draft text.

concrete-fix: move the whole `deal_action` safety slice that controls manager approval and high-risk actions into bucket 2, preferably as a dedicated class like `deal_action_safety_floor`, or into `output_claim_pii_floor` if no new class is allowed. At minimum include all duplicated `text_table` + `regex_call` rows for `_DEAL_ACTION_*`, `_DEAL_ACTION_MANAGER_APPROVAL_ACTIONS`, `_deal_action_payment_confirmed`, `_deal_action_crm_identity_ok`, `_deal_action_text_sync`, schedule/material facts, and objection/exit checks; link the class to `tests/test_deal_action_decision.py` and `tests/test_p0_money_promise_output_floor.py`.

### finding-002

severity: P1/blocker

row_id/file:line:

- missing row_id / `tests/test_adr003_regex_understanding_moratorium.py:543`
- missing row_id / `src/mango_mvp/channels/answer_safety_classifier.py:95`
- missing row_id / `src/mango_mvp/channels/answer_safety_classifier.py:96`
- missing row_id / `src/mango_mvp/channels/answer_safety_classifier.py:355`
- missing row_id / `src/mango_mvp/channels/answer_safety_classifier.py:363`

mechanism: `string_contains` inventory is incomplete by construction. The collector records only comparisons where the left side is a string literal and the right-side expression name contains one of `client/draft/lower/message/normalized/query/question/text/utterance/value` (`tests/test_adr003_regex_understanding_moratorium.py:543-550`). A safe AST scan found 154 literal-left `in/not in` comparisons outside the 167 snapshot `string_contains` rows. Examples are P0/safety membership checks such as `"refund" in haystack_codes`, `"refund" not in current_codes`, and priority checks over `current_present`.

business-risk: the map can claim `string_contains` coverage while high-risk P0/money/safety branch literals have no row_id and cannot be assigned to bucket 2. This is a measurement bug: green tests and `832` rows do not prove that all text/safety string gates are mapped.

concrete-fix: either explicitly narrow the term `string_contains` in the YAML/test to "literal substring checks against text-like variables only" and add an audited exclusion count, or extend the collector with a safety allowlist for `codes`, `flags`, `risk`, `actions`, `status`, `categories`, `slots`, `result.safety_flags`, etc. If the collector is extended, update the canon only through an explicit owner/architect decision because it changes the 832 denominator.

### finding-003

severity: P2

row_id/file:line:

- `adr003:bfe3696743f043371df433eb` / `docs/adr003_understanding_map.yaml:1488`
- source: `src/mango_mvp/channels/subscription_llm_parts/post_layers.py:596`
- live use: `src/mango_mvp/channels/subscription_llm_parts/post_layers.py:3299` and `src/mango_mvp/channels/subscription_llm_parts/post_layers.py:3302`

mechanism: `OUTPUT_SANITIZER_RAW_DETAIL_HANDOFF_RE` is classified as bucket 3 `output_format_hygiene`, but the code uses it to detect raw handoff detail and replace question-like/long child-detail text with `SAFE_FALLBACK_DRAFT_TEXT`. That is a safety fallback against raw unsupported handoff/detail leaking into the client draft, not pure formatting.

business-risk: because bucket 3 is allowed to be treated as format/hygiene, this protection can be removed or weakened during cleanup as "cosmetic". That can expose raw unverified detail or an unsafe handoff sentence to the client-facing draft.

concrete-fix: move this row to bucket 2 under `output_claim_pii_floor` or a dedicated `raw_detail_handoff_safety_floor`. Add/point to a regression that mutates a raw-detail handoff with a child/question detail and proves the sanitizer returns the safe fallback.

### finding-004

severity: P2

row_id/file:line:

- `tests/test_adr003_regex_understanding_moratorium.py:817`
- `tests/test_adr003_regex_understanding_moratorium.py:828`

mechanism: bucket-2 reproduction is too weak. The map test only checks that each class' `evidence_tests` paths exist, not that they exercise the mapped row/class. The dedicated P0/money guard checks only two targets: `PAYMENT_CONFIRMATION_RE` and `_ROUTE_REFUND_RE`. Current bucket 2 has 313 rows, so a large number of rows can be mislabeled as safety floor or left unprotected while tests stay green.

business-risk: `formal_pass` can hide false bucket-2 membership and missed safety rows. This is why finding-001 passes all current tests even though the `deal_action` safety slice is unassigned.

concrete-fix: require row-level or class-level reproduction metadata for every bucket-2 class. At minimum, add a test that each bucket-2 class has a concrete regression test name/scenario, and add explicit target checks for high-risk families: P0 recall, payment/refund, brand, PII/client data, fact grounding, manager action/deal action, and output downgrade/fallback.

## checked but not blocking

- Канон 832 and uniqueness: current snapshot has 832 rows and 832 unique `row_id`.
- Row-id line shift: current test excludes `lineno/col_offset` from identity and checks a synthetic line shift. I did not find a current blocker here.
- Machine counts in YAML match the current JSON under the current broad `text_table_rule`.
- I did not confirm a clear false "understanding" row inside bucket 2 after checking the suspicious `policy_routing.py` rows; the confirmed issue is the opposite: safety/action rows left unassigned.

## residual risks

- I did not run broad test suite, live bot, LLM, ASR, AMO/CRM/Tallanto, or heavy batch scripts by instruction.
- The term `text_table` is still semantically broad: the 154 rows include container-like tables, scalar strings, `re.compile(...)` assignments, expressions, and numeric thresholds. Counts are machine-consistent, but the report should not present them as "154 actual tables" without that caveat.

## closure/recheck 2026-07-31

Повторно проверил текущую карту после исправлений Дмитрия, без расширения периметра.

Команда:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_adr003_regex_understanding_moratorium.py` -> `12 passed`.

Текущие пересчитанные counts:

- snapshot rows: `832`, unique `row_id`: `832`, snapshot sha256: `c6a94c894a23d4a73a36d070bdec96ceb178efafd61473e50bfbdc062af10cb2`.
- node kinds: `marker_helper_call=130`, `regex_call=381`, `string_contains=167`, `text_table=154`.
- classification: bucket 2 = `336`, bucket 3 = `27`, unassigned = `469`.
- literal-left membership: total `321` = canonical text scope `167` + audited outside-canonical exclusion `154`; denominator remains `832`.
- unknown YAML row ids: `0`; duplicate YAML row ids: `0`.
- required row fields present for all mapped rows: `symbol`, `display_symbol`, `content_hash_kind`, `content_sha256`.
- all bucket-2 classes have exact `evidence_cases`; file and function definitions are checked by the target test.

Closure by prior finding:

- finding-001: closed in the revised D1 boundary. The safety subset of `deal_action` is now bucket 2 as `deal_action_safety_floor` (`24` rows), including manager approval, payment, CRM data, documents, text/action sync, amount/schedule/material checks. `_DEAL_ACTION_FACT_QUESTION_RE` remains unassigned by design as intent/fact-question routing for the bucket-1 owner; I do not keep it as a D1 blocker.
- finding-002: closed as a measurement bug. The map and test now explicitly separate `321` literal-left membership checks into `167` canonical `string_contains` rows and `154` audited exclusions without changing the `832` canon.
- finding-003: closed. `OUTPUT_SANITIZER_RAW_DETAIL_HANDOFF_RE` is now bucket 2 under `output_claim_pii_floor`.
- finding-004: closed for D1 formal map controls. The target test now requires exact `evidence_cases`, checks that the referenced test functions exist, validates `display_symbol` and content hash against the snapshot, and pins reviewed safety families including `OUTPUT_SANITIZER_RAW_DETAIL_HANDOFF_RE`, `_DEAL_ACTION_*`, P0/money/brand targets.

New scoped verdict: no new blocker found in buckets 2/3 after this recheck.

Residual risk after closure: I did not run the full referenced evidence test suite, only the requested target test; live/LLM/ASR/AMO/Tallanto were not run. Evidence cases are validated by existence and exact function name in the target test, not executed one by one in this recheck.
