# ADR-003 Package 2 Inventory

counted at HEAD `f570b7bf985f1f97402188a4c888c690a393ab66`

Graphify map: rebuilt on `f570b7bf985f1f97402188a4c888c690a393ab66`; used only as navigation. Counts below are from `tests/fixtures/adr003_direct_path_text_patterns_snapshot.json` on the same HEAD.

## Scope

This is a data-only inventory for the next "minus lasagna" package. No code behavior changes are included here.

Classification:

- `understanding`: regex/marker code trying to infer client meaning, intent, product, slots, scope, or route.
- `floor`: safety/output floor that should stay deterministic until a later explicit P0/fabrication phase.
- `mixed`: contains both understanding and floor/template guards; split before removal.
- `dead`: not observed on current direct-path runtime; delete only after separate usage proof.

Risk class:

- `S`: simple candidate for SemanticFrame/semantic-reading replacement after paired eval.
- `D`: dangerous candidate: money, P0, brand, live availability, unsupported facts, or output sanitizer. Needs stricter floor and ручной регрейд.

## Count Summary

| File | Total snapshot points | marker_helper | string_contains | regex_call | text_table | Current class | Risk |
|---|---:|---:|---:|---:|---:|---|---|
| `src/mango_mvp/channels/answer_quality_rewriter.py` | 133 | 72 | 24 | 33 | 4 | mixed | D |
| `src/mango_mvp/channels/conversation_intent_plan.py` | 55 | 36 | 0 | 18 | 1 | understanding | D |
| `src/mango_mvp/channels/subscription_llm_parts/policy_routing.py` | 276 | 56 | 81 | 121 | 18 | mixed | D |
| `src/mango_mvp/channels/subscription_llm_parts/post_layers.py` | 208 | 38 | 29 | 114 | 27 | mixed | D |
| `src/mango_mvp/channels/semantic_roles.py` | 35 | 18 | 0 | 3 | 14 | understanding/floor | D |
| `src/mango_mvp/channels/new_lead_funnel.py` | 39 | 28 | 1 | 8 | 2 | understanding | S |
| `src/mango_mvp/channels/dialogue_memory.py` | 83 | 19 | 16 | 34 | 14 | mixed | D |

Raw snapshot total across the seven package-2 files: `829`.

## Proposed Slices

### P2-A: Conversation Intent Plan

Files:

- `src/mango_mvp/channels/conversation_intent_plan.py`
- dependent reads in `src/mango_mvp/channels/subscription_llm_parts/policy_routing.py`
- metadata reads in `src/mango_mvp/channels/subscription_llm_parts/direct_path.py`

Hot points counted:

- `_asks_live_availability`: 7 snapshot points.
- `_required_fact_keys`: 9 marker calls.
- `_primary_intent`: 2 points.
- `_requested_slots`: 3 marker calls.
- `_product_focus`: 5 marker calls.
- `_asks_price_fix`: 4 marker calls.
- `_has_price_objection_signal`: 6 points.
- `_has_exit_signal`: 4 regex calls.

Class: `understanding`, with dangerous live-status/money subcases.

Replacement:

- `semantic_frame.requested_action` for `check_availability`, `enroll`, `answer_question`, `send_payment_link`, `refund_or_cancel`.
- `semantic_frame.requested_product` for brand, subject, grade, format, venue, program kind.
- `semantic_frame.payment_readiness` for payment/price/ready-to-pay distinctions.
- `semantic_frame.risk_class` + existing floor for P0/money disputes.

NEG cases to add before removal:

- "Есть ли физика онлайн для 9 класса?" -> `answer_question`, not live availability.
- "Есть места 6-17 июля?" -> `check_availability`, manager.
- "Место занятий где?" -> address/venue, not seats.
- "Можете зафиксировать цену?" -> price terms, not enrollment.
- "Если оплатил и доступа нет" -> payment/support, manager.

Recommended next action: start with shadow agreement report. Do not remove `_asks_live_availability` until false-negative seats cases from E3 are covered.

### P2-B: Policy Routing Terminal Templates

File: `src/mango_mvp/channels/subscription_llm_parts/policy_routing.py`

Hot points counted:

- `_terminal_safe_template`: 58 points.
- `_soften_current_price_deadline_text`: 36 points.
- `_informational_fact_matches_question`: 16 points.
- `_migrated_rule_intent_from_dialogue_contract`: 14 regex calls.
- `_draft_addresses_question`: 12 marker calls.
- `_prefer_format_facts`: 8 marker calls.
- `find_redundant_questions_for_known_context`: 7 regex calls.
- `_presale_refund_policy_template`: 5 points.

Class: `mixed`.

Keep as floor for now:

- P0/refund/legal/payment templates.
- Brand/cross-brand safe templates.
- Unsupported fact and live-status manager check.
- Output text sanitizers that prevent client harm.

Replace candidates:

- `_draft_addresses_question` -> `semantic_frame.intent=address` or `requested_action=answer_question` + exact address fact.
- `_prefer_format_facts` -> `semantic_frame.requested_product.format` + selected fact scope.
- `_informational_fact_matches_question` -> `context_used`/selected fact ids + `semantic_frame.answerability`.
- redundant known-data reask -> semantic reading slot provenance + dialogue memory state.

NEG cases:

- Foton address question with Foton exact fact -> answer, no manager.
- UNPK address question with Foton adjacent fact -> manager/brand-safe, no cross-brand.
- Live seats question with stable course fact only -> manager.
- Price deadline question with stale fact -> manager, no invented deadline.
- Tax/refund/document question -> existing floor unchanged.

Risk: `D`, because template routing can either wrongly self-answer or expose unsupported facts.

### P2-C: Post Layers Output Guards

File: `src/mango_mvp/channels/subscription_llm_parts/post_layers.py`

Hot points counted:

- module-level regex/text tables: 91 points.
- `_direct_path_p0_text`: 9 string contains.
- `_humanity_weekend_schedule_no_format_lock_answer`: 8 points.
- `_humanity_discount_percent_answer`: 8 points.
- `_humanity_unpk_weekend_address_answer`: 7 points.
- `_format_choice_is_disjunctive_question`: 6 points.
- `_asks_money_price_question`: 5 points.
- `_humanity_generic_fact_answer_blocked`: 4 points.

Class: `mixed`.

Keep as floor for now:

- `_direct_path_p0_text` and P0 safe text choices.
- output sanitizer, internal-token removal, unsupported promises.
- semantic output verifier and authoritative output gate.
- money/price calculation hard blocks unless exact fact support exists.

Replace candidates:

- tone/close/reask semantics -> `semantic_frame.intent`, `deal_stage`, `requested_action`, `answerability`.
- format disjunction -> `requested_product.format` + explicit client wording.
- generic fact answer blocked -> exact fact support + `answerability=answer_self`.

NEG cases:

- "Спасибо, всё понятно" -> close only when no pending manager/action.
- "Онлайн или очно?" -> format choice, not off-topic, no invented comparison.
- "Сколько процентов скидка?" -> no calculation unless exact fact.
- "По выходным есть занятия?" -> answer only with exact weekend fact.
- "Нужна справка" after lessons context -> clarify, not document-sale assumption.

Risk: `D`, because this layer is the final client-output floor.

### P2-D: Answer Quality Rewriter

File: `src/mango_mvp/channels/answer_quality_rewriter.py`

Hot points counted:

- `_answers_direct_question`: 26 points.
- `_wrong_scope_fact_selected_finding`: 12 points.
- `_deterministic_rewrite`: 12 points.
- `_clean_fact_text`: 8 regex calls.
- `_fact_matches_known_selection`: 7 points.
- `_best_camp_fact`: 5 marker calls.
- `_brand_safe_fact_texts`: 4 marker calls.

Class: `mixed`.

Keep as floor:

- fact cleaning and brand-safe filtering.
- wrong-scope fact detection until exact fact provenance replaces it.
- unsupported/foreign-brand blocks.

Replace candidates:

- `_answers_direct_question` -> `semantic_frame.intent` + `context_used`/selected exact fact ids.
- camp/product scope checks -> `requested_product.program_kind` + selected fact scope.
- direct-question answer coverage -> coverage plan / answerability.

NEG cases:

- Address exact fact in answer should not be `wrong_intent_fact`.
- Camp/LVSH fact in explicit camp context is allowed.
- Camp fact in regular course question remains blocked.
- Foreign-brand fact in answer remains blocked.
- Direct question about format must be covered by format fact, not price fact.

Risk: `D`. This file caused previous false demotions; remove only with paired output-health gate.

### P2-E: Semantic Roles

File: `src/mango_mvp/channels/semantic_roles.py`

Hot points counted:

- module-level role tables: 14 text tables.
- `_explicit_multi_format_request`: 5 points.
- `_recording_followup_from_context`: 4 marker calls.
- `is_negated_refund_topic`: 3 points.
- `_is_tax_deduction_return_question`: 3 marker calls.
- `_enrollment_vs_recording`: 2 marker calls.
- `_refund_frame`: 2 marker calls.

Class: `understanding/floor`.

Replace candidates:

- enrollment vs recording -> `requested_action` + `intent`.
- multi-format request -> `requested_product.format`.
- tax/refund distinction -> `risk_class`, `requested_action`, `payment_readiness`, with deterministic P0 floor retained.

Keep as floor:

- refund/legal/tax ambiguity until the payment/refund split is accepted on real transcripts.
- any child/privacy/document PII guard.

NEG cases:

- "Можно запись урока?" -> recording/materials, not enrollment.
- "Запишите на курс" -> enrollment, not recording.
- "Возврат НДФЛ" -> tax deduction, not course refund.
- "Верните оплату" -> refund P0.

Risk: `D` because refund/tax ambiguity can route money questions incorrectly.

### P2-F: New Lead Funnel Extractors

File: `src/mango_mvp/channels/new_lead_funnel.py`

Hot points counted:

- `extract_format`: 9 points.
- `detect_product_scope`: 6 marker calls.
- `extract_goal`: 5 marker calls.
- `extract_city_location`: 5 marker calls.
- `extract_product`: 4 marker calls.
- `extract_grade`: 3 points.
- `extract_camp_direction`: 2 marker calls.

Class: `understanding`, not dead. It is imported by `dialogue_memory.py` and `conversation_intent_plan.py`.

Replacement:

- `semantic_reading` slot candidates for grade, subject, format.
- `semantic_frame.requested_product.program_kind` for camp/course/intensive.
- Slot provenance must remain `semantic_reading_llm` or weaker, not `client_confirmed`, unless the client text explicitly says it.

NEG cases:

- "1го класса" -> grade 1 if client-authored, not price digit.
- "8 класс, 9 000 ₽" -> grade 8, amount not grade 9.
- "курс по ИИ" -> subject/program kind, not off-topic.
- "в классе" as classroom/location should not force school grade.
- "или онлайн или очно" -> multi-format, not a single confirmed format.

Risk: `S` for non-money slots; `D` if slot influences price or route.

### P2-G: Dialogue Memory

File: `src/mango_mvp/channels/dialogue_memory.py`

Hot points counted:

- module-level regex/tables: 41 points.
- `_risk_flags_from_safety`: 13 string contains.
- `_is_current_terms_question`: 9 marker calls.
- `_answer_closes_question`: 7 points.
- `_memory_llm_slot_supported_by_latest_client`: 4 points.
- `_normalize_format`: 2 marker calls.
- `_parse_recent_messages`: 2 string contains.
- `_summary_has_unsupported_number`: 1 string contains.

Class: `mixed`.

Keep as floor:

- `_risk_flags_from_safety` for memory latches from safety flags.
- `_summary_has_unsupported_number`.
- JSON parsing/normalization infrastructure.

Replace candidates:

- `_is_current_terms_question` -> `semantic_frame.requested_action` + `payment_readiness`.
- `_answer_closes_question` -> `semantic_frame.intent`/`answerability` plus post-answer route.
- slot extraction patterns -> semantic reading slot provenance.
- format normalization -> semantic reading normalized slot with explicit evidence.

NEG cases:

- A manager-only refund answer must keep P0 latch.
- A safe price answer should close price question only if answer contains exact current price fact.
- A slot inferred from model must not become client-confirmed.
- Dialogue summary must not retain unsupported numbers/dates.
- "место занятий" should not become live seats memory.

Risk: `D`, because memory can silently affect future turns.

## Package 2 Suggested Order

1. `new_lead_funnel` slot extractors that are already shadowed by semantic reading (`S` subset only).
2. `conversation_intent_plan` non-money/non-P0 requested-action slices, starting with format/address/product-existence distinctions.
3. `dialogue_memory` slot provenance and no-reask helpers.
4. `answer_quality_rewriter` wrong-scope/direct-answer checks, guarded by inline text-health gate.
5. `policy_routing` terminal template understanding slices.
6. `post_layers` tone/relevance slices.
7. Money/refund/P0 floors last, after separate acceptance.

## Acceptance Template For Each Slice

Before removal:

- frozen snapshot diff shows exactly which marker/regex points are removed;
- plus-frame and minus-regex happen in the same commit;
- paired eval on the same machine;
- route/text regressions checked by inline text-health gate;
- `too_confident=0`;
- P0/floor/brand/fabrication regressions = 0;
- all affected NEG cases are in a stable set;
- `CHANNEL_REGEX_BUDGET` / `CHANNEL_MARKER_HELPER_BUDGET` changes in the same commit if budgets are touched.

Stop conditions:

- any manager-only/P0 route becomes self-answer unexpectedly;
- any unsupported price/date/count appears in customer text;
- any cross-brand text appears;
- semantic reader lacks evidence for a removed deterministic check;
- judge disagreement cannot be resolved from raw transcript.

