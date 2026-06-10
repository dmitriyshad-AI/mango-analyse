# CODE INVENTORY — клиентский слой Mango/Foton/UNPK, 2026-06-09

Документ для внешнего архитектурного аудита. Составлен read-only по рабочему дереву `codex/wave6-llm-retriever` / `7389d07a`.

Связанный контекст: `D1_audit_backlog/PROJECT_DEEP_CONTEXT_for_external_audit_2026-06-09.md`.

## Легенда статусов

| Статус | Что означает |
|---|---|
| ПРЯМОЙ ПУТЬ | Исполняется в боевом пилотном пути при `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1` или явном `TELEGRAM_DIRECT_PATH=1`. |
| PIPELINE | Исполняется только в старом dialogue-contract pipeline / legacy-ветке, когда direct path не перехватил ход. |
| ТЕСТЫ/API | Используется тестами, алиасами или публичной совместимостью, но не подтверждён как live-клиентский путь. |
| МЁРТВЫЙ-кандидат | По статическому поиску нет runtime-вызовов в охваченных файлах. Перед удалением нужен отдельный grep по всему repo и тестовый commit. |
| ФЛАГ-OFF | Код есть, но по умолчанию выключен или включается отдельным env-флагом. |
| ИЗМЕРЕНИЕ | Runner/judge/summary/watcher, не клиентский ответ. |

Важно: `pilot_gold_v1` включает прямой путь только через env задачи/ручного запуска. Сам `BUNDLE_INFO.txt` не включает прямой путь автоматически.

## Раздел A. Живое ядро прямого пути

### A1. Верхняя развилка ответа

| Блок | файл:строка | назначение | статус | флаг |
|---|---:|---|---|---|
| `SubscriptionLlmDraftProvider.build_draft` | `src/mango_mvp/channels/subscription_llm.py:3268` | первая развилка: direct path → pipeline → legacy prompt | ПРЯМОЙ ПУТЬ | `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1` / `TELEGRAM_DIRECT_PATH` |
| `_direct_path_enabled` | `subscription_llm.py:2028` | определяет, перехватывает ли ход direct path | ПРЯМОЙ ПУТЬ | `TELEGRAM_DIRECT_PATH_PILOT_CONFIG`, `TELEGRAM_DIRECT_PATH` |
| `_build_direct_path_draft` | `subscription_llm.py:3371` | основной поток: preblock/facts/prompt/model/verifier/gate/metadata | ПРЯМОЙ ПУТЬ | direct path |
| `_direct_path_preblocked_result` | `subscription_llm.py:3056` | до LLM уводит P0/high-risk/force-manager/unknown-brand | ПРЯМОЙ ПУТЬ | direct path; при `TELEGRAM_LLM_RETRIEVE=1` срабатывает до retriever |

### A2. Сбор фактов и контекста

| Блок | файл:строка | назначение | статус | флаг/условие |
|---|---:|---|---|---|
| `build_telegram_pilot_context` | `src/mango_mvp/channels/telegram_pilot_context_builder.py:104` | собирает память, intent-plan, confirmed facts и prompt context до бота | ПРЯМОЙ ПУТЬ+PIPELINE | без флага |
| `build_dialogue_memory` | `dialogue_memory.py:300` | строит `dialogue_memory_view` из истории/слотов/латча | ПРЯМОЙ ПУТЬ | без флага |
| `build_conversation_intent_plan` | `conversation_intent_plan.py:102` | keyword intent, topic, required facts, scope и route hints | ПРЯМОЙ ПУТЬ | без флага |
| `_direct_path_client_safe_snapshot_fact` | `subscription_llm.py:2106` | фильтр snapshot facts: active_brand + client-safe + valid_until | ПРЯМОЙ ПУТЬ | direct path |
| `_direct_path_wide_fact_pack` | `subscription_llm.py:2699` | собирает fact pack из snapshot, LLM/keyword retrieval и legacy fallback | ПРЯМОЙ ПУТЬ | direct path; `TELEGRAM_LLM_RETRIEVE` меняет отбор |
| `_direct_path_selected_categories` | `subscription_llm.py:2243` | keyword-категории вопроса, максимум 2 категории | ПРЯМОЙ ПУТЬ | default path, если LLM retriever off |
| `_direct_path_keyword_fact_pack_from_records` | `subscription_llm.py:2465` | keyword exact/adjacent split по категориям, слотам и score | ПРЯМОЙ ПУТЬ | fallback для Wave 6 |
| `fact_retrieval.select_confirmed_facts` | `fact_retrieval.py:60` | старый confirmed-facts retrieval в context builder | ПРЯМОЙ ПУТЬ+PIPELINE | без флага |
| `fact_scope_spec.detect_fact_scopes` | `fact_scope_spec.py:154` | scope facts для лагерей, скидок, форматов, оплаты и т.д. | ПРЯМОЙ ПУТЬ+PIPELINE | без флага |

Замечание для аудитора: strict brand/client-safe фильтр есть на snapshot path (`_direct_path_client_safe_snapshot_fact`). `legacy_context` fallback в `_direct_path_legacy_context_fact_items` доверяет upstream context и сам бренд/client-safe заново не проверяет.

### A3. Промпт и вызов модели

| Блок | файл:строка | назначение | статус | флаг |
|---|---:|---|---|---|
| `_build_direct_path_prompt` | `subscription_llm.py:2942` | короткая mission-инструкция + facts + memory + known slots + gold examples | ПРЯМОЙ ПУТЬ | direct path |
| `_direct_path_select_gold_real_examples` | `subscription_llm.py:2886` | выбирает real-manager few-shot examples | ПРЯМОЙ ПУТЬ | `TELEGRAM_BOT_GOLD_REAL`; auto при `pilot_gold_v1` |
| `_direct_path_gold_prompt_block` | `subscription_llm.py:2922` | рендерит few-shot pack в prompt | ПРЯМОЙ ПУТЬ | `TELEGRAM_BOT_GOLD_REAL` / pilot config |
| `_direct_path_draft_runner` | `subscription_llm.py:4172` | один LLM-вызов draft-мозга | ПРЯМОЙ ПУТЬ | direct path |
| `_normalize_direct_path_payload` | `subscription_llm.py:4414` | нормализует JSON direct response в `SubscriptionDraftResult` | ПРЯМОЙ ПУТЬ | direct path |
| `_direct_path_metadata` | `subscription_llm.py:3005` | начальная metadata direct path | ПРЯМОЙ ПУТЬ | direct path |
| `_direct_path_finalize_metadata` | `subscription_llm.py:3172` | фиксирует route_before_gate/route_after/downgraded/source | ПРЯМОЙ ПУТЬ | direct path |

### A4. Верификатор и финальный gate

| Блок | файл:строка | назначение | статус | флаг |
|---|---:|---|---|---|
| `apply_semantic_output_verifier` | `subscription_llm.py:7209` | смысловой LLM-verifier финального текста против фактов; advisory/downgrade/regen | ФЛАГ-OFF / жив в pilot stack | `TELEGRAM_SEMANTIC_OUTPUT_VERIFIER` |
| `build_semantic_output_verifier_prompt` | `subscription_llm.py:7119` | prompt verifier-а | ФЛАГ-OFF | `TELEGRAM_SEMANTIC_OUTPUT_VERIFIER` |
| `_semantic_output_verifier_runner` | `subscription_llm.py:3635` | LLM-вызов verifier-а через draft provider | ФЛАГ-OFF | `TELEGRAM_SEMANTIC_OUTPUT_VERIFIER` |
| `_semantic_output_regen_runner` | `subscription_llm.py:3649` | один regen-вызов verifier-а | ФЛАГ-OFF | `TELEGRAM_SEMANTIC_OUTPUT_VERIFIER` |
| `apply_authoritative_output_gate` | `subscription_llm.py:5029` | последнее слово: sanitizers + verify_output + answer_safety + existing guard findings | ПРЯМОЙ ПУТЬ+PIPELINE | всегда вызывается в direct |
| `_authoritative_gate_direct_path_keep_text` | `subscription_llm.py:5144` | для direct мягкие findings могут сохранить текст, но route → draft_for_manager | ПРЯМОЙ ПУТЬ | direct metadata |
| `apply_output_sanitizer` | `subscription_llm.py:5167` | чистит клиентский текст от meta/PII/internal markers; PII de-echo работает даже когда sanitizer flag off | ПРЯМОЙ ПУТЬ+PIPELINE | `TELEGRAM_OUTPUT_SANITIZER` + always PII de-echo |
| `strip_internal_service_markers` | `subscription_llm.py:5955` | режет source/fact/path/service markers | ПРЯМОЙ ПУТЬ+PIPELINE | через sanitizer |
| `verify_dialogue_contract_output` | `dialogue_contract_pipeline.py:4065` | hard output verifier: brand, meta, P0, numbers, scope | SHARED через authoritative gate | часть findings фильтруется, если нет pipeline |

## Раздел B. За флагами: волны и presale-пакет

| Пакет | статус в текущей ревизии `7389d07a` | функции/точки | флаги | комментарий |
|---|---|---|---|---|
| Волна 1: scope-aware number gate | код есть | `dialogue_contract_pipeline.py:41`, `verify_output:4065`, `subscription_llm.py:119` | `TELEGRAM_NUMBER_GATE_SCOPE_AWARE` | принята, должна быть ON в `pilot_gold_v1` задачах; это output-level guard, не direct prompt |
| Волна 1: verifier handoff claims | код есть | `_verifier_handoff_claims_enabled:7344`, `_semantic_verifier_is_whitelisted_pure_handoff:7352` | `TELEGRAM_VERIFIER_HANDOFF_CLAIMS` | закрывает pure-handoff skip-дыру verifier-а |
| Волна 4: graded gate | НЕ в текущем дереве | ветка `codex/wave4-graded-gate`, head `d80f8ed6`; ТЗ `TZ_wave4_graded_gate_2026-06-09.md` | `TELEGRAM_GRADED_GATE` | не исполняется в `7389d07a`; регрейд показал, что корень не в gate |
| Волна 5: anti-promise | НЕ в текущем дереве | ветка `codex/wave5-anti-promise`, head `b203b637`; ТЗ `TZ_wave5_anti_promise_2026-06-09.md` | `TELEGRAM_ANTI_PROMISE` | не исполняется в `7389d07a`; пакет не смержен в main/current branch |
| Волна 6: LLM retriever | код есть в текущей ветке | `build_direct_path_llm_retriever_prompt:2545`, `_direct_path_llm_retrieve_fact_pack:2600`, `_direct_path_llm_retrieve_runner:3659` | `TELEGRAM_LLM_RETRIEVE` | default OFF; кандидат к включению после регрейда |
| Presale safety fixes П1-П4 | НЕ в текущем дереве | ветка `codex/presale-safety-fixes`; ТЗ `TZ_presale_safety_fixes_2026-06-09.md` | `TELEGRAM_PRESALE_SAFETY`, `TELEGRAM_PRESALE_PII_MEMORY`, `TELEGRAM_PRESALE_VERIFIER_FAILSOFT`, `TELEGRAM_PRESALE_META_RU`, `TELEGRAM_PRESALE_SOURCE_ID` | ревью PASS по ТЗ/сырью, но не присутствует в `7389d07a` |
| Semantic verifier | код есть, в pilot stack обычно ON | `apply_semantic_output_verifier:7209` | `TELEGRAM_SEMANTIC_OUTPUT_VERIFIER` | один дополнительный LLM-вызов на ход, кроме skip/fail-soft |
| Gold examples | код есть, auto для pilot config | `_direct_path_gold_real_enabled:2830`, `_direct_path_select_gold_real_examples:2886` | `TELEGRAM_BOT_GOLD_REAL`, `TELEGRAM_BOT_GOLD_REAL_PACK` | `pilot_gold_v1` включает без отдельного env-флага |

## Раздел C. Legacy / «лазанья»

### C1. Dialogue-contract pipeline

`dialogue_contract_pipeline.py` — opt-in модуль за `TELEGRAM_DIALOGUE_CONTRACT_PIPELINE`. При `pilot_gold_v1` direct path возвращается раньше и этот путь не исполняется.

| Блок | файл:строка | назначение | статус | флаг |
|---|---:|---|---|---|
| `dialogue_contract_pipeline_enabled` | `dialogue_contract_pipeline.py:591` | включает pipeline | PIPELINE | `TELEGRAM_DIALOGUE_CONTRACT_PIPELINE` |
| `run_pipeline` | `dialogue_contract_pipeline.py:3005` | understand → retrieve → draft → verify/repair/warmth | PIPELINE | `TELEGRAM_DIALOGUE_CONTRACT_PIPELINE` |
| `build_understanding_prompt` / `parse_contract` | `dialogue_contract_pipeline.py:741/830` | contract/planner для pipeline | PIPELINE | pipeline |
| `build_fact_store` / `retrieve_facts` | `dialogue_contract_pipeline.py:1778/1814` | active-brand fact store + retrieval | PIPELINE | pipeline |
| `build_draft_prompt` / `build_estimate_prompt` | `dialogue_contract_pipeline.py:1974/2038` | LLM draft/estimate prompt для pipeline | PIPELINE | pipeline + estimate флаги |
| `build_faithfulness_prompt` | `dialogue_contract_pipeline.py:2143` | старый faithfulness critic | PIPELINE | pipeline |
| `verify_output` | `dialogue_contract_pipeline.py:4065` | hard verifier, также переиспользуется authoritative gate | SHARED | pipeline/direct gate |
| `_safe_fallback_text` | `dialogue_contract_pipeline.py:6388` | старые handoff-шаблоны | PIPELINE | pipeline |
| `_is_pure_handoff_text` | `dialogue_contract_pipeline.py:6312` | распознаёт чистый handoff для verifier skip | PIPELINE/SHARED | `TELEGRAM_VERIFIER_HANDOFF_CLAIMS` меняет skip |

### C2. Legacy guard-chain в `subscription_llm.py`

Если direct path выключен, `build_draft` идёт либо в pipeline branch, либо в старый `build_draft_prompt` branch. После этого проходят слои:

| Блок | файл:строка | назначение | статус | флаг |
|---|---:|---|---|---|
| `_build_dialogue_contract_pipeline_draft` | `subscription_llm.py:3619` | wrapper вокруг `run_dialogue_contract_pipeline` | PIPELINE | `TELEGRAM_DIALOGUE_CONTRACT_PIPELINE` |
| `_apply_dialogue_contract_v2_guard_chain` | `subscription_llm.py:3690` | post-pipeline guard chain | PIPELINE | pipeline + guard flags |
| legacy prompt branch | `subscription_llm.py:3308` | старый prompt builder + bot_draft call | LEGACY | работает только direct/pipeline off |
| `apply_payment_confirmation_guard` | `subscription_llm.py:8462` | payment-status guard | LEGACY/PIPELINE | нет отдельного direct вызова |
| `apply_brand_separation_guard` | `subscription_llm.py:9155` | cross-brand separation | LEGACY/PIPELINE | direct держит brand через fact filter + gate |
| `apply_input_policy_guards` | `subscription_llm.py:7746` | input policy guards | LEGACY/PIPELINE | direct preblock/gate вместо этого |
| `apply_high_risk_content_guards` | `subscription_llm.py:7778` | high-risk manager handoff | LEGACY/PIPELINE | direct preblock/gate вместо этого |
| `apply_unstated_subject_guard` | `subscription_llm.py:8959` | subject ambiguity guard | LEGACY/PIPELINE | - |
| `apply_unsupported_promise_guard` | `subscription_llm.py:6065` | unsupported promise guard | LEGACY/PIPELINE | - |
| `apply_unconfirmed_operational_specificity_guard` | `subscription_llm.py:6636` | unsupported operational claims | LEGACY/PIPELINE | - |
| `apply_known_context_redundant_question_guard` | `subscription_llm.py:8780` | не переспросить известное | LEGACY/PIPELINE | - |
| `apply_funnel_policy_guard` | `subscription_llm.py:8825` | funnel policy | LEGACY/PIPELINE | - |
| `apply_autonomy_matrix_guard` | `subscription_llm.py:8190` | autonomy matrix | LEGACY/PIPELINE | не direct |
| `apply_humanity_guards` / `apply_humanity_x2_rewriter` | `subscription_llm.py:6754/6988` | старый humanity layer | FLAG-OFF/PIPELINE | `TELEGRAM_DRAFT_X2_REWRITE` и related |
| `apply_phase2_tone_layer` | `subscription_llm.py:7070` | tone rewrite | FLAG-OFF/PIPELINE | `TELEGRAM_PH2_TONE` |
| `apply_a2_proactive_layer` | `subscription_llm.py:4458` | proactive/contact capture | FLAG-OFF/PIPELINE | `TELEGRAM_A_PROACTIVE` |
| `apply_tone_close_detect_layer` | `subscription_llm.py:4605` | закрытие/CTA | FLAG-OFF/PIPELINE | `TELEGRAM_TONE_CLOSE_DETECT` |
| `apply_tone_sell_prompt_observer` | `subscription_llm.py:4509` | observer for selling step missing | FLAG-OFF/PIPELINE | `TELEGRAM_TONE_SELL_PROMPT` |
| `apply_semantic_diagnosis_guard` | `subscription_llm.py:7512` | старый diagnosis guard fallback | FLAG-OFF/PIPELINE | `TELEGRAM_SEMANTIC_DIAGNOSIS_GUARD`; direct использует semantic verifier |

## Раздел D. Мёртвый код / кандидаты на снос

Это не команда на удаление. Это список для отдельного cleanup-ТЗ. Риск удаления оценивает риск скрытых импортов/старых тестов, а не бизнес-важность.

| объект | файл:строка | evidence | статус | риск удаления |
|---|---:|---|---|---|
| `_direct_path_context_fact_items` | `subscription_llm.py:2774` | по `rg` только definition; wide fact pack использует `_direct_path_context_fact_pack` | МЁРТВЫЙ-кандидат | medium |
| `_has_trial_retrieved_fact` | `subscription_llm.py:6230` | только definition | МЁРТВЫЙ-кандидат | medium |
| `_is_enrollment_process_question` | `subscription_llm.py:9521` | только definition; рядом есть живой `_is_enrollment_signup_question` | МЁРТВЫЙ-кандидат | medium |
| `_mentions_schedule_day_or_time` | `subscription_llm.py:9542` | только definition | МЁРТВЫЙ-кандидат | low |
| `_price_question_explicitly_supersedes_installment` | `subscription_llm.py:9579` | только definition | МЁРТВЫЙ-кандидат | medium |
| `_manager_handoff_request_text` | `subscription_llm.py:9595` | только definition | МЁРТВЫЙ-кандидат | medium |
| `_presale_refund_handoff_ack_text` | `subscription_llm.py:9608` | только definition | МЁРТВЫЙ-кандидат | medium |
| `_price_fix_process_text` | `subscription_llm.py:9625` | только definition | МЁРТВЫЙ-кандидат | medium |
| `_enrollment_signup_process_text` | `subscription_llm.py:9638` | только definition | МЁРТВЫЙ-кандидат | medium |
| `_defer_direct_process_to_format_choice_template` | `subscription_llm.py:10402` | только definition | МЁРТВЫЙ-кандидат | medium |
| `_recent_format_choice_was_ambiguous` | `subscription_llm.py:10434` | только definition | МЁРТВЫЙ-кандидат | medium |
| `_without_known_grade_reask` / `_known_subject_or_format` / `_known_grade_int` | `subscription_llm.py:10620/10644/10660` | только definition | МЁРТВЫЙ-кандидаты | low-medium |
| `_asks_money_price_question` | `subscription_llm.py:11627` | только definition в `subscription_llm`; живой дубль есть в `conversation_intent_plan.py` | МЁРТВЫЙ duplicate | low |
| `_is_generic_price_question_without_selection` | `subscription_llm.py:11638` | только definition | МЁРТВЫЙ-кандидат | low |
| `_normalized_fact_text` | `subscription_llm.py:11694` | только definition | МЁРТВЫЙ-кандидат | low |
| `draft_has_internal_service_markers` | `subscription_llm.py:5985` | используется тестами; runtime переехал к sanitizer/gate | ТЕСТЫ | low после правки тестов |
| `parse_llm_json`, `CodexExecConfig`, aliases at bottom | `subscription_llm.py:5948/4342/12412` | экспорт/тесты/compat | ТЕСТЫ/API | medium, может сломать внешние импорты |

Не считать мёртвым без отдельного анализа: `dialogue_contract_pipeline.py` (legacy, но opt-in), `fact_retrieval.py`/`fact_scope_spec.py` (живы в context builder/gate), `telegram_native_draft.py` и `telegram_business_runtime.py` (scaffold/test/export), `telegram_bot_polling.py` (используется manager draft pilot scripts).

## Раздел E. Поток одного хода на `pilot_gold_v1`

1. Runner (`scripts/run_telegram_dynamic_client_sim.py:1375`) берёт сценарий и историю.
2. `build_bot_prompt_context` (`run_telegram_dynamic_client_sim.py:1575`) строит контекст через `build_telegram_pilot_context`.
3. `build_telegram_pilot_context` (`telegram_pilot_context_builder.py:104`) строит `dialogue_memory_view`, `conversation_intent_plan`, confirmed facts и snippets из KB snapshot.
4. `build_bot_provider` (`run_telegram_dynamic_client_sim.py:1051`) создаёт `CountingSubscriptionLlmDraftProvider`; для codex по умолчанию `codex_isolated=True`.
5. `SubscriptionLlmDraftProvider.build_draft` (`subscription_llm.py:3268`) видит `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1` и сразу уходит в `_build_direct_path_draft`; `TELEGRAM_DIALOGUE_CONTRACT_PIPELINE` ниже уже не влияет.
6. При `TELEGRAM_LLM_RETRIEVE=1` сначала `_direct_path_preblocked_result` отсекает P0/high-risk/unknown brand; иначе preblock идёт после fact pack.
7. `_direct_path_context_fact_pack` / `_direct_path_wide_fact_pack` собирают active-brand client-safe facts. При Wave 6 ON `_direct_path_llm_retrieve_fact_pack` выбирает fact ids; при fail-soft откат на keyword.
8. `_build_direct_path_prompt` собирает миссию, active brand, exact/adjacent facts, память, known slots и gold examples.
9. `_direct_path_draft_runner` делает основной LLM-вызов. LLM возвращает JSON route + draft_text.
10. `_normalize_direct_path_payload` нормализует ответ модели.
11. `apply_semantic_output_verifier` (если включён) проверяет смысл, пишет findings и может сделать один regen. Если verifier недоступен, fail-soft.
12. `apply_authoritative_output_gate` последним применяет sanitizer, hard verifier, P0/brand/number/meta/PII findings; не повышает route, только pass/annotate/downgrade/block.
13. `_direct_path_finalize_metadata` фиксирует `route_before_gate`, `route_after`, `downgraded`, `text_composition_source`, `reason_class`.
14. Runner пишет `dynamic_dialog_transcripts.jsonl`, `dynamic_summary.json`, `bot_direct_path`, `bot_authoritative_output_gate`, `llm_calls` и judge v9 result.

## Раздел F. Измерение, watcher, judge

| Блок | файл:строка | назначение | статус | флаг/CLI |
|---|---:|---|---|---|
| `run_telegram_dynamic_client_sim.py:538 main` | `scripts/run_telegram_dynamic_client_sim.py:538` | CLI вход симулятора | ИЗМЕРЕНИЕ | `--scenarios`, `--snapshot`, `--bot-mode`, `--judge-prompt-version` |
| `build_bot_provider` | `run_telegram_dynamic_client_sim.py:1051` | создаёт bot provider, кэш, codex/claude runner | ИЗМЕРЕНИЕ | `--bot-mode`, `--codex-isolated` |
| `_direct_path_config_invalid` | `run_telegram_dynamic_client_sim.py:1211` | fail-fast, если direct path не зовёт модель в первых завершённых диалогах | ИЗМЕРЕНИЕ DIRECT | `TELEGRAM_DIRECT_PATH*` |
| `build_judge_prompt` / `judge_dialog` | `run_telegram_dynamic_client_sim.py:1694/1790` | judge v2/v9 | ИЗМЕРЕНИЕ | `--judge-prompt-version v9` |
| `build_summary` | `run_telegram_dynamic_client_sim.py:2585` | итоги, gates, calls, tone, over-handoff | ИЗМЕРЕНИЕ | - |
| `_llm_call_summary` | `run_telegram_dynamic_client_sim.py:2872` | роли LLM: `bot_direct_draft`, `bot_retriever`, `bot_semantic_output_verifier`, etc. | ИЗМЕРЕНИЕ | - |
| `M1Watcher` | `scripts/m1_watcher.py:351` | deterministic task runner for M1 | ИНФРА | task yaml |
| `PRODUCTION_ENV_STACK` | `scripts/m1_watcher.py:50` | стандартный стек флагов M1 | ИНФРА | task env delta over stack |

Риск измерения: watcher `PRODUCTION_ENV_STACK` включает pipeline/semantic/tone flags, но не включает `TELEGRAM_DIRECT_PATH_PILOT_CONFIG`; task env обязан добавить `pilot_gold_v1`.

## Раздел G. Полный индекс функций/классов по охваченным файлам

Автоматический индекс ниже создан AST-проходом по файлам из охвата. `refs` — грубый текстовый счётчик упоминаний имени в охваченных файлах; он помогает найти кандидатов на снос, но не заменяет полноценный call graph.

### `src/mango_mvp/channels/subscription_llm.py`
| строка | объект | назначение | статус | флаг | refs |
|---:|---|---|---|---|---:|
| 837 | `SubscriptionDraftResult` | helper/block: SubscriptionDraftResult | LEGACY/HELPER | нет явного флага | 228 |
| 863 | `SubscriptionDraftResult.__post_init__` | helper/block: post init | LEGACY/HELPER | нет явного флага | 2 |
| 913 | `SubscriptionDraftResult.to_json_dict` | helper/block: to json dict | LEGACY/HELPER | нет явного флага | 30 |
| 946 | `SafeTemplateSpec` | helper/block: SafeTemplateSpec | LEGACY/HELPER | нет явного флага | 9 |
| 958 | `_produce_cross_brand_template` | helper/block: produce cross brand template | PIPELINE/SHARED | смотри конкретный guard/env | 2 |
| 966 | `_produce_terminal_template` | helper/block: produce terminal template | LEGACY/HELPER | нет явного флага | 2 |
| 974 | `_produce_result_guarantee_template` | helper/block: produce result guarantee template | LEGACY/HELPER | нет явного флага | 2 |
| 982 | `_produce_admission_guarantee_template` | helper/block: produce admission guarantee template | LEGACY/HELPER | нет явного флага | 2 |
| 1031 | `_is_informational_terminal_template` | helper/block: is informational terminal template | LEGACY/HELPER | нет явного флага | 4 |
| 1041 | `_safe_template_already_applied` | helper/block: safe template already applied | PIPELINE/SHARED | смотри конкретный guard/env | 2 |
| 1066 | `_safe_template_can_yield_to_dispatcher` | helper/block: safe template can yield to dispatcher | PIPELINE/SHARED | смотри конкретный guard/env | 2 |
| 1092 | `_safe_template_route` | helper/block: safe template route | PIPELINE/SHARED | смотри конкретный guard/env | 2 |
| 1106 | `_is_approved_policy_c_identity_text` | helper/block: is approved policy c identity text | LEGACY/HELPER | нет явного флага | 3 |
| 1115 | `_policy_c_identity_allowed` | helper/block: policy c identity allowed | LEGACY/HELPER | нет явного флага | 2 |
| 1130 | `_is_terminal_direct_info_template` | helper/block: is terminal direct info template | LEGACY/HELPER | нет явного флага | 3 |
| 1147 | `_apply_safe_template_spec` | helper/block: apply safe template spec | PIPELINE/SHARED | смотри конкретный guard/env | 2 |
| 1185 | `_dialogue_contract_retrieved_facts` | helper/block: dialogue contract retrieved facts | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1192 | `_dialogue_contract_mapping` | helper/block: dialogue contract mapping | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 7 |
| 1199 | `_migrated_rule_intent_from_dialogue_contract` | helper/block: migrated rule intent from dialogue contract | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 5 |
| 1257 | `_float_value` | helper/block: float value | LEGACY/HELPER | нет явного флага | 3 |
| 1264 | `_rules_engine_planner_intent_enabled` | helper/block: rules engine planner intent enabled | LEGACY/HELPER | нет явного флага | 2 |
| 1273 | `_planner_intent_candidate` | helper/block: planner intent candidate | LEGACY/HELPER | нет явного флага | 2 |
| 1282 | `_is_policy_c_identity_question` | helper/block: is policy c identity question | LEGACY/HELPER | нет явного флага | 6 |
| 1309 | `_rules_engine_intent_shadow` | helper/block: rules engine intent shadow | LEGACY/HELPER | нет явного флага | 3 |
| 1353 | `_with_rules_engine_intent_shadow` | helper/block: with rules engine intent shadow | LEGACY/HELPER | нет явного флага | 2 |
| 1367 | `_rules_engine_facts` | helper/block: rules engine facts | LEGACY/HELPER | нет явного флага | 3 |
| 1379 | `_apply_rules_engine_outcome` | helper/block: apply rules engine outcome | LEGACY/HELPER | нет явного флага | 2 |
| 1413 | `_apply_migrated_rules_engine` | helper/block: apply migrated rules engine | LEGACY/HELPER | нет явного флага | 2 |
| 1518 | `_context_with_selling_thread_slots` | helper/block: context with selling thread slots | LEGACY/HELPER | нет явного флага | 2 |
| 1544 | `_a_thread_enabled` | helper/block: a thread enabled | LEGACY/HELPER | нет явного флага | 2 |
| 1552 | `_selling_slots_from_contract_and_text` | helper/block: selling slots from contract and text | LEGACY/HELPER | нет явного флага | 2 |
| 1572 | `_selling_slots_from_memory` | helper/block: selling slots from memory | LEGACY/HELPER | нет явного флага | 2 |
| 1588 | `_merge_selling_slot_values` | helper/block: merge selling slot values | LEGACY/HELPER | нет явного флага | 3 |
| 1600 | `_selling_slots_from_text` | helper/block: selling slots from text | LEGACY/HELPER | нет явного флага | 3 |
| 1629 | `_text_explicitly_mentions_selling_slot` | helper/block: text explicitly mentions selling slot | LEGACY/HELPER | нет явного флага | 2 |
| 1633 | `_phase2_objection_signal` | helper/block: phase2 objection signal | ФЛАГ-OFF / PIPELINE | TELEGRAM_PH2_* | 2 |
| 1640 | `_phase2_anxiety_signal` | helper/block: phase2 anxiety signal | ФЛАГ-OFF / PIPELINE | TELEGRAM_PH2_* | 2 |
| 1647 | `_merged_selling_signals` | helper/block: merged selling signals | LEGACY/HELPER | нет явного флага | 3 |
| 1684 | `_manager_route_migrated_rules_override_allowed` | helper/block: manager route migrated rules override allowed | LEGACY/HELPER | нет явного флага | 4 |
| 1704 | `_migrated_rules_keep_existing_verified_answer` | helper/block: migrated rules keep existing verified answer | LEGACY/HELPER | нет явного флага | 2 |
| 1736 | `_rules_engine_result_applied` | helper/block: rules engine result applied | LEGACY/HELPER | нет явного флага | 3 |
| 1746 | `_pipeline_travel_estimate_applied` | helper/block: pipeline travel estimate applied | LEGACY/HELPER | нет явного флага | 2 |
| 1754 | `_yield_dispatcher_to_travel_estimate` | helper/block: yield dispatcher to travel estimate | LEGACY/HELPER | нет явного флага | 2 |
| 1763 | `_metadata_with_self_route_deferral_cleared` | helper/block: metadata with self route deferral cleared | LEGACY/HELPER | нет явного флага | 5 |
| 1780 | `_metadata_with_guarded_original_text` | helper/block: metadata with guarded original text | LEGACY/HELPER | нет явного флага | 6 |
| 1800 | `_safe_template_yield_result` | helper/block: safe template yield result | PIPELINE/SHARED | смотри конкретный guard/env | 2 |
| 1830 | `apply_dialogue_contract_v2_template_dispatcher` | helper/block: apply dialogue contract v2 template dispatcher | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 2028 | `_direct_path_enabled` | helper/block: direct path enabled | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2038 | `_llm_retrieve_enabled` | helper/block: llm retrieve enabled | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ | TELEGRAM_LLM_RETRIEVE | 3 |
| 2046 | `_direct_path_pilot_config` | helper/block: direct path pilot config | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 5 |
| 2055 | `_direct_path_brand_label` | helper/block: direct path brand label | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2064 | `_direct_path_snapshot_path_from_context` | helper/block: direct path snapshot path from context | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2075 | `_direct_path_load_snapshot` | helper/block: direct path load snapshot | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2088 | `_direct_path_snapshot_facts` | helper/block: direct path snapshot facts | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2095 | `_direct_path_valid_until_ok` | helper/block: direct path valid until ok | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2106 | `_direct_path_client_safe_snapshot_fact` | helper/block: direct path client safe snapshot fact | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2118 | `_direct_path_snapshot_fact_key` | helper/block: direct path snapshot fact key | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 9 |
| 2122 | `_direct_path_snapshot_fact_text` | helper/block: direct path snapshot fact text | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 7 |
| 2130 | `_direct_path_fact_text` | helper/block: direct path fact text | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2144 | `_direct_path_add_fact` | helper/block: direct path add fact | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 6 |
| 2151 | `_direct_path_legacy_context_fact_items` | helper/block: direct path legacy context fact items | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2191 | `_direct_path_fact_categories` | helper/block: direct path fact categories | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 3 |
| 2217 | `_direct_path_category_from_hint` | helper/block: direct path category from hint | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2243 | `_direct_path_selected_categories` | helper/block: direct path selected categories | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2272 | `_direct_path_slot_scope` | helper/block: direct path slot scope | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 3 |
| 2288 | `_direct_path_format_scope` | helper/block: direct path format scope | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 3 |
| 2297 | `_direct_path_grade_in_fact` | helper/block: direct path grade in fact | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2311 | `_direct_path_fact_conflicts_slots` | helper/block: direct path fact conflicts slots | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 4 |
| 2330 | `_direct_path_fact_relevance_score` | helper/block: direct path fact relevance score | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2351 | `_direct_path_render_fact_line` | helper/block: direct path render fact line | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 3 |
| 2358 | `_direct_path_render_fact_block` | helper/block: direct path render fact block | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 3 |
| 2372 | `_direct_path_fact_pack_char_count` | helper/block: direct path fact pack char count | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2376 | `_direct_path_core_fact` | helper/block: direct path core fact | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 3 |
| 2388 | `_direct_path_empty_fact_pack` | helper/block: direct path empty fact pack | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2399 | `_direct_path_records_to_fact_pack` | helper/block: direct path records to fact pack | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 3 |
| 2413 | `_direct_path_records_to_fact_pack.add_record` | helper/block: add record | LEGACY/HELPER | нет явного флага | 3 |
| 2465 | `_direct_path_keyword_fact_pack_from_records` | helper/block: direct path keyword fact pack from records | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2534 | `_direct_path_retriever_candidate_summary` | helper/block: direct path retriever candidate summary | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ | TELEGRAM_LLM_RETRIEVE | 2 |
| 2545 | `build_direct_path_llm_retriever_prompt` | helper/block: build direct path llm retriever prompt | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ | TELEGRAM_LLM_RETRIEVE | 2 |
| 2585 | `_direct_path_retriever_ids` | helper/block: direct path retriever ids | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ | TELEGRAM_LLM_RETRIEVE | 3 |
| 2600 | `_direct_path_llm_retrieve_fact_pack` | helper/block: direct path llm retrieve fact pack | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ | TELEGRAM_LLM_RETRIEVE | 2 |
| 2699 | `_direct_path_wide_fact_pack` | helper/block: direct path wide fact pack | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2753 | `_direct_path_context_fact_pack` | helper/block: direct path context fact pack | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 4 |
| 2774 | `_direct_path_context_fact_items` | helper/block: direct path context fact items | МЁРТВЫЙ-кандидат | нет; проверить перед сносом | 1 |
| 2786 | `_direct_path_recent_messages` | helper/block: direct path recent messages | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 3 |
| 2795 | `_direct_path_known_slots` | helper/block: direct path known slots | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 4 |
| 2830 | `_direct_path_gold_real_enabled` | helper/block: direct path gold real enabled | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2840 | `_direct_path_gold_pack_path` | helper/block: direct path gold pack path | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2847 | `_load_direct_path_gold_real_examples` | helper/block: load direct path gold real examples | LEGACY/HELPER | нет явного флага | 2 |
| 2865 | `_direct_path_topic_hints` | helper/block: direct path topic hints | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2886 | `_direct_path_select_gold_real_examples` | helper/block: direct path select gold real examples | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2922 | `_direct_path_gold_prompt_block` | helper/block: direct path gold prompt block | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2942 | `_build_direct_path_prompt` | helper/block: build direct path prompt | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 2994 | `_direct_path_p0_text` | helper/block: direct path p0 text | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 3005 | `_direct_path_metadata` | helper/block: direct path metadata | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 6 |
| 3056 | `_direct_path_preblocked_result` | helper/block: direct path preblocked result | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 3 |
| 3162 | `_direct_path_merge_metadata` | helper/block: direct path merge metadata | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 3172 | `_direct_path_finalize_metadata` | helper/block: direct path finalize metadata | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 4 |
| 3221 | `SubscriptionLlmDraftProvider` | helper/block: SubscriptionLlmDraftProvider | LEGACY/HELPER | нет явного флага | 4 |
| 3222 | `SubscriptionLlmDraftProvider.__init__` | helper/block: init | LEGACY/HELPER | нет явного флага | 12 |
| 3251 | `SubscriptionLlmDraftProvider._build_codex_command` | helper/block: build codex command | LEGACY/HELPER | нет явного флага | 6 |
| 3268 | `SubscriptionLlmDraftProvider.build_draft` | helper/block: build draft | LEGACY/HELPER | нет явного флага | 8 |
| 3371 | `SubscriptionLlmDraftProvider._build_direct_path_draft` | helper/block: build direct path draft | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 3461 | `SubscriptionLlmDraftProvider._build_dialogue_contract_pipeline_draft` | helper/block: build dialogue contract pipeline draft | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 3570 | `SubscriptionLlmDraftProvider._dialogue_contract_understanding_runner` | helper/block: dialogue contract understanding runner | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 3589 | `SubscriptionLlmDraftProvider._dialogue_contract_draft_runner` | helper/block: dialogue contract draft runner | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 3597 | `SubscriptionLlmDraftProvider._dialogue_contract_faithfulness_runner` | helper/block: dialogue contract faithfulness runner | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 5 |
| 3609 | `SubscriptionLlmDraftProvider._dialogue_contract_semantic_match_runner` | helper/block: dialogue contract semantic match runner | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 3622 | `SubscriptionLlmDraftProvider._semantic_diagnosis_guard_runner` | helper/block: semantic diagnosis guard runner | LEGACY/HELPER | нет явного флага | 5 |
| 3635 | `SubscriptionLlmDraftProvider._semantic_output_verifier_runner` | helper/block: semantic output verifier runner | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_SEMANTIC_OUTPUT_VERIFIER | 6 |
| 3649 | `SubscriptionLlmDraftProvider._semantic_output_regen_runner` | helper/block: semantic output regen runner | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_SEMANTIC_OUTPUT_VERIFIER | 6 |
| 3659 | `SubscriptionLlmDraftProvider._direct_path_llm_retrieve_runner` | helper/block: direct path llm retrieve runner | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ | TELEGRAM_LLM_RETRIEVE | 4 |
| 3673 | `SubscriptionLlmDraftProvider._dialogue_contract_repair_runner` | helper/block: dialogue contract repair runner | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 3681 | `SubscriptionLlmDraftProvider._dialogue_contract_warmth_runner` | helper/block: dialogue contract warmth runner | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 3690 | `SubscriptionLlmDraftProvider._apply_dialogue_contract_v2_guard_chain` | v2 post-chain: safety verifiers only; no old intent/template rewrites. | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 3700 | `SubscriptionLlmDraftProvider._apply_dialogue_contract_v2_guard_chain.record_step` | helper/block: record step | LEGACY/HELPER | нет явного флага | 12 |
| 3775 | `SubscriptionLlmDraftProvider._reverify_dialogue_contract_text_change` | helper/block: reverify dialogue contract text change | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 9 |
| 3928 | `SubscriptionLlmDraftProvider._dialogue_contract_v2_route_permission_guard` | helper/block: dialogue contract v2 route permission guard | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 3997 | `SubscriptionLlmDraftProvider._run_prompt_text` | helper/block: run prompt text | LEGACY/HELPER | нет явного флага | 11 |
| 4030 | `SubscriptionLlmDraftProvider._answer_quality_llm_rewrite_runner` | helper/block: answer quality llm rewrite runner | LEGACY/HELPER | нет явного флага | 4 |
| 4073 | `SubscriptionLlmDraftProvider._humanity_x2_rewrite_runner` | helper/block: humanity x2 rewrite runner | LEGACY/PIPELINE | humanity/autonomy флаги | 5 |
| 4094 | `SubscriptionLlmDraftProvider.generate` | helper/block: generate | LEGACY/HELPER | нет явного флага | 20 |
| 4097 | `SubscriptionLlmDraftProvider.generate_from_prompt` | helper/block: generate from prompt | LEGACY/HELPER | нет явного флага | 6 |
| 4137 | `SubscriptionLlmDraftProvider._run_once` | helper/block: run once | LEGACY/HELPER | нет явного флага | 4 |
| 4172 | `SubscriptionLlmDraftProvider._direct_path_draft_runner` | helper/block: direct path draft runner | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 4 |
| 4202 | `SubscriptionLlmDraftProvider._cache_get` | helper/block: cache get | LEGACY/HELPER | нет явного флага | 2 |
| 4214 | `SubscriptionLlmDraftProvider._cache_put` | helper/block: cache put | LEGACY/HELPER | нет явного флага | 2 |
| 4222 | `FakeSubscriptionLlmDraftProvider` | helper/block: FakeSubscriptionLlmDraftProvider | LEGACY/HELPER | нет явного флага | 2 |
| 4223 | `FakeSubscriptionLlmDraftProvider.__init__` | helper/block: init | LEGACY/HELPER | нет явного флага | 12 |
| 4229 | `FakeSubscriptionLlmDraftProvider.build_draft` | helper/block: build draft | LEGACY/HELPER | нет явного флага | 8 |
| 4266 | `FakeSubscriptionLlmDraftProvider.generate` | helper/block: generate | LEGACY/HELPER | нет явного флага | 20 |
| 4269 | `FakeSubscriptionLlmDraftProvider.generate_from_prompt` | helper/block: generate from prompt | LEGACY/HELPER | нет явного флага | 6 |
| 4281 | `build_codex_exec_command` | helper/block: build codex exec command | LEGACY/HELPER | нет явного флага | 5 |
| 4310 | `codex_isolation_cwd` | helper/block: codex isolation cwd | LEGACY/HELPER | нет явного флага | 8 |
| 4318 | `_with_codex_exec_metadata` | helper/block: with codex exec metadata | LEGACY/HELPER | нет явного флага | 3 |
| 4333 | `build_codex_exec_env` | helper/block: build codex exec env | LEGACY/HELPER | нет явного флага | 6 |
| 4342 | `CodexExecConfig` | helper/block: CodexExecConfig | ТЕСТЫ/API-совместимость | нет | 1 |
| 4349 | `CodexExecConfig.build_command` | helper/block: build command | LEGACY/HELPER | нет явного флага | 1 |
| 4360 | `normalize_subscription_draft_payload` | helper/block: normalize subscription draft payload | LEGACY/HELPER | нет явного флага | 8 |
| 4400 | `safe_fallback_draft` | helper/block: safe fallback draft | LEGACY/HELPER | нет явного флага | 12 |
| 4414 | `_normalize_direct_path_payload` | helper/block: normalize direct path payload | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 4458 | `apply_a2_proactive_layer` | A2.1 callback/contact capture plus deterministic rich-format guard. | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_PROACTIVE / TELEGRAM_A_RICH_FORMAT | 4 |
| 4482 | `_tone_sell_prompt_step_observation` | helper/block: tone sell prompt step observation | ФЛАГ-OFF / PIPELINE | TELEGRAM_TONE_* | 2 |
| 4509 | `apply_tone_sell_prompt_observer` | helper/block: apply tone sell prompt observer | ФЛАГ-OFF / PIPELINE | TELEGRAM_TONE_* | 4 |
| 4605 | `apply_tone_close_detect_layer` | helper/block: apply tone close detect layer | ФЛАГ-OFF / PIPELINE | TELEGRAM_TONE_* | 3 |
| 4666 | `_tone_close_metadata` | helper/block: tone close metadata | ФЛАГ-OFF / PIPELINE | TELEGRAM_TONE_* | 4 |
| 4690 | `_tone_close_detect_is_close_message` | helper/block: tone close detect is close message | ФЛАГ-OFF / PIPELINE | TELEGRAM_TONE_* | 2 |
| 4708 | `_tone_close_has_unanswered_or_problem_continuation` | helper/block: tone close has unanswered or problem continuation | ФЛАГ-OFF / PIPELINE | TELEGRAM_TONE_* | 2 |
| 4717 | `_tone_close_detect_is_p0` | helper/block: tone close detect is p0 | ФЛАГ-OFF / PIPELINE | TELEGRAM_TONE_* | 2 |
| 4731 | `_tone_close_pending_manager` | helper/block: tone close pending manager | ФЛАГ-OFF / PIPELINE | TELEGRAM_TONE_* | 2 |
| 4744 | `_tone_close_old_p0_history` | helper/block: tone close old p0 history | ФЛАГ-OFF / PIPELINE | TELEGRAM_TONE_* | 2 |
| 4760 | `_tone_close_message_references_pending` | helper/block: tone close message references pending | ФЛАГ-OFF / PIPELINE | TELEGRAM_TONE_* | 2 |
| 4764 | `_tone_close_next_step_text` | helper/block: tone close next step text | ФЛАГ-OFF / PIPELINE | TELEGRAM_TONE_* | 2 |
| 4795 | `_tone_close_contact_requested_from_memory` | helper/block: tone close contact requested from memory | ФЛАГ-OFF / PIPELINE | TELEGRAM_TONE_* | 4 |
| 4802 | `_tone_close_contact_requested_after_step` | helper/block: tone close contact requested after step | ФЛАГ-OFF / PIPELINE | TELEGRAM_TONE_* | 2 |
| 4816 | `_tone_close_previous_contact_requested` | helper/block: tone close previous contact requested | ФЛАГ-OFF / PIPELINE | TELEGRAM_TONE_* | 3 |
| 4820 | `_tone_close_previous_trial_requested` | helper/block: tone close previous trial requested | ФЛАГ-OFF / PIPELINE | TELEGRAM_TONE_* | 2 |
| 4824 | `_tone_close_refused_previous_step` | helper/block: tone close refused previous step | ФЛАГ-OFF / PIPELINE | TELEGRAM_TONE_* | 2 |
| 4831 | `_tone_close_pending_text` | helper/block: tone close pending text | ФЛАГ-OFF / PIPELINE | TELEGRAM_TONE_* | 2 |
| 4835 | `_a2_contact_capture_handoff` | helper/block: a2 contact capture handoff | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_PROACTIVE / TELEGRAM_A_RICH_FORMAT | 2 |
| 4892 | `_a2_apply_rich_format_guard` | helper/block: a2 apply rich format guard | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_PROACTIVE / TELEGRAM_A_RICH_FORMAT | 2 |
| 4918 | `_a2_proactive_enabled` | helper/block: a2 proactive enabled | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_PROACTIVE / TELEGRAM_A_RICH_FORMAT | 3 |
| 4926 | `_a2_rich_format_enabled` | helper/block: a2 rich format enabled | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_PROACTIVE / TELEGRAM_A_RICH_FORMAT | 3 |
| 4938 | `_a2_p0_or_high_risk` | helper/block: a2 p0 or high risk | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_PROACTIVE / TELEGRAM_A_RICH_FORMAT | 2 |
| 4957 | `_a2_extract_phone` | helper/block: a2 extract phone | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_PROACTIVE / TELEGRAM_A_RICH_FORMAT | 5 |
| 4962 | `_a2_has_time` | helper/block: a2 has time | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_PROACTIVE / TELEGRAM_A_RICH_FORMAT | 2 |
| 4966 | `_a2_mask_phone` | helper/block: a2 mask phone | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_PROACTIVE / TELEGRAM_A_RICH_FORMAT | 2 |
| 4973 | `_a2_context_phone_known` | helper/block: a2 context phone known | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_PROACTIVE / TELEGRAM_A_RICH_FORMAT | 2 |
| 4997 | `_a2_context_tag` | helper/block: a2 context tag | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_PROACTIVE / TELEGRAM_A_RICH_FORMAT | 4 |
| 5015 | `_a2_enforce_emoji_limit` | helper/block: a2 enforce emoji limit | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_PROACTIVE / TELEGRAM_A_RICH_FORMAT | 3 |
| 5029 | `apply_authoritative_output_gate` | Final safety gate over every provider output | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 16 |
| 5144 | `_authoritative_gate_direct_path_keep_text` | helper/block: authoritative gate direct path keep text | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 2 |
| 5158 | `_direct_path_generic_replacement_text` | helper/block: direct path generic replacement text | ПРЯМОЙ ПУТЬ | TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 / TELEGRAM_DIRECT_PATH | 2 |
| 5167 | `apply_output_sanitizer` | helper/block: apply output sanitizer | ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_OUTPUT_SANITIZER частично; PII de-echo всегда | 2 |
| 5218 | `_sanitize_output_client_text` | helper/block: sanitize output client text | ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_OUTPUT_SANITIZER частично; PII de-echo всегда | 2 |
| 5325 | `_sanitize_client_pii_echo` | helper/block: sanitize client pii echo | ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_OUTPUT_SANITIZER частично; PII de-echo всегда | 3 |
| 5352 | `_client_pii_echo_context` | helper/block: client pii echo context | ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_OUTPUT_SANITIZER частично; PII de-echo всегда | 2 |
| 5379 | `_client_name_echoes` | helper/block: client name echoes | ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_OUTPUT_SANITIZER частично; PII de-echo всегда | 2 |
| 5408 | `_client_name_echoed` | helper/block: client name echoed | ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_OUTPUT_SANITIZER частично; PII de-echo всегда | 2 |
| 5412 | `_flexible_name_pattern` | helper/block: flexible name pattern | LEGACY/HELPER | нет явного флага | 3 |
| 5419 | `_name_word_pattern` | helper/block: name word pattern | LEGACY/HELPER | нет явного флага | 2 |
| 5437 | `_replace_echoed_phone` | helper/block: replace echoed phone | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 2 |
| 5453 | `_sanitize_raw_detail_handoff_text` | helper/block: sanitize raw detail handoff text | LEGACY/HELPER | нет явного флага | 2 |
| 5456 | `_sanitize_raw_detail_handoff_text.repl` | helper/block: repl | LEGACY/HELPER | нет явного флага | 4 |
| 5466 | `_sanitize_raw_detail_handoff_match` | helper/block: sanitize raw detail handoff match | LEGACY/HELPER | нет явного флага | 2 |
| 5473 | `_raw_detail_handoff_looks_like_question` | helper/block: raw detail handoff looks like question | LEGACY/HELPER | нет явного флага | 2 |
| 5488 | `_normalize_output_sanitizer_text` | helper/block: normalize output sanitizer text | LEGACY/HELPER | нет явного флага | 4 |
| 5497 | `_output_sanitizer_degenerate` | helper/block: output sanitizer degenerate | LEGACY/HELPER | нет явного флага | 2 |
| 5510 | `_authoritative_gate_action` | helper/block: authoritative gate action | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 5 |
| 5514 | `_authoritative_gate_downgraded_route` | helper/block: authoritative gate downgraded route | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 2 |
| 5523 | `_authoritative_gate_finding` | helper/block: authoritative gate finding | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 14 |
| 5537 | `_authoritative_gate_findings` | helper/block: authoritative gate findings | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 2 |
| 5609 | `_authoritative_gate_text_guard_findings` | helper/block: authoritative gate text guard findings | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 2 |
| 5623 | `_authoritative_gate_a2_findings` | helper/block: authoritative gate a2 findings | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 2 |
| 5650 | `_authoritative_gate_semantic_output_findings` | helper/block: authoritative gate semantic output findings | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_SEMANTIC_OUTPUT_VERIFIER | 2 |
| 5678 | `_semantic_output_finding_detail` | helper/block: semantic output finding detail | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_SEMANTIC_OUTPUT_VERIFIER | 2 |
| 5689 | `_semantic_output_manager_note` | helper/block: semantic output manager note | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_SEMANTIC_OUTPUT_VERIFIER | 3 |
| 5709 | `_a2_is_proactive_result` | helper/block: a2 is proactive result | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_PROACTIVE / TELEGRAM_A_RICH_FORMAT | 4 |
| 5725 | `_a2_phone_echoed` | helper/block: a2 phone echoed | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_PROACTIVE / TELEGRAM_A_RICH_FORMAT | 3 |
| 5733 | `_authoritative_gate_existing_guard_findings` | helper/block: authoritative gate existing guard findings | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 2 |
| 5786 | `_authoritative_guard_changed` | helper/block: authoritative guard changed | LEGACY/HELPER | нет явного флага | 2 |
| 5795 | `_authoritative_gate_fact_texts` | helper/block: authoritative gate fact texts | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 3 |
| 5822 | `_authoritative_gate_skip_backed_finding` | helper/block: authoritative gate skip backed finding | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 4 |
| 5848 | `_authoritative_gate_verified_content_flag` | helper/block: authoritative gate verified content flag | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 4 |
| 5863 | `_authoritative_gate_has_pipeline` | helper/block: authoritative gate has pipeline | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 2 |
| 5868 | `_authoritative_gate_slot_text` | helper/block: authoritative gate slot text | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 2 |
| 5878 | `_authoritative_gate_p0_already_guarded` | helper/block: authoritative gate p0 already guarded | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 2 |
| 5901 | `_dedupe_gate_findings` | helper/block: dedupe gate findings | LEGACY/HELPER | нет явного флага | 3 |
| 5928 | `extract_json_object` | helper/block: extract json object | LEGACY/HELPER | нет явного флага | 18 |
| 5948 | `parse_llm_json` | helper/block: parse llm json | ТЕСТЫ/API-совместимость | нет | 1 |
| 5955 | `strip_internal_service_markers` | helper/block: strip internal service markers | ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_OUTPUT_SANITIZER частично; PII de-echo всегда | 13 |
| 5985 | `draft_has_internal_service_markers` | helper/block: draft has internal service markers | LEGACY/HELPER | нет явного флага | 1 |
| 6001 | `draft_has_identity_disclosure` | helper/block: draft has identity disclosure | LEGACY/HELPER | нет явного флага | 4 |
| 6005 | `find_identity_disclosure_phrases` | helper/block: find identity disclosure phrases | LEGACY/HELPER | нет явного флага | 3 |
| 6010 | `_identity_phrase_present` | helper/block: identity phrase present | LEGACY/HELPER | нет явного флага | 2 |
| 6021 | `guard_identity_disclosure` | helper/block: guard identity disclosure | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 7 |
| 6037 | `guard_draft_placeholder` | helper/block: guard draft placeholder | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 4 |
| 6051 | `guard_promocode_leak` | helper/block: guard promocode leak | ПРЯМОЙ ПУТЬ+PIPELINE | финальный authoritative gate | 4 |
| 6065 | `apply_unsupported_promise_guard` | helper/block: apply unsupported promise guard | PIPELINE/SHARED | смотри конкретный guard/env | 11 |
| 6125 | `_context_with_dialogue_contract_retrieved_facts` | helper/block: context with dialogue contract retrieved facts | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 7 |
| 6214 | `_pipeline_fact_texts` | helper/block: pipeline fact texts | LEGACY/HELPER | нет явного флага | 6 |
| 6230 | `_has_trial_retrieved_fact` | helper/block: has trial retrieved fact | МЁРТВЫЙ-кандидат | нет; проверить перед сносом | 1 |
| 6238 | `_pipeline_contract` | helper/block: pipeline contract | LEGACY/HELPER | нет явного флага | 6 |
| 6253 | `_verified_informational_answer` | helper/block: verified informational answer | LEGACY/HELPER | нет явного флага | 3 |
| 6304 | `_strict_informational_yield_ok` | helper/block: strict informational yield ok | LEGACY/HELPER | нет явного флага | 2 |
| 6325 | `_informational_yield_has_unbacked_concrete_anchors` | helper/block: informational yield has unbacked concrete anchors | LEGACY/HELPER | нет явного флага | 2 |
| 6339 | `_mentions_unbacked_children_rule` | helper/block: mentions unbacked children rule | LEGACY/HELPER | нет явного флага | 2 |
| 6348 | `_asks_non_tax_document_or_contract` | helper/block: asks non tax document or contract | LEGACY/HELPER | нет явного флага | 3 |
| 6358 | `_asks_non_matkap_document_or_contract` | helper/block: asks non matkap document or contract | LEGACY/HELPER | нет явного флага | 3 |
| 6368 | `_answers_tax_deduction_scope` | helper/block: answers tax deduction scope | LEGACY/HELPER | нет явного флага | 3 |
| 6373 | `_answers_matkap_scope` | helper/block: answers matkap scope | LEGACY/HELPER | нет явного флага | 3 |
| 6378 | `_safe_template_applied_name` | helper/block: safe template applied name | PIPELINE/SHARED | смотри конкретный guard/env | 5 |
| 6389 | `_has_informational_safe_template` | helper/block: has informational safe template | PIPELINE/SHARED | смотри конкретный guard/env | 2 |
| 6400 | `_manager_only_recovery_yield_allowed` | helper/block: manager only recovery yield allowed | LEGACY/HELPER | нет явного флага | 2 |
| 6429 | `_safe_template_yield_before_fallback` | helper/block: safe template yield before fallback | PIPELINE/SHARED | смотри конкретный guard/env | 2 |
| 6457 | `_validated_guardchain_recovery_candidate` | helper/block: validated guardchain recovery candidate | LEGACY/HELPER | нет явного флага | 4 |
| 6512 | `_recovery_candidate_from_informational_facts` | helper/block: recovery candidate from informational facts | LEGACY/HELPER | нет явного флага | 2 |
| 6546 | `_informational_fact_matches_question` | helper/block: informational fact matches question | LEGACY/HELPER | нет явного флага | 2 |
| 6589 | `find_unsupported_numeric_promises` | helper/block: find unsupported numeric promises | LEGACY/HELPER | нет явного флага | 3 |
| 6603 | `_is_verified_safe_numeric_template` | helper/block: is verified safe numeric template | LEGACY/HELPER | нет явного флага | 3 |
| 6636 | `apply_unconfirmed_operational_specificity_guard` | helper/block: apply unconfirmed operational specificity guard | PIPELINE/SHARED | смотри конкретный guard/env | 8 |
| 6688 | `find_unsupported_followup_deadline_claims` | helper/block: find unsupported followup deadline claims | LEGACY/HELPER | нет явного флага | 3 |
| 6696 | `find_unsupported_schedule_assumption_claims` | helper/block: find unsupported schedule assumption claims | LEGACY/HELPER | нет явного флага | 3 |
| 6704 | `find_unsupported_offline_visit_invitation_claims` | helper/block: find unsupported offline visit invitation claims | LEGACY/HELPER | нет явного флага | 3 |
| 6712 | `find_unsupported_content_delivery_action_claims` | helper/block: find unsupported content delivery action claims | LEGACY/HELPER | нет явного флага | 3 |
| 6720 | `_unsupported_claims_by_pattern` | helper/block: unsupported claims by pattern | LEGACY/HELPER | нет явного флага | 5 |
| 6734 | `_operational_specificity_guarded_result` | helper/block: operational specificity guarded result | LEGACY/HELPER | нет явного флага | 5 |
| 6754 | `apply_humanity_guards` | Final conversational guard: remove meta leaks and avoid useless handoff/repeats | LEGACY/PIPELINE | humanity/autonomy флаги | 3 |
| 6988 | `apply_humanity_x2_rewriter` | Optional X2 form rewrite after all deterministic draft guards | LEGACY/PIPELINE | humanity/autonomy флаги | 4 |
| 7031 | `apply_humanity_x2_rewriter.validate_candidate` | helper/block: validate candidate | LEGACY/HELPER | нет явного флага | 2 |
| 7034 | `apply_humanity_x2_rewriter.sanitize_candidate` | helper/block: sanitize candidate | LEGACY/HELPER | нет явного флага | 2 |
| 7070 | `apply_phase2_tone_layer` | helper/block: apply phase2 tone layer | ФЛАГ-OFF / PIPELINE | TELEGRAM_PH2_* | 4 |
| 7119 | `build_semantic_output_verifier_prompt` | helper/block: build semantic output verifier prompt | LEGACY/HELPER | нет явного флага | 2 |
| 7183 | `build_semantic_output_regen_prompt` | helper/block: build semantic output regen prompt | LEGACY/HELPER | нет явного флага | 2 |
| 7209 | `apply_semantic_output_verifier` | helper/block: apply semantic output verifier | PIPELINE/SHARED | смотри конкретный guard/env | 4 |
| 7344 | `_verifier_handoff_claims_enabled` | helper/block: verifier handoff claims enabled | LEGACY/HELPER | нет явного флага | 2 |
| 7352 | `_semantic_verifier_is_whitelisted_pure_handoff` | helper/block: semantic verifier is whitelisted pure handoff | LEGACY/HELPER | нет явного флага | 2 |
| 7368 | `_normalized_handoff_template_text` | helper/block: normalized handoff template text | LEGACY/HELPER | нет явного флага | 3 |
| 7372 | `_run_semantic_output_verifier_once` | helper/block: run semantic output verifier once | LEGACY/HELPER | нет явного флага | 4 |
| 7398 | `_semantic_output_findings_from_payload` | helper/block: semantic output findings from payload | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_SEMANTIC_OUTPUT_VERIFIER | 2 |
| 7436 | `_semantic_output_filter_findings` | helper/block: semantic output filter findings | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_SEMANTIC_OUTPUT_VERIFIER | 3 |
| 7449 | `_semantic_output_verifier_highest_action` | helper/block: semantic output verifier highest action | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_SEMANTIC_OUTPUT_VERIFIER | 2 |
| 7458 | `_normalize_semantic_relation` | helper/block: normalize semantic relation | LEGACY/HELPER | нет явного флага | 2 |
| 7465 | `_semantic_output_verifier_override` | helper/block: semantic output verifier override | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_SEMANTIC_OUTPUT_VERIFIER | 2 |
| 7472 | `_semantic_output_verifier_timeout_sec` | helper/block: semantic output verifier timeout sec | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_SEMANTIC_OUTPUT_VERIFIER | 3 |
| 7479 | `_llm_retrieve_timeout_sec` | helper/block: llm retrieve timeout sec | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ | TELEGRAM_LLM_RETRIEVE | 2 |
| 7486 | `build_semantic_diagnosis_prompt` | helper/block: build semantic diagnosis prompt | LEGACY/HELPER | нет явного флага | 2 |
| 7512 | `apply_semantic_diagnosis_guard` | helper/block: apply semantic diagnosis guard | PIPELINE/SHARED | смотри конкретный guard/env | 4 |
| 7580 | `_semantic_diagnosis_classifier_override` | helper/block: semantic diagnosis classifier override | LEGACY/HELPER | нет явного флага | 2 |
| 7587 | `_semantic_diagnosis_locked_deferral` | helper/block: semantic diagnosis locked deferral | LEGACY/HELPER | нет явного флага | 3 |
| 7599 | `_semantic_diagnosis_high_risk_flagged` | helper/block: semantic diagnosis high risk flagged | LEGACY/HELPER | нет явного флага | 2 |
| 7610 | `_semantic_diagnosis_plain_deferral_text` | helper/block: semantic diagnosis plain deferral text | LEGACY/HELPER | нет явного флага | 2 |
| 7624 | `_has_diagnosis_hedge_and_transfer` | helper/block: has diagnosis hedge and transfer | LEGACY/HELPER | нет явного флага | 3 |
| 7638 | `_hard_p0_in_client_text` | helper/block: hard p0 in client text | PIPELINE/SHARED | смотри конкретный guard/env | 2 |
| 7642 | `_phase2_tone_rewrite` | helper/block: phase2 tone rewrite | ФЛАГ-OFF / PIPELINE | TELEGRAM_PH2_* | 2 |
| 7665 | `_phase2_text_change_violation` | helper/block: phase2 text change violation | ФЛАГ-OFF / PIPELINE | TELEGRAM_PH2_* | 2 |
| 7695 | `_phase2_tone_rewrite_override` | helper/block: phase2 tone rewrite override | ФЛАГ-OFF / PIPELINE | TELEGRAM_PH2_* | 2 |
| 7703 | `_humanity_x2_identity_policy_locked` | helper/block: humanity x2 identity policy locked | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 7712 | `apply_subscription_policy_guards` | helper/block: apply subscription policy guards | PIPELINE/SHARED | смотри конкретный guard/env | 2 |
| 7746 | `apply_input_policy_guards` | helper/block: apply input policy guards | PIPELINE/SHARED | смотри конкретный guard/env | 6 |
| 7778 | `apply_high_risk_content_guards` | helper/block: apply high risk content guards | PIPELINE/SHARED | смотри конкретный guard/env | 5 |
| 7873 | `apply_high_risk_content_guards.cross_brand_guarded` | helper/block: cross brand guarded | PIPELINE/SHARED | смотри конкретный guard/env | 9 |
| 8190 | `apply_autonomy_matrix_guard` | helper/block: apply autonomy matrix guard | LEGACY/PIPELINE | humanity/autonomy флаги | 3 |
| 8206 | `apply_autonomy_matrix_guard.demote` | helper/block: demote | LEGACY/HELPER | нет явного флага | 9 |
| 8319 | `_is_verified_client_safe_template` | helper/block: is verified client safe template | PIPELINE/SHARED | смотри конкретный guard/env | 6 |
| 8363 | `_result_has_live_status_missing_fact` | helper/block: result has live status missing fact | LEGACY/HELPER | нет явного флага | 2 |
| 8381 | `_asks_live_status_or_booking_question` | helper/block: asks live status or booking question | LEGACY/HELPER | нет явного флага | 3 |
| 8401 | `_live_status_manager_check_text` | helper/block: live status manager check text | LEGACY/HELPER | нет явного флага | 2 |
| 8462 | `apply_payment_confirmation_guard` | helper/block: apply payment confirmation guard | PIPELINE/SHARED | смотри конкретный guard/env | 4 |
| 8485 | `apply_conversation_intent_plan_guard` | Align draft topic/route with the context-level conversation plan | PIPELINE/SHARED | смотри конкретный guard/env | 5 |
| 8588 | `_conversation_intent_plan` | helper/block: conversation intent plan | LEGACY/HELPER | нет явного флага | 19 |
| 8595 | `_answer_contract` | helper/block: answer contract | LEGACY/HELPER | нет явного флага | 3 |
| 8602 | `_answer_contract_green_template_reduction_enabled` | helper/block: answer contract green template reduction enabled | LEGACY/HELPER | нет явного флага | 2 |
| 8613 | `_conversation_plan_controls_green_templates` | helper/block: conversation plan controls green templates | LEGACY/HELPER | нет явного флага | 2 |
| 8670 | `_conversation_plan_template_blocked_by_substantive_answer` | helper/block: conversation plan template blocked by substantive answer | LEGACY/HELPER | нет явного флага | 2 |
| 8687 | `_normalize_for_template_decision` | helper/block: normalize for template decision | LEGACY/HELPER | нет явного флага | 5 |
| 8691 | `_looks_like_low_value_handoff_only` | helper/block: looks like low value handoff only | LEGACY/HELPER | нет явного флага | 3 |
| 8700 | `_looks_like_generic_template` | helper/block: looks like generic template | LEGACY/HELPER | нет явного флага | 3 |
| 8710 | `_draft_addresses_question` | helper/block: draft addresses question | LEGACY/HELPER | нет явного флага | 3 |
| 8739 | `_skip_missing_fact_template_by_answer_contract` | helper/block: skip missing fact template by answer contract | LEGACY/HELPER | нет явного флага | 2 |
| 8751 | `_compact_conversation_intent_plan_for_metadata` | helper/block: compact conversation intent plan for metadata | LEGACY/HELPER | нет явного флага | 2 |
| 8780 | `apply_known_context_redundant_question_guard` | Catch drafts that ask again for data already known from safe context. | PIPELINE/SHARED | смотри конкретный guard/env | 5 |
| 8825 | `apply_funnel_policy_guard` | helper/block: apply funnel policy guard | PIPELINE/SHARED | смотри конкретный guard/env | 6 |
| 8853 | `_known_context_repair_text` | Replace a repeated-data question with a useful answer that keeps known context. | LEGACY/HELPER | нет явного флага | 2 |
| 8907 | `_remove_repeated_known_data_questions` | helper/block: remove repeated known data questions | LEGACY/HELPER | нет явного флага | 2 |
| 8932 | `find_redundant_questions_for_known_context` | helper/block: find redundant questions for known context | LEGACY/HELPER | нет явного флага | 2 |
| 8959 | `apply_unstated_subject_guard` | helper/block: apply unstated subject guard | PIPELINE/SHARED | смотри конкретный guard/env | 6 |
| 8988 | `_unstated_subject_safe_text` | helper/block: unstated subject safe text | LEGACY/HELPER | нет явного флага | 2 |
| 9016 | `_allowed_subjects_from_context` | helper/block: allowed subjects from context | LEGACY/HELPER | нет явного флага | 2 |
| 9032 | `_subjects_from_retrieved_facts` | helper/block: subjects from retrieved facts | LEGACY/HELPER | нет явного флага | 2 |
| 9047 | `_retrieved_fact_matches_active_brand` | helper/block: retrieved fact matches active brand | PIPELINE/SHARED | смотри конкретный guard/env | 2 |
| 9058 | `_mentioned_subjects` | helper/block: mentioned subjects | LEGACY/HELPER | нет явного флага | 6 |
| 9067 | `known_context_fields` | helper/block: known context fields | LEGACY/HELPER | нет явного флага | 19 |
| 9106 | `_merge_known_context_fields` | helper/block: merge known context fields | LEGACY/HELPER | нет явного флага | 8 |
| 9130 | `_known_fields_from_text` | helper/block: known fields from text | LEGACY/HELPER | нет явного флага | 2 |
| 9155 | `apply_brand_separation_guard` | helper/block: apply brand separation guard | PIPELINE/SHARED | смотри конкретный guard/env | 6 |
| 9178 | `_is_unpk_bank_installment_question` | helper/block: is unpk bank installment question | LEGACY/HELPER | нет явного флага | 2 |
| 9197 | `apply_taxonomy_topic_guard` | helper/block: apply taxonomy topic guard | PIPELINE/SHARED | смотри конкретный guard/env | 2 |
| 9229 | `is_high_risk_result` | helper/block: is high risk result | LEGACY/HELPER | нет явного флага | 14 |
| 9246 | `detect_high_risk_input_markers` | helper/block: detect high risk input markers | LEGACY/HELPER | нет явного флага | 11 |
| 9257 | `_conversation_plan_semantic_non_p0` | helper/block: conversation plan semantic non p0 | PIPELINE/SHARED | смотри конкретный guard/env | 4 |
| 9265 | `_strip_false_p0_flags` | helper/block: strip false p0 flags | PIPELINE/SHARED | смотри конкретный guard/env | 3 |
| 9279 | `_answer_quality_was_rewritten` | helper/block: answer quality was rewritten | LEGACY/HELPER | нет явного флага | 2 |
| 9286 | `_is_refund_case` | helper/block: is refund case | LEGACY/HELPER | нет явного флага | 2 |
| 9290 | `_is_legal_threat_case` | helper/block: is legal threat case | LEGACY/HELPER | нет явного флага | 2 |
| 9294 | `_is_complaint_case` | helper/block: is complaint case | LEGACY/HELPER | нет явного флага | 2 |
| 9298 | `_is_reputation_only_case` | helper/block: is reputation only case | LEGACY/HELPER | нет явного флага | 2 |
| 9302 | `_is_combined_high_risk_case` | helper/block: is combined high risk case | LEGACY/HELPER | нет явного флага | 4 |
| 9318 | `_is_future_price_case` | helper/block: is future price case | LEGACY/HELPER | нет явного флага | 2 |
| 9341 | `_is_result_guarantee_case` | helper/block: is result guarantee case | LEGACY/HELPER | нет явного флага | 3 |
| 9368 | `_is_admission_guarantee_case` | helper/block: is admission guarantee case | LEGACY/HELPER | нет явного флага | 3 |
| 9395 | `_presale_refund_policy_template` | helper/block: presale refund policy template | LEGACY/HELPER | нет явного флага | 2 |
| 9491 | `_has_presale_refund_policy_context` | helper/block: has presale refund policy context | LEGACY/HELPER | нет явного флага | 2 |
| 9512 | `_is_enrollment_signup_question` | helper/block: is enrollment signup question | LEGACY/HELPER | нет явного флага | 3 |
| 9521 | `_is_enrollment_process_question` | helper/block: is enrollment process question | МЁРТВЫЙ-кандидат | нет; проверить перед сносом | 1 |
| 9531 | `_is_lesson_recording_question` | helper/block: is lesson recording question | LEGACY/HELPER | нет явного флага | 2 |
| 9542 | `_mentions_schedule_day_or_time` | helper/block: mentions schedule day or time | МЁРТВЫЙ-кандидат | нет; проверить перед сносом | 1 |
| 9547 | `_has_word_marker` | helper/block: has word marker | LEGACY/HELPER | нет явного флага | 3 |
| 9551 | `_asks_installment` | helper/block: asks installment | LEGACY/HELPER | нет явного флага | 2 |
| 9571 | `_asks_invoice_monthly_payment` | helper/block: asks invoice monthly payment | PIPELINE/SHARED | смотри конкретный guard/env | 2 |
| 9579 | `_price_question_explicitly_supersedes_installment` | helper/block: price question explicitly supersedes installment | МЁРТВЫЙ-кандидат | нет; проверить перед сносом | 1 |
| 9595 | `_manager_handoff_request_text` | helper/block: manager handoff request text | МЁРТВЫЙ-кандидат | нет; проверить перед сносом | 1 |
| 9608 | `_presale_refund_handoff_ack_text` | helper/block: presale refund handoff ack text | МЁРТВЫЙ-кандидат | нет; проверить перед сносом | 1 |
| 9625 | `_price_fix_process_text` | helper/block: price fix process text | МЁРТВЫЙ-кандидат | нет; проверить перед сносом | 1 |
| 9638 | `_enrollment_signup_process_text` | helper/block: enrollment signup process text | МЁРТВЫЙ-кандидат | нет; проверить перед сносом | 1 |
| 9656 | `_terminal_safe_template` | helper/block: terminal safe template | PIPELINE/SHARED | смотри конкретный guard/env | 3 |
| 9808 | `_cross_brand_safe_template` | helper/block: cross brand safe template | PIPELINE/SHARED | смотри конкретный guard/env | 3 |
| 9838 | `_scope_fact_missing_guard_template` | helper/block: scope fact missing guard template | LEGACY/HELPER | нет явного флага | 2 |
| 9859 | `_requested_fact_scope_context` | helper/block: requested fact scope context | LEGACY/HELPER | нет явного флага | 2 |
| 9875 | `_scope_guard_has_missing_intent_fact` | helper/block: scope guard has missing intent fact | LEGACY/HELPER | нет явного флага | 3 |
| 9894 | `_scope_guard_required_fact_keys` | helper/block: scope guard required fact keys | LEGACY/HELPER | нет явного флага | 3 |
| 9914 | `_scope_guard_missing_fact_keys` | helper/block: scope guard missing fact keys | LEGACY/HELPER | нет явного флага | 3 |
| 9937 | `_fact_key_root` | helper/block: fact key root | LEGACY/HELPER | нет явного флага | 3 |
| 9967 | `_scope_guard_has_foreign_concrete_fact` | helper/block: scope guard has foreign concrete fact | LEGACY/HELPER | нет явного флага | 2 |
| 9992 | `_scope_fact_detail_label` | helper/block: scope fact detail label | LEGACY/HELPER | нет явного флага | 4 |
| 10022 | `_scope_fact_narrow_handoff_text` | helper/block: scope fact narrow handoff text | LEGACY/HELPER | нет явного флага | 3 |
| 10036 | `_strict_antirepeat_fallback_text` | helper/block: strict antirepeat fallback text | LEGACY/HELPER | нет явного флага | 2 |
| 10057 | `_core_handoff_detail` | helper/block: core handoff detail | LEGACY/HELPER | нет явного флага | 2 |
| 10074 | `_is_core_handoff_fallback_repeat` | helper/block: is core handoff fallback repeat | LEGACY/HELPER | нет явного флага | 2 |
| 10087 | `_select_nonrepeating_text` | helper/block: select nonrepeating text | LEGACY/HELPER | нет явного флага | 8 |
| 10095 | `_p0_text_with_antirepeat` | helper/block: p0 text with antirepeat | PIPELINE/SHARED | смотри конкретный guard/env | 11 |
| 10110 | `_fact_scope_guard_template` | helper/block: fact scope guard template | LEGACY/HELPER | нет явного флага | 2 |
| 10166 | `_forbidden_pair_guard_template` | helper/block: forbidden pair guard template | LEGACY/HELPER | нет явного флага | 2 |
| 10180 | `_answer_fact_scopes` | helper/block: answer fact scopes | LEGACY/HELPER | нет явного флага | 3 |
| 10184 | `_missing_fact_helpful_template` | helper/block: missing fact helpful template | LEGACY/HELPER | нет явного флага | 2 |
| 10248 | `_has_missing_fact_signal` | helper/block: has missing fact signal | LEGACY/HELPER | нет явного флага | 3 |
| 10254 | `_context_has_missing_fact_signal` | helper/block: context has missing fact signal | LEGACY/HELPER | нет явного флага | 5 |
| 10270 | `_draft_is_low_value_without_exact_fact` | helper/block: draft is low value without exact fact | LEGACY/HELPER | нет явного флага | 3 |
| 10281 | `_promoted_verified_fact_text` | helper/block: promoted verified fact text | LEGACY/HELPER | нет явного флага | 2 |
| 10340 | `_confirmed_fact_texts` | helper/block: confirmed fact texts | LEGACY/HELPER | нет явного флага | 9 |
| 10358 | `_client_clean_fact_text` | helper/block: client clean fact text | LEGACY/HELPER | нет явного флага | 10 |
| 10377 | `_prefer_format_facts` | helper/block: prefer format facts | LEGACY/HELPER | нет явного флага | 2 |
| 10395 | `_ensure_sentence` | helper/block: ensure sentence | LEGACY/HELPER | нет явного флага | 4 |
| 10402 | `_defer_direct_process_to_format_choice_template` | helper/block: defer direct process to format choice template | МЁРТВЫЙ-кандидат | нет; проверить перед сносом | 1 |
| 10426 | `_format_choice_is_disjunctive_question` | helper/block: format choice is disjunctive question | LEGACY/HELPER | нет явного флага | 2 |
| 10434 | `_recent_format_choice_was_ambiguous` | helper/block: recent format choice was ambiguous | МЁРТВЫЙ-кандидат | нет; проверить перед сносом | 1 |
| 10447 | `_foton_offline_free_trial_guard_template` | helper/block: foton offline free trial guard template | LEGACY/HELPER | нет явного флага | 2 |
| 10479 | `_soften_current_price_deadline_text` | helper/block: soften current price deadline text | LEGACY/HELPER | нет явного флага | 4 |
| 10606 | `_dedupe_sentence` | helper/block: dedupe sentence | LEGACY/HELPER | нет явного флага | 2 |
| 10620 | `_without_known_grade_reask` | helper/block: without known grade reask | МЁРТВЫЙ-кандидат | нет; проверить перед сносом | 1 |
| 10644 | `_known_subject_or_format` | helper/block: known subject or format | МЁРТВЫЙ-кандидат | нет; проверить перед сносом | 1 |
| 10660 | `_known_grade_int` | helper/block: known grade int | МЁРТВЫЙ-кандидат | нет; проверить перед сносом | 1 |
| 10672 | `_autonomy_policy` | helper/block: autonomy policy | LEGACY/PIPELINE | humanity/autonomy флаги | 4 |
| 10684 | `_autonomy_enabled` | helper/block: autonomy enabled | LEGACY/PIPELINE | humanity/autonomy флаги | 5 |
| 10694 | `_default_autonomy_flip_enabled` | helper/block: default autonomy flip enabled | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 10709 | `_autonomy_topic_allowed` | helper/block: autonomy topic allowed | LEGACY/PIPELINE | humanity/autonomy флаги | 3 |
| 10722 | `RouteDecision` | helper/block: RouteDecision | LEGACY/HELPER | нет явного флага | 10 |
| 10731 | `decide_route` | Central route-permission decision point | LEGACY/HELPER | нет явного флага | 2 |
| 10799 | `_memory_followup_answered_topic` | helper/block: memory followup answered topic | LEGACY/HELPER | нет явного флага | 2 |
| 10820 | `_memory_text_items` | helper/block: memory text items | LEGACY/HELPER | нет явного флага | 4 |
| 10827 | `_memory_norm` | helper/block: memory norm | LEGACY/HELPER | нет явного флага | 6 |
| 10831 | `_memory_mentions_focus` | helper/block: memory mentions focus | LEGACY/HELPER | нет явного флага | 2 |
| 10841 | `_memory_short_followup` | helper/block: memory short followup | LEGACY/HELPER | нет явного флага | 2 |
| 10854 | `_memory_mentions_different_topic` | helper/block: memory mentions different topic | LEGACY/HELPER | нет явного флага | 2 |
| 10882 | `_memory_topic_aliases` | helper/block: memory topic aliases | LEGACY/HELPER | нет явного флага | 2 |
| 10912 | `_has_client_safe_current_fact` | helper/block: has client safe current fact | LEGACY/HELPER | нет явного флага | 3 |
| 10922 | `_mapping_has_client_safe_current_fact` | helper/block: mapping has client safe current fact | LEGACY/HELPER | нет явного флага | 5 |
| 10965 | `_humanity_p0_required` | helper/block: humanity p0 required | LEGACY/PIPELINE | humanity/autonomy флаги | 5 |
| 10977 | `_humanity_allows_dry_p0_text` | helper/block: humanity allows dry p0 text | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 10992 | `_humanity_previous_bot_texts` | helper/block: humanity previous bot texts | LEGACY/PIPELINE | humanity/autonomy флаги | 13 |
| 11014 | `_has_humanity_answer_fact` | helper/block: has humanity answer fact | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 11018 | `_humanity_block_a_route_fix_enabled` | helper/block: humanity block a route fix enabled | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 11029 | `_scope_fact_guard_enabled` | helper/block: scope fact guard enabled | LEGACY/HELPER | нет явного флага | 2 |
| 11040 | `_antirepeat_strict_enabled` | helper/block: antirepeat strict enabled | LEGACY/HELPER | нет явного флага | 2 |
| 11051 | `_humanity_can_trim_cosmetic_opening` | helper/block: humanity can trim cosmetic opening | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 11072 | `_trim_repeated_cosmetic_opening` | helper/block: trim repeated cosmetic opening | LEGACY/HELPER | нет явного флага | 2 |
| 11093 | `_humanity_block_a_direct_answer` | helper/block: humanity block a direct answer | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 11122 | `_humanity_presale_refund_rules_answer` | helper/block: humanity presale refund rules answer | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 11155 | `_humanity_unpk_address_confirmation_answer` | helper/block: humanity unpk address confirmation answer | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 11182 | `_humanity_unpk_tax_certificate_followup_answer` | helper/block: humanity unpk tax certificate followup answer | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 11207 | `_humanity_foton_bank_transfer_monthly_answer` | helper/block: humanity foton bank transfer monthly answer | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 11239 | `_humanity_unpk_weekend_address_answer` | helper/block: humanity unpk weekend address answer | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 11284 | `_humanity_generic_fact_answer_blocked` | Do not replace an unresolved operational question with a neighboring fact. | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 11318 | `_humanity_preserve_existing_answer` | helper/block: humanity preserve existing answer | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 11340 | `_humanity_guarded_handoff_reason` | helper/block: humanity guarded handoff reason | LEGACY/PIPELINE | humanity/autonomy флаги | 4 |
| 11362 | `_first_humanity_fact_text` | helper/block: first humanity fact text | LEGACY/PIPELINE | humanity/autonomy флаги | 3 |
| 11373 | `_humanity_fact_answer` | helper/block: humanity fact answer | LEGACY/PIPELINE | humanity/autonomy флаги | 5 |
| 11392 | `_humanity_precise_fact_answer` | helper/block: humanity precise fact answer | LEGACY/PIPELINE | humanity/autonomy флаги | 3 |
| 11406 | `_humanity_context_correction_answer` | helper/block: humanity context correction answer | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 11420 | `_humanity_weekend_schedule_no_format_lock_answer` | helper/block: humanity weekend schedule no format lock answer | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 11452 | `_humanity_discount_percent_answer` | helper/block: humanity discount percent answer | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 11475 | `_humanity_discount_percent_answer.matches_format` | helper/block: matches format | LEGACY/HELPER | нет явного флага | 2 |
| 11518 | `_humanity_installment_amount_answer` | helper/block: humanity installment amount answer | LEGACY/PIPELINE | humanity/autonomy флаги | 3 |
| 11545 | `_humanity_next_step` | helper/block: humanity next step | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 11558 | `_sanitize_humanity_meta_text` | helper/block: sanitize humanity meta text | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 11577 | `_semantic_haystack` | helper/block: semantic haystack | LEGACY/HELPER | нет явного флага | 4 |
| 11601 | `_dialog_context_haystack` | helper/block: dialog context haystack | LEGACY/HELPER | нет явного флага | 5 |
| 11618 | `_client_message_contains_pii` | helper/block: client message contains pii | LEGACY/HELPER | нет явного флага | 4 |
| 11627 | `_asks_money_price_question` | helper/block: asks money price question | МЁРТВЫЙ-кандидат | нет; проверить перед сносом | 4 |
| 11638 | `_is_generic_price_question_without_selection` | helper/block: is generic price question without selection | МЁРТВЫЙ-кандидат | нет; проверить перед сносом | 1 |
| 11649 | `_foton_online_price_text_from_facts` | helper/block: foton online price text from facts | LEGACY/HELPER | нет явного флага | 2 |
| 11671 | `_price_amount_from_facts` | helper/block: price amount from facts | LEGACY/HELPER | нет явного флага | 3 |
| 11694 | `_normalized_fact_text` | helper/block: normalized fact text | МЁРТВЫЙ-кандидат | нет; проверить перед сносом | 1 |
| 11698 | `_is_unpk_installment_case` | helper/block: is unpk installment case | LEGACY/HELPER | нет явного флага | 2 |
| 11746 | `_is_unpk_zvsh_case` | helper/block: is unpk zvsh case | LEGACY/HELPER | нет явного флага | 2 |
| 11767 | `_draft_confirms_payment` | helper/block: draft confirms payment | PIPELINE/SHARED | смотри конкретный guard/env | 2 |
| 11773 | `_payment_context` | helper/block: payment context | PIPELINE/SHARED | смотри конкретный guard/env | 2 |
| 11805 | `_payment_status` | helper/block: payment status | PIPELINE/SHARED | смотри конкретный guard/env | 3 |
| 11816 | `_payment_guarded_result` | helper/block: payment guarded result | PIPELINE/SHARED | смотри конкретный guard/env | 3 |
| 11827 | `_active_brand` | helper/block: active brand | PIPELINE/SHARED | смотри конкретный guard/env | 51 |
| 11841 | `_topic_id_from_context` | helper/block: topic id from context | LEGACY/HELPER | нет явного флага | 2 |
| 11853 | `_dialogue_contract_tone_guide` | helper/block: dialogue contract tone guide | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 11873 | `_dialogue_contract_style_examples` | helper/block: dialogue contract style examples | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 11893 | `_dialogue_contract_safety_flags` | helper/block: dialogue contract safety flags | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 11922 | `_sanitize_dialogue_contract_client_text` | helper/block: sanitize dialogue contract client text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 11968 | `_brand_guarded_result` | helper/block: brand guarded result | PIPELINE/SHARED | смотри конкретный guard/env | 3 |
| 12010 | `_extract_numeric_promise_claims` | helper/block: extract numeric promise claims | LEGACY/HELPER | нет явного флага | 3 |
| 12027 | `_fresh_fact_texts` | helper/block: fresh fact texts | LEGACY/HELPER | нет явного флага | 5 |
| 12067 | `_has_dialogue_contract_retrieved_facts` | helper/block: has dialogue contract retrieved facts | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 12073 | `_append_fact_texts` | helper/block: append fact texts | LEGACY/HELPER | нет явного флага | 6 |
| 12104 | `_claim_supported_by_facts` | helper/block: claim supported by facts | LEGACY/HELPER | нет явного флага | 5 |
| 12125 | `_keep_answer_supported` | helper/block: keep answer supported | LEGACY/HELPER | нет явного флага | 3 |
| 12141 | `_keep_answer_hard_anchors` | helper/block: keep answer hard anchors | LEGACY/HELPER | нет явного флага | 3 |
| 12152 | `_fact_match_anchors` | helper/block: fact match anchors | LEGACY/HELPER | нет явного флага | 6 |
| 12173 | `_fact_match_unit_anchors` | helper/block: fact match unit anchors | LEGACY/HELPER | нет явного флага | 2 |
| 12185 | `_fact_match_schedule_condition_anchors` | helper/block: fact match schedule condition anchors | LEGACY/HELPER | нет явного флага | 2 |
| 12200 | `_normalize_fact_match_text` | helper/block: normalize fact match text | LEGACY/HELPER | нет явного флага | 16 |
| 12205 | `_truthy_value` | helper/block: truthy value | LEGACY/HELPER | нет явного флага | 72 |
| 12211 | `_step4_keep_answer_enabled` | helper/block: step4 keep answer enabled | LEGACY/HELPER | нет явного флага | 6 |
| 12219 | `_output_sanitizer_enabled` | helper/block: output sanitizer enabled | LEGACY/HELPER | нет явного флага | 2 |
| 12227 | `_phase2_tone_enabled` | helper/block: phase2 tone enabled | ФЛАГ-OFF / PIPELINE | TELEGRAM_PH2_* | 2 |
| 12235 | `_phase2_objection_enabled` | helper/block: phase2 objection enabled | ФЛАГ-OFF / PIPELINE | TELEGRAM_PH2_* | 2 |
| 12243 | `_phase2_anxiety_enabled` | helper/block: phase2 anxiety enabled | ФЛАГ-OFF / PIPELINE | TELEGRAM_PH2_* | 2 |
| 12251 | `_semantic_diagnosis_guard_enabled` | helper/block: semantic diagnosis guard enabled | LEGACY/HELPER | нет явного флага | 4 |
| 12259 | `_semantic_output_verifier_enabled` | helper/block: semantic output verifier enabled | ФЛАГ-OFF / ПРЯМОЙ ПУТЬ+PIPELINE | TELEGRAM_SEMANTIC_OUTPUT_VERIFIER | 4 |
| 12269 | `_answer_quality_llm_rewrite_enabled` | helper/block: answer quality llm rewrite enabled | LEGACY/HELPER | нет явного флага | 3 |
| 12277 | `_answer_quality_llm_rewrite_mode` | helper/block: answer quality llm rewrite mode | LEGACY/HELPER | нет явного флага | 2 |
| 12285 | `_answer_quality_llm_polish_sales_enabled` | helper/block: answer quality llm polish sales enabled | LEGACY/HELPER | нет явного флага | 2 |
| 12312 | `_humanity_x2_rewrite_enabled` | helper/block: humanity x2 rewrite enabled | LEGACY/PIPELINE | humanity/autonomy флаги | 4 |
| 12320 | `_humanity_x2_rewrite_mode` | helper/block: humanity x2 rewrite mode | LEGACY/PIPELINE | humanity/autonomy флаги | 4 |
| 12330 | `_humanity_x2_confirmed_facts` | helper/block: humanity x2 confirmed facts | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 12340 | `_extract_humanity_x2_text` | helper/block: extract humanity x2 text | LEGACY/PIPELINE | humanity/autonomy флаги | 2 |
| 12378 | `_humanity_x2_repo_gate` | helper/block: humanity x2 repo gate | LEGACY/PIPELINE | humanity/autonomy флаги | 3 |
| 12418 | `subscription_llm_safety_contract` | helper/block: subscription llm safety contract | ТЕСТЫ/API-совместимость | нет | 1 |
| 12433 | `_clean_list` | helper/block: clean list | LEGACY/HELPER | нет явного флага | 22 |
| 12453 | `_clean_crm_recommendations` | helper/block: clean crm recommendations | LEGACY/HELPER | нет явного флага | 4 |
| 12473 | `_clamp_float` | helper/block: clamp float | LEGACY/HELPER | нет явного флага | 11 |
| 12481 | `_optional_text` | helper/block: optional text | LEGACY/HELPER | нет явного флага | 4 |
| 12488 | `_cache_key` | helper/block: cache key | LEGACY/HELPER | нет явного флага | 2 |
| 12495 | `_with_metadata` | helper/block: with metadata | LEGACY/HELPER | нет явного флага | 2 |
| 12499 | `_guard_cache_dir` | helper/block: guard cache dir | LEGACY/HELPER | нет явного флага | 2 |
| 12506 | `_is_retryable` | helper/block: is retryable | LEGACY/HELPER | нет явного флага | 3 |
| 12511 | `_CodexRetryableError` | helper/block: CodexRetryableError | LEGACY/HELPER | нет явного флага | 4 |

### `src/mango_mvp/channels/dialogue_contract_pipeline.py`
| строка | объект | назначение | статус | флаг | refs |
|---:|---|---|---|---|---:|
| 278 | `Subquestion` | helper/block: Subquestion | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 29 |
| 286 | `Subquestion.to_json_dict` | helper/block: to json dict | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 30 |
| 298 | `Slot` | helper/block: Slot | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 6 |
| 302 | `Slot.to_json_dict` | helper/block: to json dict | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 30 |
| 307 | `AnswerContract` | helper/block: AnswerContract | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 134 |
| 343 | `AnswerContract.composite_subquestions` | helper/block: composite subquestions | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 6 |
| 347 | `AnswerContract.needed_fact_keys` | helper/block: needed fact keys | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 69 |
| 350 | `AnswerContract.manager_only` | helper/block: manager only | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 174 |
| 353 | `AnswerContract.all_needed_fact_keys` | helper/block: all needed fact keys | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 12 |
| 359 | `AnswerContract.assertable_slots` | helper/block: assertable slots | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 8 |
| 362 | `AnswerContract.unsourced_slots` | helper/block: unsourced slots | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 5 |
| 365 | `AnswerContract.to_json_dict` | helper/block: to json dict | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 30 |
| 401 | `FactStore` | helper/block: FactStore | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 9 |
| 408 | `RetrievalResult` | helper/block: RetrievalResult | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 62 |
| 415 | `VerificationFinding` | helper/block: VerificationFinding | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 54 |
| 421 | `FaithfulnessClaim` | helper/block: FaithfulnessClaim | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 427 | `FaithfulnessClaim.to_json_dict` | helper/block: to json dict | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 30 |
| 437 | `FaithfulnessResult` | helper/block: FaithfulnessResult | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 13 |
| 444 | `FormFinding` | helper/block: FormFinding | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 9 |
| 450 | `DialogueContractPipelineResult` | helper/block: DialogueContractPipelineResult | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 37 |
| 490 | `DialogueContractPipelineResult.__post_init__` | helper/block: post init | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 520 | `_pipeline_reason_class` | helper/block: pipeline reason class | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 563 | `_force_draft_for_manager_reason_class` | helper/block: force draft for manager reason class | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 584 | `Toggles` | helper/block: Toggles | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 16 |
| 591 | `pipeline_enabled` | helper/block: pipeline enabled | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 597 | `faithfulness_shadow_enabled` | helper/block: faithfulness shadow enabled | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 603 | `faithfulness_shadow_record` | helper/block: faithfulness shadow record | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 612 | `faithfulness_shadow_events` | helper/block: faithfulness shadow events | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 621 | `_record_faithfulness_shadow` | helper/block: record faithfulness shadow | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 638 | `estimate_mode_enabled` | helper/block: estimate mode enabled | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 644 | `free_number_gate_enabled` | helper/block: free number gate enabled | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 6 |
| 656 | `number_gate_scope_aware_enabled` | helper/block: number gate scope aware enabled | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 662 | `travel_compose_enabled` | helper/block: travel compose enabled | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 8 |
| 668 | `quality_partial_yield_enabled` | helper/block: quality partial yield enabled | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 4 |
| 674 | `quality_thread_memory_enabled` | helper/block: quality thread memory enabled | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 680 | `quality_composite_enabled` | helper/block: quality composite enabled | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 4 |
| 690 | `quality_next_step_enabled` | helper/block: quality next step enabled | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 696 | `quality_clarify_scope_enabled` | helper/block: quality clarify scope enabled | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 702 | `quality_useful_handoff_enabled` | helper/block: quality useful handoff enabled | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 708 | `_normalize_warmth_mode` | helper/block: normalize warmth mode | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 713 | `build_conversation` | helper/block: build conversation | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 741 | `build_understanding_prompt` | helper/block: build understanding prompt | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 830 | `parse_contract` | helper/block: parse contract | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 908 | `_clean_planner_intent` | helper/block: clean planner intent | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 913 | `_clean_answer_mode` | helper/block: clean answer mode | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 5 |
| 917 | `_clean_estimate_domain` | helper/block: clean estimate domain | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 5 |
| 922 | `_clean_selling` | helper/block: clean selling | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 948 | `_clean_planner_slots` | helper/block: clean planner slots | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 963 | `_context_with_conversation_messages` | helper/block: context with conversation messages | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 985 | `understand` | helper/block: understand | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 1028 | `_augment_contract_with_memory_topic` | helper/block: augment contract with memory topic | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1069 | `_augment_contract_with_composite_course_camp` | helper/block: augment contract with composite course camp | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 1147 | `_mentions_regular_course_topic` | helper/block: mentions regular course topic | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 1155 | `_mentions_camp_topic` | helper/block: mentions camp topic | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 1159 | `_regular_course_composite_detail` | helper/block: regular course composite detail | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 1167 | `_regular_course_composite_keys` | helper/block: regular course composite keys | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 1182 | `_camp_composite_keys` | helper/block: camp composite keys | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 1201 | `_memory_focus_for_contract` | helper/block: memory focus for contract | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1231 | `_thread_memory_view_for_contract` | helper/block: thread memory view for contract | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 1246 | `_suppress_stale_thread_memory` | helper/block: suppress stale thread memory | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1315 | `_explicit_product_family_from_text` | helper/block: explicit product family from text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1326 | `_explicit_service_topic_from_text` | helper/block: explicit service topic from text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1337 | `_canonical_subject` | helper/block: canonical subject | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1356 | `_explicit_subject_from_text` | helper/block: explicit subject from text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 1364 | `_camp_scope_from_memory` | helper/block: camp scope from memory | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1370 | `_memory_slot_value` | helper/block: memory slot value | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 6 |
| 1376 | `_memory_slot_source` | helper/block: memory slot source | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1382 | `_memory_slot_source_allowed` | helper/block: memory slot source allowed | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1393 | `_contract_has_topic` | helper/block: contract has topic | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1397 | `_compose_topic_question` | helper/block: compose topic question | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1422 | `replace_contract_topic` | helper/block: replace contract topic | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1472 | `_memory_focus_value_for_contract` | helper/block: memory focus value for contract | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1490 | `_keys_for_topic` | helper/block: keys for topic | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 1532 | `_focus_aliases` | helper/block: focus aliases | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 9 |
| 1578 | `_contract_query_aliases` | helper/block: contract query aliases | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1594 | `_format_from_text` | helper/block: format from text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 7 |
| 1603 | `_grade_from_text` | helper/block: grade from text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 9 |
| 1608 | `_key_has_any_topic_alias` | helper/block: key has any topic alias | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 9 |
| 1627 | `_p0_latch_sources` | helper/block: p0 latch sources | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 1641 | `_latch_is_active` | helper/block: latch is active | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 1647 | `_first_p0_latch_reason` | helper/block: first p0 latch reason | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 1654 | `_has_presale_refund_evidence` | helper/block: has presale refund evidence | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 1673 | `_active_hard_p0_latch_reason` | helper/block: active hard p0 latch reason | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 1706 | `_presale_refund_latch_can_release` | helper/block: presale refund latch can release | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1718 | `_has_only_benign_refund_latch` | helper/block: has only benign refund latch | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1740 | `p0_pre_gate` | helper/block: p0 pre gate | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 12 |
| 1778 | `build_fact_store` | helper/block: build fact store | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1814 | `retrieve_facts` | helper/block: retrieve facts | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 1837 | `_resolve_answer_mode` | helper/block: resolve answer mode | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 1869 | `_is_product_question` | helper/block: is product question | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 1903 | `_estimate_policy_context` | helper/block: estimate policy context | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 1937 | `_quality_partial_yield_travel_domain` | helper/block: quality partial yield travel domain | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 1974 | `build_draft_prompt` | helper/block: build draft prompt | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 5 |
| 2038 | `build_estimate_prompt` | helper/block: build estimate prompt | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 3 |
| 2072 | `_format_memory_block` | helper/block: format memory block | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 2098 | `_format_established_topic_block` | helper/block: format established topic block | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 2113 | `_established_topic_from_context` | helper/block: established topic from context | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 2143 | `build_faithfulness_prompt` | helper/block: build faithfulness prompt | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 2187 | `check_claim_faithfulness` | helper/block: check claim faithfulness | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 2264 | `build_semantic_match_prompt` | helper/block: build semantic match prompt | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 2279 | `_semantic_match` | helper/block: semantic match | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 2311 | `_semantic_match_question_text` | helper/block: semantic match question text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 2325 | `_quality_handoff_estimate_domain` | helper/block: quality handoff estimate domain | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 3 |
| 2364 | `_semantic_recover_or_handoff` | helper/block: semantic recover or handoff | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 1 |
| 2440 | `_cite_only_recover_result_before_handoff` | helper/block: cite only recover result before handoff | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 8 |
| 2529 | `_quality_estimate_result_before_handoff` | helper/block: quality estimate result before handoff | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 3 |
| 2632 | `_estimate_number_gate_context` | helper/block: estimate number gate context | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 2656 | `_quality_next_step_result` | helper/block: quality next step result | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 9 |
| 2724 | `_has_next_step` | helper/block: has next step | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 2729 | `_quality_next_step_text` | helper/block: quality next step text | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 2750 | `_explicit_contract_next_step` | helper/block: explicit contract next step | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 2758 | `_clean_next_step_text` | helper/block: clean next step text | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 2770 | `_known_slot_value` | helper/block: known slot value | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 7 |
| 2836 | `claim_anchors_supported_by_fact` | helper/block: claim anchors supported by fact | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 2844 | `concrete_anchors` | helper/block: concrete anchors | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 10 |
| 2862 | `new_concrete_anchors` | helper/block: new concrete anchors | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 2869 | `unsupported_named_entities` | helper/block: unsupported named entities | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 2890 | `_entity_anchors` | helper/block: entity anchors | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 5 |
| 2906 | `_active_brand_entity_anchors` | helper/block: active brand entity anchors | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 2915 | `_is_safe_entity_echo` | helper/block: is safe entity echo | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 2920 | `_normalize_date_anchor` | helper/block: normalize date anchor | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 2944 | `form_check` | helper/block: form check | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 2961 | `build_warmth_prompt` | helper/block: build warmth prompt | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 2987 | `warmth_rewrite` | helper/block: warmth rewrite | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 3005 | `run_pipeline` | helper/block: run pipeline | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 4065 | `verify_output` | helper/block: verify output | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 6 |
| 4164 | `_answer_mode_number_findings` | helper/block: answer mode number findings | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 4207 | `_free_number_gate_findings` | helper/block: free number gate findings | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 4290 | `_ScopeAwareNumberFact` | helper/block: ScopeAwareNumberFact | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 6 |
| 4299 | `_scope_aware_number_facts` | helper/block: scope aware number facts | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 4323 | `_scope_aware_number_supported` | helper/block: scope aware number supported | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 4354 | `_scope_aware_number_fact_allowed` | helper/block: scope aware number fact allowed | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 4372 | `_number_scope_query_text` | helper/block: number scope query text | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 4405 | `_number_scope_fact_types` | helper/block: number scope fact types | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 4421 | `_scope_aware_product_number_context` | helper/block: scope aware product number context | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 3 |
| 4437 | `_client_number_context_text` | helper/block: client number context text | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 4457 | `_free_number_tokens` | helper/block: free number tokens | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 4461 | `_free_number_token_matches` | helper/block: free number token matches | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 4 |
| 4470 | `_free_number_surfaces` | helper/block: free number surfaces | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 5 |
| 4480 | `_payment_plan_count_surfaces_from_text` | helper/block: payment plan count surfaces from text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 4490 | `_payment_plan_count_surfaces_for_token` | helper/block: payment plan count surfaces for token | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 4512 | `_normalize_free_number_token` | helper/block: normalize free number token | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 3 |
| 4569 | `_normalize_decimal_surface` | helper/block: normalize decimal surface | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 5 |
| 4576 | `_multiply_thousand_surface` | helper/block: multiply thousand surface | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 4587 | `_is_free_product_number_context` | helper/block: is free product number context | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 4 |
| 4616 | `_is_free_structural_number` | helper/block: is free structural number | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 4631 | `_has_free_uncertainty_marker_near` | helper/block: has free uncertainty marker near | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 4651 | `_has_free_uncertainty_marker` | helper/block: has free uncertainty marker | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 4656 | `_ensure_estimate_uncertainty_marker` | helper/block: ensure estimate uncertainty marker | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 3 |
| 4665 | `_estimate_text_needs_uncertainty_marker` | helper/block: estimate text needs uncertainty marker | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 4681 | `_is_client_grade_number_context_at` | helper/block: is client grade number context at | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 3 |
| 4687 | `_free_number_context_window` | helper/block: free number context window | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 6 |
| 4701 | `_is_decimal_year_range` | helper/block: is decimal year range | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 4706 | `_free_number_word_surfaces` | helper/block: free number word surfaces | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 4716 | `_number_token_map` | helper/block: number token map | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 4727 | `_is_product_number_context` | helper/block: is product number context | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 4739 | `_is_route_estimate_number_context` | helper/block: is route estimate number context | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 3 |
| 4755 | `_is_client_grade_number_context` | helper/block: is client grade number context | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 4 |
| 4767 | `_has_uncertainty_marker` | helper/block: has uncertainty marker | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 4772 | `_general_advice_estimate_findings` | helper/block: general advice estimate findings | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 4782 | `_individual_child_diagnosis_findings` | helper/block: individual child diagnosis findings | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 4790 | `_estimate_gate_payload_from_context` | helper/block: estimate gate payload from context | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 4 |
| 4804 | `_gate_answer_mode` | helper/block: gate answer mode | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 4820 | `_gate_estimate_domain` | helper/block: gate estimate domain | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 4836 | `_gate_is_estimate` | helper/block: gate is estimate | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 4850 | `_preemptive_format_choice_finding` | helper/block: preemptive format choice finding | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 4874 | `_unconfirmed_schedule_finding` | helper/block: unconfirmed schedule finding | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 4897 | `_schedule_specificity_anchors` | helper/block: schedule specificity anchors | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 4906 | `_schedule_alias_present` | helper/block: schedule alias present | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 4913 | `_schedule_specificity_is_declined` | helper/block: schedule specificity is declined | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 4924 | `_self_contradiction_finding` | helper/block: self contradiction finding | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 4960 | `_discount_scope_anchors` | helper/block: discount scope anchors | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 4969 | `_hard_check` | helper/block: hard check | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 13 |
| 5038 | `_CoverageFinding` | helper/block: CoverageFinding | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 17 |
| 5045 | `_coverage_findings` | helper/block: coverage findings | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 5 |
| 5103 | `_answer_cites_fact` | helper/block: answer cites fact | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 5 |
| 5121 | `_coverage_value_anchors` | helper/block: coverage value anchors | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 5141 | `_coverage_terms` | helper/block: coverage terms | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 5161 | `_coverage_repair_prompt` | helper/block: coverage repair prompt | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 5181 | `_coverage_cite_only_answer` | helper/block: coverage cite only answer | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 5192 | `_key_coverage_cite_only_answer` | helper/block: key coverage cite only answer | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 5196 | `_coverage_cite_only_answer_from_findings` | helper/block: coverage cite only answer from findings | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 6 |
| 5214 | `_key_coverage_findings` | helper/block: key coverage findings | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 5243 | `_key_coverage_ok` | helper/block: key coverage ok | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 5255 | `_quality_composite_result_before_draft` | helper/block: quality composite result before draft | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 5384 | `_contract_subquestions` | helper/block: contract subquestions | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 7 |
| 5396 | `_composite_has_hard_p0_part` | helper/block: composite has hard p0 part | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 5408 | `_model_composite_candidate` | helper/block: model composite candidate | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 5472 | `_composite_candidate_from_parts` | helper/block: composite candidate from parts | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 5486 | `_partial_yield_result_before_handoff` | helper/block: partial yield result before handoff | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 3 |
| 5574 | `_partial_yield_full_check` | helper/block: partial yield full check | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 3 |
| 5624 | `_partial_yield_candidate` | helper/block: partial yield candidate | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 5642 | `_partial_yield_findings_and_missing` | helper/block: partial yield findings and missing | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 3 |
| 5695 | `_partial_yield_missing_text` | helper/block: partial yield missing text | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 3 |
| 5704 | `_composition_answer` | helper/block: composition answer | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 5717 | `_verified_empty_handoff_replacement` | helper/block: verified empty handoff replacement | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 5 |
| 5749 | `_cite_only_recover_before_handoff` | helper/block: cite only recover before handoff | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 3 |
| 5820 | `_stashed_recovery_candidate` | helper/block: stashed recovery candidate | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 5847 | `_original_failure_allows_cite_only_recover` | helper/block: original failure allows cite only recover | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 5856 | `_unsupported_item_is_missing_answer` | helper/block: unsupported item is missing answer | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 5861 | `_cite_only_recover_blocked` | helper/block: cite only recover blocked | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 9 |
| 5877 | `_exact_scope_cite_only_answer` | helper/block: exact scope cite only answer | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 5881 | `_exact_scope_coverage_findings` | helper/block: exact scope coverage findings | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 5912 | `_should_replace_empty_handoff` | helper/block: should replace empty handoff | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 5934 | `_draft_cites_any_retrieved_self_fact` | helper/block: draft cites any retrieved self fact | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 5962 | `_facts_with_derived_answer` | helper/block: facts with derived answer | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 7 |
| 5969 | `_compose_n_subjects_discount` | helper/block: compose n subjects discount | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 5993 | `_compose_nearest_camp_shift` | helper/block: compose nearest camp shift | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6023 | `_compose_price_plus_format` | helper/block: compose price plus format | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6047 | `_compose_installment_summary` | helper/block: compose installment summary | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6067 | `_requested_subject_count` | helper/block: requested subject count | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6090 | `_price_for_composition` | helper/block: price for composition | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6118 | `_second_subject_discount_pct` | helper/block: second subject discount pct | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6146 | `_first_money_amount` | helper/block: first money amount | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6154 | `_format_rub` | helper/block: format rub | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 6158 | `_unsupported_claims_without_current_fact_support` | helper/block: unsupported claims without current fact support | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 6175 | `_claim_supported_by_current_subquestion_fact` | helper/block: claim supported by current subquestion fact | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6191 | `_fact_matches_current_subquestion` | helper/block: fact matches current subquestion | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6235 | `_specific_semantic_topics` | helper/block: specific semantic topics | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 6240 | `_fact_key_matches_required_key` | helper/block: fact key matches required key | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6246 | `_semantic_topic_anchors` | helper/block: semantic topic anchors | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 6 |
| 6290 | `_handoff_factual_claim_text` | helper/block: handoff factual claim text | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 5 |
| 6312 | `_is_pure_handoff_text` | helper/block: is pure handoff text | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 5 |
| 6321 | `_dry_p0_text` | helper/block: dry p0 text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6328 | `_p0_handoff_text` | helper/block: p0 handoff text | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 6341 | `_p0_handoff_kind` | helper/block: p0 handoff kind | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 3 |
| 6357 | `_refund_policy_handoff_text` | helper/block: refund policy handoff text | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 3 |
| 6364 | `_payment_dispute_handoff_text` | helper/block: payment dispute handoff text | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 6381 | `_complaint_handoff_text` | helper/block: complaint handoff text | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 3 |
| 6388 | `_safe_fallback_text` | helper/block: safe fallback text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 16 |
| 6398 | `_safe_fallback_text_with_reason` | helper/block: safe fallback text with reason | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 6404 | `_safe_fallback_text_with_reason.traced` | helper/block: traced | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 13 |
| 6468 | `_safe_fallback_reason_is_punt` | helper/block: safe fallback reason is punt | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6472 | `_useful_handoff_text` | helper/block: useful handoff text | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 6492 | `_handoff_open_point_label` | helper/block: handoff open point label | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 6505 | `_client_safe_question_detail` | helper/block: client safe question detail | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 11 |
| 6527 | `_question_detail_topic_label` | helper/block: question detail topic label | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6554 | `_looks_like_raw_question_detail` | helper/block: looks like raw question detail | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6562 | `_secondary_fact_text` | helper/block: secondary fact text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6583 | `_partial_orientation_text` | helper/block: partial orientation text | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 3 |
| 6604 | `_fact_is_safe_partial_orientation` | helper/block: fact is safe partial orientation | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 6636 | `_is_contact_hours_fact` | helper/block: is contact hours fact | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 6643 | `_is_address_fact` | helper/block: is address fact | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6648 | `_generic_handoff_text` | helper/block: generic handoff text | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 6652 | `_detail_handoff_text` | helper/block: detail handoff text | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 6657 | `_short_fact_sentence` | helper/block: short fact sentence | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 16 |
| 6666 | `_avoid_repeating_text` | helper/block: avoid repeating text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 27 |
| 6701 | `_select_unused_handoff_variant` | helper/block: select unused handoff variant | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 6 |
| 6714 | `_is_refund_handoff_text` | helper/block: is refund handoff text | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 6719 | `_is_complaint_handoff_text` | helper/block: is complaint handoff text | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 2 |
| 6724 | `_is_handoff_text` | helper/block: is handoff text | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 10 |
| 6729 | `_looks_like_handoff` | helper/block: looks like handoff | ФЛАГ-OFF / PIPELINE | TELEGRAM_Q_* | 4 |
| 6743 | `_near_repeat` | helper/block: near repeat | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 6762 | `_repeat_norm` | helper/block: repeat norm | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 6766 | `_specialize_grade_range_answer` | helper/block: specialize grade range answer | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 6796 | `_client_grade_from_contract` | helper/block: client grade from contract | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6813 | `_augment_with_soft_guidance` | helper/block: augment with soft guidance | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6845 | `_augment_with_format_guidance` | helper/block: augment with format guidance | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6874 | `_augment_with_known_absence` | helper/block: augment with known absence | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6904 | `_augment_with_presale_refund_policy` | helper/block: augment with presale refund policy | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6937 | `_scope_camp_retrieval_for_contract` | helper/block: scope camp retrieval for contract | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 6974 | `_scope_required_retrieval_for_contract` | helper/block: scope required retrieval for contract | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7037 | `_asks_weekend_or_slot` | helper/block: asks weekend or slot | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 7049 | `_soft_weekend_guidance_text` | helper/block: soft weekend guidance text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 7057 | `_wrong_intent_fact_findings` | helper/block: wrong intent fact findings | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7088 | `_class_schedule_publication_answer` | helper/block: class schedule publication answer | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 7119 | `_format_context_prefix` | helper/block: format context prefix | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7131 | `_asks_training_format_choice` | helper/block: asks training format choice | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 6 |
| 7136 | `_asks_class_schedule_days` | helper/block: asks class schedule days | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 6 |
| 7146 | `_contract_intent_text` | helper/block: contract intent text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 22 |
| 7162 | `_draft_uses_contact_hours_as_schedule` | helper/block: draft uses contact hours as schedule | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7177 | `_draft_uses_address_fact` | helper/block: draft uses address fact | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7191 | `_draft_uses_camp_or_lvsh_fact` | helper/block: draft uses camp or lvsh fact | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7198 | `_camp_or_lvsh_facts` | helper/block: camp or lvsh facts | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 7212 | `_is_camp_or_lvsh_fact` | helper/block: is camp or lvsh fact | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 7 |
| 7217 | `_contract_mentions_camp_or_lvsh` | helper/block: contract mentions camp or lvsh | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 11 |
| 7221 | `_camp_scope_from_contract` | helper/block: camp scope from contract | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 7227 | `_camp_scope_from_text` | helper/block: camp scope from text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 5 |
| 7241 | `_camp_scope_from_fact` | helper/block: camp scope from fact | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 7245 | `_has_self_answerable_subquestion` | helper/block: has self answerable subquestion | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7249 | `_has_retrieved_self_answer_part` | helper/block: has retrieved self answer part | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 7264 | `_has_exact_retrieved_answer_part` | True only for a fact matched to the current subquestion and its scope | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 6 |
| 7292 | `_direct_exact_fact_answer` | helper/block: direct exact fact answer | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 7317 | `_hard_failure_exact_fact_fallback` | helper/block: hard failure exact fact fallback | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 7335 | `_can_autonomously_replace_failed_draft` | Only factual drift may be repaired into an autonomous exact-fact answer | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7345 | `_direct_price_answer_from_facts` | helper/block: direct price answer from facts | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 7392 | `_direct_format_answer_from_facts` | helper/block: direct format answer from facts | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 7415 | `_direct_camp_format_answer_from_facts` | helper/block: direct camp format answer from facts | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 7431 | `_direct_recording_answer_from_facts` | helper/block: direct recording answer from facts | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7450 | `_asks_address` | helper/block: asks address | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 7461 | `_asks_price` | helper/block: asks price | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 7473 | `_first_address_from_facts` | helper/block: first address from facts | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7489 | `_direct_payment_answer_from_facts` | helper/block: direct payment answer from facts | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 7508 | `_fact_tail` | helper/block: fact tail | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 7517 | `_retrieved_keys_match_question_scope` | helper/block: retrieved keys match question scope | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 6 |
| 7561 | `_matched_scope_fact_text_for_required_keys` | helper/block: matched scope fact text for required keys | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7574 | `_matched_scope_fact_keys_for_required_key` | helper/block: matched scope fact keys for required key | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 10 |
| 7587 | `_scope_matched_facts_for_contract` | helper/block: scope matched facts for contract | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 7603 | `_has_foreign_brand_matched_self_fact` | helper/block: has foreign brand matched self fact | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 7621 | `_fact_scope_matches_question` | helper/block: fact scope matches question | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 5 |
| 7646 | `_format_values_from_text` | helper/block: format values from text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 7 |
| 7656 | `_grade_values_from_fact_scope` | helper/block: grade values from fact scope | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 7684 | `_matched_fact_text_for_required_keys` | helper/block: matched fact text for required keys | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 7693 | `_matched_fact_mapping_for_required_keys` | helper/block: matched fact mapping for required keys | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7697 | `_subquestion_scope_text` | helper/block: subquestion scope text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 7712 | `_asks_refund_policy` | helper/block: asks refund policy | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 17 |
| 7727 | `_current_turn_asks_refund_policy` | helper/block: current turn asks refund policy | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 1 |
| 7744 | `_presale_refund_policy_text` | helper/block: presale refund policy text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 8 |
| 7752 | `_client_presale_refund_text` | helper/block: client presale refund text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7762 | `_dialogue_had_hard_p0_claim` | helper/block: dialogue had hard p0 claim | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 7788 | `_current_refund_dispute_signal` | helper/block: current refund dispute signal | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7822 | `_scope_clarification_question` | helper/block: scope clarification question | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7853 | `_format_values_from_facts` | helper/block: format values from facts | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7860 | `_grade_values_from_retrieved_facts` | helper/block: grade values from retrieved facts | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7867 | `_single_missing_slot_question` | helper/block: single missing slot question | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7885 | `_is_existence_yes_no_contract` | helper/block: is existence yes no contract | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 7891 | `_contract_existence_text` | helper/block: contract existence text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7899 | `_existence_target_anchors` | helper/block: existence target anchors | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 7913 | `_fact_has_existence_anchors` | helper/block: fact has existence anchors | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 7926 | `_is_negative_existence_fact_for_target` | helper/block: is negative existence fact for target | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 5 |
| 7933 | `_is_positive_existence_fact_for_target` | helper/block: is positive existence fact for target | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7942 | `_known_absence_text` | helper/block: known absence text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 7952 | `_existence_yes_no_findings` | helper/block: existence yes no findings | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 7989 | `_payment_method_target_anchors` | helper/block: payment method target anchors | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 8 |
| 8010 | `_payment_method_anchors_from_text` | helper/block: payment method anchors from text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 8 |
| 8026 | `_has_monthly_no_bank_support` | helper/block: has monthly no bank support | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 8033 | `_fact_supports_payment_target` | helper/block: fact supports payment target | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 8040 | `_payment_method_findings` | helper/block: payment method findings | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 8075 | `_repair_prompt` | helper/block: repair prompt | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 8086 | `_parse_subquestions` | helper/block: parse subquestions | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 8114 | `_normalize_question_type` | helper/block: normalize question type | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 8124 | `_clean_slots` | helper/block: clean slots | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 8143 | `_valid_contract_key` | helper/block: valid contract key | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 3 |
| 8154 | `_matched_fact_keys` | helper/block: matched fact keys | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 8164 | `_prioritize_catalog` | helper/block: prioritize catalog | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 8191 | `_key_mentions_current_text` | helper/block: key mentions current text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 8208 | `_snapshot_path_from_context` | helper/block: snapshot path from context | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 8218 | `_load_snapshot` | helper/block: load snapshot | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 8228 | `_snapshot_facts` | helper/block: snapshot facts | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 8235 | `_client_safe_fact` | helper/block: client safe fact | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 8244 | `_join_fact_text` | helper/block: join fact text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 8250 | `_fact_value_text` | helper/block: fact value text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 8259 | `_seq` | helper/block: seq | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 8 |
| 8267 | `_extract_json_object` | helper/block: extract json object | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 6 |
| 8283 | `_numbers` | helper/block: numbers | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 7 |
| 8288 | `_is_allowed_ungrounded_number` | helper/block: is allowed ungrounded number | ФЛАГ-OFF / PIPELINE | TELEGRAM_A_FREE_NUMBER_GATE / TELEGRAM_STEP4_NUMBER_GROUNDING / TELEGRAM_A_ESTIMATE_MODE | 2 |
| 8300 | `_sanitize_blocks` | helper/block: sanitize blocks | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 8312 | `_client_asked_identity` | helper/block: client asked identity | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 8316 | `_brand_token_present` | helper/block: brand token present | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |
| 8325 | `_normalize_brand` | helper/block: normalize brand | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 19 |
| 8334 | `_normalize_lookup` | helper/block: normalize lookup | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 5 |
| 8338 | `_clamp_float` | helper/block: clamp float | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 11 |
| 8345 | `_truthy` | helper/block: truthy | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 43 |
| 8349 | `_norm_text` | helper/block: norm text | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 4 |
| 8353 | `_similarity` | helper/block: similarity | PIPELINE | TELEGRAM_DIALOGUE_CONTRACT_PIPELINE | 2 |

### `src/mango_mvp/channels/dialogue_memory.py`
| строка | объект | назначение | статус | флаг | refs |
|---:|---|---|---|---|---:|
| 151 | `DialogueTurn` | helper/block: DialogueTurn | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 23 |
| 155 | `DialogueTurn.to_json_dict` | helper/block: to json dict | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 30 |
| 160 | `DialogueSlot` | helper/block: DialogueSlot | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 17 |
| 165 | `DialogueSlot.to_json_dict` | helper/block: to json dict | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 30 |
| 170 | `DialogueQuestion` | helper/block: DialogueQuestion | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 18 |
| 175 | `DialogueQuestion.to_json_dict` | helper/block: to json dict | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 30 |
| 180 | `DialogueP0Latch` | helper/block: DialogueP0Latch | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 17 |
| 189 | `DialogueP0Latch.to_json_dict` | helper/block: to json dict | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 30 |
| 202 | `DialogueMemory` | helper/block: DialogueMemory | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 40 |
| 232 | `DialogueMemory.to_json_dict` | helper/block: to json dict | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 30 |
| 264 | `DialogueMemory.to_prompt_view` | helper/block: to prompt view | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 11 |
| 300 | `build_dialogue_memory` | helper/block: build dialogue memory | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 5 |
| 391 | `update_dialogue_memory_after_answer` | helper/block: update dialogue memory after answer | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 4 |
| 475 | `_proactive_state_after_answer` | helper/block: proactive state after answer | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 483 | `update_memory_llm` | Optional post-answer memory enrichment | ФЛАГ-OFF / SIM | memory-mode codex/claude в симуляторе | 2 |
| 507 | `build_memory_llm_prompt` | helper/block: build memory llm prompt | ФЛАГ-OFF / SIM | memory-mode codex/claude в симуляторе | 2 |
| 537 | `_apply_memory_llm_update` | helper/block: apply memory llm update | ФЛАГ-OFF / SIM | memory-mode codex/claude в симуляторе | 2 |
| 575 | `_memory_llm_slots` | helper/block: memory llm slots | ФЛАГ-OFF / SIM | memory-mode codex/claude в симуляторе | 2 |
| 587 | `_merge_memory_llm_slots` | helper/block: merge memory llm slots | ФЛАГ-OFF / SIM | memory-mode codex/claude в симуляторе | 2 |
| 608 | `_memory_llm_can_override_slot` | helper/block: memory llm can override slot | ФЛАГ-OFF / SIM | memory-mode codex/claude в симуляторе | 2 |
| 624 | `_memory_llm_slot_supported_by_latest_client` | helper/block: memory llm slot supported by latest client | ФЛАГ-OFF / SIM | memory-mode codex/claude в симуляторе | 2 |
| 671 | `_latest_client_text` | helper/block: latest client text | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 678 | `_memory_llm_topic` | helper/block: memory llm topic | ФЛАГ-OFF / SIM | memory-mode codex/claude в симуляторе | 2 |
| 695 | `_memory_llm_open_question` | helper/block: memory llm open question | ФЛАГ-OFF / SIM | memory-mode codex/claude в симуляторе | 2 |
| 710 | `_memory_llm_commitments` | helper/block: memory llm commitments | ФЛАГ-OFF / SIM | memory-mode codex/claude в симуляторе | 2 |
| 720 | `_memory_llm_summary` | helper/block: memory llm summary | ФЛАГ-OFF / SIM | memory-mode codex/claude в симуляторе | 2 |
| 729 | `_coerce_turn` | helper/block: coerce turn | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 737 | `_extract_json_object` | helper/block: extract json object | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 6 |
| 758 | `dialogue_memory_from_mapping` | helper/block: dialogue memory from mapping | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 6 |
| 805 | `next_best_action_hint` | helper/block: next best action hint | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 3 |
| 820 | `safe_next_action` | Prompt metadata for current-terms requests; never promises booking or payment. | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 3 |
| 845 | `_parse_recent_messages` | helper/block: parse recent messages | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 864 | `_extract_slots_from_turns` | helper/block: extract slots from turns | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 869 | `_extract_slots_from_text` | helper/block: extract slots from text | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 3 |
| 880 | `_merge_slots` | helper/block: merge slots | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 6 |
| 916 | `_detect_open_question` | helper/block: detect open question | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 939 | `_detect_risk_flags` | helper/block: detect risk flags | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 4 |
| 943 | `_detect_commitments` | helper/block: detect commitments | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 3 |
| 955 | `_answer_closes_question` | helper/block: answer closes question | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 3 |
| 974 | `_needs_current_terms_action` | helper/block: needs current terms action | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 984 | `_first_missing_current_terms_slot` | helper/block: first missing current terms slot | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 992 | `_is_current_terms_question` | helper/block: is current terms question | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 3 |
| 1021 | `_sales_stage` | helper/block: sales stage | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 4 |
| 1039 | `_turns_from_previous` | helper/block: turns from previous | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1047 | `_slots_from_previous` | helper/block: slots from previous | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1055 | `_answered_questions` | helper/block: answered questions | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1063 | `_route_history` | helper/block: route history | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1071 | `_fact_refs` | helper/block: fact refs | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1082 | `_plain_str_mapping` | helper/block: plain str mapping | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 6 |
| 1088 | `_p0_latch_from_mapping` | helper/block: p0 latch from mapping | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1102 | `_next_p0_latch` | helper/block: next p0 latch | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 8 |
| 1226 | `_latchable_p0_codes` | helper/block: latchable p0 codes | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 4 |
| 1238 | `_primary_p0_risk` | helper/block: primary p0 risk | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1245 | `_p0_latch_release_event` | helper/block: p0 latch release event | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1255 | `_autonomous_p0_latch_release_event` | helper/block: autonomous p0 latch release event | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1274 | `_has_hard_p0_latch_code` | helper/block: has hard p0 latch code | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1278 | `_has_hard_p0_history_code` | helper/block: has hard p0 history code | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 3 |
| 1282 | `_p0_latch_released` | helper/block: p0 latch released | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1286 | `_previous_autonomous_p0_latch_released` | helper/block: previous autonomous p0 latch released | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1290 | `_slots_by_source` | helper/block: slots by source | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 5 |
| 1298 | `_do_not_reask_slots` | helper/block: do not reask slots | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 3 |
| 1302 | `_topic_focus` | helper/block: topic focus | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 3 |
| 1322 | `_unanswered_questions` | helper/block: unanswered questions | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1338 | `_safe_answered_parts_from_previous` | helper/block: safe answered parts from previous | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1346 | `_bot_inferred_slots` | helper/block: bot inferred slots | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1354 | `_pending_manager_actions` | helper/block: pending manager actions | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 6 |
| 1362 | `_safe_answered_parts` | helper/block: safe answered parts | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1383 | `_conversation_summary_short` | helper/block: conversation summary short | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1401 | `_open_loop_summary` | helper/block: open loop summary | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 4 |
| 1416 | `_risk_flags_from_safety` | helper/block: risk flags from safety | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1433 | `_stable_session_id` | helper/block: stable session id | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 1439 | `_normalize_format` | helper/block: normalize format | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 7 |
| 1448 | `_clean` | helper/block: clean | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 15 |

### `src/mango_mvp/channels/conversation_intent_plan.py`
| строка | объект | назначение | статус | флаг | refs |
|---:|---|---|---|---|---:|
| 26 | `ConversationIntentPlan` | helper/block: ConversationIntentPlan | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 3 |
| 62 | `ConversationIntentPlan.to_prompt_view` | helper/block: to prompt view | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 11 |
| 102 | `build_conversation_intent_plan` | Build an internal conversation plan | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 3 |
| 256 | `_selling_signals` | helper/block: selling signals | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 278 | `_has_price_objection_signal` | helper/block: has price objection signal | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 291 | `_has_exit_signal` | helper/block: has exit signal | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 301 | `_primary_intent` | helper/block: primary intent | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 382 | `_asks_live_availability` | helper/block: asks live availability | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 393 | `_asks_price_fix` | helper/block: asks price fix | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 401 | `_asks_price_without_installment_focus` | helper/block: asks price without installment focus | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 416 | `_keyword_signals` | helper/block: keyword signals | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 3 |
| 437 | `_is_payment_terms_question` | helper/block: is payment terms question | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 3 |
| 444 | `_asks_money_price_question` | helper/block: asks money price question | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 4 |
| 452 | `_camp_scope_signals` | Return city-day and residential-LVSH signals from the current message | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 468 | `_risk_signals` | helper/block: risk signals | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 479 | `_product_focus` | helper/block: product focus | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 515 | `_topic_switch_decision` | helper/block: topic switch decision | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 539 | `_topic_for_intent` | helper/block: topic for intent | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 571 | `_required_fact_keys` | helper/block: required fact keys | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 657 | `_fact_scope_constraints` | helper/block: fact scope constraints | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 680 | `_scope_from_roles` | helper/block: scope from roles | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 738 | `_scope_tuple` | helper/block: scope tuple | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 742 | `_answer_policy` | helper/block: answer policy | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 754 | `_requested_slots` | helper/block: requested slots | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 767 | `_next_step_hint` | helper/block: next step hint | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 787 | `_fact_query_text` | helper/block: fact query text | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 812 | `_decision_notes` | helper/block: decision notes | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 832 | `_intent_confidence` | helper/block: intent confidence | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 842 | `_direct_question` | helper/block: direct question | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 849 | `_merge_slots` | helper/block: merge slots | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 6 |
| 862 | `_extract_slots` | helper/block: extract slots | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 873 | `_format_from_roles` | helper/block: format from roles | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 881 | `_held_tagger_context` | helper/block: held tagger context | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 892 | `_held_active_fact_scope` | helper/block: held active fact scope | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 897 | `_held_active_topics` | helper/block: held active topics | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 905 | `_held_p0_latched` | helper/block: held p0 latched | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 910 | `_is_camp_followup_from_held` | helper/block: is camp followup from held | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 3 |
| 940 | `_camp_product_scope_from_fact_scope` | helper/block: camp product scope from fact scope | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 947 | `_roles_from_memory_view` | helper/block: roles from memory view | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 969 | `_slot_key` | helper/block: slot key | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 983 | `_slot_value` | helper/block: slot value | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 2 |
| 992 | `_normalize_format` | helper/block: normalize format | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 7 |
| 1001 | `_is_followup` | helper/block: is followup | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 5 |
| 1008 | `_normalize_brand` | helper/block: normalize brand | ПРЯМОЙ ПУТЬ+PIPELINE | нет; строится context builder | 19 |

### `src/mango_mvp/channels/fact_retrieval.py`
| строка | объект | назначение | статус | флаг | refs |
|---:|---|---|---|---|---:|
| 51 | `key_matches` | helper/block: key matches | ПРЯМОЙ ПУТЬ+PIPELINE | нет; контекст/гейты | 12 |
| 60 | `select_confirmed_facts` | helper/block: select confirmed facts | ПРЯМОЙ ПУТЬ+PIPELINE | нет; контекст/гейты | 2 |

### `src/mango_mvp/channels/fact_scope_spec.py`
| строка | объект | назначение | статус | флаг | refs |
|---:|---|---|---|---|---:|
| 138 | `normalize_scope_text` | helper/block: normalize scope text | ПРЯМОЙ ПУТЬ+PIPELINE | нет; контекст/гейты | 3 |
| 142 | `blocked_neighbors_for` | helper/block: blocked neighbors for | ПРЯМОЙ ПУТЬ+PIPELINE | нет; контекст/гейты | 5 |
| 149 | `scope_family_for` | helper/block: scope family for | ПРЯМОЙ ПУТЬ+PIPELINE | нет; контекст/гейты | 3 |
| 154 | `detect_fact_scopes` | helper/block: detect fact scopes | ПРЯМОЙ ПУТЬ+PIPELINE | нет; контекст/гейты | 9 |
| 196 | `fact_scopes_allowed` | helper/block: fact scopes allowed | ПРЯМОЙ ПУТЬ+PIPELINE | нет; контекст/гейты | 5 |
| 220 | `answer_scopes_allowed` | helper/block: answer scopes allowed | ПРЯМОЙ ПУТЬ+PIPELINE | нет; контекст/гейты | 5 |

### `src/mango_mvp/channels/telegram_pilot_context_builder.py`
| строка | объект | назначение | статус | флаг | refs |
|---:|---|---|---|---|---:|
| 95 | `KnowledgeSnapshotContext` | helper/block: KnowledgeSnapshotContext | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 5 |
| 104 | `build_telegram_pilot_context` | Build PilotContext for Telegram manager drafts from a compact KC snapshot. | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 274 | `_contextual_message_for_knowledge_lookup` | helper/block: contextual message for knowledge lookup | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 292 | `_contextual_message_with_recent_product` | helper/block: contextual message with recent product | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 330 | `build_telegram_pilot_context_from_snapshot` | Compatibility wrapper for dry-run scripts created during the KB build. | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 357 | `build_knowledge_snapshot_context` | helper/block: build knowledge snapshot context | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 521 | `merge_theme_and_rop_policy` | helper/block: merge theme and rop policy | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 543 | `required_fact_keys_for_message` | helper/block: required fact keys for message | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 637 | `_missing_snapshot_context` | helper/block: missing snapshot context | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 665 | `_load_snapshot` | helper/block: load snapshot | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 4 |
| 684 | `_snapshot_version` | helper/block: snapshot version | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 700 | `_chunk_records` | helper/block: chunk records | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 735 | `_select_confirmed_facts` | helper/block: select confirmed facts | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 828 | `_effective_recall_blocked_scopes` | helper/block: effective recall blocked scopes | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 849 | `_missing_fact_keys` | helper/block: missing fact keys | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 900 | `_knowledge_snippets` | helper/block: knowledge snippets | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 912 | `_manager_pattern_snippets` | helper/block: manager pattern snippets | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 943 | `_chunk_matches_context` | helper/block: chunk matches context | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 967 | `_chunk_matches_scope_only` | helper/block: chunk matches scope only | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 992 | `_record_matches_scope_only` | helper/block: record matches scope only | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 6 |
| 1007 | `_scope_match_text` | helper/block: scope match text | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 1032 | `_record_matches_context` | helper/block: record matches context | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 1082 | `_record_matches_retrieval_core` | helper/block: record matches retrieval core | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 1114 | `_record_matches_fact_scope` | helper/block: record matches fact scope | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 4 |
| 1137 | `_record_fact_scopes` | helper/block: record fact scopes | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 1141 | `_record_matches_product_markers` | helper/block: record matches product markers | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 1159 | `_record_is_objection_pattern` | helper/block: record is objection pattern | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 1163 | `_record_is_unrequested_special_product` | helper/block: record is unrequested special product | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 1185 | `_record_matches_requested_format` | helper/block: record matches requested format | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 1199 | `_record_matches_requested_class` | helper/block: record matches requested class | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 1213 | `_query_class_numbers` | helper/block: query class numbers | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 1222 | `_query_asks_for_date` | helper/block: query asks for date | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 1226 | `_record_answers_date_question` | helper/block: record answers date question | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 1239 | `_expand_required_fact_types` | helper/block: expand required fact types | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 1259 | `_fact_match_score` | helper/block: fact match score | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 1314 | `_record_mentions_number_or_range` | helper/block: record mentions number or range | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 1326 | `_looks_like_date_fact` | helper/block: looks like date fact | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 1336 | `_normalize_match_text` | helper/block: normalize match text | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 11 |
| 1340 | `_usable_for_precise_answer` | helper/block: usable for precise answer | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 1351 | `_record_allowed_for_active_brand` | helper/block: record allowed for active brand | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 4 |
| 1370 | `_fact_text` | helper/block: fact text | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 6 |
| 1380 | `_fact_types` | helper/block: fact types | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 9 |
| 1389 | `_topic_id` | helper/block: topic id | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 6 |
| 1403 | `_records` | helper/block: records | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 6 |
| 1413 | `_text_list` | helper/block: text list | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 13 |
| 1425 | `_query_terms` | helper/block: query terms | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 1429 | `_has_precise_claim` | helper/block: has precise claim | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 1437 | `_stable_fact_key` | helper/block: stable fact key | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 1444 | `_stable_status` | helper/block: stable status | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 4 |
| 1448 | `_clean_text` | helper/block: clean text | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 45 |
| 1453 | `_clip_text` | helper/block: clip text | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 3 |
| 1460 | `_truthy` | helper/block: truthy | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 43 |
| 1466 | `_normalize_active_brand` | helper/block: normalize active brand | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 4 |
| 1475 | `_brand_neutral_text_is_safe` | helper/block: brand neutral text is safe | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 2 |
| 1493 | `_dedupe` | helper/block: dedupe | ПРЯМОЙ ПУТЬ+PIPELINE | нет; сбор контекста до бота | 10 |

### `scripts/run_telegram_dynamic_client_sim.py`
| строка | объект | назначение | статус | флаг | refs |
|---:|---|---|---|---|---:|
| 58 | `DynamicSimInput` | helper/block: DynamicSimInput | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 64 | `FakeClientModel` | helper/block: FakeClientModel | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 65 | `FakeClientModel.generate` | helper/block: generate | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 20 |
| 73 | `FakeJudgeModel` | helper/block: FakeJudgeModel | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 2 |
| 74 | `FakeJudgeModel.generate` | helper/block: generate | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 20 |
| 95 | `FakeMemoryModel` | helper/block: FakeMemoryModel | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 96 | `FakeMemoryModel.generate` | helper/block: generate | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 20 |
| 106 | `FakeSemanticMatchModel` | helper/block: FakeSemanticMatchModel | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 107 | `FakeSemanticMatchModel.generate` | helper/block: generate | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 20 |
| 111 | `FakeSemanticOutputVerifierModel` | helper/block: FakeSemanticOutputVerifierModel | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 112 | `FakeSemanticOutputVerifierModel.generate` | helper/block: generate | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 20 |
| 116 | `FakeSellingComposeModel` | helper/block: FakeSellingComposeModel | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 117 | `FakeSellingComposeModel.generate` | helper/block: generate | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 20 |
| 121 | `FakeBotProvider` | helper/block: FakeBotProvider | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 122 | `FakeBotProvider.build_draft` | helper/block: build draft | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 8 |
| 138 | `LlmCallCounter` | helper/block: LlmCallCounter | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 5 |
| 139 | `LlmCallCounter.__init__` | helper/block: init | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 12 |
| 143 | `LlmCallCounter.increment` | helper/block: increment | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 150 | `LlmCallCounter.snapshot` | helper/block: snapshot | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 40 |
| 155 | `CountingGenerateModel` | helper/block: CountingGenerateModel | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 156 | `CountingGenerateModel.__init__` | helper/block: init | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 12 |
| 161 | `CountingGenerateModel.generate` | helper/block: generate | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 20 |
| 167 | `maybe_counting_model` | helper/block: maybe counting model | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 9 |
| 173 | `CountingSubscriptionLlmDraftProvider` | helper/block: CountingSubscriptionLlmDraftProvider | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 174 | `CountingSubscriptionLlmDraftProvider.__init__` | helper/block: init | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 12 |
| 178 | `CountingSubscriptionLlmDraftProvider._count_llm_call` | helper/block: count llm call | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 15 |
| 182 | `CountingSubscriptionLlmDraftProvider._dialogue_contract_understanding_runner` | helper/block: dialogue contract understanding runner | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 4 |
| 186 | `CountingSubscriptionLlmDraftProvider._dialogue_contract_draft_runner` | helper/block: dialogue contract draft runner | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 4 |
| 190 | `CountingSubscriptionLlmDraftProvider._dialogue_contract_repair_runner` | helper/block: dialogue contract repair runner | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 4 |
| 194 | `CountingSubscriptionLlmDraftProvider._dialogue_contract_warmth_runner` | helper/block: dialogue contract warmth runner | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 198 | `CountingSubscriptionLlmDraftProvider._dialogue_contract_faithfulness_runner` | helper/block: dialogue contract faithfulness runner | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 5 |
| 202 | `CountingSubscriptionLlmDraftProvider._dialogue_contract_semantic_match_runner` | helper/block: dialogue contract semantic match runner | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 4 |
| 206 | `CountingSubscriptionLlmDraftProvider._semantic_diagnosis_guard_runner` | helper/block: semantic diagnosis guard runner | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 5 |
| 210 | `CountingSubscriptionLlmDraftProvider._semantic_output_verifier_runner` | helper/block: semantic output verifier runner | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 6 |
| 214 | `CountingSubscriptionLlmDraftProvider._semantic_output_regen_runner` | helper/block: semantic output regen runner | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 6 |
| 218 | `CountingSubscriptionLlmDraftProvider._answer_quality_llm_rewrite_runner` | helper/block: answer quality llm rewrite runner | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 4 |
| 234 | `CountingSubscriptionLlmDraftProvider._humanity_x2_rewrite_runner` | helper/block: humanity x2 rewrite runner | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 5 |
| 238 | `CountingSubscriptionLlmDraftProvider._run_once` | helper/block: run once | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 4 |
| 242 | `CountingSubscriptionLlmDraftProvider._direct_path_draft_runner` | helper/block: direct path draft runner | ИЗМЕРЕНИЕ DIRECT | TELEGRAM_DIRECT_PATH* | 4 |
| 246 | `CountingSubscriptionLlmDraftProvider._direct_path_llm_retrieve_runner` | helper/block: direct path llm retrieve runner | ИЗМЕРЕНИЕ DIRECT | TELEGRAM_DIRECT_PATH* | 4 |
| 251 | `CodexJsonModel` | helper/block: CodexJsonModel | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 7 |
| 252 | `CodexJsonModel.__init__` | helper/block: init | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 12 |
| 267 | `CodexJsonModel.generate` | helper/block: generate | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 20 |
| 306 | `_claude_reasoning_args` | helper/block: claude reasoning args | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 313 | `_claude_env` | helper/block: claude env | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 317 | `build_claude_print_command` | helper/block: build claude print command | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 350 | `ClaudeJsonModel` | helper/block: ClaudeJsonModel | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 1 |
| 351 | `ClaudeJsonModel.__init__` | helper/block: init | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 12 |
| 366 | `ClaudeJsonModel.generate` | helper/block: generate | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 20 |
| 387 | `ClaudeCliRunner` | helper/block: ClaudeCliRunner | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 388 | `ClaudeCliRunner.__init__` | helper/block: init | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 12 |
| 406 | `ClaudeCliRunner.drain_events` | helper/block: drain events | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 412 | `ClaudeCliRunner.__call__` | helper/block: call | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 1 |
| 454 | `_claude_cli_event_if_visible_failure` | helper/block: claude cli event if visible failure | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 479 | `_claude_cli_stage_from_requested_cmd` | helper/block: claude cli stage from requested cmd | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 491 | `_shell_join` | helper/block: shell join | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 495 | `_tail_compact` | helper/block: tail compact | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 500 | `_format_claude_cli_event_log` | helper/block: format claude cli event log | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 518 | `_consume_claude_cli_events` | helper/block: consume claude cli events | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 528 | `_claude_model_from_args` | helper/block: claude model from args | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 538 | `main` | helper/block: main | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 4 |
| 871 | `write_dynamic_outputs` | helper/block: write dynamic outputs | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 4 |
| 914 | `load_dynamic_sim_input` | helper/block: load dynamic sim input | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 935 | `load_transcripts` | helper/block: load transcripts | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 945 | `build_client_model` | helper/block: build client model | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 959 | `build_judge_model` | helper/block: build judge model | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 4 |
| 973 | `build_memory_model` | helper/block: build memory model | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 4 |
| 990 | `build_semantic_match_model` | helper/block: build semantic match model | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 1007 | `build_semantic_output_verifier_model` | helper/block: build semantic output verifier model | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 1029 | `build_selling_compose_model` | helper/block: build selling compose model | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 1051 | `build_bot_provider` | helper/block: build bot provider | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 1088 | `run_one_dialog_isolated` | helper/block: run one dialog isolated | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 1118 | `_dialog_completed` | helper/block: dialog completed | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 1125 | `_should_rerun_existing_dialog` | helper/block: should rerun existing dialog | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 1140 | `build_infra_error_dialog` | helper/block: build infra error dialog | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 1170 | `sort_transcripts_by_persona_order` | helper/block: sort transcripts by persona order | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 1183 | `extract_judge_results` | helper/block: extract judge results | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 3 |
| 1191 | `_truthy_env_value` | helper/block: truthy env value | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 1195 | `_direct_path_fail_fast_enabled` | helper/block: direct path fail fast enabled | ИЗМЕРЕНИЕ DIRECT | TELEGRAM_DIRECT_PATH* | 2 |
| 1201 | `_dialog_direct_model_called` | helper/block: dialog direct model called | ИЗМЕРЕНИЕ DIRECT | TELEGRAM_DIRECT_PATH* | 3 |
| 1211 | `_direct_path_config_invalid` | helper/block: direct path config invalid | ИЗМЕРЕНИЕ DIRECT | TELEGRAM_DIRECT_PATH* | 4 |
| 1242 | `build_turn_rows` | helper/block: build turn rows | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 4 |
| 1313 | `attach_context_facts_to_dialog` | helper/block: attach context facts to dialog | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 1375 | `run_one_dialog` | helper/block: run one dialog | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 1575 | `build_bot_prompt_context` | helper/block: build bot prompt context | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 1655 | `build_client_prompt` | helper/block: build client prompt | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 1679 | `normalize_judge_prompt_version` | helper/block: normalize judge prompt version | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 5 |
| 1686 | `_is_judge_prompt_v9` | helper/block: is judge prompt v9 | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 6 |
| 1690 | `judge_prompt_version_id` | helper/block: judge prompt version id | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 6 |
| 1694 | `build_judge_prompt` | helper/block: build judge prompt | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 3 |
| 1759 | `_semantic_verifier_block_for_judge` | helper/block: semantic verifier block for judge | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 2 |
| 1772 | `_judge_v9_rules` | helper/block: judge v9 rules | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 2 |
| 1790 | `judge_dialog` | helper/block: judge dialog | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 3 |
| 1816 | `_should_reask_judge_gates` | helper/block: should reask judge gates | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 2 |
| 1829 | `_apply_judge_gate_reask` | helper/block: apply judge gate reask | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 2 |
| 1866 | `build_judge_gate_reask_prompt` | helper/block: build judge gate reask prompt | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 2 |
| 2017 | `normalize_judge_result` | helper/block: normalize judge result | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 2 |
| 2059 | `_normalize_judge_gate_list` | helper/block: normalize judge gate list | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 4 |
| 2069 | `_normalize_judge_gate_name` | helper/block: normalize judge gate name | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 2 |
| 2079 | `_needs_judge_gate_inference` | helper/block: needs judge gate inference | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 3 |
| 2083 | `_infer_failed_hard_gates` | helper/block: infer failed hard gates | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2096 | `_infer_hard_gates_from_text` | helper/block: infer hard gates from text | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2116 | `_truthy_judge_value` | helper/block: truthy judge value | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 2 |
| 2126 | `_normalize_first_failing_turn` | helper/block: normalize first failing turn | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2136 | `build_human_review_rows` | helper/block: build human review rows | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2179 | `review_priority` | helper/block: review priority | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 5 |
| 2196 | `manual_check_hint` | helper/block: manual check hint | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 4 |
| 2215 | `hard_gate_cause` | helper/block: hard gate cause | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 5 |
| 2242 | `hard_gate_cause_evidence` | helper/block: hard gate cause evidence | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 4 |
| 2261 | `dialog_number_audit_levels` | helper/block: dialog number audit levels | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 4 |
| 2275 | `dialog_judge_fact_audit_levels` | helper/block: dialog judge fact audit levels | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 3 |
| 2289 | `dialog_number_audit_worst_level` | helper/block: dialog number audit worst level | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 2 |
| 2298 | `_claude_cli_error_summary` | helper/block: claude cli error summary | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2326 | `_turn_fallback_reason_summary` | helper/block: turn fallback reason summary | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2338 | `_turn_primary_fallback_reason` | helper/block: turn primary fallback reason | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 2353 | `_turn_authoritative_gate_reason` | helper/block: turn authoritative gate reason | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2366 | `_manager_deferral_summary` | helper/block: manager deferral summary | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2409 | `_close_detect_summary` | helper/block: close detect summary | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2436 | `_tone_sell_prompt_summary` | helper/block: tone sell prompt summary | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2461 | `_non_p0_self_route_transcripts` | helper/block: non p0 self route transcripts | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2474 | `_is_non_p0_self_tone_turn` | helper/block: is non p0 self tone turn | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2487 | `_rich_format_summary` | helper/block: rich format summary | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2522 | `_turn_fact_keys` | helper/block: turn fact keys | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2541 | `_text_composition_source_summary` | helper/block: text composition source summary | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2585 | `build_summary` | helper/block: build summary | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2764 | `_scenario_metadata` | helper/block: scenario metadata | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2790 | `_semantic_output_verifier_summary` | helper/block: semantic output verifier summary | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2853 | `_semantic_verifier_deduped_by_deterministic_gate` | helper/block: semantic verifier deduped by deterministic gate | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2872 | `_llm_call_summary` | helper/block: llm call summary | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2894 | `_branch_count_metrics` | helper/block: branch count metrics | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 2922 | `judge_fact_audit_summary` | helper/block: judge fact audit summary | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 2 |
| 2973 | `_over_handoff_metrics` | helper/block: over handoff metrics | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3010 | `_handoff_trace_enabled` | helper/block: handoff trace enabled | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 3014 | `_handoff_trace_summary` | helper/block: handoff trace summary | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3037 | `_handoff_trace_summary_reason` | helper/block: handoff trace summary reason | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3041 | `_handoff_trace_summary_fallback_reason` | helper/block: handoff trace summary fallback reason | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3045 | `_handoff_trace_for_turn` | helper/block: handoff trace for turn | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3089 | `_handoff_trace_layer_guard` | helper/block: handoff trace layer guard | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3136 | `_handoff_trace_reason` | helper/block: handoff trace reason | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3169 | `_is_over_handoff_turn` | helper/block: is over handoff turn | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 3185 | `_turn_is_real_p0` | helper/block: turn is real p0 | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3193 | `_handoff_fact_level` | helper/block: handoff fact level | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 4 |
| 3209 | `_turn_has_retrieved_match_for_contract` | helper/block: turn has retrieved match for contract | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3238 | `_summary_scope_exact` | helper/block: summary scope exact | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3273 | `_fact_key_matches_required` | helper/block: fact key matches required | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3284 | `_send_unedited_proxy` | helper/block: send unedited proxy | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3309 | `build_metric_intervals` | helper/block: build metric intervals | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3364 | `wilson_interval` | helper/block: wilson interval | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 4 |
| 3374 | `tone_stats` | helper/block: tone stats | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 1 |
| 3395 | `compact_confirmed_facts` | helper/block: compact confirmed facts | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 3412 | `_dialogue_contract_metadata_from_result` | helper/block: dialogue contract metadata from result | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3420 | `_direct_path_metadata_from_result` | helper/block: direct path metadata from result | ИЗМЕРЕНИЕ DIRECT | TELEGRAM_DIRECT_PATH* | 2 |
| 3437 | `_manager_deferral_metadata_from_result` | helper/block: manager deferral metadata from result | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3497 | `_reason_class_from_runtime_channels` | helper/block: reason class from runtime channels | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3541 | `_authoritative_output_gate_metadata_from_result` | helper/block: authoritative output gate metadata from result | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3575 | `_semantic_output_verifier_metadata_from_result` | helper/block: semantic output verifier metadata from result | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3618 | `_authoritative_gate_finding_codes` | helper/block: authoritative gate finding codes | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 4 |
| 3632 | `facts_for_judge` | Facts visible to judge should match what the bot actually retrieved | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 2 |
| 3671 | `_compact_fact_for_judge` | helper/block: compact fact for judge | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 3 |
| 3681 | `_compact_dialogue_contract_for_judge` | helper/block: compact dialogue contract for judge | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 2 |
| 3694 | `audit_number_claims` | helper/block: audit number claims | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 3 |
| 3774 | `extract_number_claims` | helper/block: extract number claims | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 6 |
| 3813 | `ignored_number_spans` | helper/block: ignored number spans | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3828 | `normalize_audit_number` | helper/block: normalize audit number | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 4 |
| 3853 | `normalize_date_claim` | helper/block: normalize date claim | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3882 | `claim_matches_text` | helper/block: claim matches text | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3898 | `_number_claim_index_key` | helper/block: number claim index key | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 4 |
| 3906 | `snapshot_number_index` | helper/block: snapshot number index | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 3931 | `worst_number_audit_level` | helper/block: worst number audit level | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 2 |
| 3939 | `_filter_judge_confirmed_facts` | helper/block: filter judge confirmed facts | ИЗМЕРЕНИЕ/СУДЬЯ | --judge-prompt-version v2|v9 | 2 |
| 3951 | `compact_knowledge_snippets` | helper/block: compact knowledge snippets | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 3968 | `render_summary_md` | helper/block: render summary md | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 4012 | `render_full_transcripts_md` | helper/block: render full transcripts md | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 4022 | `render_one_dialog_md` | helper/block: render one dialog md | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 4084 | `known_dialog_fields_from_client_messages` | helper/block: known dialog fields from client messages | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 4123 | `write_jsonl` | helper/block: write jsonl | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 4129 | `write_csv` | helper/block: write csv | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 4153 | `write_human_review_csv` | helper/block: write human review csv | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |
| 4181 | `format_list` | helper/block: format list | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 9 |
| 4185 | `safe_filename` | helper/block: safe filename | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 3 |
| 4190 | `extract_json_object` | helper/block: extract json object | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 18 |
| 4208 | `_codex_env` | helper/block: codex env | ИЗМЕРЕНИЕ/СИМУЛЯТОР | CLI/env | 2 |

### `scripts/m1_watcher.py`
| строка | объект | назначение | статус | флаг | refs |
|---:|---|---|---|---|---:|
| 77 | `WatcherError` | helper/block: WatcherError | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 49 |
| 78 | `WatcherError.__init__` | helper/block: init | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 12 |
| 86 | `TaskSpec` | helper/block: TaskSpec | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 10 |
| 99 | `RunOutcome` | helper/block: RunOutcome | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 5 |
| 112 | `utc_now` | helper/block: utc now | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 8 |
| 116 | `sha256_file` | helper/block: sha256 file | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 5 |
| 124 | `write_text_atomic` | helper/block: write text atomic | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 5 |
| 131 | `write_json_atomic` | helper/block: write json atomic | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 5 |
| 135 | `_tail_text` | helper/block: tail text | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 8 |
| 140 | `run_command_probe` | helper/block: run command probe | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 161 | `_strip_quotes` | helper/block: strip quotes | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 3 |
| 170 | `parse_task_yaml` | helper/block: parse task yaml | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 3 |
| 200 | `validate_task_dict` | helper/block: validate task dict | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 264 | `effective_task_env` | helper/block: effective task env | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 4 |
| 271 | `validate_set_path` | helper/block: validate set path | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 292 | `load_json` | helper/block: load json | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 6 |
| 299 | `verify_bundle_manifest` | helper/block: verify bundle manifest | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 337 | `parse_bundle_info` | helper/block: parse bundle info | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 351 | `M1Watcher` | helper/block: M1Watcher | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 352 | `M1Watcher.__init__` | helper/block: init | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 12 |
| 382 | `M1Watcher.inbox` | helper/block: inbox | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 3 |
| 386 | `M1Watcher.running` | helper/block: running | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 17 |
| 390 | `M1Watcher.done` | helper/block: done | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 5 |
| 394 | `M1Watcher.failed` | helper/block: failed | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 21 |
| 398 | `M1Watcher.status_path` | helper/block: status path | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 402 | `M1Watcher.logs_dir` | helper/block: logs dir | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 3 |
| 405 | `M1Watcher.ensure_layout` | helper/block: ensure layout | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 418 | `M1Watcher.load_state` | helper/block: load state | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 4 |
| 422 | `M1Watcher.save_state` | helper/block: save state | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 5 |
| 425 | `M1Watcher._today_log_path` | helper/block: today log path | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 4 |
| 428 | `M1Watcher._log_event` | helper/block: log event | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 5 |
| 440 | `M1Watcher._log_tail` | helper/block: log tail | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 446 | `M1Watcher.write_status` | helper/block: write status | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 13 |
| 466 | `M1Watcher._heartbeat_path` | helper/block: heartbeat path | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 3 |
| 469 | `M1Watcher._heartbeat_counter` | helper/block: heartbeat counter | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 478 | `M1Watcher._write_heartbeat` | helper/block: write heartbeat | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 12 |
| 499 | `M1Watcher._cli_versions` | helper/block: cli versions | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 510 | `M1Watcher._base_run_env` | helper/block: base run env | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 516 | `M1Watcher._readiness_checks` | helper/block: readiness checks | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 553 | `M1Watcher.unacked_executed_count` | helper/block: unacked executed count | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 566 | `M1Watcher.terminal_report_exists` | helper/block: terminal report exists | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 569 | `M1Watcher.running_tasks` | helper/block: running tasks | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 572 | `M1Watcher._process_alive` | helper/block: process alive | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 581 | `M1Watcher._handle_existing_running` | helper/block: handle existing running | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 595 | `M1Watcher._task_signature_stable` | helper/block: task signature stable | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 607 | `M1Watcher._ready_marker_ok` | helper/block: ready marker ok | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 617 | `M1Watcher._parse_wait_or_fail` | helper/block: parse wait or fail | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 636 | `M1Watcher._parse_task_spec_or_wait` | helper/block: parse task spec or wait | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 650 | `M1Watcher._next_task_path` | helper/block: next task path | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 675 | `M1Watcher._claim` | helper/block: claim | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 693 | `M1Watcher._fail_inbox_task` | helper/block: fail inbox task | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 7 |
| 707 | `M1Watcher._finish_running_task` | helper/block: finish running task | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 713 | `M1Watcher._write_report` | helper/block: write report | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 6 |
| 771 | `M1Watcher._wait_or_fail_bundle` | helper/block: wait or fail bundle | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 784 | `M1Watcher._deploy_bundle` | helper/block: deploy bundle | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 805 | `M1Watcher._build_command` | helper/block: build command | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 834 | `M1Watcher._run_subprocess` | helper/block: run subprocess | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 874 | `M1Watcher._execute` | helper/block: execute | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 975 | `M1Watcher.process_once` | helper/block: process once | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 3 |
| 994 | `M1Watcher.loop` | helper/block: loop | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 2 |
| 1000 | `main` | helper/block: main | ИНФРА/ШИНА M1 | task yaml + PRODUCTION_ENV_STACK | 4 |

