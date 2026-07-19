# ADR-003 Live Direct-Path Understanding Map

Сгенерировано: 2026-07-05

Репозиторий: `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff`

HEAD: `843c0b844b7029f59b2d0dec4c29f3f61d938d22`

Ветка: `codex/adr003-semanticframe-migration`

Статус: историческая карта. На 2026-07-19 владелец отказался от DCP fallback; `dialogue_contract_pipeline.py`, `humanity_guards.py`, `rules_engine.py` и `answer_quality_rewriter.py` удалены после выноса живого output-floor в `output_verification_floor.py`.

## Короткий вывод

Мы еще не избавились от regex/marker-понимания полностью.

Что уже доказано:

- Живой `direct-path` возвращает результат без старого монолитного конвейера. Владелец окончательно отказался от fallback; legacy-хвост, `answer_quality_rewriter`, `humanity_guards`, DCP и rules-engine удалены, а реально вызываемый output-floor сохранён отдельно.
- Срезы `rewrite_quality`, `post_semantics` и `route_templates/redundant_guard` в локальной trace-диагностике дали 0 записей на 173 ходах. Для live direct-path их нельзя считать проверенными и нельзя строить для них apply-режим.
- Живая точка `route_templates/autonomy_matrix` исполняется: в trace-диагностике 153 сравнимых записи, 1 расхождение. Расхождение безопасное: legacy выбрал более осторожный `manager_only` на оплатном/P0-соседнем ходе.

Что остается:

- В живом пути все еще есть несколько regex/marker/floor-слоев. Часть из них можно дальше переводить на SemanticFrame, но часть является safety-floor и не должна удаляться как "понимание" без отдельного safety-дизайна.

## Источники проверки

- Graphify: карта свежая на `843c0b84`; использована только как навигация.
- Source of truth: сырой код текущего HEAD.
- Trace-отчет: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-04_REPORT_srezy2_4_trace_agreement_D1.md`.
- Audit pack: `audits/_inbox/adr003_package2_slices_2_4_20260704182000/`.

## Фактический живой путь

Под `pilot_gold_v1` direct-path включается профилем: `src/mango_mvp/channels/subscription_llm_parts/direct_path.py:274-281`.

Фактический порядок в `SubscriptionLlmDraftProvider.build_draft`:

1. `provider.py:962` проверяет `_direct_path_enabled(context)`.
2. `provider.py:963` вызывает `_build_direct_path_draft(...)`.
3. `provider.py:964-966` при включенном deal-action применяет `_direct_path_autonomy_matrix_topic_result(...)` и `apply_autonomy_matrix_guard(...)`.
4. `provider.py:967-989` применяет direct-path post-layers и возвращает `apply_semantic_reading_trace_finalize(...)`.
5. Все, что ниже этого return, не исполняется на direct-path.

Старый монолитный хвост начинается после direct-path return. В частности:

- `apply_answer_quality_rewriter(...)` удалён вместе с legacy-хвостом в Пакете 2.
Старый хвост и перечисленные только в нём вызовы удалены. Источник истины для текущих строк — сырой код актуального HEAD, а не номера строк этой исторической карты.

## Direct-path pipeline

| Шаг | Где | Что делает | Статус для live direct-path |
|---|---|---|---|
| Direct-path enable | `direct_path.py:274-281` | Включает direct-path по профилю/флагу | live, routing switch |
| P0/high-risk/policy preblock до модели | `provider.py:1101-1113`, `provider.py:1127-1137`, `post_layers.py:1006-1216` | Может вернуть `manager_only`/`draft_for_manager` без вызова LLM: P0, high-risk, reliable-answerer bypass, cross-brand, force-manager, unknown-brand | live, floor |
| Fact selection / retrieval | `provider.py:1114-1118`, `direct_path.py:753-840`, `direct_path.py:1000-1103`, `direct_path.py:1529-2030`, `direct_path.py:2032-2104` | Выбирает категории/факты, проверяет slot conflicts, LLM-retriever, venue scope, fallback; `_direct_path_context_fact_pack` - входная функция | live, business-understanding + scope/safety |
| Prompt build | `provider.py:1139-1142`, `direct_path.py:581-720`, `direct_path.py:1622-1712`, `direct_path.py:2328-2450`, `direct_path.py:2599-2825` | Собирает prompt: факты, bot-safe память, known slots, do-not-reask, slot/topic shadow metadata, reliable block, инструкции, SemanticFrame-поля | live, mixed |
| Direct LLM call | `provider.py:1153-1185` | Получает JSON-черновик | live, model understanding |
| Route rubric regen | `provider.py:1186-1201`, `direct_path.py:3105-3121` | Один повторный вызов, если `draft_for_manager` без `missing_facts` при фактах или fallback open question | live, model+rule mixed |
| Model P0 route | `provider.py:1202-1206`, `provider.py:2507-2555` | Поднимает route до `manager_only`, если модель/floor видит P0 | live, safety mixed |
| P0 text hygiene | `provider.py:1207-1211`, `text_hygiene.py:113-185`, `text_hygiene.py:222-427` | Чистит текст по refund/payment/complaint/legal, различает presale refund / refund claim / payment dispute / forward payment | live, safety floor |
| Conversation intent plan guard | `provider.py:1212-1220`, `policy_routing.py:3432-3458` | Legacy/LLM comparison for route/action; после `да №5` `intent_actions` входит в профильный default, а legacy output-ветка live_availability удалена | live, mixed |
| Reliable answerer guard | `provider.py:1221-1225`, `direct_path.py:2611-2614`, `direct_path.py:2641`, `direct_path.py:2794`, `reliable_answerer.py:149-244`, `reliable_answerer.py:303-419` | Строит план покрытия вопроса, вставляет prompt-block и блокирует обещания мест/групп без live-факта, trace `sense_seats` | live, safety floor + trace |
| Assumed scope guard | `provider.py:1226`, `direct_path.py:2773-2779`, `direct_path.py:3049-3102` | Не дает утверждать неподтвержденные параметры как клиентские; `2773-2779` - prompt-инструкция, `3049-3102` - runtime guard | live, safety/memory |
| Semantic output verifier | `provider.py:1228-1234`, `post_layers.py:5064-5201` | Финальная проверка вывода/выдумок/claim issues | live, output safety |
| Bot-safe memory step guard | `provider.py:1235`, `post_layers.py:4394-4435` | Не дает утверждать следующий шаг из памяти, если статус требует проверки менеджером | live, memory safety floor |
| Authoritative output gate | `provider.py:1236-1238`, `post_layers.py:2677-2804` | Финальный gate; только понижает/блокирует, не улучшает | live, final floor |
| Deal action decision layer | `provider.py:967-971`, `post_layers.py:1641-1710` | Превращает action proposal/model/facts в action decision; no live execution | live, mixed |
| Direct keyword fallback reask | `provider.py:972`, `direct_path.py:3165-3205` | Может заменить draft_for_manager на уточняющий вопрос при fallback | live, understanding/reask |
| Tone close detect | `provider.py:973`, `post_layers.py:2224-2284` | Теплое закрытие, suppress для P0/pending | live, tone+regex |
| Second P0 text scrub | `provider.py:974-978`, `text_hygiene.py:113-148` | Повторная защита после deal/reask/close | live, safety floor |
| SemanticFrame posthoc/gates | `provider.py:979-988`, `provider.py:1915-1952`, `provider.py:3103-3429` | Posthoc shadow extraction; existence/proof/self-answer/decision в основном shadow; manager-action gate при отдельном флаге может реально понизить autonomous route до `draft_for_manager` | live, mostly shadow; one optional active gate |
| Reading trace finalize | `provider.py:989`, `provider.py:3432-3441` | Финализирует trace-метаданные | live, telemetry |

## Live understanding/floor inventory

### 1. P0/high-risk/policy preblock

- Код: `post_layers.py:1006-1216`.
- Источники смысла: `dialogue_contract_p0_pre_gate(...)`, `detect_high_risk_input_markers(...)`, дополнительные complaint backstop условия, reliable-answerer bypass, cross-brand bypass, `force_manager_only`, unknown-brand.
- Живость: вызывается до LLM из `_build_direct_path_draft(...)` в `provider.py:1101-1113` и `provider.py:1127-1137`.
- Тип: `floor`.
- Что делать: не удалять как "регекс-лазанью". Это safety-floor. Возможна калибровка только отдельным P0-проектом с отдельными воротами.

### 2. Fact selection / retrieval

- Код: `direct_path.py:753-840`, `direct_path.py:1000-1103`, `direct_path.py:1529-2030`, `direct_path.py:2032-2104`.
- Источники смысла: category aliases, fact/category regex, grade/format/slot conflict checks, LLM retriever, requested venue scope, exact/adjacent fact selection.
- Живость: `_direct_path_context_fact_pack(...)` вызывается перед prompt build в `provider.py:1114-1118`.
- Тип: `business-understanding + scope/safety`.
- Replacement-кандидат: да, но не простым удалением regex. Это слой выбора фактов; ошибка здесь меняет весь downstream ответ. Нужна отдельная B/ON-пара по fact selection.

### 3. Direct-path prompt understanding

- Код: `direct_path.py:581-720`, `direct_path.py:1622-1712`, `direct_path.py:2328-2450`, `direct_path.py:2609-2641`, `direct_path.py:2794-2806`.
- Источники смысла: LLM-инструкции для route, P0, model_intent, SemanticFrame, action_proposal, dialog_summary, known slots, bot-safe CRM memory, reliable-answerer block, slot/topic shadow metadata.
- Живость: prompt строится перед direct LLM call в `provider.py:1139-1154`.
- Тип: `model understanding`.
- Что делать: это целевой источник понимания по ADR-003, но внутри prompt уже есть safety/coverage рамки до LLM. Расширять можно, но нельзя смешивать модельное понимание с hard floors и нельзя превращать inferred slots в client-confirmed slots.

### 4. Gold-example topic hints

- Код: `direct_path.py:2424-2450`, `direct_path.py:2479-2597`.
- Источники смысла: suppression по known slots / active next step, keyword hints и few-shot examples для выбора темы примеров.
- Живость: `_direct_path_select_gold_real_examples(...)` вызывается перед prompt build в `provider.py:1141`.
- Тип: `understanding helper`, но не route/floor.
- Риск: может влиять на качество/тон модели через подбор примеров; suppression означает, что влияние зависит от уже известных слотов и текущего шага, а не только от текста вопроса.
- Replacement-кандидат: можно позже заменить на SemanticFrame/topic metadata, но это не первый приоритет, потому что не принимает route-решение напрямую.

### 5. Model P0 route + P0 text hygiene

- Код: `provider.py:2507-2555`, `text_hygiene.py:35-84`, `text_hygiene.py:113-185`, `text_hygiene.py:222-427`.
- Источники смысла: модельное `is_p0/p0_kind`, floor reason, regex scrub refund/payment/legal/complaint, presale refund exception, `forward_payment`, `payment_dispute`, `refund_claim`, `p0_kind` normalization.
- Живость: применяется внутри direct-path после model result и повторно после close/reask.
- Тип: `mixed`, но safety-critical.
- Что делать: не удалять сейчас. Payment/refund split уже отдельный трек; P0/floor не трогать до поздней фазы.

### 6. Conversation intent plan

- Код: `conversation_intent_plan.py:146-230`, `conversation_intent_plan.py:300-527`, `conversation_intent_plan.py:590-619`; direct-path применяет guard в `provider.py:1212-1220`.
- Источники смысла: `semantic_roles`, known slots, current message, previous focus, held state, keyword/risk signals.
- Живость: plan участвует в `required_fact_keys`, `fact_scope`, `answer_topics`, `route_bias`, prompt и guard.
- Тип: `mixed`.
- Replacement-кандидат: частично уже заменяется SemanticReading/SemanticFrame (`slots_gsf`, `intent_actions`; `route_templates` остаётся отдельным apply-кандидатом). Нужна поузловая резка, не массовое удаление.

### 7. Semantic roles

- Код: `semantic_roles.py:21-82`, `semantic_roles.py:261-330`, `semantic_roles.py:341-420`, `semantic_roles.py:450-533`, `semantic_roles.py:557-587`.
- Источники смысла: marker tables и helper'ы для payment/refund/transfer/format/negation/camp/schedule/topic roles.
- Живость: `conversation_intent_plan.py:166` вызывает `tag_message_roles(...)`.
- Тип: `understanding helper`.
- Replacement-кандидат: да, но это не один детектор. Разные helper'ы отвечают за разные риски; резать можно только после покрытия SemanticFrame для соответствующих полей и regression-set на refund/payment/transfer/format/negation/camp.

### 8. Dialogue memory / slots / next step

- Код: `dialogue_memory.py:45-73`, `dialogue_memory.py:308-488`, `dialogue_memory.py:960-1052`, `dialogue_memory.py:1102-1255`, `dialogue_memory.py:1408-1512`, `dialogue_memory.py:1688-1848`, `dialogue_memory.py:1888-1934`, `dialogue_memory.py:1969-1995`.
- Источники смысла: question_kind markers, confirmed slot provenance, topic focus, `do_not_reask`, commitments, p0 latch/release, pending manager facts, next action.
- Живость: direct prompt читает known slots / do-not-reask / memory view в `direct_path.py:2328-2450`; `conversation_intent_plan.py` тоже читает memory.
- Тип: `mixed memory`.
- Replacement-кандидат: не "удалить regex", а переводить происхождение слотов на `semantic_reading_llm` с provenance. При этом нужно сохранить P0 latch/release, confirmed slot provenance, `do_not_reask`, topic focus и pending-manager facts; это safety/memory contract, а не просто понимание.

### 9. Reliable answerer / sense_seats

- Код: `reliable_answerer.py:37-96`, `reliable_answerer.py:149-244`, `reliable_answerer.py:303-419`, `direct_path.py:2611-2614`, `direct_path.py:2641`, `direct_path.py:2794`, `policy_routing.py:3039-3047`.
- Источники смысла: facet regex, coverage plan, prompt block, availability-promise regex, P0/cross-brand bypass markers, preservation of partial live-status answer, SemanticReading trace.
- Живость: direct-path вызывает `apply_reliable_answerer_output_guard(...)` в `provider.py:1221-1225`.
- Тип: `safety floor + understanding trace`.
- Replacement-кандидат: `sense_seats` уже подключен к SemanticReading trace, но floor шире trace: он держит prompt coverage и post-filter. Удалять floor нельзя, пока нет отдельного live-availability safety design.

### 10. Autonomy matrix / route_templates

- Код: `policy_routing.py:2931-3118`, `policy_routing.py:3188-3222`, `policy_routing.py:3432-3505`, `actions.py:89-146`, `post_layers.py:1227-1708`.
- Источники смысла: high-risk markers, funnel state, message type, brand, topic allowlist, live-status checks, missing fact signals, Fix1b corridor, action policy, payment/enroll action markers, SemanticReading trace.
- Живость: direct-path вызывает `apply_autonomy_matrix_guard(...)` в `provider.py:964-966`; `apply_conversation_intent_plan_guard(...)` в `provider.py:1212-1220` также трассирует `route_templates/autonomy_matrix`.
- Тип: `mixed`.
- Текущий замер: 153 comparable, 1 mismatch, legacy more conservative.
- Replacement-кандидат: да, но только как следующий эксперимент/замер для узкой live-точки `route_templates/autonomy_matrix`; не распространять на dead-on-direct-path stages. Текущая диагностика не является B/ON-приемкой.

### 11. Direct keyword fallback reask

- Код: `direct_path.py:3105-3205`.
- Источники смысла: fallback_reason, open question, requested_slots, do_not_reask, P0 flag regex.
- Живость: вызывается в `provider.py:972`.
- Тип: `understanding/reask`.
- Replacement-кандидат: возможно, но отдельно от route_templates. Нужна проверка "переспрашивает ли известное" и "не снимает ли manager handoff".

### 12. Tone close detect

- Код: `post_layers.py:2224-2284`, `post_layers.py:2372-2414`.
- Источники смысла: close-message heuristics, pending manager regex, recent handoff regex, p0 history.
- Живость: direct-path вызывает `apply_tone_close_detect_layer(...)` в `provider.py:973`.
- Тип: `tone/mixed`.
- Replacement-кандидат: да, но не в текущем срезе 3-4. Это живая direct-path точка и должна быть отдельной картой/замером, потому что она может менять route/text.

### 13. Off-topic / identity / prompt injection

- Код: `policy_routing.py:822-867`, `post_layers.py:3690-3700`, `post_layers.py:4108-4157`, `direct_path.py:162-165`, `direct_path.py:668-672`; legacy dispatcher: `policy_routing.py:4768-4805`.
- Источники смысла: service topic, SemanticReading `off_topic`, marker checks for prompt injection, identity questions, negative feedback.
- Живость: SemanticReading `off_topic` живет как профильный class и может участвовать в direct-path trace. Output-side checks в `post_layers.py` живые как финальные guards. Старый safe-template dispatcher в `policy_routing.py:4768-4805` и `rules_engine.py` не доказан как прямой live direct-path apply; он относится к dispatcher/монолитной ветке.
- Тип: `mixed`, часть safety.
- Replacement-кандидат: off-topic уже частично заменяется SemanticReading. Identity/prompt-injection safe templates нельзя считать закрытыми этим срезом и нельзя удалять без отдельного safety replacement.

### 14. Bot-safe memory step guard

- Код: `post_layers.py:4394-4435`, вызов `provider.py:1235`.
- Источники смысла: `next_step_status` из bot-safe memory, проверка спорных claims в тексте.
- Живость: direct-path вызывает guard после semantic output verifier и до authoritative gate.
- Тип: `memory safety floor`.
- Replacement-кандидат: не текущий regex-to-LLM срез. Guard защищает от утверждения непроверенного следующего шага из памяти; его можно менять только вместе с contract'ом bot-safe memory.

### 15. SemanticFrame posthoc / manager-action gate

- Код: `provider.py:979-988`, `provider.py:1915-1952`, `provider.py:3103-3429`.
- Источники смысла: posthoc SemanticFrame, proof shadows, manager-action gate.
- Живость: вызывается после direct-path post-layers, до финального return.
- Тип: `mostly shadow`, но `apply_semantic_frame_manager_action_gate` при отдельном флаге может реально понизить autonomous route до `draft_for_manager`.
- Replacement-кандидат: это не regex-to-LLM cleanup, а уже LLM-gate. Держать отдельно от "резки regex".

### 16. Semantic output verifier / authoritative output gate

- Код: `post_layers.py:2677-2804`, `post_layers.py:3690-3709`, `post_layers.py:3849-3899`, `post_layers.py:3902-3936`, `post_layers.py:4108-4157`, `post_layers.py:5064-5201`.
- Источники смысла: verifier/gate findings, output sanitizer, prose quality guard, authoritative fact checks, prompt-injection/identity/fallback output guards.
- Живость: direct-path вызывает оба слоя в `provider.py:1228-1238`.
- Тип: `final floor`.
- Что делать: не считать "пониманием клиента" и не удалять в рамках regex-lasanьya. Это финальные safety floors: они не должны становиться шире, но и не должны исчезать без отдельного safety replacement.

## Dead-on-direct-path / frozen-monolith-only

Эти точки не получили trace в локальной диагностике и находятся за direct-path return:

| Точка | Код | Доказательство | Статус |
|---|---|---|---|
| `rewrite_quality/rewriter` | удалённый `answer_quality_rewriter.py` | trace report: 0 records; владелец отказался от legacy fallback | `removed_in_refactoring_package_2` |
| `post_semantics/humanity` | `post_layers.py:4533+`, вызов `provider.py:1060` | trace report: 0 records; direct-path return at `provider.py:989` | `dead_on_direct_path/frozen_monolith_only` |
| `route_templates/redundant_guard` | `policy_routing.py:4005-4065`, вызовы старого хвоста `provider.py:1039` и `provider.py:1057` | trace report: 0 records; direct-path return at `provider.py:989` | `dead_on_direct_path/frozen_monolith_only` |
| `rules_engine.py` dispatcher path | `rules_engine.py:175-222`, `policy_routing.py:1123-1225`, `policy_routing.py:1519-1623` | not called in `provider.py:962-989` direct-path branch; appears in monolith/dispatcher imports and old branch | `not_direct_path_live / dispatcher_or_monolith` |

Важно: "dead-on-direct-path" не значит "можно удалить сейчас". Это значит, что эти точки не надо включать в ближайший apply-mode для живого Telegram direct-path. Их удаление или перенос - отдельная уборка старого монолита.

## Что уже можно считать закрытым

- Срез 1 (`sense_seats`, `slots_gsf`, `off_topic`) уже прошел отдельные замеры/регрейды и вошел в профильный default через прежние решения. Это не означает полного удаления всех regex, а только закрытие конкретных классов.
- `intent_actions` после регрейда пары 1a и `да №5` входит в профильный default; legacy output-ветка live_availability удалена, входной `conversation_intent_plan.py` не тронут.
- `route_templates/autonomy_matrix` показал высокий agreement в локальной диагностике и является кандидатом на следующий узкий эксперимент/замер, но не на включение без B/ON приемки.

## Что нельзя утверждать

- Нельзя говорить, что regex-понимание полностью удалено.
- Нельзя говорить, что `rewrite_quality/post_semantics` проверены на live direct-path: они не исполнились.
- Нельзя считать отсутствие trace у dead-on-direct-path точек доказательством, что код безопасно удалить из репозитория. Это только доказательство, что он не часть текущего live direct-path.
- Нельзя удалять P0/floor/preblock по общей логике "модель понимает лучше": это другой класс риска.

## Рекомендованный пересчет Пакета-2

1. Зафиксировать `rewrite_quality`, `post_semantics`, `route_templates/redundant_guard` как `dead_on_direct_path/frozen_monolith_only` в манифесте Пакета-2.
2. Не строить будущий apply-switch для срезов 3-4 до отдельного решения: либо перенос trace в реально живую ветку, либо уборка старого монолита.
3. Следующий экспериментальный кандидат для будущего apply-switch: только `route_templates/autonomy_matrix`, с fail-closed условиями:
   - есть inline SemanticFrame/SemanticReading;
   - confidence >= согласованного порога;
   - нет P0/high-risk/floor flags;
   - нет live-status/availability missing fact;
   - нет unknown brand;
   - legacy считается в тени, и при конфликте на P0/money/live-status выигрывает более осторожное решение.
4. Перед любым включением apply или удалением legacy построить B/ON пару на живом direct-path и raw-регрейд.
5. Отдельно спланировать будущие живые точки:
   - tone close detect;
   - direct keyword fallback reask;
   - reliable answerer/sense seats floor;
   - dialogue memory slots/provenance;
   - semantic roles/conversation intent plan;
   - payment/refund text hygiene;
   - final output gates.

## Аудиторская сверка

Карта прошла независимую проверку по трем направлениям:

- Call-chain audit: подтвержден ранний return direct-path и то, что `rewrite_quality/post_semantics/redundant_guard` не являются live direct-path точками в текущем профиле.
- Inventory audit: карта расширена по живым зонам, которые легко спрятать за общими словами: bot-safe memory, reliable-answerer prompt+guard, text hygiene, semantic roles, dialogue memory, final output gates.
- M1 audit: отдельный M1-замер для самой карты сейчас не нужен, потому что карта не меняет поведение. M1 понадобится перед apply/inclusion/delete решениями, где меняется route/text или удаляется legacy.

Принятые правки аудиторов:

- После `да №5` старое аудиторское ограничение про explicit-only снято: `intent_actions` теперь profile-default; новые target-классы всё ещё добавляются только явным env.
- `rewrite_quality`, `post_semantics`, `route_templates/redundant_guard` помечены как `dead_on_direct_path/frozen_monolith_only`, а не как кандидаты на apply.
- Bot-safe memory step guard вынесен отдельной live safety-точкой.
- Identity/prompt-injection и final output guards не считаются закрытыми срезом `off_topic`.
- Semantic roles, dialogue memory и text hygiene описаны как набор разных helper/floor-механизмов, а не как один удаляемый regex.

## Практический ответ на вопрос Дмитрия

Нет, мы не полностью разобрали regex-лазанью.

Мы сделали важный шаг: перестали путать живые regex-точки с мертвым старым монолитом. Теперь резать нужно не "все, где есть regex", а только реально исполняемые direct-path точки, по одной, с замером и сохранением safety-floor.
