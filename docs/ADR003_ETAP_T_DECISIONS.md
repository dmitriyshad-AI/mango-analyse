# ADR003 этап T — журнал решений Codex

Дата: 2026-07-03
Ветка: `codex/adr003-semanticframe-migration`
Стартовый HEAD: `7676a902bd0ac0dc55dc5df18e461565e22339cb`

## D-001. `PROJECT_NOW.md` не коммитить

Решение: обновлять `docs/PROJECT_NOW.md` только локально через `python3 scripts/project_now.py`; exact HEAD и ссылки на артефакты фиксировать в этом журнале и audit pack.

Обоснование: `docs/PROJECT_NOW.md` является generated/ignored файлом по `AGENTS.md` и `.gitignore`. Коммит такого файла нарушил бы проектную дисциплину и смешал runtime-снимок с кодовым изменением.

## D-002. Делать repo-wrapper ТЗ

Решение: создать tracked wrapper в `tasks/_inbox_codex/`, затем перевести его через `scripts/task_move.py --take` и запускать `scripts/preflight.py`.

Обоснование: исходное ТЗ лежит в Foton и не имеет машиночитаемой шапки. Для крупной реализации нужен штатный preflight, иначе границы зон и безопасный collect-only не проверяются.

## D-003. Сузить реализацию до этапа T

Решение: реализовывать только trace, `sense_seats`, `off_topic`, `slots_gsf`, env-matrix, `e3_paired` runner и deletion manifest. Fix1b/Fix2/slots_reask/deal_action не реализовывать.

Обоснование: эти блоки относятся к другим подсистемам и сделают будущий замер неатрибутируемым. `deal_action` уже включён профилем, `slots_reask` требует передачи semantic reading в симуляторную память, Fix1b/Fix2 требуют отдельных смысловых критериев.

## D-004. Trace финализировать после `apply_semantic_frame_decision_shadow`

Решение: добавить `apply_semantic_reading_trace_finalize(...)` после текущего последнего direct-path shadow слоя; старый `frame_decision_shadow` не переименовывать и не менять.

Обоснование: иначе trace может не увидеть итоговые изменения и может сломать существующие отчёты, которые читают старый ключ.

## D-005. `sense_seats` зависит от reliable step1

Решение: если маска `sense_seats` включена, но `TELEGRAM_RELIABLE_ANSWERER_STEP1` выключен, писать trace `suppressed(reason="reliable_step1_off")`; в `e3_paired` включать reliable step1 в обоих плечах.

Обоснование: `apply_reliable_answerer_output_guard` не работает при выключенном reliable step1. Без явной suppressed-записи класс молча стал бы инертным.

## D-006. `semantic_reading_slots` — hidden storage без читателей

Решение: хранить модельные slot-кандидаты отдельно от `known_slots`, `client_confirmed_slots`, `topic_focus`, prompt-view и direct-path prompt. В этом этапе нет потребителя этих слотов.

Обоснование: модельная догадка не является подтверждённым словом клиента. Любой merge в общий slot-map создаст старую ошибку: бот перестанет переспросить неподтверждённые данные.

## D-007. Маска `slots_gsf` в memory-layer

Решение: не добавлять новый параметр `context` в `update_dialogue_memory_after_answer`; гейтить запись `semantic_reading_slots` через существующий env/context resolver `reading_class_enabled(None, "slots_gsf")`, а caller-side включение оставить future work.

Обоснование: у функции уже есть параметр `semantic_reading`, но нет `context`. Добавление `context` расширило бы публичный контракт памяти и затронуло бы больше вызовов. Env-only гейт достаточен для этого этапа, потому что `TELEGRAM_SEMANTIC_READING_CLASSES` уже является глобальным флагом reading-классов.

## D-008. `e2_triple` не трогать

Решение: не менять `scripts/run_adr003_semantic_reading_e2_triple.sh`; для будущего ON-замера создать отдельный `scripts/run_adr003_semantic_reading_e3_paired.sh`.

Обоснование: `e2_triple` нужен для теневой проверки inline/posthoc frame и специально держит `TELEGRAM_SEMANTIC_READING_CLASSES=` пустым. Смешивать его с active reading-масками нельзя.

## D-009. Deletion manifest — только данные

Решение: создать `docs/ADR003_DELETION_MANIFEST.md` со статусом `awaiting_green`; ничего не удалять.

Обоснование: удаление legacy regex возможно только после per-class зелёного замера и отдельного разрешения Дмитрия.

## D-010. E3 мерит reading-дельту поверх одинакового inline SemanticFrame

Решение: в `e3_paired` включить `TELEGRAM_SEMANTIC_FRAME_SHADOW=1` в обоих плечах, а ON-плечо отличается от B-плеча только `TELEGRAM_SEMANTIC_READING_CLASSES`.

Обоснование: `slots_gsf` читает `requested_product` из inline SemanticFrame. Если frame отсутствует в обоих плечах, paired-замер не измеряет reading-класс. Это не меняет runtime default/profilе и не включает флаги в живом боте; это только решение измерительного runner.

## D-011. Локальный timeout E3 dry-check не блокирует кодовый этап

Решение: локальный E3 `--dry-check`, остановившийся на `provider_error=timeout` в первых двух диалогах, классифицировать как `infrastructure_bug`. Для закрытия кодового этапа добавить unit-инвариант fake direct-provider: при всех reading-масках OFF `semantic_reading_trace` отсутствует, а metadata/route/text остаются исходными. Поведенческий E3 dry-check остаётся обязательным fail-fast предусловием будущего M1/E3 прогона.

Обоснование: timeout случился до появления inline SemanticFrame и не доказывает дефект trace/vrezka-кода. Unit-инвариант закрывает красную строку OFF без модельных вызовов, а runner сохраняет поведенческий gate там, где доступен валидный model-run.

## D-012. `sense_seats` не снимает floor на обещание наличия мест

Решение: `sense_seats=not_seats` больше не пропускает deterministic `availability_promise` guard, если в ответе уже есть обещание наличия места/группы/записи. В таком случае старый floor остаётся, а trace пишет `reason="availability_promise_floor_kept"`.

Обоснование: LLM должна понимать смысл слова «место», но не должна отключать защиту от клиентского обещания «места есть / запишем» без live-факта. Детерминизм здесь остаётся верификатором, а не понимальщиком.

## D-013. PR-A Fix1b пропускает только полностью проверенный клиентский ответ

Решение: `TELEGRAM_FIX1B_AUTONOMY_VERIFIED_FACTS` оставлен default-OFF и не добавлен в профиль. Коридор может снять только два ложных демоута (`autonomy_default_cautious_missing_facts`, `autonomy_default_cautious_unverified_fact`) и только если весь черновик поддержан свежими client-safe фактами, не содержит неподтвержденных чисел/дат, чужого бренда или live-обещаний мест. Trace-класс `fix1b` не входит в `TELEGRAM_SEMANTIC_READING_CLASSES`; это диагностическая запись применения коридора, а не новый пользовательский reading-класс.

Обоснование: найденный баг находится в проверке выхода: бот уже дал проверенный ценовой/адресный ответ, но старый cautious-layer понижал его в `draft_for_manager`. Расширять понимание клиента здесь не нужно. Reader agreement и будущий регрейд должны видеть, где коридор сработал, но P0/live/brand/fabrication полы остаются выше и не обходятся.

## D-014. PR-B slots_reask остается hidden-storage/read-only механизмом

Решение: `TELEGRAM_SLOTS_REASK` оставлен default-OFF и не добавлен в профиль. Он не создаёт `semantic_reading_slots`; hidden-слоты создаются только активной маской `TELEGRAM_SEMANTIC_READING_CLASSES=slots_gsf`. PR-B только читает имена уже записанных hidden-слотов и добавляет эти имена в `do_not_ask_again`, чтобы бот не переспрашивал grade/subject/format. Значения hidden-слотов не попадают в `known_slots`, `client_confirmed_slots` или `to_prompt_view()`.

Обоснование: это анти-переспрос, а не merge semantic slots into memory. Старые regex G/S/F и `known_slots` остаются до отдельного `slots_gsf -> known_slots` решения с `source=semantic_reading_llm`. В текущем HEAD три sim/update точки уже пробрасывают `semantic_reading=` в память; блок PR-B зафиксирован как инвентаризация существующего механизма и проверка его границ.

## D-015. PR-D rolling dialog summary производится inline и пишется fail-closed

Решение: `TELEGRAM_DIALOG_SUMMARY_ROLLING` оставлен default-OFF и не добавлен в профиль. Поле `dialog_summary` добавляется только в основной direct-path JSON при включённом флаге, без отдельного LLM-вызова. Запись в `DialogueMemory.conversation_summary_short` идёт отдельной веткой `update_dialogue_memory_after_answer(dialog_summary=...)`, до раннего возврата memory-provenance, но не через `_apply_memory_llm_update`.

Обоснование: старый memory-LLM слой трогает слоты и исторически конфликтовал с provenance. PR-D должен хранить только короткую смысловую сводку диалога, без записи фактов в `known_slots` и без ПДн/чужого бренда/цен и дат. Direct path получает предыдущую сводку как безопасный prompt context, а Wappi использует её вместо сырьевой 6-строчной склейки только при включённом флаге.

## D-016. Package-2 Srez-1a `intent_actions` остается explicit-env и same-stage

Решение: `intent_actions` добавлен только в allowlist `TELEGRAM_SEMANTIC_READING_CLASSES`, но не в `PILOT_PROFILE_DEFAULT_READING_CLASSES`. В переходном режиме `apply_conversation_intent_plan_guard` считает legacy-решение как раньше, затем читает только inline `SemanticFrame` на той же стадии direct-path pipeline и применяет fail-closed/max-conservative слой.

Обоснование: срез заменяет выходное route-решение, а не входные чтения `conversation_intent_plan`. Posthoc frame появляется позже и не является той же стадией. При `requested_action=check_availability` frame может переустановить protective-сигнал `conversation_intent_plan_live_availability`, чтобы Fix1b/autonomy не начали обещать живые места. При отсутствии inline frame, низкой confidence, невалидном enum или source mismatch применяется legacy. Frame-based false-P0 repair в этом коммите не включается; его продолжает делать только legacy-логика до отдельного замера. Профильное включение и deletion legacy-вычисления остаются отдельными решениями после пары и регрейда.

## D-017. PaymentFix и PR-D включаются в `pilot_gold_v1` как профильные дефолты

Решение: после M1-регрейда трех приемочных пар добавить `TELEGRAM_TEXT_HYGIENE_PAYMENT_FIX` и `TELEGRAM_DIALOG_SUMMARY_ROLLING` в профильный default-ON контур `pilot_gold_v1`, сохранив явный env/context override `=0`.

Обоснование: PaymentFix и PR-D прошли независимые пары; это не включает live runtime и не трогает AMO/CRM/Tallanto. PaymentFix чинит текстовую болезнь #16, PR-D хранит короткую безопасную rolling summary. `TELEGRAM_FIX1B_AUTONOMY_VERIFIED_FACTS` в профиль не добавляется из-за найденных дыр коридора.

## D-018. Fix1b hardening до любого включения

Решение: `TELEGRAM_FIX1B_AUTONOMY_VERIFIED_FACTS` остается default-OFF. Коридор дополнительно блокирует отрицательные утверждения о существовании курса/группы/программы и входящий paid-context (`чек`, `квитанция`, `скрин оплаты`, `оплатил/оплачено`).

Обоснование: M1-регрейд подтвердил безопасность Fix1b, но не разрешил включение: коридор мог пропустить отрицательное существование и оплаченный контекст. Эти стопы относятся к верификации готового черновика/операционного контекста, а не к новому чтению client intent.

## D-019. Package-2 runner добавляет целевой reading-класс поверх профиля только env-ом

Решение: `scripts/run_adr003_semantic_reading_e3_paired.sh` принимает `TARGET_READING_CLASS` и добавляет его к `READING_CLASSES` только для ON-ноги. B-нога не задаёт `TELEGRAM_SEMANTIC_READING_CLASSES=` пустой строкой, а запускается как чистый профиль `pilot_gold_v1`. Валидатор разрешает профильные traces в B, но запрещает target-class trace; ON обязан иметь target-class trace. Профильный default reading classes на момент D-019 не менялся; после D-032 `intent_actions` стал профильным default, и раннер теперь отказывает, если target уже входит в профиль/base.

Обоснование: срез-1a должен был измерить `intent_actions` как единственную новую переменную поверх текущего профиля. Добавлять `intent_actions` в профиль до пары было нельзя: baseline стал бы загрязнен и неатрибутируем. Это ограничение снято только после D-032.

## D-020. Inline text gate верифицирует адресные числа и учебный год только из источников хода

Решение: числа адресов вроде `20`/`30` считаются verified, если они пришли из selected exact address fact или уже подтверждены `number_audit`; учебный год `2026/27` считается verified только из selected exact fact id, текста selected exact fact или metadata этого же selected exact fact (`product`/`academic_year`/`school_year`). Произвольный raw blob фактов не становится источником истины, а служебные даты вида `2026_06_11` не считаются учебным годом.

Обоснование: это снимает ложные тревоги гейта на адресах и строках `2026/27`, но не открывает проход любым числам из базы. Граница сохраняет правило: проверяется только источник текущего хода, adjacent facts остаются warning, а не pass.

## D-021. Deletion №1 остановлен до свежей пары после фикса B-ноги

Решение: не удалять legacy live-availability ветку `conversation_intent_plan` на основании пары `adr003_srez1a_pair_72c84090_20260704_ready`. Сначала нужно переснять pair после исправления runner: B = чистый профиль, ON = профиль + `TARGET_READING_CLASS=intent_actions`. На момент D-021 `intent_actions` не добавлялся в `PILOT_PROFILE_DEFAULT_READING_CLASSES`, а профильное включение/deletion оставались отдельным решением после свежего регрейда. Это решение superseded by D-032 после свежей пары и `да №5`.

Обоснование: старая B-нога явно задавала `TELEGRAM_SEMANTIC_READING_CLASSES=` и тем самым глушила профильные default-классы `sense_seats,slots_gsf,off_topic`. Такая пара полезна для разведки, но не является достаточным доказательством безопасности deletion на боевом профиле. Удаление legacy ветки без профильного replacement могло снять защитный `conversation_intent_plan_live_availability`, на который опираются live-status/Fix1b/autonomy полы; D-032 делает deletion атомарно с replacement и fail-closed fallback.

## D-022. Apply-класс задаётся точкой, а не широким reading-классом

Решение: добавить отдельный env `TELEGRAM_READING_APPLY_CLASSES` с allowlist точек применения, сейчас только `route_templates/autonomy_matrix`. Значение `route_templates` в `TELEGRAM_SEMANTIC_READING_CLASSES` включает trace/read-класс, но не включает apply. Apply срабатывает только при одновременном `TELEGRAM_SEMANTIC_READING_CLASSES=route_templates` и `TELEGRAM_READING_APPLY_CLASSES=route_templates/autonomy_matrix`.

Обоснование: широкий apply по имени `route_templates` случайно включил бы все будущие route-template точки одним флагом и сделал бы замеры неатрибутируемыми. Точечный stage-key позволяет резать «лазанью» по одному живому месту и сохраняет дисциплину default-OFF.

## D-023. `route_templates/autonomy_matrix` применяет frame только как узкий safe-bypass

Решение: apply-ветка `route_templates/autonomy_matrix` может вернуть исходный безопасный ответ вместо legacy-demote только когда inline SemanticFrame имеет `confidence>=0.90`, `requested_action=answer_question`, `answerability=answer_self`, `must_handoff=false`, `risk_class=safe`; при любом P0/high-risk/manager_only/blocked/live-availability floor, отсутствии frame, posthoc source, низкой confidence или замене текста выбирается legacy. Direct-path вызывает `apply_conversation_intent_plan_guard` для этой точки даже без `intent_model_led`.

Обоснование: это первый apply-механизм после trace, поэтому он должен быть fail-closed и не иметь права снимать полы безопасности. Цель — измерить точечное снятие ложного route-template понижения, а не переписать routing целиком.

## D-024. Ж2 строится как trace-only на реально живых live-status точках

Решение: добавить класс `live_status_read` в allowlist reading-классов, но не в профиль. Сам по себе `live_status_read` не вызывает `conversation_intent_plan` guard, потому что этот guard меняет route; trace рядом с `conversation_intent_plan` пишется только если guard уже вызван другой активной причиной (`intent_model_led`/`intent_actions`/apply stage). Дополнительно trace пишется в `reliable_answerer` output guard: legacy-решение, frame `requested_action`, нормализованные grade/subject/format, фасеты и availability-promise status. `requested_product.raw_text` и сырой клиентский текст в trace не пишутся.

Обоснование: Ж2 должен сначала доказать исполняемость и agreement на живом direct-path. Писать trace в dead monolith или в `build_answer_coverage_plan` нельзя: первое не влияет на бота, второе раздует trace и смешает prompt-time вычисления с output-guard решением. Удаление стема «мест» и apply для Ж2 остаются после agreement-регрейда.

## D-025. Ж3/Ж4 сейчас закрываются sentinel-гейтами, без нового поведения

Решение: для Ж3 усилить тест, что hidden `semantic_reading_slots` не становятся `known_slots`/`client_confirmed_slots` и не попадают в direct-path prompt как подтверждённые значения. Для Ж4 усилить тесты: «сколько можно вернуть по налоговому вычету» остаётся tax/non-refund, а «уже оплатил, хочу вернуть оплату» даже с упоминанием налогового вычета остаётся `refund_frame=dispute` и `manager_only`.

Обоснование: Ж3 и Ж4 имеют высокую цену ошибки: слоты могут превратиться в ложные клиентские подтверждения, а деньги — в P0-пропуск. До отдельной trace-пары и per-class регрейда здесь нельзя включать apply или менять runtime-поведение; быстрый безопасный шаг — закрепить границы автоматическими sentinel-тестами.

## D-026. Apply `route_templates/autonomy_matrix` усиливается дополнительными floors

Решение: `TELEGRAM_READING_APPLY_CLASSES=route_templates/autonomy_matrix` остаётся default-OFF и может выбирать frame-safe original только если legacy не сработал как brand/payment/manual/live/topic floor. В floor добавлены `brand_separation_guarded`/cross-brand, `payment_confirmation_without_two_sources`/`payment_source_conflict`, `manager_approval_required+no_auto_send` без известного autonomy-cautious false-positive, и `topic_id` mismatch.

Обоснование: аудитор нашёл, что первый scaffold доказывал P0/live floor, но не доказывал brand/payment floors прямо в apply-ветке. Поздние final gates всё ещё существуют, но apply должен быть fail-closed сам по себе, чтобы не зависеть от неявной очередности слоёв.

## D-027. Ж2 `live_status_read` получает входной trace-observer

Решение: если `live_status_read` включён, но настоящий `apply_conversation_intent_plan_guard` не вызван соседним режимом, direct-path пишет trace-only observer рядом с `conversation_intent_plan` без применения route/text/safety изменений. Если guard уже вызван профилем/`intent_actions`/apply, используется существующий trace на той же стадии, без дубля.

Обоснование: разведка показала, что один `live_status_read` хорошо видит `reliable_answerer_output_guard`, но не всегда видит входной классификатор `conversation_intent_plan`. Observer нужен для honest agreement Ж2: legacy live-status vs `SemanticFrame.requested_action`, но до регрейда это не имеет права менять поведение.

## D-028. Ж3 начинается с final-text `reask_read` trace, не с переноса legacy guard

Решение: добавить default-OFF class `reask_read`, который на финальном direct-path тексте пишет trace-only запись о повторном вопросе уже известных `grade/subject/format` и о hidden `semantic_reading_slots` только по именам слотов, без значений. Runtime text/route не меняются.

Обоснование: живой direct-path почти не доходит до старого `apply_known_context_redundant_question_guard`; переносить его вслепую означало бы снова мерить мёртвый код. Сначала нужен наблюдатель на финальном тексте, а hidden slots не должны стать client-confirmed или утечь в prompt/trace значениями.

## D-029. Ж4 начинается с `roles_read` trace, без попытки чинить деньги маршрутом

Решение: добавить default-OFF class `roles_read`, который пишет trace-only запись по смысловым ролям денег/записи: `payment_source`, `refund_frame`, `enrollment_vs_recording`, `transfer_sense`, поля `SemanticFrame` и итоговый route/topic. Runtime text/route/safety flags не меняются.

Обоснование: роли вроде «возврат оплаты курса» vs «налоговый вычет» и «записать на курс» vs «запись урока» стоят рядом с P0/деньгами. До отдельной пары и регрейда здесь нельзя делать apply. Trace нужен, чтобы увидеть реальные расхождения frame с legacy на живом direct-path, а не переносить regex-узлы из замороженного монолита.

## D-030. Hidden semantic slots запрещены в prompt-памяти значениями

Решение: `dialogue_memory_view.semantic_reading_slots` не передаётся в direct-path prompt как память диалога даже при выключенном `PRESALE_PII_MEMORY`. В trace для `reask_read` разрешены только имена hidden-слотов, без `value`; в prompt hidden-значения не попадают.

Обоснование: `semantic_reading_slots` — это вывод модели, а не подтверждённые клиентом `known_slots`. Аудит показал, что старый путь мог вставить эти значения в блок «Память диалога». Это могло сделать LLM-вывод похожим на клиентское подтверждение. Для Ж3 это критичный boundary: не переспросить уже известное можно только после отдельного решения, но нельзя тихо считать hidden-slot фактом.

## D-031. Ж2 apply строится как смысловое объединение, а не 1:1 замена legacy

Решение: добавить default-OFF apply point `live_status_read/conversation_intent_plan`. Он работает только при включённых `TELEGRAM_SEMANTIC_READING_CLASSES=live_status_read` и `TELEGRAM_READING_APPLY_CLASSES=live_status_read/conversation_intent_plan`. `SemanticFrame.requested_action=check_availability` ставит страховку живых мест: `draft_for_manager` для автономного route, флаг `conversation_intent_plan_live_availability` и manager checklist «не обещать место до проверки». `risk_class=manager_action/missing_facts` для `check_availability` не блокирует apply, потому что это ожидаемый смысл «проверить наличие у менеджера», а не P0. `send_document`/`enroll` без availability-смысла не ставят live-status demote и могут снять ложный legacy live-status. Paid/P0/high-risk/blocked/manager_only остаются fail-closed floor; P0/high-risk проверяется не только по flags, но и через `is_high_risk_result()` и `direct_path_model_p0` metadata.

Обоснование: регрейд Ж2 показал: 10/12 legacy-хитов подтверждены frame, 2 only-legacy — ложные или более правильно закрытые другим смыслом, а 10 only-frame — слепая зона legacy. Поэтому замена должна идти по объединению смыслов: availability защищаем, paid отправляем менеджеру, document/enroll без availability не считаем «местами». Аудит дополнительно поймал опасный обход: high-risk мог быть задан через `topic_id/risk_level/direct_path_model_p0`, а не через flags; поэтому floor использует общий high-risk классификатор. Профильное включение и deletion legacy по-прежнему запрещены до микро-пары и отдельного решения.

## D-032. Профильное включение `intent_actions` и deletion №1 после регрейда пары 1a

Решение: после raw-регрейда свежей пары 1a и отдельного «да» владельца `intent_actions` добавлен в `PILOT_PROFILE_DEFAULT_READING_CLASSES`, а legacy output-ветка `primary_intent == "live_availability"` внутри `conversation_intent_plan` guard удалена. Входной `conversation_intent_plan.py` и детектор `_asks_live_availability` не тронуты. Отдельный apply-класс `live_status_read/conversation_intent_plan` не включается в профиль. Если inline-frame отсутствует или невалиден, но старый план видит `live_availability`, `intent_actions` fail-closed ставит manager/check-live floor, чтобы не было тихого автономного ответа без frame.

Обоснование: пара 1a показала, что inline `SemanticFrame.requested_action=check_availability` переустанавливает защитный сигнал `conversation_intent_plan_live_availability` там, где старый output-guard был слеп, и не понижает `manager_only`/`blocked`. Удаляем именно выходное legacy-применение, а не источник метаданных или safety floor. Это первый шаг «минус-лазанья»: live availability на этом участке теперь идёт через frame; legacy `primary_intent` остаётся только fail-closed запасным полом при неисправном frame.

## D-033. Ж3 `reask_read` не переводится в apply до исправления источника known-slots

Решение: не включать apply для `reask_read` в финальный M1-пакет текущего захода. Класс остаётся trace-only. Следующий безопасный шаг по Ж3 — отдельный фикс качества памяти/known-slots и повторная микро-пара, а не применение текущего наблюдателя.

Обоснование: локальная прослушка Ж3 (`2026-07-05_Zh3_reask_read_export_6896673b`) дала 3 `would_flag` на 101 ход. Два случая полезны: бот действительно повторно просит уже названный предмет/класс. Но один случай небезопасен: в `zh3_reask_known_slot_01` ход 3 память перед trace содержит `child_name='Записи'`, и `reask_read` считает просьбу прислать ФИО ученика повторным запросом `student_name`. Это ложное client-confirmed значение, возникшее из слова «записи», и active-apply мог бы удалить нужный запрос ФИО ученика. Пока источник `known_context_fields`/dialogue memory не перестанет принимать такие ложные имена, `reask_read` нельзя применять.

## D-034. Ж4 `roles_read` не переводится в apply: деньги/НДФЛ/запись требуют отдельного фикса

Решение: не включать apply для `roles_read` и не включать его в следующий большой ON-пакет как поведенческий класс. Класс остаётся trace-only. Для Ж4 нужен отдельный regression set и upstream-фикс plan/frame/text hygiene: НДФЛ не должен превращаться в refund/dispute, payment-dispute без возврата не должен получать возвратный шаблон, real refund/dispute остаётся manager-only.

Обоснование: локальная прослушка Ж4 (`2026-07-05_Zh4_roles_read_export_6896673b`) показала 106/106 `shadow_only`; готового apply-механизма нет. Аудит сырья нашёл блокеры: `Возврат НДФЛ оформляете?` размечен как `plan_primary_intent=refund`, `refund_frame=dispute`, хотя это налоговый вычет; `Если оплатили, а доступа к уроку нет` получает текст про возврат вместо статуса доступа; P0 `ребёнок один остался` остаётся `manager_only`, но текст уходит в объяснение вместо сухого handoff. Эти дефекты не лечатся включением `roles_read`; сначала нужен точечный money/tax/recording fix с регрессиями.
