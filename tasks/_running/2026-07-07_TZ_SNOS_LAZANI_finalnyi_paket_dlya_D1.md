> TAKE 2026-07-07 05:03 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/channels/, src/mango_mvp/integrations/, scripts/, tests/, docs/, product_data/telegram_dynamic_test_sets/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_subscription_llm_draft_provider.py tests/test_dialogue_memory.py tests/test_semantic_reading.py tests/test_adr003_semantic_reading_e3_runner.py tests/test_adr003_regex_understanding_moratorium.py
Семантический-аудит: да

# ТЗ Д1: «СНОС ЛАЗАНЬИ» — одно финальное ТЗ: включить всё готовое LLM-понимание, снести регулярки-дублёры, один M1-экзамен

**Владельческое решение (Дмитрий, 07.07.2026, зафиксировано):** полностью снести регулярки-ПОНИМАНИЕ везде, где есть LLM-замена; раскрыть готовые LLM-блоки, запертые default-OFF флагами; замер — ОДИН финальный M1-экзамен пакетом, без пофлаговых пар. Полы безопасности (ПДн/бренд/числа/P0/live-места) — НЕ трогать (§12 передачи, подтверждено).
**База:** ветка `codex/adr003-semanticframe-migration` (сейчас 10667f90 + твой комбо-заход). Это ТЗ поглощает комбо-ТЗ: комбо-пара, которая уже бежит, остаётся единственной локальной проверкой; новых локалок ноль (slots-1b идёт в M1-пакет без своей пары — риск принят владельцем).

## ПРАВКИ Д1 + АУДИТОРА ПЕРЕД ИСПОЛНЕНИЕМ (07.07.2026)

1. **Решение по `factsel_fallback_wrong_venue_lvsh_29`:** свежий регрейд Fable (`Foton/2026-07-07_REGRADE_STOP_wrong_venue_lvsh_29_verdict.md`) признал стоп корректным, но дефект — в экзаменационной персоне/судье, не в runtime. Runtime-код, порог `FACT_SELECT_FRAME_MIN_CONFIDENCE=0.90` и полы не трогаем.
2. **Фикс экзамена:** persona 29 делится на 29a (явное исключение всех летних программ: «обычные занятия в течение учебного года, никакие летние программы — ни лагерь, ни летняя школа») и 29b (прежняя двусмысленная формулировка; success = уточнение или аккуратная разводка учебный год/городская летняя программа). Judge не ставит `fabrication/wrong_product_fact` за факты своего бренда/площадки на двусмысленный запрос при уточнении/разводке, но ставит FAIL при игноре явного исключения клиента.
3. **slots-1b не входит в пачку без локальной микро-пары:** `TELEGRAM_SLOTS_GSF_KNOWN_MERGE` сначала проходит локальную микро-пару 15-20 slot-fixtures и отдельное «да» Дмитрия. До этого B1 не выполняется, запись slot-regex в `client_confirmed` не отключается.
4. **Политика мест:** A10 разрешает только безопасную формулировку «на регулярные группы сейчас идёт набор», если это подтверждено KB-фактом. Запрещено писать «места есть», «забронирую», «закреплю» без live-проверки. Все live-seat floors из C остаются неизменными.
5. **Перед M1-пакетом — мини-дым fact_select, не пара:** ON-конфиг на 6-8 `factsel_*` персонах, без судьи; смотрим только `fact_select` trace. Если есть хотя бы один `applied`/`shadow_only` с непустым `product`, `fact_select` едет в пачку; если всё `fail_closed/empty_product`, `fact_select` исключается из профиля/пачки, это фиксируется в README.
6. **M1-состав:** пакет собирается по спеке после A/B-коммитов и мини-дыма: канон-156 + focus_32 с 29a/29b + фокус-35 + latch/слоты/seats + 2-3 сезонные персоны. В ON идут только зелёные A-классы; slots-1b только после отдельного «да».
7. **Шаг E:** уборка мёртвого монолита начинается только после сборки M1-пакета и safety-аудита текущих живых классов; не смешивать с A/B-коммитами.

## РЕШЕНИЯ ИСПОЛНИТЕЛЯ D1 В ХОДЕ РЕАЛИЗАЦИИ (07.07.2026)

1. `roles_read/refund_tax`: исправляет клиентский смысл текста при ложном «возврате НДФЛ», но не снимает уже поставленный `manager_only`/`no_auto_send`. Причина: финальный инвариант трека — LLM/SemanticFrame владеет смыслом клиентского шаблона, legacy/P0 слой может только ужесточать маршрут.
2. B1 (`TELEGRAM_SLOTS_GSF_KNOWN_MERGE`) не реализуется в этом заходе: нужен отдельный локальный slot-fixtures micro-pair и отдельное «да» Дмитрия, как указано в правках перед исполнением.
3. B2/B5 не удаляются механически до мини-дыма/M1: keyword-ветки там частично являются retriever/fail-closed fallback и floor-контекстом. До измерения можно раскрыть `fact_select_read`, но нельзя сносить fallback так, будто это только старая «лазанья».
4. B6 (`verified-fact` старые prose-ветки) остаётся как explicit `TELEGRAM_PROSE_MODEL_LED=0` rollback. В профиле `pilot_gold_v1` они не активны; физическое удаление будет корректно только после M1, если rollback больше не нужен.
5. B7 (`DIRECT_PATH_GOLD_TOPIC_KEYWORDS`) удалён: это был keyword-подбор few-shot gold-примеров, не route/safety-floor. После удаления подсказки берутся только из уже рассчитанного контекста/плана.

## Git-инвентаризация (сделана Fable, 07.07)
Ветки: ценного невлитого НЕТ. tz147-p0-deep — суть в `p0_recall_spec.py:154+` (безусловно); reliable-step1/venue/scope-precision/botsafe — суть в каноне; старые remote codex/* — история, кандидаты на архив после твоего подтверждения (список в конце). `email-pipeline-restore`, `tz135-wow-tone` (live), `main` — не трогать.
Флагов TELEGRAM_* на HEAD: ~152, из них смысловых default-OFF с ГОТОВЫМ кодом — 12 (ниже). Остальное: инфра/schema (живут), монолитные мертвецы (сносятся шагом E).

## ШАГ A — ВКЛЮЧИТЬ В ПРОФИЛЬ pilot_gold_v1 (один коммит на флаг, default-ON в DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS / PILOT_PROFILE_DEFAULT_READING/APPLY_CLASSES)

| # | Что | Где | Примечание |
|---|---|---|---|
| A1 | `TELEGRAM_FACT_SELECT_FRAME` + reading `fact_select_read` | direct_path | выбор фактов по смыслу (ретривер self-report), fail-closed каскад уже есть |
| A2 | `TELEGRAM_TONE_CLOSE_FRAME_VETO` | provider | не прощаться с покупающим; frame решает |
| A3 | `TELEGRAM_P0_LATCH_AUTORELEASE_V2` | dialogue_memory | разлипание ложного P0-латча (из комбо-ТЗ 0.3) |
| A4 | reading `roles_read` + apply `roles_read/refund_tax` | policy_routing:4682 | НДФЛ ≠ возврат (D-039/D-042, сужен) |
| A5 | reading `reask_read` + apply `reask_read/final_text` | policy_routing:4634 | не переспрашивать известное (D-038) |
| A6 | apply `route_templates/autonomy_matrix` (+reading route_templates) | policy_routing:3895 | frame снимает ложный демоут (D-023/D-026; agreement 153/1) |
| A7 | `TELEGRAM_P0_MODEL_LED` | support:509 | модельное смягчение complaint-preblock — сейчас мертво на профиле |
| A8 | `TELEGRAM_PROSE_MODEL_LED` | support:529, policy_routing:6116 | живой текст вместо робо-скелета «Да, по формату есть проверенная информация…» |
| A9 | `TELEGRAM_PAYMENT_REFUND_DISPUTE_SPLIT` | direct_path:138,441 | tz154: проверь фактический резолв — если OFF на профиле, включить |
| A10 | `TELEGRAM_SEATS_DEFAULT_OPEN` | policy_routing:106,3628,3716 | РЕШЕНИЕ ВЛАДЕЛЬЦА §0.1-3, но в безопасной формулировке: «на регулярные группы сейчас идёт набор», НЕ «места есть»; сверь исключения-полы с ТЗ politika_mest (ЛВШ распродано, 2-я смена ЛШ, paid/брони → менеджер); KB-факт — Fable сделает semantic-review параллельно |
| A11 | `TELEGRAM_SLOTS_GSF_KNOWN_MERGE` — НАПИСАТЬ по дизайну `2026-07-07_DIZAIN_slots1b_pamyat_slotov_na_LLM.md` | dialogue_memory/direct_path | единственный новый код >2ч; фикстуры из дизайна — в M1-набор |
| A12 | НЕ включать: `FIX1B` (польза не доказана, D-017), `SEMANTIC_FRAME_MANAGER_ACTION_GATE` (понижающий гейт — против скорости продаж), `RELIABLE_ANSWERER_STEP1` (пол со спящим дефектом R22; отдельное решение после freeze) | | |

## ШАГ B — СНОС регулярок-дублёров (АТОМАРНО: тот же коммит, что включение соответствующего A; грабля №2)

| # | Регулярка (file:line на 10667f90, сверяй по имени) | Действие | Вместе с |
|---|---|---|---|
| B1 | Слот-экстракторы в client_confirmed: `_FORMAT_PATTERNS`, `_GRADE_PATTERNS`, `_SUBJECT_PATTERNS`, `_CHILD_NAME_MARKER_RE` (dialogue_memory:1102-1153) | выключить ЗАПИСЬ в client_confirmed/known_slots (merge их заменяет); do_not_reask — только от client_confirmed; функции остаются мёртвым кодом до шага E | A11 |
| B2 | `DIRECT_PATH_CATEGORY_ALIASES` first-match + `_direct_path_fact_categories` keyword-скоринг (direct_path:741-809, :665) | понизить до fail-closed fallback ТОЛЬКО при недоступном retriever (timeout/empty); при живом ретривере keyword-ветка не участвует | A1 |
| B3 | `_refund_frame` `dispute:bare_refund_mention` (semantic_roles:533) как выбор ТЕКСТА | перестаёт красить текст при валидном frame/plan (apply refund_tax решает); маркер остаётся полом маршрута | A4 |
| B4 | tone_close `_TONE_CLOSE_GRATITUDE_RE` как РЕШАТЕЛЬ | остаётся детектором кандидата; решение — за frame-veto | A2 |
| B5 | `QUESTION_KIND_MARKERS` live_availability(«мест»,«налич»)/price(«сколько») (dialogue_memory:45-73) + `_keyword_signals` price/identity (conversation_intent_plan:490-506) | понизить до tie-breaker: при валидном inline frame (confidence≥0.90) kind/intent берётся из frame; keyword — только при отсутствии frame | A1+A6 (тот же заход) |
| B6 | verified-fact шаблон: после A8 робо-ветки с «По…есть проверенная информация» удалить, остаётся prose_model_led | A8 |
| B7 | gold topic keywords (direct_path:2467-2478) | удалить (LOW-риск хелпер; примеры выбирает не route) | отдельный коммит |

## ШАГ C — НЕ ТРОГАТЬ (полы; любое «заодно» здесь = СТОП)
ПДн-маскер (с фиксами 0.1), phone/email; бренд-полы и cross-brand шаблон; числовые гейты/number_audit; P0-recall/preblock/латч (кроме релиза A3) — включая CHILD_SAFETY (689423d); 4 пола «мест» (D-037) — `_asks_live_availability` floor-сигнал, live-status гварды, `_AVAILABILITY_PROMISE_RE` (kept_as_output_floor), safe-шаблон; semantic output verifier; authoritative gate; output sanitizer; bot-safe memory guard. SEATS_DEFAULT_OPEN (A10) меняет ДЕФОЛТ ответа о местах, но его исключения-полы обязаны остаться.

## ШАГ D — ЕДИНСТВЕННЫЙ ЗАМЕР: M1-экзамен пакета
По спеке `2026-07-07_SPEKA_konsolidirovannyi_M1_ekzamen_pachka.md`, состав ON = зелёные пункты шага A (обнови env-контракт пакета; `fact_select` зависит от мини-дыма, `slots-1b` — от отдельного «да»). Набор: канон-156 + focus_32 с 29a/29b + фокус-35 (D-043) + latch-фикстуры + слот-фикстуры (из дизайна A11, если слот-блок допущен) + seats-кейсы (регулярная группа → «идёт набор»; ЛВШ → «распродано/лист ожидания»; бронь/paid → менеджер) ≈ 250 персон. Гейты и правило «красный класс выкидываем из пачки без второй ночи» — из спеки. После зелёного → один профиль-коммит (A×всё зелёное + B атомарно) → деплой по чек-листу freeze/свапа.
Внимание: комбо-пара, которая уже бежит, — ранний дым для A1/A2/маскера. Если она красная по классу — этот класс чинится ДО сборки M1-пакета, остальное не ждёт.

## ШАГ E — уборка мёртвого монолита (параллельно ожиданию M1, НЕ блокирует пакет)
«Да» владельца на уборку получено 07.07 (этот документ). Function-level manifest (D-041): `answer_quality_rewriter`, humanity-слои, `known_context_redundant_question_guard`, rules_engine dispatcher-ветка, монолитный хвост provider.py ниже :999 (кроме используемого совместимостью), их мёртвые флаги (ANSWER_QUALITY_*, DRAFT_X2_*, PH2_*, Q_*, A_SELLING/COVERAGE/ESTIMATE/TRAVEL, STEP4_*, HUMANITY_*, ANTIREPEAT_STRICT, INTENT_STATE_REPAIR, SEMANTIC_DIAGNOSIS_GUARD — каждый подтверди импорт-аудитом). КРИТЕРИЙ: pytest зелёный + смоук-пара байт-в-байт на 10 диалогах + `dialogue_contract_pipeline` НЕ удалять без доказательства недостижимости. Отдельная серия коммитов ПОСЛЕ сборки M1-пакета (чтобы экзамен не тащил гигантский дифф).

## Ветки на архив (после твоего «мои, суть влита»)
origin/codex/tz147-p0-deep-output-carry, yandex+origin: reliable-answerer-step1-with-closing, autonomy-scope-precision, release-venue-autonomy, botsafe-summary-builder, sales-exam-progression-v2, progression-judge, wappi-history, tz139-*, phase1-dossier-enrich, tz148-env-isolation, wappi-controlled-watch-observe, tz155-light-git-bundles (0 невлитых), mango-call-increment, etap1-crm-card-assembler. Удалять только remote-указатели, объекты остаются в git навсегда (восстановимо).

## Стоп-правила и границы
≤2 итерации на пункт → СТОП и вопрос. Любой красный в C-списке → СТОП. Push в свою ветку можно; live/AMO/Wappi/CRM — нет; M1 запускает человек по PROMPT_M1. Мораторий: новые понимающие regex запрещены по-прежнему (сносим, не добавляем); каждое B-понижение — запись в мораторий-док. Каждый A/B-коммит атомарный с тестами; фикстуры аудита 07.07 (child_left_alone, «Нет претензий, продлеваем», «Онлайн не подходит», «Меня зовут Ольга», «зовут Максим», hot-lead «запишите нас») — обязательные регрессии пакета.
