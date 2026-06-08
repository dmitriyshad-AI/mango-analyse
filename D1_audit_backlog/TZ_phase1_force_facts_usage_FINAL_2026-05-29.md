# ТЗ Phase 1 — принуждение использовать факты (ФИНАЛ). 2026-05-29

Автор: второй Claude. Снимок: HEAD `36e23cb8`. Read-only. Старт у Кодекса — ПОСЛЕ Phase 12 + Phase 2 (диспетчер precedence к этому моменту уже в v2).
Обоснование приоритета: разбор Части 1 (`part1_remaining_failures_analysis`) — Phase 1 закрывает **12-13 из 26 PWN** на выборке 33; это главный движок роста PASS после того, как Волна 3 убрала пустые хендоффы и hard_gate.

## §0. Контекст
- v2-цепочка `_apply_dialogue_contract_v2_guard_chain` (`subscription_llm.py:994-1029`); pipeline `run_dialogue_contract_pipeline` (`dialogue_contract_pipeline.py`).
- В драфтер-промпте УЖЕ есть мягкая инструкция «ты обязан из фактов» (`dialogue_contract_pipeline.py:531-549`) и «Нет факта по ключам → узкий честный хендофф, не подставляй соседний» (`:539`) — но это soft-LLM, НЕ энфорс. Phase 1 = детерминированный энфорс.
- Частичный coverage уже есть: `answer_quality_rewriter._answerable_parts` (`:990`) + проверка «односоставный ответ при ≥2 частях» (`:181`). Phase 1.1 обобщает до «retrieved_match-факт не процитирован».
- **Обновление по ответам Дмитрия (Б.3):** онлайн-цена УНПК 41 800/69 900 ПОДТВЕРЖДЕНА актуальной (квалификатор «2 раза в неделю») → в Phase 2 факт появляется → Phase 1 сможет принудить его использование (закрывает C2_03/C1_03 как «цена есть, но не названа»).

---

## §1. Кластеризация 12-13 Phase 1 PWN Части 1

| Кластер | Тип «не использовал факт» | Диалоги (Часть 1) | Что должно быть |
|---|---|---|---|
| **K1 — не назвал значение из rfk** | retrieved_match (дата/цена) есть, ответ = пустой хендофф/manager | `foton_S1_camp_dates_02 t1`, `foton_S1_camp_dates_04 t1`, `unpk_S1_camp_dates_02 t1`, `unpk_S1_camp_dates_03 t1`, `C1_format_price_04 t2`, `E5_payment_01` | назвать дату/цену из факта; по живым местам — узкий честный хендофф |
| **K2 — не сложил/не посчитал** | факты-слагаемые есть (цена + скидка %), ответ = формула без итога | `C2_multi_subj_01 t2`, `C2_multi_subj_02 t1` (74 500+59 600=**134 100**), `C2_multi_subj_04 t1`, `C2_multi_subj_05 t2` (47 250+33 075=**80 325**) | детерминированный расчёт итоговой суммы |
| **K3 — пустой хендофф при retrieved_match** | answerability=answer_self + rfk≥1, но «Передам менеджеру» | `E6_discount_01 t3`, частично `E5_payment_01`, остаток `C1_format_price_04 t2` | запретить пустой хендофф; собрать ответ из rfk |
| **K4 — KB-gap → Phase 2, потом Phase 1** | факта не было (онлайн-цена УНПК), теперь будет (Б.3) | `unpk_C1_format_price_03`, `unpk_E1_price_03`, `C2_multi_subj_03` | после Phase 2 — принудить использование внесённого факта |

Распределение по подпунктам: K1→1.1+1.3, K2→1.2, K3→1.3, K4→Phase 2 + 1.1. Доминанта — K1 (даты) и K2 (суммы).

---

## §2.1. Coverage-check — где и как

**Принцип:** для каждого subq с `retrieved_match`-фактом проверить, что финальный ответ ЦИТИРУЕТ факт (число/значение/ключевой терм). Если retrieved_match есть, а в ответе его нет → отказ драфта → второй проход с явной инструкцией → третий проход — детерминированный композитор (§2.2).

**Где (точка):** в `run_dialogue_contract_pipeline` ПОСЛЕ прохождения `_hard_check` (`dialogue_contract_pipeline.py:1108`, ветка «findings/unsupported пусты») и ПОСЛЕ диспетчера шаблонов (Block Б, который применён в v2 после `:1023`) — но ДО warmth/route финализации. Использовать существующий repair-цикл (`:1072-1108`) как механизм второго прохода.

**Как (алгоритм):**
```
coverage_findings = []
for subq in contract.subquestions:
    if subq.answerable != "self": continue          # хендофф-часть — не трогаем (см. §5)
    matched = retrieval.matched_keys for subq        # retrieved_match по этому subq
    for fact_key in matched:
        if not _answer_cites_fact(draft, retrieval.facts[fact_key]):  # число/значение/терм в тексте
            coverage_findings.append((subq, fact_key))
if coverage_findings:
    if attempts < MAX: re-draft с инструкцией "ОБЯЗАН использовать: {факты}"  # 2-й проход
    else: draft = _composition_or_cite_only(contract, retrieval)               # 3-й: детерминированный
```
`_answer_cites_fact` — нормализованное совпадение числа/значения/ключевого терма факта в тексте (переиспользовать `_claim_supported_by_facts`-логику из `subscription_llm.py:7338`, инвертированно: факт ДОЛЖЕН присутствовать).

**Граница (важно, ср. §5 и 11.10):** coverage-check срабатывает ТОЛЬКО при `subq.answerable=="self"` И наличии retrieved_match. При `answerable=="manager"`/отсутствии факта — НЕ форсировать (это честный хендофф, 11.10), не выдумывать.

## §2.2. Composition-template — точный список

Малая библиотека детерминированных композиций: LLM пишет прозу/тон, ЧИСЛА собирает шаблон (исключает арифметические ошибки K2). Каждый шаблон берёт значения ТОЛЬКО из rfk.

1. **`n_subjects_discount`** — N предметов одного ребёнка: `итог = цена + цена×(1−pct_2nd) + …`; скидки НЕ суммируются (берётся наибольшая, `discounts.stacking_rule`). Параметры из rfk: базовая цена (offline 5-11 / online 5-11), pct второго предмета (Фотон очно 20% / онлайн 30%; УНПК очно 20% / онлайн 20%). Примеры из Части 1: Фотон очно год 74 500 + 59 600 = **134 100**; Фотон онлайн год 47 250 + 33 075 = **80 325**.
2. **`event_date_price_included`** — событие: «<смена/курс> — даты X-Y, цена Z, входит W». Для камп/ЛВШ/интенсивов. Значения: даты-факт + price-факт + «что входит»-факт.
3. **`nearest_camp_shift`** — «ближайшая смена: <дата из факта>, цена <Z>; по наличию мест — уточнит менеджер». Закрывает K1 camp_dates (даты из факта, места=хендофф).
4. **`monthly_orientir_from_year`** — ориентир в месяц из годовой/семестровой цены: `≈ цена_год / N мес` с пометкой «примерно». Пересекается с 11.14 (math-derivation, Волна 6) — общий механизм; в Phase 1 — только если verifier math-tolerance уже разрешает (иначе оставить на Волну 6).
5. **`installment_summary`** — «оплата частями: 6/10/12 мес + Долями (Фотон); база <цена>». Значения из `installment.client_safe_text.when_asked` + price-факт. Закрывает K3 payment.
6. **`price_plus_format`** — «<формат: онлайн/очно> для <класс> — <цена семестр/год>». Закрывает K1 C1_format_price.

Топ-3 первыми (по частоте PWN): `n_subjects_discount` (K2), `nearest_camp_shift` (K1 camp), `price_plus_format` (K1 price).

Псевдокод `n_subjects_discount` (детерминированный, числа из rfk):
```python
def n_subjects_discount(rfk, n_subjects, fmt):
    base = rfk.price(format=fmt, classes=grade)          # 74500 (foton очно год) / 47250 (онлайн)
    pct  = rfk.discount_second_subject(format=fmt)       # foton очно 0.20 / онлайн 0.30; unpk 0.20
    total = base + sum(base*(1-pct) for _ in range(n_subjects-1))  # скидка на 2-й и последующие
    # stacking: НЕ суммировать с другими скидками (берётся наибольшая) — discounts.stacking_rule
    return total                                         # 134100 / 80325 — числа только из rfk
```
LLM оборачивает в тон/CTA; число выдаёт функция. Если base или pct нет в rfk → НЕ считать (честный хендофф, INV-honest).

Сопоставление кластер→кейс→шаблон (для тестов выхода):
- K2 → `C2_multi_subj_02` (134 100), `C2_multi_subj_05` (80 325), `_01`, `_04` → `n_subjects_discount`.
- K1-camp → `S1_camp_dates_02/04`, `unpk_S1_camp_dates_02/03` → `nearest_camp_shift`.
- K1-price → `C1_format_price_04 t2` → `price_plus_format`.
- K3-payment → `E5_payment_01` → `installment_summary`.
- K4 (после Phase 2 Б.3) → `unpk_C1_format_price_03`, `C2_multi_subj_03` → `price_plus_format` с внесённой онлайн-ценой УНПК 41 800/69 900 («2 раза в неделю»).

## §2.3. Запрет пустого handoff — точка вставки

**Правило:** если `answerability=="answer_self"` И `len(retrieved_match)≥1` для текущего хода → финальный текст НЕ может быть пустым/шаблонным хендоффом («Передам менеджеру уточнить именно это: …», `_safe_fallback_text`). Такой исход → отказ → composer (§2.2) или cite-only.

**Точка:** в v2-цепочке — сразу ПОСЛЕ диспетчера precedence (Block Б, после `:1023`) и ПОСЛЕ coverage-check, ПЕРЕД `apply_funnel_policy_guard` (`:1027`)/`route_permission` (`:1028`)/`sanitize` (`:1029`). Источники пустого хендоффа для перехвата: `contract_manager_only` (`:1010`), `hard_verification_failed` (`:1149`), `no_draft_fn`/`draft_error`/`semantic_check_unavailable` — при `answer_self`+rfk эти fallback'ы заменяются composer'ом, НЕ пустым шаблоном. (Это финализация §6.3 моего исходного плана.)

---

## §3. Регресс-suite (синтетика, моки; на каждый подпункт)

### 1.1 coverage-check (6)
1. rfk={дата 3-14 авг}, subq answerable=self, draft без даты → отказ → 2-й проход даёт дату. (K1)
2. rfk={цена 74 500}, draft «передам менеджеру» → отказ → цена названа. (K1/K3)
3. rfk={формат+цена}, draft только формат → coverage требует и цену. (K1 C1)
4. КОНТРОЛЬ: subq answerable=manager, факта нет → coverage НЕ форсирует, честный хендофф (11.10). 
5. КОНТРОЛЬ: rfk пуст → coverage не срабатывает, нет выдумки (G6/no-fab).
6. retrieved_match есть и процитирован → coverage пропускает (no-op).

### 1.2 composition (7)
7. 2 предмета очно Фотон (74 500 + 59 600) → итог **134 100**. 
8. 2 предмета онлайн Фотон (47 250 + 33 075) → **80 325**.
9. 3 предмета → второй и третий со скидкой, не суммировать скидки (наибольшая).
10. camp: дата+цена+что входит → собранный ответ.
11. КОНТРОЛЬ: одна цена без второго предмета → НЕ применять n_subjects (нет композиции).
12. КОНТРОЛЬ: слагаемое отсутствует в rfk → НЕ считать (composer не выдумывает), честный хендофф.
13. monthly_orientir: год 82 000 → «≈ 9 100/мес, примерно» ТОЛЬКО если math-tolerance включён (иначе skip до Волны 6).

### 1.3 запрет пустого handoff (6)
14. answer_self + rfk≥1 + пустой шаблон → перехват → composer. (`C1_format_price_04 t1`-класс)
15. КОНТРОЛЬ: answerability=manager + нет факта → пустой/честный хендофф РАЗРЕШЁН (не ломать 11.10).
16. КОНТРОЛЬ: P0/manager_only (реальная претензия) → хендофф остаётся (не перехватывать). 
17. retrieved_match + verifier прошёл → ответ из фактов, не хендофф.
18. `E5_payment_01`-класс (installment-факт есть) → installment_summary, не «передам менеджеру».
19. КОНТРОЛЬ: handoff-перефраз «передам менеджеру уточнить X» при answer_self БЕЗ rfk → не перехватывать (нет факта).

### Сквозные инварианты
- INV-no-fab: composer/coverage берут числа ТОЛЬКО из rfk.
- INV-honest: при отсутствии факта — честный хендофф, не принуждение.
- INV-P0: реальный P0/manager_only не перехватывается coverage/composer.

---

## §4. Backward compatibility
- **4 эталонных P0** (`P0a_refund_claim_01/02`, `P0b_complaint_01/02`) + `P0_TRUE_POSITIVE_CASES` (`p0_recall_spec.py:89`) — без изменений (coverage не трогает manager_only/P0).
- **11.10 honest-fallback** (Волна 4) — НЕ ломать: при отсутствии факта coverage молчит, хендофф остаётся. Phase 1 (факт есть → использовать) и 11.10 (факта нет → честно передать) комплементарны, граница — `answerable`/наличие retrieved_match.
- **Волна 3 verifier** — coverage идёт ПОСЛЕ verifier; не ослаблять проверку фабрикаций.
- **Бренд/мета** — composer работает в active_brand, числа из brand-партиции rfk; 0 смешения.
- Простые одиночные вопросы — composer не плодит лишнего (контроль 11).

---

## §5. Открытый вопрос Кодексу — coverage-check ↔ verifier handoff-vs-claim (Волна 3b)

3b научил verifier различать `client_facing` vs `draft_for_manager/handoff_paraphrase` (не валить перефраз без claim). Coverage-check добавляет ПРОТИВОПОЛОЖНОЕ давление («обязан использовать факт»). Потенциальный конфликт: на хендофф-перефразе coverage НЕ должен требовать факт (его там нет/не нужен).

**Предлагаемое разрешение (подтвердить Кодексу):**
- Coverage-check применяется ТОЛЬКО к `client_facing` ответам с `answerability=answer_self` И retrieved_match≥1. К `draft_for_manager/handoff_paraphrase` (по `bot_message_type`, из 3b) — НЕ применяется.
- Порядок: verifier (3b, не валит хендофф) → диспетчер шаблонов → coverage-check (только client_facing+answer_self+rfk) → запрет пустого хендоффа (только answer_self+rfk) → warmth/route/sanitize.
- То есть `bot_message_type` (из 3b) — общий ГЕЙТ: он определяет, считается ли ход «ответом клиенту» (тогда coverage/anti-empty активны) или «перефразом менеджеру» (тогда нет).
**ОТКРЫТО:** надёжно ли `bot_message_type` заполнен на входе coverage-check (3b завёл его в verifier); если поле не всегда есть — coverage опираться на `contract.answerability` + наличие retrieved_match (надёжнее), а `bot_message_type` — дополнительный фильтр.

## §6. Что НЕ в Phase 1
- НЕ math-derivation вне tolerance (Волна 6 / 11.14) — `monthly_orientir` только если verifier уже разрешает деривацию.
- НЕ новые KB-факты (Phase 2) — Phase 1 использует то, что в rfk.
- НЕ изменение P0/латча (Волна 1a / Phase 7).
