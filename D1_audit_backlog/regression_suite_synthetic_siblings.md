# Регресс-suite: синтетические siblings по волнам

Автор: второй Claude. Дата: 2026-05-28. Назначение: фикстурный набор для Кодекса, отдельный от eval (do-not-tune-against).

**Правила набора:** (1) все сообщения — СИНТЕТИКА, не дословные eval-кейсы; (2) без вызовов реальной LLM — только моки фактов/контракта + детерминированные ассерты по существующим функциям; (3) каждый кейс ссылается на реальную функцию из кода (`classify_answer_safety`, `codes_from_text`, `tag_message_roles`, `find_unsupported_numeric_promises`, `find_identity_disclosure_phrases`, `apply_unstated_subject_guard`, `detect_fact_scopes`, `_mentions_contact_hours`/`_wrong_scope_fact_findings`). Сигнатуры — по факту чтения кода. Всего ~78 кейсов.

Конвенции:
```python
from mango_mvp.channels.answer_safety_classifier import classify_answer_safety
from mango_mvp.channels.p0_recall_spec import codes_from_text
from mango_mvp.channels.semantic_roles import tag_message_roles
# и т.д. — пути по факту
def p0(msg, ctx=None): return classify_answer_safety(client_message=msg, context=ctx).p0_required
def frame(msg): return tag_message_roles(msg).refund_frame
```

---

## Волна 1a — граница P0 (группы A-G + споры). ~22 кейса. Самый важный блок.

```python
# --- Группа A: реальное требование СЕЙЧАС → P0 (контроль, НЕ ослаблять) ---
# 1a-A-1
assert p0("Верните мне деньги за курс немедленно.") is True
# 1a-A-2
assert "refund" in codes_from_text("Требую возврат оплаты за обучение.")
# 1a-A-3
assert p0("Отдайте деньги обратно, я отказываюсь от обучения.") is True

# --- Группа A: НЕ P0 (намерение/субъект/объект) ---
# 1a-A-4 (гипотетика будущего — presale)
assert frame("А если потом потребую возврат, это возможно?") == "presale_policy"
assert p0("А если потом потребую возврат, это возможно?") is False
# 1a-A-5 (третье лицо: налоговая) — ЦЕЛЕВОЙ ФИКС tax
assert p0("Подскажите, когда налоговая возвращает деньги за обучение?") is False
# 1a-A-6 (объект не деньги)
assert p0("А когда вы возвращаете тетради после проверки?") is False
# 1a-A-7 (возврат к теме)
assert p0("Вернёмся к теме: сколько стоит годовой курс?") is False

# --- Группы B/C: всегда P0 (контроль) ---
# 1a-B-1
assert p0("С карты дважды списали оплату за один курс.") is True
# 1a-B-2
assert p0("Оплатил, а доступ к курсу так и не открыли.") is True
# 1a-C-1
assert p0("Буду жаловаться в Роспотребнадзор.") is True
# 1a-C-2
assert p0("Обращусь в суд, если не решите вопрос.") is True
# 1a-C-3
assert p0("Вы мошенники, это обман.") is True

# --- Группа D: «недовольство» БЕЗ фин/услуга-маркера → НЕ P0 (целевой фикс над-триггера) ---
# 1a-D-1
assert p0("Меня беспокоит, точно ли качественное преподавание у вас?") is False
# 1a-D-2 (после фикса COMPLAINT_RE: «недоволен» сам по себе не P0)
assert p0("Я пока не очень доволен тем, как идёт подбор курса.") is False
# 1a-D-3 (комбинация с фин-маркером → P0)
assert p0("Я недоволен, что списали деньги дважды.") is True

# --- Группа E: эмоция к боту → НЕ P0 (школьники тестируют) ---
# 1a-E-1
assert p0("Вы вообще не отвечаете нормально, бесполезно.") is False
# 1a-E-2
assert p0("Толку от вас нет, вы не понимаете вопрос.") is False
# 1a-E-3
assert p0("Зря потратил время на этот чат.") is False

# --- Группа F: нейтральное завершение → НЕ P0 ---
# 1a-F-1
assert p0("Ладно, посмотрим, попозже подумаю.") is False
# 1a-F-2
assert p0("Спасибо, я понял, пока всё.") is False

# --- Группа G: гипотетический возврат до оплаты → НЕ P0 (presale) ---
# 1a-G-1
assert frame("А если ребёнку не подойдёт формат, деньги вернёте?") == "presale_policy"
assert p0("А если ребёнку не подойдёт формат, деньги вернёте?") is False
# 1a-G-2
assert p0("Перед оплатой хочу понять: если не доучимся, что с возвратом?") is False

# --- Споры ---
# 1a-S1 (оплатил-передумал-верните → soft handoff: НЕ молчаливый hard P0, но и не автономный ответ)
# После фикса: decision.escalation == "soft_handoff" (новое поле). До фикса — фиксируем текущее как baseline.
assert classify_answer_safety(client_message="Я оплатил, но передумал, верните, пожалуйста.").manager_only is True
# ожидание после Волны 1a: escalation == "soft_handoff", бот даёт безопасный ответ + пометку менеджеру
# 1a-S3 (месяц не получил справку → P0 после расширения)
assert p0("Уже месяц не могу получить справку, ничего не присылаете.") is True
# 1a-S5 (претензия к преподавателю → P0/manager после расширения)
assert p0("Мне совсем не нравится, как преподаватель ведёт занятия.") is True
```

---

## Волна 0 — измерительная база + contact-hours regex. ~8 кейсов.

```python
from mango_mvp.channels.fact_claim_audit import _wrong_scope_fact_findings  # через него тестируем _mentions_contact_hours
# 0c — расписание-вопрос + контактные часы в ответе → wrong_scope (ЦЕЛЕВОЙ ФИКС regex)
# 0c-1 (формулировка «на связи ежедневно с X до Y» — сейчас НЕ ловится)
findings = _wrong_scope_fact_findings("Мы на связи ежедневно с 10:00 до 18:00.",
                                      "А по каким дням проходят занятия?", active_brand="foton", registry={})
assert any(f["claim_type"]=="contact_hours_as_class_schedule" for f in findings)
# 0c-2 (вариант «Пн-Вс 10-18»)
findings = _wrong_scope_fact_findings("Пн-Вс 10:00-18:00.", "В какие дни занятия?", active_brand="foton", registry={})
assert any(f["level"]=="wrong_scope" for f in findings)
# 0c-3 КОНТРОЛЬ: вопрос про часы связи + часы связи → НЕ wrong_scope
findings = _wrong_scope_fact_findings("Мы на связи с 10 до 18.", "Когда вы на связи?", active_brand="foton", registry={})
assert not any(f["claim_type"]=="contact_hours_as_class_schedule" for f in findings)
# 0c-4 КОНТРОЛЬ: реальное расписание занятий (с 16:00 до 17:30 по вторникам) → НЕ помечать как контактные часы
findings = _wrong_scope_fact_findings("Занятия по вторникам с 16:00 до 17:30.", "Когда занятия?", active_brand="foton", registry={})
assert not any(f["claim_type"]=="contact_hours_as_class_schedule" for f in findings)

# 0a — should_auto_trip только в live (мок replay)
# 0a-1
assert should_auto_trip(is_live_runtime=False, gate_anomaly=False) is False  # replay/синтетика не валит
# 0a-2
assert should_auto_trip(is_live_runtime=True, gate_anomaly=True) is True     # live + реальная аномалия

# 0b — evaluate_night_gate принимает Mapping[str,str]
# 0b-1
res = evaluate_night_gate(retrieved_facts={"price.year": "82 000 ₽"})
assert res.fact_levels.get("retrieved_match", 0) >= 1
```

---

## Волна 3a — unsupported_promise при retrieved_match. ~8 кейсов.

```python
from mango_mvp.channels.subscription_llm import find_unsupported_numeric_promises
def ctx_facts(*texts): return {"confirmed_facts": {f"k{i}": t for i,t in enumerate(texts)}}
# 3a-1: процент совпадает с фактом → НЕ unsupported
assert find_unsupported_numeric_promises("Скидка на второй предмет 20%.",
        context=ctx_facts("скидка на второй предмет 20%")) == ()
# 3a-2: срок совпадает с фактом → НЕ unsupported
assert find_unsupported_numeric_promises("Справку пришлём за 10 рабочих дней.",
        context=ctx_facts("справка до 10 рабочих дней")) == ()
# 3a-3: цена совпадает с фактом → НЕ unsupported
assert find_unsupported_numeric_promises("Очно год — 82 000 ₽.",
        context=ctx_facts("очно год 82 000 ₽")) == ()
# 3a-4 КОНТРОЛЬ: число, которого НЕТ в фактах → unsupported (не ослаблять)
assert find_unsupported_numeric_promises("Гарантируем 100 баллов на ЕГЭ.",
        context=ctx_facts("скидка 20%")) != ()
# 3a-5 КОНТРОЛЬ: выдуманная цена → unsupported
assert find_unsupported_numeric_promises("Онлайн стоит 5 000 ₽ в месяц.",
        context=ctx_facts("очно год 82 000 ₽")) != ()
# 3a-6: процент в другой формулировке, но факт есть → НЕ unsupported
assert find_unsupported_numeric_promises("Второй предмет — минус 20 процентов.",
        context=ctx_facts("скидка на второй предмет 20%")) == ()
# 3a-7: пустой context → не падать, числа без подтверждения → unsupported
assert find_unsupported_numeric_promises("Цена 49 000 ₽.", context={}) != ()
# 3a-8: verified safe template → () (whitelist _is_verified_safe_numeric_template)
# (мок: draft == один из *_SAFE_TEXT) → ()
```

---

## Волна 3b — verifier различает handoff-перефраз vs claim. ~6 кейсов.

```python
# Тестируем поведение _hard_check/route: handoff-перефраз без утверждения НЕ должен валиться в hard_verification_failed.
def run(msg, draft, contract, facts): ...  # обёртка над пайплайном с мок draft_fn
# 3b-1: «Передам менеджеру уточнить: <перефраз вопроса>» при answer_self+rfk → НЕ hard_verification_failed, не пустой хендофф
res = run("онлайн или очно и цена?", draft="<facts answer>", contract=mk(answerability="answer_self", subqs=2), facts={"online":"...","offline":"..."})
assert res.fallback_reason != "hard_verification_failed"
assert res.fallback_reason != "contract_manager_only"
# 3b-2: чистый handoff-перефраз (msg_type=draft_for_manager) → claim-сверка пропускается
assert verify_mode(bot_message_type="draft_for_manager") == "skip_claim_check"
# 3b-3 КОНТРОЛЬ: client_facing с выдуманным числом → всё ещё валится
res = run("цена?", draft="онлайн 5 000 ₽", contract=mk(answerability="answer_self"), facts={"y":"82 000"})
assert res.findings  # фабрикация ловится
# 3b-4: пустой safe_fallback не должен содержать «Клиент спрашивает …» (правка 11.12, связка)
assert "Клиент" not in res_safe_fallback_text
# 3b-5: при answer_self + непустой rfk пустой handoff запрещён (§6.3)
assert not is_empty_handoff(run("цена 7 класс онлайн?", draft="", contract=mk(answerability="answer_self"), facts={"sem":"29 750","year":"47 250"}))
# 3b-6 КОНТРОЛЬ: при answerability=manager и нет факта → handoff допустим
assert run("дайте телефон директора", draft="", contract=mk(answerability="manager"), facts={}).route in ("draft_for_manager","manager_only")
```

---

## Волна 3c — verifier и сужающие подвопросы. ~6 кейсов.

```python
# 3c-1: после общего ответа узкий подвопрос «а второй предмет дешевле?» — сверка с подмножеством rfk
res = run_narrow(prev="цены...", msg="а второй предмет дешевле?", facts={"second_pct":"20%","stack":"не суммируются"})
assert res.fallback_reason != "hard_verification_failed"
# 3c-2: «помесячно?» при наличии installment-факта → не валить
res = run_narrow(prev="...", msg="а помесячно можно?", facts={"installment":"6/10/12 + Долями"})
assert res.fallback_reason != "hard_verification_failed"
# 3c-3: узкий «во сколько в субботу?» при факте про слоты → ответ из релевантного факта
# 3c-4 КОНТРОЛЬ: узкий вопрос, факта НЕТ → честный микро-handoff (не выдумка)
res = run_narrow(prev="...", msg="а есть скидка многодетным?", facts={})
assert res.route in ("draft_for_manager","manager_only")
# 3c-5: drafted-claim точно соответствует ОДНОМУ факту rfk + ключевые слова subq → verifier пропускает
# 3c-6 КОНТРОЛЬ: claim не соответствует ни одному факту подмножества → валится
```

---

## Волна 2 — редактор-шум + sanitize + brand-relationship. ~8 кейсов.

```python
from mango_mvp.channels.subscription_llm import find_identity_disclosure_phrases
# 2b редакторы (insights/sanitizers): client-safe бренд-контакт НЕ редактируется
# 2b-1
assert "phone_redacted" not in sanitize_flags("Телефон: 8 (495) 500-25-88.", brand_whitelist=True)
# 2b-2: топоним Москвы не person_name_redacted
assert "person_name_redacted" not in sanitize_flags("Адрес: Сретенка.", brand_whitelist=True)
# 2b-3: «5-6 раз в год» не deadline_redacted
assert "deadline_redacted" not in sanitize_flags("Контрольные 5-6 раз в год.")
# 2b-4 КОНТРОЛЬ: чужой телефон клиента в тексте → всё ещё редактируется
assert "phone_redacted" in sanitize_flags("Мой номер 8 900 123 45 67.")
# 2a sanitize 3-е лицо «Клиент …» (включая шаблон _safe_fallback_text)
# 2a-1
assert not leaks_third_person("Передам менеджеру уточнить: Клиент спрашивает цену.")  # после фикса
# 2a-2 КОНТРОЛЬ: нормальный ответ без «Клиент» проходит
assert leaks_third_person("Подскажу цену: год — 82 000 ₽.") is False
# 2c identity (подстрока → граница слова)
# 2c-1: бенайн-драфт с «как и интенсив» НЕ должен ловиться (после перехода на границы слов)
assert find_identity_disclosure_phrases("Подготовим к олимпиадам, как и к интенсиву.") == ()
# 2c-2 КОНТРОЛЬ: реальная утечка «я бот» / «GPT» → ловится
assert find_identity_disclosure_phrases("Я бот на базе GPT.") != ()
```

---

## Волна 4 — honest fallback / compound / CTA. ~8 кейсов.

```python
# 4a honest-fallback: subq с answerable=manager → бот явно «по X нет данных, передам менеджеру», НЕ подмена
# 4a-1
res = run("как зайти в личный кабинет?", facts={})  # факта про ЛК нет
assert says_honest_handoff(res, topic="личный кабинет")
assert not substitutes_other_topic(res)
# 4a-2 КОНТРОЛЬ: subq с answerable=self и факт есть → отвечает, не handoff
res = run("сколько стоит онлайн 7 класс?", facts={"sem":"29 750","year":"47 250"})
assert gives_price(res)
# 4c compound: «X и Y» → subqs >= 2
# 4c-1
assert len(decompose("онлайн или очно, и сколько стоит для 6 класса?")) >= 2
# 4c-2 КОНТРОЛЬ: одиночный вопрос → не плодит лишние subqs
assert len(decompose("сколько стоит онлайн?")) == 1
# 4c-3: обе части compound адресованы (coverage-check)
res = run("формат и цена 7 класс онлайн?", facts={"fmt":"вебинары МТС","sem":"29 750"})
assert covers_all_subqs(res)
# 4b CTA на «как записаться»
# 4b-1
res = run("как записаться на пробное?", facts={"trial":"..."})
assert has_cta(res)  # конкретное предложение времени/«менеджер свяжется», не голый список контактов
# 4b-2 КОНТРОЛЬ: «расскажите про курс» → CTA не обязателен
assert run("расскажите про курс", facts={"course":"..."}).ok
```

---

## Волна 5 — unstated_subject / locked slots. ~6 кейсов.

```python
from mango_mvp.channels.subscription_llm import apply_unstated_subject_guard
def flags_after_guard(draft, msg, ctx): 
    return apply_unstated_subject_guard(mk_result(draft), client_message=msg, context=ctx).safety_flags
# 5a-1: subject назван в текущем сообщении, бот ответил про него → guard НЕ срабатывает
assert "unstated_subject_guarded" not in flags_after_guard("По физике для 9 класса есть курс.", "Есть физика 9 класс?", {})
# 5a-2: subject в confirmed_slots (назван ранее) → guard НЕ срабатывает
ctx = {"dialogue_memory_view": {"client_confirmed_slots": {"subject": "математика"}}}
assert "unstated_subject_guarded" not in flags_after_guard("По математике есть онлайн.", "а онлайн есть?", ctx)
# 5a-3 КОНТРОЛЬ: бот дописал предмет, который клиент НЕ называл → guard срабатывает
assert "unstated_subject_guarded" in flags_after_guard("Есть математика, физика и информатика.", "что есть по физике?", {})
# 5a-4: склонение «по физике» → распознан как тот же subject
assert "unstated_subject_guarded" not in flags_after_guard("Физика есть.", "интересует занятие по физике", {})
# 5b locked slot: класс назван на t1, не теряется на t3
# 5b-1
assert locked_slots_after(["9 класс", "...", "а онлайн?"]).get("grade") == "9"
# 5b-2 КОНТРОЛЬ: слот не выставлен, если клиент не называл
assert "grade" not in locked_slots_after(["сколько стоит?"])
```

---

## Волна 6 — недетерминизм verifier + math-derivation. ~8 кейсов.

```python
# 6a кэш (claim_signature, rfk_signature) в пределах диалога
# 6a-1: тот же claim+rfk на t2 после прохождения на t1 → не валится повторно
assert verify_cached(claim="год 82 000 ₽", rfk_ids=["price.year"], prev_passed=True) == "pass_cached"
# 6a-2: перефраз того же claim (норм. до ядра) → тоже из кэша
assert verify_cached(claim="82 000 в год", rfk_ids=["price.year"], prev_passed=True) == "pass_cached"
# 6a-3 КОНТРОЛЬ: другой rfk (другой контекст/бренд) → НЕ из кэша, проверяется заново
assert verify_cached(claim="год 82 000 ₽", rfk_ids=["other.price"], prev_passed=True) != "pass_cached"
# 6a-4 КОНТРОЛЬ: 3 hard_gate валятся через scope (Phase 3), НЕ через кэш
# 6b math-derivation
# 6b-1: месячный ориентир из годовой цены (82 000/9 ≈ 9 100) с «примерно» → НЕ unsupported
assert find_unsupported_numeric_promises("Это примерно 9 100 ₽ в месяц.",
        context=ctx_facts("год 82 000 ₽")) == ()
# 6b-2: подтверждение арифметики клиента (82 000 + 82 000*0.8 = 147 600), слагаемые в rfk → НЕ unsupported
assert find_unsupported_numeric_promises("Да, 147 600 ₽ верно.",
        context=ctx_facts("предмет 82 000 ₽","скидка на второй 20%")) == ()
# 6b-3 КОНТРОЛЬ: произвольное число, не деривация из rfk → unsupported
assert find_unsupported_numeric_promises("Это 5 000 ₽ в месяц.",
        context=ctx_facts("год 82 000 ₽")) != ()
# 6b-4: округление до сотен + «примерно» (не дробные копейки)
```

---

## Сквозные контрольные инварианты (включать в каждый прогон волны)

```python
# INV-1: 4 эталонных P0 ВСЕГДА срабатывают (любая волна, проверка регресса)
for case,_ in P0_TRUE_POSITIVE_CASES:  # p0_recall_spec.py:89
    assert p0(case) is True
# INV-2: P0_BENIGN_CASES (p0_recall_spec.py:110) ВСЕГДА не-P0
for case in P0_BENIGN_CASES:
    assert p0(case) is False
# INV-3: бренд-разделение — ни один ответ не содержит обоих брендов одновременно (на синтетике)
# INV-4: identity — реальная «я бот/GPT» в драфте всегда ловится
```

## Открытые вопросы

1. Кейсы 3b/3c/4 написаны как поведенческие (обёртка `run`/`run_narrow` над пайплайном с мок `draft_fn`) — Кодексу определить точную тестовую обёртку (функции `_hard_check`/`run_dialogue_contract_pipeline` приватные/требуют draft_fn-мок).
2. `should_auto_trip`/`evaluate_night_gate` (Волна 0) — сигнатуры по плану, не по прямому чтению `night_funnel_shadow.py` (не открывал за этот проход).
3. `sanitize_flags`/`leaks_third_person`/`has_cta`/`decompose` — псевдо-обёртки; реальные имена в `insights/sanitizers.py` и `answer_quality_rewriter.py` Кодексу подставить.
4. Кейс 1a-S1 (soft handoff) и 1a-S3/S5 (расширения) предполагают правки Волны 1a; до них — фиксируют целевое поведение, не текущее (помечено в комментариях).
5. Набор — siblings, не eval. Числа/фразы синтетические; при совпадении с eval-кейсом заменить.
