# ТЗ MAIN — 3 правки остаточных выдумок одним проходом (полуфабрикат). 2026-05-31.

Автор: Клод 1. Точки проверены чтением 9e874ccf. Три отдельных коммита в одной ветке. Лимиты Кодекса
сброшены — Кодекс САМ гонит pytest+smoke перед каждым коммитом; Клод 1 финально проверяет в песочнике
перед мержем. Один прогон в конце (можно на 2 мака — половина набора на каждый).

## Почему детерминированно (вывод регрейда 31.05)

Остаток выдумок после цепочки памяти — 3 типа, и LLM-критик их не ловит надёжно (на пустом/частичном
факте не вызывается или саботируется обогащением). Поэтому чиним в `verify_output` (pipeline.py:1778) —
детерминированный верификатор, который УЖЕ ловит fact_grounding (числа вне фактов), brand_leak, meta_leak
по тексту. Добавляем 3 чека там же. Это надёжнее промпта судьи.

`verify_output` сигнатуру и доступ к `facts`, `contract`, `previous_bot_texts` подтвердить чтением
(previous_bot_texts собираются в `_hard_check` ~1713: `[item.text for item in conversation if role=="bot"]`
— для коммита 3 прокинуть их в verify_output новым параметром).

---

## КОММИТ 1 — преждевременный выбор формата (t1)

Проблема: клиент «онлайн или очно для 6 класса?» → бот «Это онлайн». `_augment_with_format_guidance`
(2785) добавляет ТОЛЬКО онлайн-факт (`format.online`), и LLM-критик считает «это онлайн» supported.
Детектор вопроса уже есть: `_asks_training_format_choice` (2968).

Правка — в `verify_output` добавить чек:
```python
    # Преждевременный формат: клиент спросил выбор онлайн/очно, а бот навязал один формат.
    if _asks_training_format_choice(contract) and not _contract_mentions_camp_or_lvsh(contract):
        asserts_single = re.search(r"\bэто\s+онлайн\b|\bтолько\s+онлайн\b|\bэто\s+очно\b|\bтолько\s+очно\b", low)
        mentions_both = ("онлайн" in low and "очно" in low)
        if asserts_single and not mentions_both:
            findings.append(VerificationFinding(
                "preemptive_format",
                "клиент спросил выбор формата, а ответ навязывает один формат без альтернативы"))
```
Доп. (желательно) — инструкция в `build_draft_prompt`: «если клиент спрашивает онлайн или очно — назови
доступные форматы, не выбирай один за клиента». Но verify-чек — основной барьер.

Тест: «онлайн или очно для 6» + draft «Это онлайн» → finding preemptive_format.
NEG: клиент сам сказал «онлайн» (не выбор) → не флагать; ответ «есть и онлайн, и очно» → не флагать;
вопрос про лагерь/смену → не флагать (там формат фиксирован).

---

## КОММИТ 2 — выдуманная специфика расписания («в будни»)

Проблема: тема найдена (олимпиада/курс), но бот добавил «занятия проходят в будни» без факта-расписания.
Критик-правило 5.1 (pipeline.py:902 «РАСПИСАНИЕ/ДНИ/ВРЕМЯ … unsupported») есть, но просачивается.
Маркеры дней уже есть: `condition:weekdays` (1022: «по будням», «будни», «будний»).

Правка — в `verify_output` детерминированный чек (как meta_leak):
```python
    _SCHEDULE_DAY_MARKERS = ("по будням", "в будни", "будни", "по выходным",
                             "по вторникам", "по понедельникам", "по средам", "по четвергам",
                             "по пятницам", "вечерам", "вечером", "утрам", "по утрам")
    fact_text_low = " ".join(str(v) for v in facts.values()).casefold().replace("ё", "е")
    said_days = [m for m in _SCHEDULE_DAY_MARKERS if m in low]
    fact_has_schedule = any(m in fact_text_low for m in _SCHEDULE_DAY_MARKERS) or "расписани" in fact_text_low
    if said_days and not fact_has_schedule and not _client_mentioned(said_days, client_words):
        findings.append(VerificationFinding(
            "unconfirmed_schedule",
            f"ответ называет дни/время без факта-расписания: {said_days}"))
```
Тест: draft «занятия проходят в будни», facts без расписания → finding unconfirmed_schedule.
NEG: факт-расписание содержит «по будням» → не флагать; клиент сам спросил «по будням?» → не флагать
(деталь из вопроса, не выдумка); ответ без дней → не флагать.

---

## КОММИТ 3 — противоречие собственному числу/обещанию

Проблема (нить unpk_06): t2 бот «скидка 14%», t3 «скидка 10%». fact_grounding не ловит (10% валидно
где-то в KB), но это противоречит ранее СКАЗАННОМУ ботом по той же теме. `last_bot_commitments` хранит
только действия, не числа. Но `previous_bot_texts` доступны (прокинуть в verify_output).

Правка — новый параметр `previous_bot_texts` в verify_output + чек:
```python
def verify_output(..., previous_bot_texts: Sequence[str] = ()) -> list[VerificationFinding]:
    ...
    # Противоречие ранее названному ботом проценту по той же сущности (скидка/цена).
    cur_pcts = set(re.findall(r"(\d{1,2})\s*%", text))
    if cur_pcts and "скидк" in low:
        for prev in previous_bot_texts:
            prev_low = prev.casefold()
            if "скидк" not in prev_low:
                continue
            prev_pcts = set(re.findall(r"(\d{1,2})\s*%", prev))
            conflict = prev_pcts and cur_pcts and prev_pcts.isdisjoint(cur_pcts)
            if conflict:
                findings.append(VerificationFinding(
                    "self_contradiction",
                    f"процент скидки противоречит ранее названному ботом: было {sorted(prev_pcts)}, стало {sorted(cur_pcts)}"))
                break
```
В `_hard_check` прокинуть `previous_bot_texts` (они уже собираются там, ~1713) в вызов verify_output.
Тест: prev bot «скидка 14%», draft «скидка 10%» → finding self_contradiction.
NEG: то же число (14%→14%) → не флагать; разные сущности (скидка vs предоплата) → не флагать; нет
previous_bot_texts → не флагать.

---

## Маршрутизация findings

Все 3 finding-кода (preemptive_format, unconfirmed_schedule, self_contradiction) должны вести к
repair-петле (как другие findings verify_output) → если repair не убрал, безопасный fallback /
draft_for_manager (НЕ выдумка клиенту). Подтвердить, что новые коды попадают в существующую обработку
findings в run_pipeline (как fact_grounding). Если нужен явный список — добавить туда.

## Замер

- Кодекс гонит pytest+smoke перед каждым коммитом (лимиты сброшены).
- Клод 1 финально проверяет каждый коммит в песочнике перед мержем.
- Один прогон в конце: батч 45 (контроль — упали ли E8/S2_04 формат, E7 расписание) + нить 14
  (unpk_06 противоречие, unpk_02 расписание). Можно 2 мака: батч на одном, нить на другом.

## Ограничения

- Каждый защитный чек — с НЕГАТИВНЫМ контролем (не флагать законные случаи: клиент сам назвал формат/дни,
  факт содержит расписание, то же число). Это критично — иначе вырастет over-handoff.
- Хирургично: 3 чека в verify_output + 1 проброс параметра + опц. draft-инструкция. Не трогать память
  и критик-LLM (они уже работают на своих типах).
- Правило #1: сигнатуру verify_output, _asks_training_format_choice, сбор previous_bot_texts, обработку
  findings — подтвердить чтением.
