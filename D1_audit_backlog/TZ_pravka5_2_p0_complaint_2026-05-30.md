# ТЗ MAIN — правка 5.2: P0b-жалоба, чистый хендофф без сбора данных (полуфабрикат). 2026-05-30.

Автор: Клод 1. Полуфабрикат — Кодексу проверить чтением и перенести. Идёт в ОДИН батч с правкой 5.1
(обе мелкие, локальные; прогон M1 один на обе).

## Проблема (подтверждено чтением кода 6dd7cf2a)

2 FAIL `p0_mishandled` (P0b_complaint_02 foton/unpk): маршрут `manager_only` верный, но ТЕКСТ плохой:
- **unpk** — ответил про скидку вместо принятия жалобы;
- **foton** — попросил данные ребёнка (ПДн) на жалобе.

Корень: `_safe_fallback_text` (dialogue_contract_pipeline.py ~2179) не различает жалобу — на complaint
уходит в `_secondary_fact_text` (выдаёт вторичную справку = скидка) или в ветку «уточнить деталь»
(`question_detail`, звучит как запрос данных). При этом `classify_answer_safety` УЖЕ вычисляет
`zero_collect_required=True` для `{refund, legal, complaint}` (answer_safety_classifier.py:162), но
этот флаг нигде не читается. Для refund ветка есть (`_refund_policy_handoff_text`), для complaint — нет.

## Правка 1 — добавить complaint-хендофф (по образцу refund)

Рядом с `_REFUND_POLICY_TEXTS` / `_refund_policy_handoff_text` (~2165) добавить:

```python
_COMPLAINT_HANDOFF_TEXTS: tuple[str, ...] = (
    "Понимаю, что ситуация неприятная, и хочу, чтобы её разобрали внимательно. "
    "Передам менеджеру — он свяжется с вами и поможет.",
    "Спасибо, что написали. Такую ситуацию правильнее разобрать с менеджером — "
    "передам ему, он свяжется и во всём разберётся.",
    "Понимаю вас. Чтобы решить вопрос по существу, передам менеджеру — "
    "он свяжется с вами напрямую.",
)


def _complaint_handoff_text(*, conversation: Sequence[Mapping[str, str]] | None = None) -> str:
    bot_turns = 0
    if conversation:
        bot_turns = sum(1 for item in conversation if str(item.get("role") or "") == "bot")
    return _COMPLAINT_HANDOFF_TEXTS[bot_turns % len(_COMPLAINT_HANDOFF_TEXTS)]
```

Тексты: эмпатия + чистый хендофф, БЕЗ сбора данных, БЕЗ обещаний по сумме/решению, БЕЗ продаж.

## Правка 2 — zero_collect-ветка в начале `_safe_fallback_text`

В `_safe_fallback_text` (~2179) сразу после определения `traced(...)` и ДО `known_absence` /
`_secondary_fact_text` / `question_detail` вставить:

```python
    # P0 zero-collect: жалоба / возврат / юр-угроза — чистый хендофф,
    # без вторичной справки и без запроса данных клиента.
    safety = classify_answer_safety(
        client_message=contract.current_question or "",
        context=context,
        route="manager_only",
    )
    if safety.zero_collect_required:
        if safety.primary_risk == "complaint":
            return traced(_complaint_handoff_text(), "complaint_zero_collect")
        if safety.primary_risk == "refund":
            return traced(_refund_policy_handoff_text(), "refund_zero_collect")
        return traced(
            "Сейчас точно ответить не могу. Передам вопрос менеджеру — он свяжется с вами.",
            "p0_zero_collect",
        )
```

`classify_answer_safety` уже импортирован в этом модуле (используется в `verify_output`, ~1480) — новый
импорт не нужен. Ветка возвращает результат ДО `secondary_fact`/`question_detail`, поэтому ни справка
про скидку, ни «уточните деталь про ребёнка» на жалобе/возврате/юр-угрозе не сформируются.

## Что НЕ трогать

- `_secondary_fact_text`, `question_detail`, `generic` — для НЕ-P0 работают как раньше.
- COMPLAINT_RE (p0_recall_spec.py:40) — НЕ расширять. Он ловит «жалоб/претензи/обман/недовол/ужасн/
  плохо учит/некомпетентн». Мягкие жалобы без этих слов («ребёнок ничего не понял») он может не
  поймать — это вопрос ДЕТЕКЦИИ, отдельный кандидат, НЕ эта правка (расширение regex рискует переловлей).
  Данная правка гарантирует чистый текст КОГДА P0 детектится — этого достаточно для двух наблюдаемых FAIL.

## Тесты

- complaint → `_safe_fallback_text` возвращает текст из `_COMPLAINT_HANDOFF_TEXTS`; НЕ содержит «скидк»,
  «укажите», «ребён», «как зовут», цифр/процентов.
- refund (zero_collect) → возвращает refund-handoff, не secondary_fact.
- НЕГАТИВНЫЙ КОНТРОЛЬ: обычный вопрос (цена/формат, zero_collect_required=False) → secondary_fact /
  question_detail работают как раньше (ветка zero_collect не перехватывает).

## Ограничения

- Тесты не гонять (лимит) — Клод 1 прогонит pipeline + smoke в песочнике. В main не мержить до зелёного.
- Правило #1: место и сигнатуру `classify_answer_safety` / `_safe_fallback_text` подтвердить чтением
  (current_question — поле AnswerContract; conversation в _complaint_handoff_text опционален, можно не
  прокидывать — тогда всегда первый вариант текста, это допустимо).
- Идёт в батч с 5.1: оба коммита → один прогон целевого 45 на M1.
