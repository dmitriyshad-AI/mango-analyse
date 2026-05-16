# ТЗ-X: Analyze.py refactor — улучшение качества извлечения контекста разговоров

Дата: 2026-05-14
Автор: Claude (наставник Дмитрия по ИИ-проекту Mango Analyse)
Адресат: Codex как primary implementation agent в отдельном диалоге
Связанный документ: `Foton/Mango_Analyse_Data_Quality_Improvement_TZ_FOR_CODEX_2026-05-14.md` (системное ТЗ из 18 правок, этот трек — часть P1 + часть P2)

---

## 0. Контекст и проблема

Дмитрий в начале работы над проектом конкретно жаловался: «Codex плохо улавливает контекст разговоров». Полный аудит pipeline (6 субагентов 14 мая) нашёл **пять конкретных мест в `services/analyze.py`**, где код теряет, искажает или неправильно агрегирует данные. Каждая точка по отдельности — это small fix. Вместе они дают **измеримое улучшение качества входа** для всего downstream pipeline (catalog, deal-aware, customer timeline).

Этот ТЗ закрывает все пять правок одним координированным треком. Файлы — изолированы от других параллельных задач Codex'а (catalog v2 D.2, deal-aware writeback rollout, ТЗ-Y sanitizer, ТЗ-Z hygiene).

**Граница ответственности этого ТЗ:** только `services/analyze.py`, `services/resolve.py`, `services/config.py` и тесты в `tests/test_analyze*.py`. **Не трогать** `quality/`, `insights/`, `customer_timeline/`, `question_catalog/`, `deal_aware/` — это зоны других треков.

---

## 1. Пять правок

### Правка X.1 — Compact/Full асимметрия rule-hints

**Файл:** `src/mango_mvp/services/analyze.py:787-825`

**Текущая логика:**
```python
if normalized == "compact":
    hints_payload = self._prune_prompt_payload(self._analysis_rule_hints(call, text))
    prompt += "Deterministic hints JSON: ..."
prompt += "Transcript: ..."
```

**Проблема:** Compact-режим получает hints, Full-режим — **НЕТ**. Full вызывается как эскалация при слабом выводе compact. Без hints Full без подсказок выдаёт тот же пустой результат → эскалация на ~4.6% звонков бесполезна. Это **инвертированная логика**: full должен иметь **больше** контекста, не меньше.

**Что сделать:**

Добавить rule-hints и в Full-промпт. Compact оптимизирован под cost, Full под качество. Full с hints + большим контекстом транскрипта должен реально давать лучше.

Изменение:
```python
# Было: только compact получает hints
if normalized in {"compact", "full"}:
    hints_payload = self._prune_prompt_payload(self._analysis_rule_hints(call, text))
    if hints_payload:
        prompt += f"Deterministic hints JSON: {json.dumps(hints_payload, ensure_ascii=False)}\n\n"
prompt += "Transcript: ..."
```

**Тест-якорь:** `tests/test_analyze.py::test_full_profile_includes_rule_hints` — проверяет, что full-промпт содержит `Deterministic hints JSON`.

**Acceptance:** на 100 эскалированных звонках (из истории, можно из `stable_runtime/`) full-результат отличается от compact-результата по ≥ 5 структурным полям (people, student, interests, commercial, objections, next_step) в ≥ 30% случаев. До правки гипотеза — отличий <10%.

---

### Правка X.2 — _compose_history_summary теряет commercial и school

**Файл:** `src/mango_mvp/services/analyze.py:1342-1484`

**Проблема:** LLM извлекает в structured_fields следующие поля:
- `student.school` (название школы ребёнка)
- `commercial.price_sensitivity` (чувствительность к цене: высокая/средняя/низкая)
- `commercial.budget` (бюджет если упоминался)
- `commercial.discount_interest` (есть ли интерес к скидкам)
- `lead_priority` (низкий/средний/высокий)

Все эти поля **доступны** в `structured_fields`, но `_compose_history_summary` их **игнорирует** при сборке финального текста для CRM. История получается обеднённой: «менеджер общался с клиентом + список тегов», без эмоциональной/мотивационной/коммерческой картины.

**Что сделать:**

Расширить `_compose_history_summary` блоками:

1. **Школа ученика** — если `student.school` непусто, включить в раздел «Об ученике»: «Школа: {school}».
2. **Коммерческий блок** — новый раздел в истории:
   ```
   Коммерческий контекст:
   - Чувствительность к цене: {commercial.price_sensitivity}
   - Бюджет: {commercial.budget}
   - Интерес к скидкам: {commercial.discount_interest}
   ```
   Включать только если хотя бы одно из трёх полей непусто.
3. **Lead priority** — если ≥ `medium`, добавлять отметку «Приоритет лида: {lead_priority}» в конце сводки.

**Тест-якорь:** `tests/test_analyze.py::test_history_summary_includes_commercial_block` — проверяет, что фикстура со заполненным commercial возвращает summary с этими полями.

**Acceptance:** на 50 случайных звонках из `call_records` SQLite, где LLM заполнил commercial поля — history_summary содержит эти поля в ≥ 90% случаев. Длина history_summary растёт в среднем на 15-25%, не на 80% (защита от перегрузки).

---

### Правка X.3 — Filler-токены семантически значимые

**Файл:** `src/mango_mvp/services/analyze.py:218-228, 594-616, 672-679`

**Текущая логика:**
```python
PROMPT_COMPACTION_FILLER_TOKENS = {"ага", "алло", "да", "ладно", "понятно", "спасибо", "угу", "хорошо", "ясно"}
```

**Проблема:** Слова `да`, `ладно`, `хорошо`, `спасибо` — могут быть полноценным согласием клиента на покупку. Кейс потери:

```
Менеджер: Запишем на пробное занятие?
Клиент: Да.
Менеджер: Тогда подтверждаю.
Клиент: Да.
```

В compact-режиме `_compact_prompt_filler_body` дедуплицирует второе «Да» (логика: две подряд реплики одного спикера, состоящие только из filler — вторая выбрасывается). **Теряется согласие на запись.**

**Что сделать:**

Сузить список filler-токенов до тех, что действительно семантически пустые:

```python
PROMPT_COMPACTION_FILLER_TOKENS = {"ага", "алло", "понятно", "угу", "ясно"}
# Убраны: "да", "ладно", "хорошо", "спасибо" — могут быть commitment'ом
```

Дополнительно проверить `_filler_only_signature` (analyze.py:608-615) — если он использует тот же список через ссылку, правка автоматическая. Если хардкод — поправить там тоже.

**Тест-якорь:** `tests/test_analyze.py::test_consent_da_preserved_in_compact_prompt`:
```python
def test_consent_da_preserved_in_compact_prompt():
    transcript = "MANAGER: Запишем?\nCLIENT: Да.\nMANAGER: Подтверждаю.\nCLIENT: Да."
    compacted = _compact_prompt_filler_body(transcript)
    assert compacted.count("Да") == 2, "Both consent 'Да' must be preserved"
```

**Acceptance:** регрессионный тест зелёный. Дополнительно: на 100 случайных звонках с шаблоном «менеджер предлагает + клиент короткое согласие» — все случаи согласия сохраняются в compacted prompt.

---

### Правка X.4 — Smart-chunking для длинных транскриптов

**Файл:** `src/mango_mvp/services/analyze.py:692-702, 626 (_compact_transcript_for_prompt)`

**Текущая логика:**
```python
return transcript[:head] + "\n[... transcript truncated ...]\n" + transcript[-tail:]
```

Тупая head+tail обрезка. Compact: head 4600 + tail 1600 + маркер truncation. Срабатывает на ~5% звонков (где транскрипт >6500 chars). Но это **самые ценные** 10-минутные звонки — теряется середина = презентация продукта + возражения + закрытие.

**Что сделать:**

Заменить head+tail на keyword-density smart-chunking. Логика deterministic, без LLM-overhead:

```python
def _smart_chunk_transcript(transcript: str, max_chars: int) -> str:
    """Выбирает наиболее информативные окна из транскрипта по keyword density."""
    if len(transcript) <= max_chars:
        return transcript
    
    # 1. Разбить на окна по 200 char (с overlap 50 char для контекста)
    windows = _split_into_windows(transcript, window_size=200, overlap=50)
    
    # 2. Скоринг каждого окна
    keyword_weights = {
        "product_pattern": 5,    # PRODUCT_PATTERNS из normalization.py
        "subject_pattern": 5,    # SUBJECT_PATTERNS
        "objection_pattern": 7,  # OBJECTION_PATTERNS
        "price_pattern": 7,      # MONEY_AMOUNT_RE или числа+руб
        "client_speech_char": 1, # 1 за каждый char реплики клиента
    }
    
    scored_windows = []
    for w in windows:
        score = _score_window(w, keyword_weights)
        scored_windows.append((score, w))
    
    # 3. Отсортировать по score, взять top-N до бюджета
    scored_windows.sort(reverse=True)
    selected = []
    used_chars = 0
    for score, w in scored_windows:
        if used_chars + len(w.text) > max_chars:
            break
        selected.append(w)
        used_chars += len(w.text)
    
    # 4. Восстановить хронологический порядок
    selected.sort(key=lambda w: w.start_pos)
    
    # 5. Склеить с маркерами пропусков
    result = []
    last_end = 0
    for w in selected:
        if w.start_pos > last_end:
            result.append("\n[... фрагмент пропущен ...]\n")
        result.append(w.text)
        last_end = w.end_pos
    if last_end < len(transcript):
        result.append("\n[... конец пропущен ...]\n")
    
    return "".join(result)
```

**Важно:** реализация должна быть deterministic (одинаковый вход → одинаковый выход), без random sampling. И **не** использовать LLM для scoring — это пере-усложнение.

**Тест-якорь:** `tests/test_analyze.py::test_smart_chunking_preserves_objections`:
```python
def test_smart_chunking_preserves_objections():
    long_transcript = (
        "MANAGER: Добрый день. (3000 char филлера)... "
        "CLIENT: Дорого, не потяну. "
        "MANAGER: (2000 char ответа)... "
        "CLIENT: Договорились, оформляем. "
        "MANAGER: (1000 char закрытия)..."
    )  # длина ~6500, как раз compact limit
    chunked = _smart_chunk_transcript(long_transcript, max_chars=4000)
    assert "Дорого, не потяну" in chunked, "Objection must be preserved"
    assert "Договорились, оформляем" in chunked, "Commitment must be preserved"
```

**Acceptance:** 
1. На 100 длинных звонках (>= 8000 chars) recall ключевых событий (PRODUCT/PRICE/OBJECTION mentions) ≥ 90% после smart-chunking vs ~60% после head+tail.
2. На звонках < max_chars (короткие) — выход идентичен входу (функция работает идемпотентно).
3. Performance: smart-chunking не должен быть медленнее head+tail более чем в 3× (это синхронный код, должен быть быстрым).

---

### Правка X.5 — Pre-LLM gate false positives

**Файл:** `src/mango_mvp/services/analyze.py:91-118 (STRONG_NON_CONVERSATION_MARKERS), 1205-1285 (_apply_non_conversation_hard_validation)`

**Проблема:**

`STRONG_NON_CONVERSATION_MARKERS` содержит фразы типа «все разговоры записываются», «вас приветствует компания» — это **стандартные compliance-преамбулы** в начале реальных звонков, не маркеры non_conversation. Когда они срабатывают, реальный звонок помечается как non_conversation, LLM не вызывается, в БД попадает заглушка.

Дополнительно: `_apply_non_conversation_hard_validation` (1205-1285) **затирает** результат LLM если эвристика говорит non_conversation, даже если LLM явно извлёк `interests.products` или `next_step.action` (то есть видел сделку).

**Что сделать:**

**Шаг 1.** Разделить STRONG_NON_CONVERSATION_MARKERS на два класса:
```python
# Безусловные маркеры — звонок гарантированно non_conversation
STRONG_NON_CONVERSATION_MARKERS = {
    "продолжение следует",
    "голосовой ассистент",
    "оставьте сообщение",
    "после сигнала",
    "набранный номер",
    "коллекторская организация",
    "целевые финансы",
    "нажмите 1",
    # ... остальное безусловное
}

# Compliance-маркеры — strong ТОЛЬКО если занимают >50% транскрипта
COMPLIANCE_OPENING_MARKERS = {
    "все разговоры записываются",
    "вас приветствует компания",
    "ваш звонок очень важен",
    # ... compliance-преамбулы
}
```

**Шаг 2.** В `_detect_call_type` (или где сейчас проверяются маркеры) — для COMPLIANCE_OPENING_MARKERS:
```python
def _is_compliance_dominated(text: str) -> bool:
    """Compliance-преамбула доминирует если занимает >50% текста."""
    compliance_chars = 0
    for marker in COMPLIANCE_OPENING_MARKERS:
        # Подсчёт длины окружения маркера до следующего реального содержания
        ...
    return compliance_chars / len(text) > 0.5
```

Compliance-маркер сам по себе **не** триггерит non_conversation. Только если >50% транскрипта — compliance.

**Шаг 3.** В `_apply_non_conversation_hard_validation` (1205-1285) добавить **escape hatch**:
```python
def _apply_non_conversation_hard_validation(self, analysis, signals, transcript):
    if analysis.get("call_type") == "non_conversation":
        # Escape hatch: если LLM явно увидел сделку — не затирать
        sf = analysis.get("structured_fields", {})
        has_product = bool(sf.get("interests", {}).get("products"))
        has_next_step = bool(sf.get("next_step", {}).get("action"))
        has_objection = bool(sf.get("objections"))
        
        if has_product or has_next_step or has_objection:
            # LLM явно увидел продажный контекст — доверяем ему, не затираем
            analysis["call_type"] = "sales_call"  # переопределить
            analysis["quality_flags"]["llm_overrode_non_conversation_gate"] = True
            return analysis
        
        # Иначе — обычное затирание
        return self._non_conversation_analysis(signals)
    return analysis
```

**Тест-якорь:**
```python
def test_compliance_preamble_does_not_trigger_non_conversation():
    transcript = (
        "Все разговоры записываются для контроля качества. "
        "MANAGER: Добрый день, по вашему запросу о подготовке к ЕГЭ по математике... "
        "CLIENT: Да, интересует, расскажите про стоимость."
    )
    result = analyze(transcript)
    assert result["call_type"] != "non_conversation"

def test_llm_override_when_extracted_sales_signal():
    """Если LLM увидел продукт + next_step, не затирать как non_conversation."""
    fixture_with_compliance_preamble_but_real_sale = ...
    result = analyze(fixture)
    assert result["call_type"] == "sales_call"
    assert result["quality_flags"].get("llm_overrode_non_conversation_gate") is True
```

**Acceptance:**
1. Регрессия на 50 звонках с compliance-преамбулой — 0 помечены non_conversation (сейчас гипотеза ~5-10).
2. Регрессия на 50 реальных non_conversation (auto-voicemail, IVR) — все 50 корректно помечены non_conversation.
3. На 20 звонках с compliance-преамбулой + явная продажа — LLM-результат сохраняется, не затирается.

---

### Бонус: правка из Трека P2 — Resolve LLM отключение

**Файл:** `src/mango_mvp/services/resolve.py`, `src/mango_mvp/config.py:234`

**Проблема (из аудита):** RESOLVE_LLM_TRIGGER_SCORE=75 даёт 213 LLM-запусков из 2911 звонков (7.3%), но только **3 успешные** (1.4% success rate). Это **трата токенов впустую**.

**Что сделать:**

Вариант A (рекомендуется): отключить resolve LLM целиком — `RESOLVE_LLM_PROVIDER=off` в `.env.example` + документировать в `docs/`. Сохранит rule-based и rescue ASR. Кеш не теряем — если решим вернуть, всё на месте.

Вариант B: поднять порог `RESOLVE_LLM_TRIGGER_SCORE` с 75 до 85. Срабатывание упадёт до ~2%, success rate (гипотеза) вырастет до ~5%.

**Рекомендую вариант A.** Если потом увидим деградацию resolve quality — вернём с поднятым порогом.

**Acceptance:**
- На неделе после отключения: 0 LLM-вызовов в resolve логах.
- Quality resolve на 100 случайных звонках не упал относительно baseline (proxy метрика — длина и cohesion финального транскрипта).

---

## 2. Использование субагентов

Codex, можешь использовать до 6 параллельных субагентов. Распределение задач:

- **Sub-A**: правка X.1 (compact/full asymmetry) — самая короткая, можно использовать как warm-up.
- **Sub-B**: правка X.2 (_compose_history_summary) — требует понимания structured_fields схемы.
- **Sub-C**: правка X.3 (filler tokens) — простая, но критично написать regression test.
- **Sub-D**: правка X.4 (smart-chunking) — самая сложная и важная. Дать ей наибольшее внимание + 1-2 subagent помощника для генерации test fixtures и performance benchmark.
- **Sub-E**: правка X.5 (pre-LLM gate FP) — требует анализа реальной выборки звонков.
- **Sub-F**: правка резерв / Resolve LLM отключение / общий sanity check всех изменений + final integration tests.

Один субагент может вести 2 связанных правки (например, Sub-A на X.1 и X.5 — обе про промпт-сборку).

---

## 3. Acceptance Criteria (вся правка X)

### Hard requirements

1. Все 5 правок реализованы и тесты зелёные.
2. Существующие тесты `tests/test_analyze*.py` всё ещё passed (не сломали legacy).
3. Smart-chunking (X.4) recall ≥ 90% на 100 длинных звонках, при head+tail baseline ~60%.
4. Compliance-преамбула (X.5) на 50 звонках — 0 помечены non_conversation.
5. История с commercial блоком (X.2) — содержит блок в ≥ 90% случаев с заполненными полями.
6. Filler tokens (X.3) — регрессия зелёная, согласие «Да» сохраняется в compact prompt.
7. Compact/Full asymmetry (X.1) — на эскалациях full отличается от compact в ≥ 30% случаев.

### Soft requirements

8. Производительность smart-chunking не хуже head+tail × 3.
9. Resolve LLM отключён, токены = 0 в логах за неделю наблюдения (если выбран вариант A).
10. Длина history_summary растёт в среднем на 15-25%, не на 80%.

---

## 4. Deliverables

Codex, по завершении пришли в чат:

1. **Изменённые файлы:**
   - `src/mango_mvp/services/analyze.py` (5 правок)
   - `src/mango_mvp/services/resolve.py` (если выбран вариант A для Resolve LLM)
   - `src/mango_mvp/services/config.py` (изменение RESOLVE_LLM настроек)
   - `.env.example` (документация отключения Resolve LLM)

2. **Тесты:**
   - `tests/test_analyze_x1_compact_full_hints.py` (новый)
   - `tests/test_analyze_x2_history_commercial_block.py` (новый)
   - `tests/test_analyze_x3_filler_tokens.py` (новый)
   - `tests/test_analyze_x4_smart_chunking.py` (новый)
   - `tests/test_analyze_x5_compliance_preamble.py` (новый)

3. **Audit pack для Claude:**
   - `audits/_inbox/analyze_refactor_x_<timestamp>/`:
     - `AUDIT_SCOPE.md` — что трогали и почему
     - `BEFORE_AFTER_50_CALLS.csv` — выборка из 50 звонков с before/after полями (call_type, history_summary length, structured_fields completeness)
     - `SMART_CHUNKING_RECALL_REPORT.md` — измерение recall ключевых событий на 100 длинных звонках
     - `COMPLIANCE_FALSE_POSITIVE_REGRESSION.md` — 50 звонков с compliance-преамбулой, before/after
     - `PERFORMANCE_BENCHMARK.json` — timing smart-chunking vs head+tail
     - `MIGRATION_NOTES.md` — что выкинуто, что добавлено, обратимо ли

4. **Документация:**
   - Обновление `docs/TOKEN_OPTIMIZATION_PLAN_2026-03-26.md` с пометкой «обновлено в трек ТЗ-X 2026-05-XX»

---

## 5. Граничные условия

— **НЕ трогать** файлы вне `services/` и `tests/test_analyze*.py`. Если кажется что нужно — это сигнал зависимости, спрашивай через `QUESTIONS_FOR_CLAUDE.md`.
— **НЕ менять схему `analysis_schema_version="v2"`** в этом ТЗ. Изменение схемы — отдельный трек.
— **НЕ trogать** `services/transcribe.py` (это ТЗ-Z hygiene) или `quality/` модули (это ТЗ-Y).
— **НЕ менять** `_apply_non_conversation_hard_validation` глубже чем escape hatch — полная переработка gate-логики выходит за scope.
— Если правка X.4 (smart-chunking) даёт regression на коротких звонках (где раньше выход был = вход, а теперь странный) — остановиться и спросить.

---

## 6. Контроль качества

Перед сдачей задачи Codex отвечает сам:

1. Все ли 5 правок реализованы?
2. Все ли 5 новых тестов зелёные?
3. Старые тесты `test_analyze*.py` всё ещё passed?
4. Smart-chunking recall ≥ 90% на 100 длинных звонках?
5. На звонках длиной < max_chars smart-chunking возвращает идентичный вход?
6. Compliance preamble regression — 0 false positives?
7. История с commercial блоком — present в ≥ 90% случаев с заполненными полями?
8. Performance smart-chunking ≤ 3× head+tail?

Если на любой ответ «нет» — дорабатывает, не сдаёт.

---

## 7. Если что-то непонятно

Codex, если в этом ТЗ есть неопределённость:
1. Создай `audits/_inbox/analyze_refactor_x_clarifications_REQUEST_<timestamp>/QUESTIONS_FOR_CLAUDE.md`
2. Список конкретных вопросов с контекстом
3. Дмитрий передаст мне для ответа

Вероятные точки уточнения:
- Точные правила scoring окон в smart-chunking (X.4) — какие keyword patterns использовать
- Какой именно % считается «занимает >50% транскрипта» для compliance markers — пороговое значение
- Что делать с `_compact_prompt_filler_body`, если он зависит от того же hardcoded списка — мигрировать на единую константу или дублировать правку

---

## 8. Ожидаемый эффект

После реализации ТЗ-X:
- **Прямой ответ на исходную жалобу Дмитрия** «теряется контекст разговоров» закрыт.
- Длинные звонки (10+ минут) перестают терять середину = существенное улучшение качества анализа сделок.
- Catalog v2 на этапах D-F получает более качественный вход — точность классификации тем должна вырасти.
- Deal-aware writeback rollout получает более полную history_summary с commercial блоком — лучше для РОПа.
- Compliance-преамбулы перестают съедать ~3-5% реальных звонков.
- Resolve LLM перестаёт жечь токены впустую.

Это **большой системный фикс** входного слоя pipeline.
