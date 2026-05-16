# ТЗ-X-v2: Analyze.py refactor — переписанная версия после реверс-аудита Codex

Дата: 2026-05-14
Автор: Claude (наставник Дмитрия) после critique Codex'а и глубокого код-разбора через 2 субагента
Заменяет: `Mango_Analyse_TZ_X_Analyze_Refactor_2026-05-14.md` (v1, имела серьёзные ошибки)
Адресат: Codex как primary implementation agent

---

## 0. Что изменилось относительно v1

Codex выявил серьёзные проблемы в v1:
- X.5 правил только `analyze.py`, но **основной non_conversation gate в `quality/non_conversation.py`** — escape hatch в analyze был слишком поздним
- X.3 имел **неправильный тест-якорь** — реальная потеря «Да» происходит не на моём синтетическом примере
- X.4 предлагал полную замену head+tail на keyword-density — это **опасно**, нужен flag + head+smart_middle+tail
- Path ошибка: config в `src/mango_mvp/config.py`, не `services/config.py`
- Acceptance через `stable_runtime/` запрещён — это read-only зона
- Resolve LLM disable требует сначала проверить, что метрика `llm_used` честно показывает реальные вызовы

После v1-feedback запущены 2 субагента на глубокий разбор `non_conversation.py` и `_compact_prompt_filler_body`. Картина прояснилась, ТЗ переписан.

**Главное изменение архитектурно:** v1 был **одним большим треком** из 5 правок. v2 разбит на **4 независимых трека** (A/B/C/D), которые можно отдавать Codex'у **по очереди или параллельно**, в зависимости от приоритета. Каждый — самостоятельная задача с своим audit pack.

---

## ТРЕК X-A: Safe pack — три локальных улучшения analyze.py

Это **первый и самый низкорисковый трек**. Реализуется одним пакетом, потому что правки локальные и не пересекаются.

### Правка X-A.1 — Compact/Full asymmetry rule-hints + SYSTEM_PROMPT_FULL update

**Файл:** `src/mango_mvp/services/analyze.py:787-825` (где собирается prompt), `:26-88` (SYSTEM_PROMPT_FULL и SYSTEM_PROMPT_COMPACT)

**Проблема:** Full-промпт не получает hints, потому что код в `_analysis_prompt_context` добавляет hints только для compact. Но в `SYSTEM_PROMPT_FULL` тоже написано «only transcript + metadata». Если добавим hints в payload без обновления промпта — модель будет в когнитивном конфликте.

**Что сделать (две части):**

**Часть 1.** В `_analysis_prompt_context` добавить hints для обоих профилей:

```python
if normalized in {"compact", "full"}:
    hints_payload = self._prune_prompt_payload(self._analysis_rule_hints(call, text))
    if hints_payload:
        prompt += f"Deterministic hints JSON (may be incomplete; use only if supported by transcript): {json.dumps(hints_payload, ensure_ascii=False)}\n\n"
```

**Часть 2.** Обновить `SYSTEM_PROMPT_FULL` (analyze.py:26-88) — добавить упоминание hints в правила:

Текущая фраза (примерно): «Use only transcript + metadata».
Новая фраза: «Use only transcript + metadata + provided deterministic hints when supported by transcript».

Это явно говорит модели: hints — это **подсказки**, использовать только если в транскрипте есть подтверждение. То есть **не галлюцинировать** только потому что в hints что-то есть.

**Bump prompt version:** `ANALYZE_PROMPT_VERSION_FULL` поднять с `"v6"` до `"v7"` (потому что system_prompt изменился). Это **обязательно** — иначе LLM cache отдаст устаревшие парсы.

**Тест-якорь:** `tests/test_analyze_xa_1_full_hints.py`:

```python
def test_full_profile_prompt_includes_hints_section():
    """В full-промпте есть Deterministic hints JSON."""
    service = AnalyzeService(make_settings())
    call = make_call_with_transcript(...)
    payload = service._analysis_prompt_context(call, "full")
    assert "Deterministic hints JSON" in payload["prompt"]

def test_system_prompt_full_v7_mentions_hints():
    """SYSTEM_PROMPT_FULL обновлён под hints-aware режим."""
    from mango_mvp.services.analyze import SYSTEM_PROMPT_FULL
    assert "deterministic hints" in SYSTEM_PROMPT_FULL.lower() or "rule hints" in SYSTEM_PROMPT_FULL.lower()
    
def test_prompt_version_v7_bumped():
    from mango_mvp.services.analyze import ANALYZE_PROMPT_VERSION_FULL
    assert ANALYZE_PROMPT_VERSION_FULL == "v7"
```

**Acceptance:**
1. Full-промпт включает hints (verified тестом).
2. SYSTEM_PROMPT_FULL обновлён.
3. ANALYZE_PROMPT_VERSION_FULL = "v7".
4. На 50 эскалированных звонках (из недавнего test fixture, **не stable_runtime**) full-результат отличается от compact-результата в ≥ 30% случаев.

---

### Правка X-A.2 — history_summary включает commercial + school (условно)

**Файл:** `src/mango_mvp/services/analyze.py:1342-1484` (_compose_history_summary)

**Проблема:** LLM извлекает `student.school`, `commercial.{price_sensitivity, budget, discount_interest}`, `lead_priority`. Эти поля доступны в structured_fields, но history_summary их не включает.

**Что сделать (с критичным условием от Codex):**

Расширить summary блоками. **Но только если в полях есть содержательные данные.** НЕ писать «Чувствительность к цене: не указана», НЕ добавлять `warm` в каждую карточку без сигнала.

```python
def _compose_history_summary(self, ...):
    # ... existing logic ...
    
    sf = structured_fields
    
    # School (опционально)
    school = sf.get("student", {}).get("school")
    if school and school not in {"", "не_указано", "не указана", None}:
        student_block_lines.append(f"Школа: {school}")
    
    # Commercial block (опционально — только если есть хотя бы 1 содержательное поле)
    commercial = sf.get("commercial", {})
    commercial_lines = []
    if commercial.get("price_sensitivity") in {"высокая", "средняя", "низкая"}:  # явные значения, не пустые
        commercial_lines.append(f"Чувствительность к цене: {commercial['price_sensitivity']}")
    if commercial.get("budget") and commercial["budget"] not in {"не указан", "не_указано", None}:
        commercial_lines.append(f"Бюджет: {commercial['budget']}")
    if commercial.get("discount_interest") is True or commercial.get("discount_interest") in {"да", "интересуется"}:
        commercial_lines.append("Интересуется скидками")
    
    if commercial_lines:
        sections.append("Коммерческий контекст:\n" + "\n".join(f"- {l}" for l in commercial_lines))
    
    # Lead priority (опционально — только если medium/high)
    lead_priority = sf.get("lead_priority")
    if lead_priority in {"medium", "high", "средний", "высокий"}:
        # добавить в конец summary как pointer
        sections.append(f"Приоритет лида: {lead_priority}")
```

**Правило:** **NEVER** включать поле если значение пустое, дефолтное, «не_указано», None, или эквивалент. Это требование прямо от Codex'а — не засорять history шумом.

**Тест-якорь:** `tests/test_analyze_xa_2_history_blocks.py`:

```python
def test_history_includes_commercial_when_filled():
    sf = {
        "commercial": {"price_sensitivity": "высокая", "budget": "50000", "discount_interest": True},
        # ... остальные поля ...
    }
    summary = compose_history_summary(sf)
    assert "Чувствительность к цене" in summary
    assert "Бюджет" in summary
    assert "скидк" in summary.lower()

def test_history_skips_commercial_when_empty():
    sf = {
        "commercial": {"price_sensitivity": None, "budget": "", "discount_interest": None},
        # ... остальные поля ...
    }
    summary = compose_history_summary(sf)
    assert "Коммерческий контекст" not in summary
    assert "не указан" not in summary  # не должно быть мусора

def test_history_skips_lead_priority_when_low():
    sf = {"lead_priority": "low"}
    summary = compose_history_summary(sf)
    assert "Приоритет лида" not in summary

def test_history_includes_lead_priority_when_medium_or_high():
    for priority in ["medium", "high", "средний", "высокий"]:
        sf = {"lead_priority": priority}
        summary = compose_history_summary(sf)
        assert "Приоритет лида" in summary
```

**Acceptance:**
1. На 50 случайных звонках из call_records SQLite с заполненными commercial полями — history содержит блок в ≥ 90%.
2. На 50 случайных звонках с **пустыми** commercial полями — history НЕ содержит блок «Коммерческий контекст» (0 случаев шума).
3. Длина history_summary растёт в среднем на 10-20%, не на 50% (нет засорения).

---

### Правка X-A.3 — filler tokens commit-preservation

**Файл:** `src/mango_mvp/services/analyze.py:594-684 (_compact_prompt_filler_body, _filler_only_signature, _compact_transcript_for_prompt)`

**Проблема (правильная формулировка, не как в v1):**

Сценарий потери:
```
CLIENT: Да.
CLIENT: Да.    ← удалится, потому что та же signature, тот же спикер, нет не-filler между
```

Это происходит **только** при плохой диаризации или быстрых back-to-back подтверждениях клиента. Если между двумя «Да» стоит реплика менеджера — потери НЕТ (как Codex и сказал).

Опасные токены: `да`, `спасибо` (commitments/closing). Их дедупликация может терять согласие на запись/покупку.

**Что сделать (точечный фикс, НЕ выкидывание токенов из whitelist):**

В `_filler_only_signature` (analyze.py:608-615): для строк, содержащих ключевые commitment-токены, возвращать **уникальный** signature, чтобы они НЕ дедуплицировались между строк:

```python
COMMITMENT_TOKENS = {"да", "спасибо"}

def _filler_only_signature(text: str) -> str | None:
    tokens = re.findall(r"[a-zа-яё0-9]+", text.lower())
    if not tokens:
        return None
    if not all(t in PROMPT_COMPACTION_FILLER_TOKENS for t in tokens):
        return None
    # Если строка содержит commitment-токен — уникальная signature, не дедуплицировать между строк
    if any(t in COMMITMENT_TOKENS for t in tokens):
        return f"keep_commitment:{id(text)}"  # unique per call
    return " ".join(sorted(set(tokens)))
```

Это сохраняет **внутри строковую** compaction («Да, да, да» → «Да» через PROMPT_COMPACTION_REPEAT_RE), но **отключает** межстроковую дедупликацию для строк с commitment.

**Тест-якорь:** `tests/test_analyze_xa_3_filler_commitment.py`:

```python
def test_consecutive_da_preserved_between_lines():
    """Два подряд 'Клиент: Да' без не-filler реплики между — оба сохраняются."""
    transcript = (
        "[00:00.1] Менеджер: Записываю вас на пробное на субботу.\n"
        "[00:00.6] Клиент: Да.\n"
        "[00:01.1] Клиент: Да.\n"
        "[00:02.0] Менеджер: Отправляю ссылку на оплату.\n"
    )
    compacted = service._compact_transcript_for_prompt(transcript, "compact")
    assert compacted["transcript"].count("Клиент: Да") == 2

def test_intra_line_da_da_da_still_compacted():
    """Внутри одной строки 'Да, да, да' → 'Да' (compaction работает как раньше)."""
    transcript = "[00:00.1] Клиент: Да, да, да.\n"
    compacted = service._compact_transcript_for_prompt(transcript, "compact")
    assert compacted["transcript"].count("Да") == 1

def test_consecutive_uga_still_dedupe():
    """Подряд 'Угу.' / 'Угу.' (без commitment) — дедуплицируется как раньше."""
    transcript = (
        "[00:00.1] Менеджер: Понимаете?\n"
        "[00:00.6] Клиент: Угу.\n"
        "[00:01.1] Клиент: Угу.\n"
    )
    compacted = service._compact_transcript_for_prompt(transcript, "compact")
    assert compacted["transcript"].count("Клиент: Угу") == 1

def test_spasibo_preserved_between_lines():
    """Два 'Спасибо' подряд от клиента — оба сохраняются (commitment-closing)."""
    transcript = (
        "[00:00.1] Менеджер: Записал вас, ссылка придёт.\n"
        "[00:00.6] Клиент: Спасибо.\n"
        "[00:01.1] Клиент: Спасибо.\n"
    )
    compacted = service._compact_transcript_for_prompt(transcript, "compact")
    assert compacted["transcript"].count("Клиент: Спасибо") == 2
```

**ВАЖНО:** существующий тест в `tests/test_analyze.py:476` сейчас утверждает обратное (что подряд два «Клиент: Да» дедуплицируется в одно). Этот тест нужно **переписать с обратным assertion** — он защищает текущий bug. Выпуск ТЗ обязательно включает обновление этого теста.

**Acceptance:**
1. 4 новых теста зелёные.
2. Старый тест test_analyze.py:476 переписан (assertion инвертирован).
3. Внутри-строковая compaction («Да, да, да» → «Да») работает как раньше.
4. Дедупликация «Угу» / «Понятно» (не commitment) работает как раньше.

---

### Трек X-A: общие deliverables

**Изменённые файлы:**
- `src/mango_mvp/services/analyze.py` (3 правки)

**Тесты:**
- `tests/test_analyze_xa_1_full_hints.py`
- `tests/test_analyze_xa_2_history_blocks.py`
- `tests/test_analyze_xa_3_filler_commitment.py`
- `tests/test_analyze.py:476` — обновление существующего теста

**Acceptance criteria трека X-A:**
1. Все 3 правки реализованы.
2. Существующие тесты passed (64 → 64+).
3. Новые тесты зелёные.
4. prompt_version v6 → v7 (cache намеренно инвалидируется на следующем прогоне analyze).

**Audit pack:** `audits/_inbox/analyze_xa_safe_pack_<timestamp>/`

---

## ТРЕК X-B: non_conversation false positives — правильно сделанный fix

Это **отдельный трек**, потому что Codex прав: основной gate в `quality/non_conversation.py`, не в `analyze.py`. Правка в analyze слишком поздняя.

### Архитектурный контекст (важно понимать)

Из глубокого аудита: pre-LLM gate работает в трёх местах, главный — `_analyze_text:2176-2184` вызывает `detect_non_conversation_signals` из `quality/non_conversation.py`. Если он сработал — LLM **вообще не вызывается**, в БД пишется заглушка.

Compliance-маркеры («все разговоры записываются», «вас приветствует компания», «для улучшения качества обслуживания») живут в **трёх местах**:
- `non_conversation.py:21-43` SYSTEM_NO_DIALOGUE_RE
- `non_conversation.py:45-59` THIRD_PARTY_IVR_RE
- `analyze.py:91-118` STRONG_NON_CONVERSATION_MARKERS (substring дубликаты)

### Что сделать (4 связанные правки)

### Правка X-B.1 — Вынести compliance-маркеры в отдельный regex

**Файл:** `src/mango_mvp/quality/non_conversation.py:21-43`

Создать `COMPLIANCE_PREAMBLE_RE` с фразами:
- «вас приветствует компан»
- «все разговоры записываются»
- «ваш звонок очень важен»
- «звонок может быть записан»

Эти фразы **убрать** из `SYSTEM_NO_DIALOGUE_RE`. То есть `SYSTEM_NO_DIALOGUE_RE` остаётся только с реальными non-live маркерами («голосовой ассистент», «оставьте сообщение», «нажмите 1»).

`COMPLIANCE_PREAMBLE_RE` **не включать** в `NO_LIVE_RE`. Использовать его только в **специальном** check, который учитывает контекст (см. X-B.4).

### Правка X-B.2 — Live_sales_context bypass для THIRD_PARTY_IVR_RE

**Файл:** `src/mango_mvp/quality/non_conversation.py:45-59, 276-283`

Текущий `live_payment_context` bypass (строки 276-283) применяется только если есть платёжные паттерны. Для обычной sales-преамбулы он не работает.

Добавить **аналогичный** `live_sales_context` bypass:

```python
def _has_live_sales_context(text: str, client_text: str, manager_text: str) -> bool:
    """Bypass для third_party_ivr-маркеров в начале менеджерской преамбулы при наличии живого диалога."""
    if not text or len(client_text) < 80:
        return False
    if not CLIENT_HUMAN_RESPONSE_RE.search(client_text):
        return False
    if not BUSINESS_TERM_RE.search(text):  # упоминание продукта/курса
        return False
    return True
```

Применить в логике detect_non_conversation_signals: если `third_party_ivr=True` И `_has_live_sales_context(...)` → понизить до warning, не force.

### Правка X-B.3 — Очистить STRONG_NON_CONVERSATION_MARKERS

**Файл:** `src/mango_mvp/services/analyze.py:91-118`

Удалить compliance-дубликаты из STRONG (они теперь живут отдельно в COMPLIANCE_PREAMBLE_RE):
- Убрать «вас приветствует компания»
- Убрать «все разговоры записываются»

Оставить undisputed:
- «продолжение следует»
- «голосовой ассистент»
- «оставьте сообщение»
- «после сигнала»
- «нажмите 1»
- «коллекторская организация»
- остальные реальные non-live

### Правка X-B.4 — Escape hatch для LLM-overrides в _apply_non_conversation_hard_validation

**Файл:** `src/mango_mvp/services/analyze.py:1205-1285`

Текущая функция затирает результат LLM, если quality_flags.call_type=="non_conversation". Это **затирает даже если LLM явно увидел продажный контекст**.

Изменение:

```python
def _apply_non_conversation_hard_validation(self, analysis, signals, transcript):
    if analysis.get("call_type") != "non_conversation":
        return analysis
    
    sf = analysis.get("structured_fields", {})
    has_product = bool(sf.get("interests", {}).get("products"))
    has_next_step = bool(sf.get("next_step", {}).get("action"))
    has_objections = bool(sf.get("objections"))
    has_target_product = bool(sf.get("target_product"))
    
    # Escape hatch: если LLM явно увидел сделку — не затирать
    if has_product or has_next_step or has_objections or has_target_product:
        # Понижаем до manual_review, не затираем
        analysis["call_type"] = "sales_call_manual_review"
        analysis["quality_flags"]["llm_overrode_non_conversation_gate"] = True
        analysis["quality_flags"]["original_non_conversation_signals"] = signals.to_dict()
        return analysis
    
    # Иначе — обычное затирание
    return self._non_conversation_analysis(signals)
```

**Это работает только для post-LLM пути** (когда LLM был вызван и вернул что-то). Pre-LLM gate (когда `should_force_non_conversation=True`) НЕ затрагивается — там LLM вообще не зовётся, escape hatch неприменим. Поэтому правки X-B.1–X-B.3 в `non_conversation.py` критичны — они уменьшают false positives на pre-LLM этапе.

### Тесты X-B

`tests/test_non_conversation_xb_compliance.py`:

```python
def test_compliance_preamble_alone_does_not_force_non_conversation():
    """Только compliance-преамбула от менеджера + содержательный sales-диалог = не non_conversation."""
    text = (
        "MANAGER: Все разговоры записываются. Добрый день, по вашему запросу о подготовке к ЕГЭ. "
        "CLIENT: Да, интересует, расскажите про стоимость и формат. "
        "MANAGER: Стоимость 50000 за полугодие, формат онлайн."
    )
    signals = detect_non_conversation_signals(text)
    assert signals.should_force_non_conversation is False

def test_third_party_ivr_with_live_sales_context_downgraded():
    """Менеджер говорит 'для улучшения качества' но клиент ведёт живой sales-диалог = не non_conversation."""
    text = (
        "MANAGER: Для улучшения качества обслуживания скажу, что курс по математике стоит 50000. "
        "CLIENT: Хорошо, расскажите подробнее, мой ребёнок в 11 классе готовится к ЕГЭ. "
        "MANAGER: Конечно, занятия проходят онлайн два раза в неделю."
    )
    signals = detect_non_conversation_signals(text)
    assert signals.third_party_ivr is False or signals.should_force_non_conversation is False

def test_pure_ivr_still_blocks():
    """Чистый IVR без живого диалога всё ещё блокируется."""
    text = (
        "Для оплаты нажмите 1. Для информации о балансе нажмите 2. Для соединения с оператором нажмите 0."
    )
    signals = detect_non_conversation_signals(text)
    assert signals.should_force_non_conversation is True

def test_llm_override_when_extracted_sales_signal():
    """Если LLM вернул interests.products и next_step.action — не затирать как non_conversation."""
    analysis = {
        "call_type": "non_conversation",  # heuristic решила что non_conversation
        "structured_fields": {
            "interests": {"products": ["годовые курсы"]},
            "next_step": {"action": "отправить ссылку на оплату"},
        },
        "quality_flags": {},
    }
    result = service._apply_non_conversation_hard_validation(analysis, signals, transcript)
    assert result["call_type"] == "sales_call_manual_review"
    assert result["quality_flags"]["llm_overrode_non_conversation_gate"] is True
    assert result["structured_fields"]["interests"]["products"] == ["годовые курсы"]  # не затёрто
```

### Acceptance трека X-B

1. 4 новых теста зелёные.
2. Существующие тесты `tests/test_non_conversation_quality.py` (~30) всё ещё passed.
3. На 50 звонках с compliance-преамбулой + sales-диалогом (test fixture, **не stable_runtime**) — 0 помечены non_conversation.
4. На 50 реальных non_conversation (auto-voicemail, IVR без живого ответа) — все 50 корректно помечены.
5. На 20 звонках где LLM явно вернул products+next_step — результат сохраняется как `sales_call_manual_review`, не затирается.

### Audit pack X-B

`audits/_inbox/non_conversation_xb_fix_<timestamp>/`

---

## ТРЕК X-C: Smart-chunking — за флагом, не как замена

Это **самый рискованный трек**, поэтому идёт **за feature flag** с A/B сравнением. Не включать по умолчанию.

### Что сделать

**Файл:** `src/mango_mvp/services/analyze.py:692-702 (текущая обрезка)`

**Этап 1:** Не выкидывать head+tail, а **дополнить**:

```python
def _smart_chunk_or_default(transcript: str, max_chars: int, mode: str = "head_tail") -> str:
    """
    mode='head_tail' (default): текущая логика head+tail
    mode='head_smart_middle_tail': head 30% + smart-selected middle 40% + tail 30%
    mode='smart_only': pure keyword-density (рискованно, только для эксперимента)
    """
    if len(transcript) <= max_chars:
        return transcript
    
    if mode == "head_tail":
        return _legacy_head_tail(transcript, max_chars)
    
    if mode == "head_smart_middle_tail":
        head_budget = int(max_chars * 0.3)
        tail_budget = int(max_chars * 0.3)
        middle_budget = max_chars - head_budget - tail_budget - 50  # 50 на маркеры
        
        head = transcript[:head_budget]
        tail = transcript[-tail_budget:]
        middle_zone = transcript[head_budget:-tail_budget]
        
        smart_middle = _select_best_windows(middle_zone, middle_budget)
        return f"{head}\n[... fragment selected from middle ...]\n{smart_middle}\n[... fragment selected from middle ...]\n{tail}"
    
    raise ValueError(f"unknown mode: {mode}")
```

**Этап 2:** Feature flag в config:

```python
# config.py
ANALYZE_TRANSCRIPT_CHUNKING_MODE = os.getenv("ANALYZE_TRANSCRIPT_CHUNKING_MODE", "head_tail")
# валидные значения: "head_tail" (default, safe), "head_smart_middle_tail" (experiment)
```

**Этап 3:** A/B сравнение **на специальной test fixture** (не stable_runtime — это read-only!).

Создать `tests/fixtures/long_transcripts_for_chunking_ab.jsonl` с 50 длинными звонками (>= 6500 chars) с известными «важными местами» (PRODUCT/PRICE/OBJECTION mentions помечены вручную).

Прогнать оба mode и измерить recall важных мест:
- head_tail baseline: ~60-70%
- head_smart_middle_tail target: ≥ 90%

### Acceptance трека X-C

1. Feature flag реализован, default `head_tail` (текущая безопасная логика).
2. Smart mode реализован как `head_smart_middle_tail` (НЕ pure smart-only).
3. A/B на 50-звоночной fixture: smart recall ≥ 90%, baseline ≤ 75%.
4. На звонках < max_chars оба mode возвращают идентичный вход (идемпотентность).
5. Performance smart не хуже head+tail × 3.
6. Документ `docs/SMART_CHUNKING_EXPERIMENT_2026-05-XX.md` с A/B результатами.
7. **Включение в production отдельным решением Дмитрия после ревью результатов.**

### Audit pack X-C

`audits/_inbox/smart_chunking_xc_experiment_<timestamp>/` + AB report

---

## ТРЕК X-D: Resolve LLM disable — с правильной метрикой

**Файл:** `src/mango_mvp/config.py:234` (RESOLVE_LLM_TRIGGER_SCORE, RESOLVE_LLM_PROVIDER)

### Pre-step: проверить метрику `llm_used`

Codex прав: перед отключением нужно убедиться, что `llm_used` flag в `call_records.resolve_metadata` или эквивалент **честно отражает реальные LLM-вызовы**. Если он legacy/неточный — отключение в config не даст видимого эффекта.

**Шаг 1:** Проверить, что метрика честная:
- grep `llm_used` в `src/mango_mvp/services/resolve.py` — где ставится
- Verify: ставится **только** после реального LLM-вызова, не на mock-пути, не до вызова
- Если метрика неправильная — поправить **до** отключения

**Шаг 2:** Отключить LLM через env:

```python
# config.py
RESOLVE_LLM_PROVIDER = os.getenv("RESOLVE_LLM_PROVIDER", "off")
# Было: default зависел от чего-то; теперь default "off"
```

Update `.env.example`:
```
# Resolve LLM отключён по умолчанию — success rate 1.4% (TOKEN_OPTIMIZATION_PLAN раздел 4)
# Включить: RESOLVE_LLM_PROVIDER=openai или ollama если нужен LLM-fallback в resolve
RESOLVE_LLM_PROVIDER=off
```

**Шаг 3:** Документировать в `docs/RESOLVE_LLM_DISABLED_2026-05-XX.md` — что отключено, почему, как вернуть.

### Acceptance трека X-D

1. Метрика `llm_used` проверена и честная (verified тестом).
2. RESOLVE_LLM_PROVIDER default = "off".
3. На прогоне 1000 случайных звонков через resolve: `llm_used=True` появляется **0 раз**.
4. Quality resolve не падает (proxy: длина и cohesion итогового транскрипта на 100 звонках сравнима с baseline).
5. Документ `docs/RESOLVE_LLM_DISABLED_2026-05-XX.md` создан.

### Audit pack X-D

`audits/_inbox/resolve_llm_disable_xd_<timestamp>/`

---

## Порядок выполнения 4 треков

Codex может выполнять параллельно или последовательно. Рекомендация:

**Параллельно (если есть 4 диалога):**
- X-A + X-D — самые низкорисковые, можно сразу
- X-B + X-C — параллельно после X-A (зависят от понимания текущего состояния analyze, которое X-A не нарушает)

**Последовательно (если один диалог):**
1. X-A (safe pack) — 1-2 дня работы
2. X-D (resolve disable) — 0.5 дня
3. X-B (non_conversation fix) — 2-3 дня (правки в 2 модулях, нужны интеграционные тесты)
4. X-C (smart-chunking experiment) — 3-4 дня (требует A/B fixture)

---

## Использование субагентов

Codex, в каждом из 4 треков можешь использовать до 6 субагентов. Конкретные сценарии — описаны в каждом треке выше.

---

## Граничные условия (общие для всех 4 треков)

— **НЕ запускать ASR/R+A на stable_runtime/.** Это read-only зона по контракту CLAUDE.md. Использовать `tests/fixtures/` или специально подготовленные test fixtures.
— **НЕ менять контракт `analysis_schema_version="v2"`** в этих треках.
— **НЕ трогать** `quality/sanitizer*`, `tenant_text_normalizer`, `crm_text_quality_detector` — это зона ТЗ-Y.
— **НЕ трогать** `transcribe.py`, `stage15_export_quality_gate.py`, `bot_safety_frozen_corpus.py` — это зона ТЗ-Z.
— **Bump prompt_version при любом изменении system_prompt** (правка X-A.1 — v6→v7).
— **Path config.py:** `src/mango_mvp/config.py`, не `services/config.py`.

---

## Если что-то непонятно

Создай `audits/_inbox/analyze_xv2_<track>_clarifications_REQUEST_<timestamp>/QUESTIONS_FOR_CLAUDE.md`. Дмитрий передаст мне.

---

## Резюме изменений v1 → v2

| Аспект | v1 | v2 |
|---|---|---|
| Структура | Один большой трек из 5 правок | 4 независимых трека (A/B/C/D) |
| X.3 (filler) | Неправильный тест-якорь | Правильный сценарий — два подряд `Клиент: Да` без не-filler между |
| X.4 (smart-chunking) | Полная замена head+tail | За флагом, режим `head_smart_middle_tail`, A/B сравнение |
| X.5 (non_conversation) | Только в analyze.py | В non_conversation.py (3 правки) + analyze.py (escape hatch) |
| X.1 (full hints) | Только payload | Payload + SYSTEM_PROMPT_FULL update + bump version |
| Resolve LLM | Сразу отключить | Pre-step: проверить метрику честная |
| Acceptance | Через stable_runtime | Через test fixtures (stable_runtime read-only) |
| Path config | `services/config.py` | `src/mango_mvp/config.py` |

После реализации всех 4 треков — Дмитрию доступен **полный набор улучшений** входного слоя pipeline, доказанных тестами и A/B-измерениями, безопасный для production.
