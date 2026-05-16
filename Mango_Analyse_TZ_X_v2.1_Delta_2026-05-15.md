# ТЗ-X-v2.1: Delta-исправления к TZ-X-v2 после 2-го реверс-аудита Codex

Дата: 2026-05-15
Автор: Claude (наставник Дмитрия) после critique Codex'а на v2 + deep grep-проверка функций
Дополняет (не заменяет): `Mango_Analyse_TZ_X_v2_Analyze_Refactor_2026-05-14.md`
Адресат: Codex как primary implementation agent

Этот документ — **delta только**. Структура треков (X-A safe pack, X-B non_conversation, X-C smart-chunking, X-D resolve disable) сохраняется. Меняются только конкретные правки и тесты внутри треков.

---

## 0. Контекст

Codex провёл 2-й реверс-аудит на TZ-X-v2 и нашёл 8 ошибок. Моя «корректировка процесса» (grep-pre-flight) поймала верхний уровень (имена), но не средний (внутренности функций, точные return fields, enum values, множественные ветки). Дополнительная корректировка — для критичных правок субагент должен **читать целиком** функции, не только grep по имени.

Запущен subagent на deep function-level reading. Все 8 ошибок Codex'а подтверждены + детали.

Codex рекомендует порядок: **X-A → X-B → X-D → X-C**. Это правильно. X-C самый рискованный, делается последним.

---

## ТРЕК X-A: Delta-правки

### X-A.1 — Compact/Full hints: тест-якорь исправлен

**Было в v2:**
```python
def test_full_profile_prompt_includes_hints_section():
    payload = service._analysis_prompt_context(call, "full")
    assert "Deterministic hints JSON" in payload["prompt"]  # ❌ KeyError — нет ключа "prompt"
```

**Стало в v2.1:**

`_analysis_prompt_context()` возвращает dict с ключами `profile`, `system_prompt`, **`user_prompt`**, `llm_prompt`, `metrics`. Не `prompt`. Тест:

```python
def test_full_profile_user_prompt_includes_hints_section():
    payload = service._analysis_prompt_context(call, text, profile="full")
    assert "Deterministic hints JSON" in payload["user_prompt"]
```

**Реализация (без изменений):** в `analyze.py:811-816` блок `if normalized == "compact":` поменять на `if normalized in {"compact", "full"}:`. Hint payload собирается в `user_prompt`.

---

### X-A.2 — _compose_history_summary: правки в ОБЕИХ ветках + правильные enum values

**Было в v2:** правка описана только в общем виде, без указания веток.

**Стало в v2.1:**

`_compose_history_summary` (analyze.py:1342-1484) имеет **две ветки сборки**:
- **Ветка 1** (1399-1452): есть LLM-черновик. Использует `_summary_mentions_any` для проверки дублей. `parts = [opening, sentence, ...]`.
- **Ветка 2** (1454-1484): fallback без черновика. Простая `blocks = [opening, student_bits, topic_parts, objections, next_step, contacts]`.

**Правка должна быть в ОБЕИХ ветках.** В ветке 1 — с проверкой `_summary_mentions_any`, в ветке 2 — без.

**Точные enum values из реальной схемы:**

```python
# lead_priority (analyze.py:1703-1707): hot / warm / cold
LEAD_PRIORITY_INCLUDED_VALUES = {"hot", "warm"}  # cold не подсвечиваем, шум

# commercial.price_sensitivity (analyze.py:1639-1645): high / medium / low (английские, не русские!)
PRICE_SENSITIVITY_INCLUDED_VALUES = {"high", "medium", "low"}

# commercial.discount_interest — BOOL, не enum
# Включаем если True
```

**Доступ к полям:**
- `lead_priority` — **плоское** поле на верхнем уровне: `structured_fields["lead_priority"]`
- `commercial.price_sensitivity` — вложенное: `structured_fields["commercial"]["price_sensitivity"]`
- `student.school` — вложенное: `structured_fields["student"]["school"]`

**Реализация:**

```python
def _build_commercial_lines(self, structured_fields: Dict[str, Any]) -> list[str]:
    """Возвращает список строк для commercial-блока. Пустой список если данных нет."""
    commercial = self._nested_dict(structured_fields, "commercial")  # уже есть helper
    lines = []
    
    price_sens = self._clean_text(commercial.get("price_sensitivity"))
    if price_sens in {"high", "medium", "low"}:
        ru = {"high": "высокая", "medium": "средняя", "low": "низкая"}[price_sens]
        lines.append(f"Чувствительность к цене: {ru}")
    
    budget = self._clean_text(commercial.get("budget"))
    if budget and budget not in {"не указан", "не_указано"}:
        lines.append(f"Бюджет: {budget}")
    
    if commercial.get("discount_interest") is True:
        lines.append("Интересуется скидками")
    
    return lines


def _build_lead_priority_line(self, structured_fields: Dict[str, Any]) -> Optional[str]:
    """Возвращает строку про lead_priority или None."""
    priority = self._clean_text(structured_fields.get("lead_priority"))
    if priority in {"hot", "warm"}:
        ru = {"hot": "горячий", "warm": "тёплый"}[priority]
        return f"Приоритет лида: {ru}"
    return None


def _build_school_line(self, structured_fields: Dict[str, Any]) -> Optional[str]:
    """Возвращает строку про школу или None."""
    student = self._nested_dict(structured_fields, "student")
    school = self._clean_text(student.get("school"))
    if school:
        return f"Школа: {school}"
    return None
```

В **ветке 1** (около analyze.py:1399-1452) — добавить с проверкой дублей:
```python
commercial_lines = self._build_commercial_lines(structured_fields)
if commercial_lines and not self._summary_mentions_any(parts, [
    "чувствительность", "бюджет", "скидк"
]):
    parts.append("Коммерческий контекст: " + "; ".join(commercial_lines))

school_line = self._build_school_line(structured_fields)
if school_line and not self._summary_mentions_any(parts, ["школ"]):
    parts.append(school_line)

priority_line = self._build_lead_priority_line(structured_fields)
if priority_line and not self._summary_mentions_any(parts, ["приоритет"]):
    parts.append(priority_line)
```

В **ветке 2** (около analyze.py:1454-1484) — добавить в `blocks` без проверок:
```python
blocks.extend([line for line in self._build_commercial_lines(structured_fields)])
school_line = self._build_school_line(structured_fields)
if school_line:
    blocks.append(school_line)
priority_line = self._build_lead_priority_line(structured_fields)
if priority_line:
    blocks.append(priority_line)
```

**Обновлённый тест:**

```python
def test_history_includes_commercial_when_filled_branch_with_draft():
    """Ветка 1: есть LLM-черновик."""
    sf = {
        "commercial": {"price_sensitivity": "high", "budget": "50000", "discount_interest": True},
        "lead_priority": "hot",
        "student": {"school": "школа №16"},
    }
    summary = service._compose_history_summary(
        call, draft_history_summary="Клиент интересуется ЕГЭ.", summary="...",
        structured_fields=sf, objections=[], next_step_action=None, due=None, follow_up_reason=None,
    )
    assert "Чувствительность к цене: высокая" in summary
    assert "Бюджет: 50000" in summary
    assert "Интересуется скидками" in summary
    assert "Приоритет лида: горячий" in summary
    assert "Школа: школа №16" in summary


def test_history_includes_commercial_when_filled_branch_no_draft():
    """Ветка 2: нет LLM-черновика."""
    sf = {
        "commercial": {"price_sensitivity": "medium", "budget": "", "discount_interest": False},
        "lead_priority": "warm",
    }
    summary = service._compose_history_summary(
        call, draft_history_summary=None, summary=None,
        structured_fields=sf, objections=[], next_step_action=None, due=None, follow_up_reason=None,
    )
    assert "Чувствительность к цене: средняя" in summary
    assert "Приоритет лида: тёплый" in summary
    # Пустой budget и False discount_interest НЕ должны появиться
    assert "Бюджет:" not in summary
    assert "скидк" not in summary.lower()


def test_history_skips_lead_priority_when_cold():
    sf = {"lead_priority": "cold"}
    summary = service._compose_history_summary(call, ..., structured_fields=sf, ...)
    assert "Приоритет лида" not in summary  # cold = шум, не подсвечиваем
```

---

### X-A.3 — _filler_only_signature: правильный фикс без id(text)

**Было в v2:**
```python
if any(t in COMMITMENT_TOKENS for t in tokens):
    return f"keep_commitment:{id(text)}"  # ❌ id() нестабилен, плохая практика
```

**Стало в v2.1:**

Caller `_filler_only_signature` — единственный, в `_compact_transcript_for_prompt` (analyze.py:671). Логика дедупа:
```python
if filler_signature and filler_signature == prev_filler_signature and (speaker or "") == (prev_speaker or ""):
    # дедуплицируется
```

**Правильный фикс:** возвращать `None` для одиночных filler-токенов и для строк с commitment-токенами. Тогда дедуп **не сработает** (потому что `None != None` в проверке == False, и условие `filler_signature` имитирует "не filler-only").

```python
COMMITMENT_TOKENS = {"да", "спасибо"}

@staticmethod
def _filler_only_signature(text: str) -> Optional[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-zа-яё0-9]+", lowered)
    if not tokens:
        return None
    if not all(t in PROMPT_COMPACTION_FILLER_TOKENS for t in tokens):
        return None
    # Одиночный filler-токен ИЛИ строка с commitment — не дедуплицировать
    if len(tokens) == 1 or any(t in COMMITMENT_TOKENS for t in tokens):
        return None
    return " ".join(sorted(set(tokens)))
```

**Это решает обе проблемы:**
- `«Клиент: Да. / Клиент: Да.»` — каждое `Да` имеет signature `None` → дедуп не срабатывает.
- `«Клиент: Ага. / Клиент: Угу.»` — разные signature, дедуп не срабатывает (как и раньше).
- `«Клиент: Угу. / Клиент: Угу.»` (повтор) — обе имеют signature `None` (одиночные), не дедуплицируются (изменение поведения — но безопасное: лучше оставить лишнее, чем выкинуть смысл).
- Внутри-строковая compaction `"да, да, да"` → `"да"` через PROMPT_COMPACTION_REPEAT_RE — работает как раньше, эту функцию не трогаем.

Тесты в v2 X-A.3 остаются валидными.

---

## ТРЕК X-B: Delta-правки

### X-B.4 — Escape hatch ДО обнуления полей, не внутри hard_validation

**Было в v2:** правка в `_apply_non_conversation_hard_validation(self, analysis, signals, transcript)` — добавить escape hatch при наличии structured_fields с продажным сигналом.

**Проблема:** к моменту вызова `_apply_non_conversation_hard_validation` поля **уже обнулены** на analyze.py:1719-1743 (внутри `_normalize_analysis`):

```python
if call_type == "non_conversation":
    tags.append("non_conversation")
    products = []; formats = []; subjects = []; ...
    lead_priority = "cold"
```

Это **до** того как `_apply_non_conversation_hard_validation` запускается (line 1901). Escape hatch внутри hard_validation **бесполезен** — там уже нет данных.

**Стало в v2.1:**

Правка в **двух местах**:

**Часть 1 — escape hatch ДО обнуления полей (analyze.py:1719):**

Перед строкой `if call_type == "non_conversation":` (line 1719) сохранить snapshot LLM-извлечённых сигналов и проверить escape условие:

```python
# Capture LLM-extracted sales signals BEFORE zeroing
llm_extracted_products = list(blocks.get("interests", {}).get("products") or [])
llm_extracted_next_step = blocks.get("next_step", {}).get("action") or ""
llm_extracted_objections = list(blocks.get("objections") or [])
llm_extracted_target_product = blocks.get("target_product")

# Escape hatch: если LLM явно увидел продажный контекст — НЕ затирать
has_sales_signal = (
    bool(llm_extracted_products)
    or bool(llm_extracted_next_step)
    or bool(llm_extracted_objections)
    or bool(llm_extracted_target_product)
)

if call_type == "non_conversation" and not has_sales_signal:
    # Существующая логика обнуления (1720-1743)
    tags.append("non_conversation")
    products = []
    # ... остальное обнуление ...
    lead_priority = "cold"
elif call_type == "non_conversation" and has_sales_signal:
    # LLM явно увидел сделку — понижаем до manual review, не затираем
    call_type = "sales_call"  # переопределить
    tags.append("manual_review_non_conversation_override")
    quality_flags["llm_overrode_non_conversation_gate"] = True
    # Поля НЕ обнуляются — оставляем LLM-extracted
```

**Часть 2 — `_apply_non_conversation_hard_validation` signature**:

Реальная signature: `(self, call: CallRecord, normalized: Dict[str, Any]) -> Dict[str, Any]`.

В функции сейчас early-return при `call_type != "non_conversation"` (line 1213-1214). После Части 1 если LLM-override сработал, `quality_flags["call_type"]` будет всё ещё `"non_conversation"` (мы только переопределили локальную переменную `call_type`). Это **тоже надо поправить** — quality_flags должен отражать override:

В analyze.py:1710-1712 (где quality_flags.call_type ставится):
```python
quality_flags["call_type"] = call_type  # уже учитывает override после Части 1
if quality_flags.get("llm_overrode_non_conversation_gate"):
    quality_flags["call_type"] = "sales_call_manual_review"
```

Тогда `_apply_non_conversation_hard_validation` на line 1213 увидит `call_type != "non_conversation"` и early-return, не затрёт ничего.

**Изменённый тест:**

```python
def test_llm_override_when_extracted_sales_signal_preserves_fields():
    """LLM вернул products + next_step → fields НЕ обнуляются."""
    # Setup: текст с compliance-преамбулой, но реальный sales-диалог
    text = (
        "Все разговоры записываются. MANAGER: По вашему вопросу о ЕГЭ по математике. "
        "CLIENT: Да, интересует, стоимость 50000?"
    )
    # Mock LLM возвращает sales-extraction
    mock_llm_response = {
        "structured_fields": {
            "interests": {"products": ["годовые курсы"]},
            "next_step": {"action": "отправить ссылку на оплату"},
            "lead_priority": "warm",
        },
    }
    result = service._analyze_text(call, text, llm_client=mock(mock_llm_response))
    
    # call_type переопределён
    assert result["call_type"] == "sales_call_manual_review"
    assert result["quality_flags"]["llm_overrode_non_conversation_gate"] is True
    
    # Поля сохранены, не обнулены
    assert result["structured_fields"]["interests"]["products"] == ["годовые курсы"]
    assert result["structured_fields"]["next_step"]["action"] == "отправить ссылку на оплату"
    assert result["structured_fields"]["lead_priority"] == "warm"


def test_no_llm_signal_still_zeros_fields():
    """Regression: пустой LLM-output → обнуление работает как раньше."""
    text = "Голосовой ассистент. Нажмите 1."
    result = service._analyze_text(call, text, llm_client=mock_empty())
    assert result["call_type"] == "non_conversation"
    assert result["structured_fields"]["interests"]["products"] == []
```

---

## ТРЕК X-C: Delta-правки

### X-C — Settings constructor compat в tests

**Было в v2:** добавляем env-var `ANALYZE_TRANSCRIPT_CHUNKING_MODE`, реализуем feature flag.

**Что упустили:** `tests/test_dialogue_format.py` использует `make_settings(...)` функцию (lines 12-107), где все ~80 полей `Settings` перечислены **явно** — без полагания на default values. Добавление нового поля без обновления make_settings вызовет Pydantic validation error.

**Стало в v2.1:**

В deliverables трека X-C добавить обязательное:

```python
# tests/test_dialogue_format.py:make_settings()
def make_settings(...):
    return Settings(
        # ... existing fields ...
        analyze_transcript_compaction_enabled=True,
        analyze_transcript_chunking_mode="head_tail",  # NEW: default-safe value
        # ... rest ...
    )
```

И в реальном `Settings` (Pydantic model) добавить:
```python
class Settings(BaseSettings):
    # ... existing fields ...
    analyze_transcript_chunking_mode: Literal["head_tail", "head_smart_middle_tail"] = "head_tail"
```

**Acceptance addition:** `tests/test_dialogue_format.py` всё ещё passes после правки.

---

## ТРЕК X-D: Delta-правки

### X-D — Метрика `llm_used` сначала чинится, потом отключается

**Было в v2:** «Pre-step: проверить метрику честная» + «Шаг 2: отключить через RESOLVE_LLM_PROVIDER=off».

**Что упустили:** Codex прав, что одного env-флага недостаточно. Конкретно:

- `_resolve_with_llm` (resolve.py:1412-1538) при `RESOLVE_LLM_PROVIDER=off` **всё равно** идёт по основному пути и возвращает кандидата с `name="llm"` (lines 1480, 1520).
- `_merge_pair_with_llm` (resolve.py:762-772) при off возвращает rule-merged dict с `provider="rule"`, но внешнее имя candidate остаётся `"llm"`.
- `llm_used += 1` (resolve.py:1805) инкрементируется как только `_resolve_with_llm` вернул не-None — без проверки реального провайдера.

То есть **отключение через env-флаг не меняет метрику** — она показывает 7.3% LLM-вызовов даже когда реальных LLM-вызовов 0.

**Стало в v2.1:**

**Шаг 1 — поправить метрику (обязательно перед disable):**

В начале `_resolve_with_llm` (resolve.py:1412) добавить early-return:

```python
def _resolve_with_llm(self, call: CallRecord, variants_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    settings = self._settings
    llm_provider = settings.resolve_llm_provider.lower() if settings.resolve_llm_provider else "off"
    
    # NEW: early return — если LLM провайдер off, не идём дальше, не помечаем как llm
    if llm_provider not in {"ollama", "openai", "codex_cli"}:
        return None
    
    # ... existing logic ...
```

Это гарантирует, что при `RESOLVE_LLM_PROVIDER=off`:
- Функция вообще не вызывает LLM
- `llm_used` не инкрементируется (потому что caller получает None)
- Resolve использует rule-merge или rescue ASR как fallback

**Шаг 2 — отключение через env:**

`.env.example`:
```
# Resolve LLM отключён по умолчанию — success rate 1.4% (TOKEN_OPTIMIZATION_PLAN раздел 4)
# Поддерживаемые значения: off, ollama, openai, codex_cli
RESOLVE_LLM_PROVIDER=off
```

**Шаг 3 — verification тест:**

```python
def test_resolve_llm_provider_off_returns_none_immediately():
    """При RESOLVE_LLM_PROVIDER=off функция возвращает None, не идёт по основному пути."""
    settings = make_settings(resolve_llm_provider="off")
    service = ResolveService(settings)
    result = service._resolve_with_llm(call, variants_payload={...})
    assert result is None  # не "llm" candidate


def test_resolve_llm_off_does_not_increment_llm_used():
    """Прогон 100 resolve с off — llm_used = 0."""
    settings = make_settings(resolve_llm_provider="off")
    service = ResolveService(settings)
    metrics = service.run_with_progress(session, limit=100)
    assert metrics.get("llm_used", 0) == 0
```

**Acceptance трека X-D (обновлено):**

1. `_resolve_with_llm` early-return при `provider="off"` — verified тестом.
2. На прогоне 100 resolve с `RESOLVE_LLM_PROVIDER=off`: `llm_used = 0`.
3. Quality resolve на 100 звонках не падает относительно baseline.
4. Документ `docs/RESOLVE_LLM_DISABLED_2026-05-XX.md` создан.

---

## Порядок выполнения (от Codex)

**X-A → X-B → X-D → X-C**

Обоснование: X-A локальные и низкорисковые. X-B архитектурный (escape hatch до обнуления) — важно сделать раньше smart-chunking. X-D простой metric fix. X-C самый рискованный (smart-chunking + Settings изменение + Tests compat) — последним, как эксперимент с отчётом.

---

## Финальные acceptance criteria (v2.1)

Все из v2 остаются + addition:

1. `_analysis_prompt_context` тест использует `payload["user_prompt"]`, не `["prompt"]`.
2. `_compose_history_summary` правка в **обеих** ветках (1399-1452 + 1454-1484).
3. Используются **точные enum values**: `hot/warm/cold` для lead_priority, `high/medium/low` для price_sensitivity.
4. `_filler_only_signature` возвращает `None` для одиночных filler-токенов и для commitment, не использует `id(text)`.
5. Escape hatch для non_conversation override — **до** обнуления полей (analyze.py:1719), не внутри `_apply_non_conversation_hard_validation`.
6. `_apply_non_conversation_hard_validation` signature правильная: `(self, call, normalized)`.
7. `tests/test_dialogue_format.py:make_settings()` обновлён с `analyze_transcript_chunking_mode`.
8. `_resolve_with_llm` early-return при provider="off" — метрика `llm_used = 0` honest.

---

## Если v2.1 опять с ошибками

Если Codex найдёт ещё проблемы — это 5-й случай. Будем разбираться, что именно я упускаю на этом уровне. Но я считаю, что после deep function-reading через субагент мы покрыли все три уровня (имена → внутренности → enum), и больше системных дыр не должно быть.
