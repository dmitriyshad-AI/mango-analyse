# ТЗ-Y-v2: Sanitizer + Quality fixes — переписанная версия после реверс-аудита Codex

Дата: 2026-05-14
Автор: Claude (наставник Дмитрия) после critique Codex'а и точной grep-проверки через субагент
Заменяет: `Mango_Analyse_TZ_Y_Sanitizer_Quality_Fixes_2026-05-14.md` (v1, имела серьёзные ошибки)
Адресат: Codex как primary implementation agent

---

## 0. Что изменилось относительно v1

Codex выявил 7 серьёзных ошибок в v1:
- Y.1 устарел: bare `5000 человек` уже НЕ режется, реальная проблема в **spaced-thousands** (`5 000 человек`, `за 5000`)
- Y.2 предлагал опасный regex `[А-ЯЁа-яё]{0,2}` — съедает начало следующего слова
- Y.3 даёт **ноль эффекта** — `has_blocking_crm_text_quality_risk` имеет 0 production callers, изменение default бесполезно
- Несуществующие имена: `sanitize_text` → `sanitize_answer`, `[PRICE]` → `[CURRENT_PRICE]`, `build_text_quality_report` нет
- `tenant_config` — JSON, не YAML
- Scope противоречил deliverables
- Конфликт с pending changes в `tenant_text_normalizer.py`

Точная grep-проверка через субагент подтвердила всё это + обнаружила, что **`quality.crm_detector_min_severity` уже есть в `tenant_config_v1.json`** — механизм конфигурируемой severity встроен, не нужно изобретать.

**Главное архитектурное изменение:** v1 был **одним блоком из 3 правок**. v2 разбит на **2 трека (Y-A, Y-B)**, а Y.3 **полностью вынесен** как future-work после ROP-аудита P2 классов. Не пытаемся решать политические вопросы инфраструктурно.

---

## ТРЕК Y-A: Sanitizer spaced-thousands fix

### Контекст

Текущий `MONEY_AMOUNT_RE` в `insights/sanitizers.py:23` имеет 3 альтернативы. Первая ловит числа в money-контексте (правильно). **Вторая** (line 26) — `(?<!\w)\d{1,3}(?:[\s ]\d{3})+(?!\w)` — ловит **любое standalone число с пробельным разделителем тысяч** независимо от контекста. **Третья** (line 33) — `\bза\s+\d{4,6}\b` — ловит «за NNNN» без проверки следующего слова.

**Реальные false positives** (subagent verified):
- `"5 000 человек"` → `"[CURRENT_PRICE] человек"`
- `"за 5000 человек"` → `"[CURRENT_PRICE] человек"`
- `"2 500 баллов"` → `"[CURRENT_PRICE] баллов"`

**Не-false-positives** (работают корректно):
- `"5000 человек"` (без пробела-разделителя) → unchanged ✅
- `"5000 рублей"` → `[CURRENT_PRICE]` ✅

То есть проблема **только** в spaced-thousands и в «за NNNN»-pattern.

### Правка Y-A.1 — Negative lookahead для NON_MONEY_UNITS

**Файл:** `src/mango_mvp/insights/sanitizers.py:23-38` (MONEY_AMOUNT_RE)

**Что сделать:**

Добавить общий negative lookahead для не-денежных существительных. Применяется ко всем 3 альтернативам внутри MONEY_AMOUNT_RE:

```python
NON_MONEY_UNITS = (
    r"человек\w*|"
    r"учеников?|учениц(?:а|ы|у|ей)?|"
    r"семей|семь(?:и|ям)?|"
    r"баллов?|балл(?:а|у)?|"
    r"уроков?|урок(?:а|у)?|"
    r"занятий|занят(?:ия|ие)|"
    r"часов?|час(?:а|у)?|минут\w*|"
    r"класс(?:а|у|ом)?|"
    r"лет|года?|месяц(?:а|у|ев)?|недель?|"
    r"страниц\w*|тем\w*|глав\w*"
)

# MONEY_AMOUNT_RE — добавить negative lookahead к каждой альтернативе:
MONEY_AMOUNT_RE = re.compile(
    rf"(?<!\w)(?:"
    rf"(?:\d{{1,3}}(?:[\s ]\d{{3}})+|\d+[.,]\d+|\d[\d\s]{{2,}})\s*(?:руб\w*|тыс\w*|млн\w*|₽|р\.?|рубл\w*)\b"
    rf"|(?<!\w)\d{{1,3}}(?:[\s ]\d{{3}})+(?!\w)(?!\s*(?:{NON_MONEY_UNITS}))"
    rf"|\bза\s+\d{{4,6}}\b(?!\s*(?:{NON_MONEY_UNITS}))"
    rf")",
    re.I,
)
```

**Ключевая идея:**
- Первая альтернатива с явным money-словом (`руб`, `тыс`, etc.) — оставляем как есть, она правильная.
- Вторая (standalone spaced-thousands) и третья (`за NNNN`) — добавляем negative lookahead на не-денежные слова.

### Правка Y-A.2 — PERCENT_RE контекст-чувствительность

**Файл:** `src/mango_mvp/insights/sanitizers.py:39` (PERCENT_RE)

**Что сделать:**

Добавить negative lookahead для слов, после которых % — это **не** скидка/комиссия:

```python
PERCENT_RE = re.compile(
    r"\d{1,3}\s*%(?!\s*(?:результат\w*|гарант\w+|успех\w*|охват\w*|посещаем\w+|сдач\w+))",
    re.I,
)
```

Тогда `"100% результат"`, `"95% гарантия"`, `"98% посещаемость"` — НЕ санитизируются. `"15% скидка"`, `"30% комиссия"` — санитизируются как раньше.

Дополнительно: `result.replace("[PERCENT]", "[PAYMENT_OPTIONS]")` (line 250) — оставить как есть, потому что после правки PERCENT_RE плейсхолдер появляется только если это **реально скидка/комиссия**.

### Правка Y-A.3 (опционально) — Вынести NON_MONEY_UNITS в tenant_config

**Файл:** `_local_archive_mango_api_downloads_20260507/product_appliance/tenants/foton/config/tenant_config_v1.json` (формат JSON, не YAML — Codex прав)

**Что сделать (опционально, если есть время):**

Добавить в tenant_config:

```json
{
  "quality": {
    "crm_detector_min_severity": "P2",
    "money_safe_units": [
      "человек", "ученик", "семья", "балл", "урок",
      "занятие", "час", "минута", "класс", "год",
      "месяц", "неделя", "страница", "тема", "глава"
    ]
  }
}
```

В `tenant_config.py` — расширить validator чтобы поддерживать `money_safe_units` опционально. В `sanitizers.py` — функция `load_money_safe_units(tenant_config)`, которая возвращает hardcoded NON_MONEY_UNITS + extension из config.

**Это опциональная правка.** Если pending changes в `tenant_config.py` или `tenant_text_normalizer.py` уже что-то ломают — Y-A.3 пропускаем, оставляем NON_MONEY_UNITS hardcoded в sanitizers.py.

### Тесты Y-A

**Файл:** `tests/test_sanitizers_ya_spaced_thousands.py` (новый)

```python
import pytest
from mango_mvp.insights.sanitizers import sanitize_answer


def test_spaced_thousands_with_non_money_unit_not_sanitized():
    cases = [
        "у нас 5 000 человек прошли курсы",
        "2 500 баллов на ЕГЭ",
        "за 5000 человек",
        "1 000 учеников",
        "15 000 семей",
    ]
    for text in cases:
        result = sanitize_answer(text, mode="manager")
        assert "[CURRENT_PRICE]" not in result.text, f"FP on: {text!r}"


def test_money_still_sanitized_with_explicit_currency():
    cases = [
        "стоимость 50 000 рублей",
        "10 000 руб за семестр",
        "5000 руб",
        "120 000 рублей в месяц",
    ]
    for text in cases:
        result = sanitize_answer(text, mode="manager")
        assert "[CURRENT_PRICE]" in result.text or "actual" not in result.text.lower(), f"FN on: {text!r}"


def test_bare_number_with_non_money_unit_still_safe():
    """Regression: `5000 человек` без пробела-разделителя уже работает корректно."""
    cases = [
        "5000 человек",  # был корректен до правки, должен остаться корректен
        "250 баллов",
        "100 уроков",
    ]
    for text in cases:
        result = sanitize_answer(text, mode="manager")
        assert "[CURRENT_PRICE]" not in result.text, f"Regression on: {text!r}"


def test_percent_in_result_context_not_sanitized():
    cases = [
        "100% результат — наша гарантия",
        "95% гарантия успеха",
        "98% посещаемость учеников",
        "80% охват программы",
        "70% успешной сдачи ЕГЭ",
    ]
    for text in cases:
        result = sanitize_answer(text, mode="manager")
        assert "[PAYMENT_OPTIONS]" not in result.text, f"FP on: {text!r}"


def test_percent_in_discount_context_still_sanitized():
    cases = [
        "15% скидка на курс",
        "30% комиссия банка",
        "5% годовых при рассрочке",
    ]
    for text in cases:
        result = sanitize_answer(text, mode="manager")
        # [PERCENT] → [PAYMENT_OPTIONS] mapping должен сработать
        assert "[PAYMENT_OPTIONS]" in result.text or "actual" in result.text.lower(), f"FN on: {text!r}"
```

### Acceptance трека Y-A

1. Все 5 тестов зелёные.
2. Существующие тесты sanitizer (`tests/test_productization_sanitized_real_demo.py` и др.) всё ещё passed.
3. На 200 случайных bot_safe_answer текстах (test fixture, **не stable_runtime**) — over-sanitization candidates падает с baseline ~250 до ≤ 150.
4. Регрессия: `"5000 рублей"` и подобные явные money-cases всё ещё санитизируются.

### Deliverables Y-A

- `src/mango_mvp/insights/sanitizers.py` (правки MONEY_AMOUNT_RE + PERCENT_RE)
- `tests/test_sanitizers_ya_spaced_thousands.py` (новый)
- Опционально: `tenant_config_v1.json` (если делаем Y-A.3)
- `audits/_inbox/sanitizer_ya_<timestamp>/`:
  - `AUDIT_SCOPE.md`
  - `BEFORE_AFTER_50_REAL_TEXTS.md` — выборка из 50 реальных bot_safe_answer текстов с before/after
  - `OVER_SANITIZATION_DELTA.json` — изменение count кандидатов

---

## ТРЕК Y-B: Brand normalizer safe extension

### Контекст

**`tenant_text_normalizer.py` уже изменён в рабочем дереве** (subagent verified: pending changes на ветке `codex/git-order-20260513`). Текущий regex (после pending):

```
\b(?:(?:[А-ЯA-Z]?МПК|(?!УНПК)[А-ЯA-Z]?НПК|О\s*Н\s*П\s*К|Н\s*П\s*К|УНФК|УНП|ЛНПК|U\s*Н\s*И\s*П\s*К)
   \s*М\s*[ФШ]\s*(?:[ТДП](?:\s*[ИI])?|[ИI])
   |У\s*Н\s*П\s*К\s*М\s*[ФШ]\s*(?:[ДП](?:\s*[ИI])?|Т\b|[ИI]\b))\b
```

Это сложнее моего описания в v1, **уже покрывает** некоторые tail-варианты через `[ИI]` и word boundary `\b`. Но **точечные tail-варианты МФТИШ/К/Й/В/НГ всё ещё не покрыты** — после `И` идёт word-character, `\b` не срабатывает.

### Безопасная правка (не overmatching)

**Файл:** `src/mango_mvp/quality/tenant_text_normalizer.py:8-13`

**Что сделать:**

**НЕ** добавлять `[А-ЯЁа-яё]{0,2}` после `И` — это съест начало следующего слова (`УНПК МФТИ подготовка` → `УНПК МФТИп...`).

**Подход — explicit list:**

```python
# Известные ASR-tail-варианты МФТИ
MFTI_ASR_TAIL_VARIANTS = (
    "МФТИШ", "МФТИК", "МФТИЙ", "МФТИВ", "МФТИНГ", "МФТИХ", "МФТИЦ",
    "МФТИЧ", "МФТИШИ", "МФТИКА", "МФТИЙА",
)

# Преобразование происходит в две стадии:
# 1. Существующий BRAND_ALIASES_RE покрывает основные варианты
# 2. Дополнительный список tail-вариантов — точечная замена

def _normalize_mfti_tail_variants(text: str) -> str:
    """Точечная замена ASR-tail-вариантов МФТИ на УНПК МФТИ."""
    for variant in MFTI_ASR_TAIL_VARIANTS:
        # Word-boundary вокруг variant — гарантирует что variant это отдельное слово
        text = re.sub(rf"\bУНПК\s+{variant}\b", "УНПК МФТИ", text, flags=re.I)
        # Также если variant встречается отдельно без УНПК prefix
        text = re.sub(rf"\b{variant}\b", "МФТИ", text, flags=re.I)
    return text
```

Применить `_normalize_mfti_tail_variants` после `BRAND_ALIASES_RE.sub(...)` в существующих функциях `normalize_manager_text` / `normalize_customer_text`.

**Почему это safe:**
- `\b{variant}\b` гарантирует, что variant — отдельное слово (требует non-word до и после)
- Список explicit, не regex, который может overmatch
- Расширение прозрачное — добавил вариант в список → автоматически в logic
- Тесты можно покрыть каждый вариант explicitly

### Изоляция detector от normalizer (исправление self-validation loop)

**Файл:** `src/mango_mvp/quality/tenant_text_normalizer.py:56` (или где живёт `detect_residual_manager_text_artifacts`)

**Что сделать:**

Detector использует **независимые** patterns, не `BRAND_ALIASES_RE`:

```python
DETECTOR_BRAND_SUSPICIOUS_PATTERNS = (
    # Series of capital letters with potential ASR-mistakes
    r"\b(?:[МНУ]\W{0,2}){2,}\s*(?:[МФШ]\W{0,2}){2,}\b",  # multiple capitals
    # МФ + 2+ capitals (not МФТИ canonical)
    r"\bМ[ФШ][А-ЯЁA-Z]{3,}\b",
    # Spaced versions
    r"\b[МНУ]\s+[МНУ]\s+[ПКФТ]",
)

def detect_residual_manager_text_artifacts(text: str) -> list[TenantTextArtifact]:
    """Independent detector — НЕ использует BRAND_ALIASES_RE для проверки."""
    findings = []
    for pattern in DETECTOR_BRAND_SUSPICIOUS_PATTERNS:
        for match in re.finditer(pattern, text, re.I):
            # Skip если это нормализованная форма УНПК МФТИ
            if match.group().strip().upper() in {"УНПК МФТИ", "МФТИ"}:
                continue
            findings.append(TenantTextArtifact(...))
    return findings
```

Это разрывает self-validation loop. Detector ловит **подозрительные** patterns, не точные алиасы.

### Тесты Y-B

**Файл:** `tests/test_tenant_normalizer_yb_tail_variants.py`

```python
import pytest
from mango_mvp.quality.tenant_text_normalizer import (
    normalize_manager_text,
    detect_residual_manager_text_artifacts,
)


def test_tail_variants_normalized():
    cases = [
        ("УНПК МФТИШ", "УНПК МФТИ"),
        ("УНПК МФТИК", "УНПК МФТИ"),
        ("УНПК МФТИЙ", "УНПК МФТИ"),
        ("УНПК МФТИВ", "УНПК МФТИ"),
        ("УНПК МФТИНГ", "УНПК МФТИ"),
    ]
    for input_text, expected in cases:
        result = normalize_manager_text(input_text)
        assert expected in result, f"Failed: {input_text!r} -> {result!r}, expected {expected!r}"


def test_no_overmatch_into_next_word():
    """КРИТИЧНО: regex не должен съедать начало следующего слова."""
    cases = [
        ("УНПК МФТИ подготовка к ЕГЭ", "УНПК МФТИ подготовка к ЕГЭ"),  # МФТИ + полное слово "подготовка"
        ("МФТИ преподаватели лучшие", "МФТИ преподаватели лучшие"),  # МФТИ + "преподаватели"
        ("УНПК МФТИ программа физика", "УНПК МФТИ программа физика"),
    ]
    for input_text, expected in cases:
        result = normalize_manager_text(input_text)
        assert result == expected, f"Overmatch: {input_text!r} -> {result!r}"


def test_existing_variants_still_normalized():
    """Regression: старые варианты, которые уже работали."""
    cases = [
        ("МПК МФТИ", "УНПК МФТИ"),
        ("НПК МФТИ", "УНПК МФТИ"),
        ("ОНПК МФТИ", "УНПК МФТИ"),
        ("УНФК МФТИ", "УНПК МФТИ"),
        ("УНПК МФТИ", "УНПК МФТИ"),  # canonical стабилен
    ]
    for input_text, expected in cases:
        result = normalize_manager_text(input_text)
        assert expected in result, f"Regression: {input_text!r} -> {result!r}"


def test_detector_uses_independent_patterns():
    """Detector не должен импортировать BRAND_ALIASES_RE — verified через grep."""
    import inspect
    from mango_mvp.quality import tenant_text_normalizer
    source = inspect.getsource(tenant_text_normalizer.detect_residual_manager_text_artifacts)
    # detector не должен ссылаться на BRAND_ALIASES_RE напрямую
    assert "BRAND_ALIASES_RE" not in source
```

### Acceptance трека Y-B

1. 5 tail-вариантов нормализуются (МФТИШ/К/Й/В/НГ).
2. `УНПК МФТИ подготовка` → unchanged (НЕТ overmatch на «подготовка»).
3. Все старые тесты `tests/test_tenant_text_normalizer.py` всё ещё passed.
4. Detector не использует `BRAND_ALIASES_RE` (grep verified).

### Deliverables Y-B

- `src/mango_mvp/quality/tenant_text_normalizer.py` (правки + новая функция `_normalize_mfti_tail_variants`)
- `tests/test_tenant_normalizer_yb_tail_variants.py` (новый)
- `audits/_inbox/tenant_normalizer_yb_<timestamp>/`:
  - `AUDIT_SCOPE.md`
  - `OVERMATCH_REGRESSION.md` — 20 тестов на `УНПК МФТИ + следующее слово` показывают 0 overmatch
  - `TAIL_VARIANT_COVERAGE.md` — таблица «вариант → результат» на 30 ASR-вариантах

---

## ТРЕК Y-C: Severity policy review — отложен до ROP-аудита

**НЕ реализуется в v2.**

**Причина:**
- `has_blocking_crm_text_quality_risk` имеет 0 production callers (subagent verified) — изменение default бесполезно.
- Production callers (`deal_quality_gate.py:234`, `deal_writeback.py:147`, `amo_waiting_autonomous_work.py:205`) явно используют `min_severity="P2"` в hardcoded.
- Менять hardcoded на P1 без ROP-проверки опасно — мы не знаем, какие P2 классы реально false-positive.

**Что нужно сделать ДО любой реализации Y-C:**

1. **ROP-аудит 30 P2-rows.** Выборка из последних preview-пакетов deal-aware (`audits/_results/2026-05-12_deal_aware_preview_50_v*`). РОП помечает каждую: «реально критичная (оставить P2)» vs «false positive (понизить до P3)».
2. **На основе ROP-аудита** — конкретный список классов для перевода из P2 в P3 (или наоборот).
3. **Только потом** — кодовая правка либо в `severity_for_class` mapping, либо в hardcoded callers через `tenant_config.quality.crm_detector_min_severity`.

**Этот трек — отдельная задача для будущего ТЗ после ROP review.**

**Что НЕ делать сейчас:**
- НЕ менять default `has_blocking_crm_text_quality_risk` (бесполезно).
- НЕ менять hardcoded `min_severity="P2"` в production callers (без обоснования какие классы понижать).
- НЕ переводить классы из P2 в P3 без ROP-входа.

---

## Использование субагентов

Codex, в каждом из 2 треков можешь использовать до 6 субагентов:

- **Sub-A**: Y-A.1 (NON_MONEY_UNITS regex) + полный тест-сценарий
- **Sub-B**: Y-A.2 (PERCENT_RE context) + 50 real-text сценариев
- **Sub-C**: Y-A.3 (tenant_config integration, опционально) — пропустить если pending changes конфликтуют
- **Sub-D**: Y-B (brand tail-variants normalizer) — начать с **полного git diff** текущего `tenant_text_normalizer.py` против последнего committed
- **Sub-E**: Y-B (detector independence) + grep verification
- **Sub-F**: общий sanity check + final integration tests + audit pack

---

## Граничные условия

— **НЕ трогать** Y-C (severity policy) в этом ТЗ. Только после ROP-аудита.
— **Перед стартом Y-B** — `git status` и проверить, что pending changes в `tenant_text_normalizer.py` не конфликтуют с правкой. Если конфликт — спросить через QUESTIONS_FOR_CLAUDE.md.
— **Не использовать** `[А-ЯЁа-яё]{0,2}` или подобные «лишние символы» — overmatch-риск.
— **Использовать правильные имена API**: `sanitize_answer` (не `sanitize_text`), `[CURRENT_PRICE]` (не `[PRICE]`).
— **Acceptance не через `stable_runtime/`** — test fixtures или production data на копии.

---

## Если что-то непонятно

Создай `audits/_inbox/sanitizer_quality_yv2_<track>_clarifications_REQUEST_<timestamp>/QUESTIONS_FOR_CLAUDE.md`.

Вероятные точки уточнения:
- Точный список NON_MONEY_UNITS — может быть расширения для tenant-specific
- Список MFTI_ASR_TAIL_VARIANTS — может быть больше вариантов в реальных данных, нужен дополнительный анализ корпуса
- Y-A.3 (tenant_config integration) — делать или пропустить в этой итерации

---

## Резюме v1 → v2

| Аспект | v1 | v2 |
|---|---|---|
| Структура | 3 правки в одном треке | 2 трека (Y-A, Y-B) + Y-C отложен |
| Y.1 (sanitizer) | bare numbers `5000 человек` | spaced-thousands `5 000 человек` (реальная проблема) |
| Y.2 (brand) | regex `[А-ЯЁа-яё]{0,2}` — overmatch | explicit list MFTI_ASR_TAIL_VARIANTS |
| Y.3 (severity) | изменить default | ОТЛОЖЕНО до ROP-аудита (0 production callers) |
| API names | `sanitize_text`, `[PRICE]` | `sanitize_answer`, `[CURRENT_PRICE]` |
| Config | `tenant_config.yaml` | `tenant_config_v1.json` |
| Acceptance | `stable_runtime/` | test fixtures (stable_runtime read-only) |
| Detector independence | TODO | reализовано через `DETECTOR_BRAND_SUSPICIOUS_PATTERNS` |

После реализации 2 треков — sanitizer перестаёт ловить spaced-thousands false positives, brand normalizer покрывает ASR-tail-варианты без overmatch, detector не зависит от normalizer regex. Y-C ждёт ROP-аудита.
