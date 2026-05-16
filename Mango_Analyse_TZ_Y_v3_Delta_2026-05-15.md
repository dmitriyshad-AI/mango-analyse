# ТЗ-Y-v3: Delta-исправления к TZ-Y-v2 после 2-го реверс-аудита Codex

Дата: 2026-05-15
Дополняет (не заменяет): `Mango_Analyse_TZ_Y_v2_Sanitizer_Quality_2026-05-14.md`
Адресат: Codex как primary implementation agent

Это **delta только**. Структура треков (Y-A, Y-B) сохраняется. Меняются только конкретные правки.

---

## Контекст

Codex провёл 2-й реверс-аудит TZ-Y-v2 и нашёл 6 ошибок (после первого v1→v2 цикла). Это 5-й случай подряд для серии Y/X/Z. Признаём системный паттерн — мой подход к написанию ТЗ имеет ограничения на детальном уровне. С следующих ТЗ переходим на новую модель (Codex пишет ТЗ под мою spec).

---

## ТРЕК Y-A: Delta-правки

### Y-A.1 — НЕ переписывать `MONEY_AMOUNT_RE` целиком

**Было в v2 (опасное):** полная замена `MONEY_AMOUNT_RE` на один новый шаблон из 3 альтернатив.

**Стало в v3:** **точечно добавить exclusion** только к двум опасным альтернативам, не трогая остальные.

**Что трогать:**
- Альтернатива standalone spaced-thousands (то место в `MONEY_AMOUNT_RE`, которое матчит `\d{1,3}(?:[\s ]\d{3})+(?!\w)` без money context) — добавить `(?!\s*(?:{NON_MONEY_UNITS}))`.
- Альтернатива `\bза\s+\d{4,6}\b` — добавить `(?!\s*(?:{NON_MONEY_UNITS}))`.

**Что НЕ трогать в `MONEY_AMOUNT_RE`:**
- Альтернативу с явным money-словом (`руб|тыс|млн|₽|р\.?|рубл`) — она правильная, оставить.
- Любые другие альтернативы которые ловят `50к`, `т.р.`, spoken money, контекст `стоимость/оплата/семестр` — все они должны продолжать работать.

**Конкретный план:**
- Codex читает текущий полный `MONEY_AMOUNT_RE` (sanitizers.py:23-38).
- Находит **только** 2 конкретные альтернативы которые дают FP (spaced-thousands standalone + `за NNNN`).
- Добавляет negative lookahead `(?!\s*(?:{NON_MONEY_UNITS}))` точечно.
- Остальные альтернативы оставляет нетронутыми.

**Acceptance расширен:**
- Регрессионные тесты на `50к`, `100 т.р.`, `пятьдесят тысяч рублей`, `стоимость 50000` — все санитизируются как раньше.
- Только spaced-thousands с non-money unit и `за NNNN человек` перестают санитизироваться.

---

### Y-A.2 — `PERCENT_RE` сохранить все формы, только добавить контекст-exclusions

**Было в v2 (опасное):** `PERCENT_RE = re.compile(r"\d{1,3}\s*%(?!...)")` — теряет `10 процентов`.

**Стало в v3:** найти текущий полный `PERCENT_RE` (включая словесные формы `процент/процента/процентов`) и добавить **только** negative lookahead для не-скидочного контекста:

```python
# Гипотетический подход (Codex проверит точный текущий regex)
# Сейчас: PERCENT_RE = re.compile(r"\d{1,3}\s*(?:%|процент(?:а|ов)?)(?!\w)|...", re.I)
# Стало: PERCENT_RE = re.compile(r"\d{1,3}\s*(?:%|процент(?:а|ов)?)(?!\s*(?:результат|гарант|успех|охват|посещаем|сдач))(?!\w)|...", re.I)
```

**Acceptance:**
- `10 процентов` всё ещё санитизируется (regression).
- `100% результат` НЕ санитизируется (fix).
- `15% скидка` санитизируется (regression).

---

### Y-A — Тесты исправить

**Было:** `assert "[CURRENT_PRICE]" in result.text or "actual" not in result.text.lower()` — вторая часть бесполезна.

**Стало:** простое `assert "[CURRENT_PRICE]" in result.text` для money cases. Для percent cases — `assert "[PAYMENT_OPTIONS]" in result.text` (через PERCENT→PAYMENT_OPTIONS replace).

---

## ТРЕК Y-B: Delta-правки

### Y-B — Учесть pending changes в `tenant_text_normalizer.py`

**Было в v2:** правка regex `BRAND_ALIASES_RE` на текущем рабочем дереве.

**Стало в v3:** перед стартом Y-B Codex обязательно:

1. **Зафиксировать текущий diff:** `git diff HEAD -- src/mango_mvp/quality/tenant_text_normalizer.py` → сохранить как `pending_changes_before_yb.patch` в audit pack.
2. **Не перезаписывать чужую работу.** Работать **поверх** current state, не возвращать к pre-pending версии.
3. Если pending changes уже решают часть Y-B задачи — отметить в audit pack «уже сделано в pending, не дублируется» и продолжить только с **дополнительными** правками.

---

### Y-B — `normalize_customer_text` не существует

**Было в v2:** «Применить `_normalize_mfti_tail_variants` после `BRAND_ALIASES_RE.sub(...)` в существующих функциях `normalize_manager_text` / `normalize_customer_text`».

**Стало в v3:** применять только в `normalize_manager_text`. Функция `normalize_customer_text` в коде **не существует**. Если бы нужна была отдельная customer-нормализация — это отдельное решение, в этом ТЗ не делается.

---

### Y-B — Detector patterns с explicit list known-variants

**Было в v2:** общие patterns типа `r"М[ФШ][А-ЯЁA-Z]{3,}\b"` — ловят `МФТИШ`, но могут пропустить `МПК МФТИ`, `НПК МФТИ` если нормализатор их пропустит.

**Стало в v3:**

```python
DETECTOR_KNOWN_BRAND_VARIANTS = (
    # Series of historical ASR-mistakes that may slip through normalizer
    "МПК МФТИ", "НПК МФТИ", "ОНПК МФТИ", "ВНПК МФТИ", "МНПК МФТИ", "УНПК МФП",
    "ЛНПК МФТИ", "УНФК МФТИ", "УНП МФТИ", "УНИПК МФТИ",
    # Tail-variants (если normalizer не догнал)
    "УНПК МФТИШ", "УНПК МФТИК", "УНПК МФТИЙ", "УНПК МФТИВ", "УНПК МФТИНГ",
)

DETECTOR_BRAND_GENERAL_PATTERNS = (
    # Catch-all patterns как safety net
    r"\bМ[ФШ][А-ЯЁA-Z]{3,}\b",
    r"\b[МНУ]\s+[МНУ]\s+[ПКФТ]",
)


def detect_residual_manager_text_artifacts(text: str) -> list[TenantTextArtifact]:
    findings = []
    # 1. Explicit known-variants check (точное совпадение)
    for variant in DETECTOR_KNOWN_BRAND_VARIANTS:
        # Skip canonical "УНПК МФТИ" (это нормализованная форма)
        if variant.upper() == "УНПК МФТИ":
            continue
        if variant.lower() in text.lower():
            findings.append(TenantTextArtifact(
                artifact_type="known_brand_variant_residual",
                sample=variant,
            ))
    # 2. General pattern catch-all
    for pattern in DETECTOR_BRAND_GENERAL_PATTERNS:
        for match in re.finditer(pattern, text, re.I):
            if match.group().strip().upper() == "УНПК МФТИ":
                continue
            findings.append(TenantTextArtifact(
                artifact_type="suspicious_brand_pattern",
                sample=match.group(),
            ))
    return findings
```

Это двухслойная защита:
1. **Explicit list** — гарантированно ловит известные historical ASR-варианты.
2. **General patterns** — catch-all для новых неожиданных вариантов.

**Acceptance:**
- На 30 test cases с known-variants — detector ловит все 30.
- На 10 случайных текстах с `УНПК МФТИ подготовка` — 0 false positives (no overmatch).

---

## ТРЕК Y-C: подтверждение, что отложен

Y-C полностью отложен. Subagent + Codex подтвердили: `has_blocking_crm_text_quality_risk` имеет **0 production callers**, изменение default бесполезно. Производственные места используют hardcoded `min_severity="P2"`. Перед любой правкой — ROP-аудит 30 P2-rows.

---

## Resume v2 → v3

| Аспект | v2 | v3 |
|---|---|---|
| Y-A.1 MONEY_AMOUNT_RE | Полная замена шаблона | Точечный exclusion в 2 опасных альтернативах |
| Y-A.2 PERCENT_RE | Только `%`, теряет `10 процентов` | Сохранить все формы, добавить context exclusions |
| Y-A тесты | `... or "actual" not in ...` | Прямое `assert "[CURRENT_PRICE]" in result.text` |
| Y-B start | Без учёта pending changes | Зафиксировать git diff перед стартом |
| Y-B function names | `normalize_customer_text` (не существует) | Только `normalize_manager_text` |
| Y-B detector | Только general patterns | Explicit list known-variants + general patterns как safety net |
