# ТЗ-Y: Sanitizer + Quality fixes — снижение false positives в санитизации

Дата: 2026-05-14
Автор: Claude (наставник Дмитрия по ИИ-проекту Mango Analyse)
Адресат: Codex как primary implementation agent в отдельном диалоге
Связанный документ: `Foton/Mango_Analyse_Data_Quality_Improvement_TZ_FOR_CODEX_2026-05-14.md` (системное ТЗ из 18 правок, этот трек — часть P2)

---

## 0. Контекст и проблема

После полного аудита pipeline (14 мая) выявлено: **защитный слой санитизации работает, но местами слишком жёстко**. Три конкретных места дают **false positives** — система санитизирует или блокирует то, что объективно нормально, или пропускает то, что должна бы поймать.

Каждое из этих мест — small surgical fix, но они **не пересекаются** с другими параллельными работами Codex'а: catalog v2, deal-aware writeback, ТЗ-X analyze.py, ТЗ-Z hygiene infrastructure.

**Граница ответственности этого ТЗ:** только `insights/sanitizers.py`, `quality/tenant_text_normalizer.py`, `quality/crm_text_quality_detector.py` и соответствующие тесты. **Не трогать** `services/`, `customer_timeline/`, `question_catalog/`, `deal_aware/`.

---

## 1. Три правки

### Правка Y.1 — MONEY_AMOUNT_RE контекст-whitelist

**Файл:** `src/mango_mvp/insights/sanitizers.py:23-37` (MONEY_AMOUNT_RE), 39 (PERCENT_RE)

**Текущая логика:**
```python
MONEY_AMOUNT_RE = re.compile(
    r"\d[\d\s]{2,}\s*(?:руб|тыс|...)?(?!\s*(?:год(?:а|у|ом|е)?|г\.?))",
    re.I
)
```

**Проблема:** Регекс ловит числа в денежном контексте, но negative lookahead исключает только «год». В реальных звонках встречаются:
- **«5000 человек»** (количество клиентов / абитуриентов / посещений) → санитизируется как money → плейсхолдер
- **«30 баллов»** (балл ЕГЭ) → санитизируется
- **«250 учеников»** → санитизируется
- **«100 уроков»** → санитизируется
- **«15 часов»** → санитизируется

И того хуже, `PERCENT_RE`:
- **«100% результат»** → `[PERCENT]` → потом `[PAYMENT_OPTIONS]` (через hardcode replace) → выход «актуальные варианты результат»
- **«100% гарантия»** → так же портится

**Что сделать:**

**Шаг 1.** Расширить negative lookahead для MONEY_AMOUNT_RE:

```python
NON_MONEY_UNITS = (
    r"год(?:а|у|ом|е)?|г\.?|"
    r"человек|учеников?|учениц(?:а|ы|у|ей)?|"
    r"семей|семь(?:и|ям)?|"
    r"баллов?|балл(?:а|у)?|"
    r"уроков?|урок(?:а|у)?|"
    r"занятий|часов?|час(?:а|у)?|минут|"
    r"класс(?:а|у|ом)?|"
    r"лет|года?|месяц(?:а|у|ев)?|недель?|"
    r"страниц|тем|глав"
)

MONEY_AMOUNT_RE = re.compile(
    rf"\d[\d\s]{{2,}}\s*(?:руб|тыс|млн|₽|р\.?|рубл\w*)?\b(?!\s*(?:{NON_MONEY_UNITS}))",
    re.I,
)
```

**Шаг 2.** Вынести список не-денежных существительных в `tenant_config.yaml`:

```yaml
quality:
  money_safe_units:
    - человек
    - ученик
    - семья
    - балл
    - урок
    - занятие
    - час
    - минута
    - класс
    - год  # уже было
    - месяц
    - неделя
```

Tenant-level список, который добавляется к hardcoded в коде. Это для будущего SaaS — каждый tenant может расширить.

**Шаг 3.** PERCENT_RE — добавить контекст-проверку для «результат», «гарантия», «успех»:

```python
PERCENT_RE = re.compile(
    r"\d{1,3}\s*%(?!\s*(?:результат\w*|гарант\w+|успех\w*|охват\w*|посещаем\w+))",
    re.I,
)
```

Числа типа «5%», «10%», «20%» **без** этих слов после — это скидка/комиссия → санитизировать. С этими словами — не санитизировать.

**Шаг 4.** Заменить hardcode `result.replace("[PERCENT]", "[PAYMENT_OPTIONS]")` (line 250) на условную логику:

```python
# Было: result.replace("[PERCENT]", "[PAYMENT_OPTIONS]")
# Стало:
if "[PERCENT]" in result:
    # Этот placeholder появился значит PERCENT_RE сработал = это скидка/комиссия
    result = result.replace("[PERCENT]", "[PAYMENT_OPTIONS]")
```

(семантика не меняется — но я фиксирую факт того, что `replace` теперь имеет смысл только потому что PERCENT_RE уже отфильтровал «результат» через lookahead).

**Тест-якорь:** `tests/test_sanitizers_y1_context_whitelist.py`:

```python
def test_money_not_sanitized_in_non_money_context():
    cases = [
        "у нас 5000 человек прошли курсы",
        "ребёнок получил 250 баллов на ЕГЭ",
        "150 учеников в потоке",
        "100 уроков в программе",
        "20 часов индивидуальных занятий",
        "класс 7 А",
        "ученики 10 классов",
    ]
    for text in cases:
        sanitized = sanitize_text(text)
        assert "[" not in sanitized or "[PRICE]" not in sanitized, f"FP on: {text}"

def test_money_still_sanitized_in_money_context():
    cases = [
        "стоимость 50000 рублей",
        "10000 руб за семестр",
        "оплата 25000",  # без явной единицы, но в money-контексте
    ]
    for text in cases:
        sanitized = sanitize_text(text)
        assert "[PRICE]" in sanitized or "[" in sanitized, f"FN on: {text}"

def test_percent_not_sanitized_in_result_context():
    cases = [
        "100% результат",
        "95% гарантия успеха",
        "98% посещаемость",
        "80% охват программы",
    ]
    for text in cases:
        sanitized = sanitize_text(text)
        assert "результат" in sanitized or "гарантия" in sanitized or "посещаемость" in sanitized or "охват" in sanitized
        assert "[PAYMENT_OPTIONS]" not in sanitized, f"FP on: {text}"

def test_percent_still_sanitized_in_discount_context():
    cases = [
        "15% скидка",
        "30% комиссия банка",
        "5% годовых",
    ]
    for text in cases:
        sanitized = sanitize_text(text)
        assert "[" in sanitized, f"FN on: {text}"
```

**Acceptance:** регрессионные тесты зелёные. На 100 случайных bot_safe_answer текстах — over-sanitization candidates падает с текущих ~250 до ≤ 100 (приблизительно, точная оценка зависит от выборки).

---

### Правка Y.2 — Tenant normalizer brand regex blind spots

**Файл:** `src/mango_mvp/quality/tenant_text_normalizer.py:8-13` (BRAND_ALIASES_RE)

**Текущая логика:**
```python
BRAND_ALIASES_RE = re.compile(
    r"\b(?:М|У)?(?:Н|МН|ОНП|УНП|УНИП|УН[ФП]|УНФ)К?\s*"
    r"М\s*[ФШ]\s*(?:[ТДП](?:\s*[ИI])?|Т\b|[ИI]\b)",
    re.I,
)
```

**Проблема:** Word boundary `\b` в конце требует, чтобы после `И` или `Т` был non-word character. Но реальные ASR-варианты:

- **«УНПК МФТИШ»** → после «И» идёт «Ш» (word-char) → `\b` не срабатывает → regex НЕ матчит
- **«УНПК МФТИК»** → после «К» (word-char) → не матчит
- **«УНПК МФТИЙ»** → не матчит
- **«УНПК МФТИВ»** → не матчит

Эти ASR-варианты пропускаются нормализатором → в выход попадают как «УНПК МФТИШ» вместо канонического «УНПК МФТИ».

Дополнительно: `detect_residual_manager_text_artifacts` (line 50) использует **тот же** regex что и `normalize_manager_text` (line 44). Это **self-validation loop**: regex проверяет сам себя.

**Что сделать:**

**Шаг 1.** Расширить regex для tail-tolerance:

```python
BRAND_ALIASES_RE = re.compile(
    r"\b(?:М|У)?(?:Н|МН|ОНП|УНП|УНИП|УН[ФП]|УНФ)К?\s*"
    r"М\s*[ФШ]\s*[ТДПИ]\s*[ИI]?\s*[А-ЯЁа-яё]{0,2}",
    # Последняя группа [А-ЯЁа-яё]{0,2} ловит 0-2 лишних символа в хвосте
    re.I,
)
```

Логика: после МФТ + опциональная И + 0-2 любые кириллические буквы. Это покроет:
- МФТИ (стандарт) — 0 хвостовых букв
- МФТИШ — 1 хвостовая буква
- МФТИНГ — 2 хвостовых буквы
- МФТ (без И) — тоже сработает (0 после МФТ)

**Шаг 2.** Разделить detector и normalizer. Сейчас:
```python
def normalize_manager_text(text): use BRAND_ALIASES_RE
def detect_residual_manager_text_artifacts(text): use BRAND_ALIASES_RE  # тот же!
```

Сделать:
```python
def normalize_manager_text(text): use BRAND_ALIASES_RE  # как сейчас

def detect_residual_manager_text_artifacts(text):
    # Независимый detector — ищет brand-artifacts по более широким признакам
    # (не just regex, а character-level patterns)
    suspicious_patterns = [
        r"\b(?:[МНУ]\W{0,2}){2,}\s*(?:[МФШ]\W{0,2}){2,}",  # >=2 заглавных одной серии
        r"М[ФШ][А-ЯЁ]{2,}",  # МФ + 2+ заглавных
    ]
    findings = []
    for pat in suspicious_patterns:
        if re.search(pat, text, re.I):
            findings.append(("possible_brand_residual", pat))
    return findings
```

Это разрывает self-validation loop. Detector использует **другие** patterns, не тот же regex что normalizer.

**Тест-якорь:** `tests/test_tenant_normalizer_y2_brand_blind_spots.py`:

```python
def test_brand_normalization_catches_tail_variants():
    variants = ["УНПК МФТИШ", "УНПК МФТИК", "УНПК МФТИЙ", "УНПК МФТИВ", "УНПК МФТИНГ"]
    for v in variants:
        normalized = normalize_manager_text(v)
        assert "УНПК МФТИ" in normalized, f"Failed to normalize: {v}"

def test_brand_normalization_regression():
    """Старые работающие варианты всё ещё нормализуются."""
    variants = ["МПК МФТИ", "НПК МФТИ", "ОНПК МФТИ", "УНФК МФТИ", "УНПК МФТИ"]
    for v in variants:
        normalized = normalize_manager_text(v)
        assert "УНПК МФТИ" in normalized

def test_detector_uses_independent_patterns():
    """Detector ловит brand-residual через свои patterns, не через normalizer regex."""
    # Текст где normalizer почему-то промахнулся
    fake_residual = "У_Н_П_К  М_Ф_Т_И"  # с разделителями
    findings = detect_residual_manager_text_artifacts(fake_residual)
    assert any("brand" in f[0].lower() for f in findings)
```

**Acceptance:**
1. 5 tail-вариантов нормализуются корректно.
2. Все старые варианты (regression) продолжают работать.
3. Detector использует независимые patterns (verified через grep — detect функция не импортирует BRAND_ALIASES_RE).
4. Population gate после правки: residual_findings ≥ 0 (раньше было 0 из-за self-validation, после правки может появиться 1-5 как сигнал реальной нагрузки).

---

### Правка Y.3 — has_blocking_crm_text_quality_risk default P2 → P1

**Файл:** `src/mango_mvp/quality/crm_text_quality_detector.py:289`

**Текущая логика:**
```python
def has_blocking_crm_text_quality_risk(findings, min_severity="P2"):
    return any(f["severity"] <= min_severity for f in findings)
```

**Проблема:** Default `min_severity=P2` означает, что **любое** Q4c `vague_next_step` (например, «связаться позже», «обсудить детали») **блокирует live writeback**. Но «связаться позже после поступления» — корректная фраза в контексте «ребёнок ещё не зачислен в школу». 

Stage5 показывает: 14/723 hard-блок (1.9%), включая false positives на корректных «связаться позже»-фразах.

**Что сделать:**

Изменить default на `P1`:

```python
def has_blocking_crm_text_quality_risk(findings, min_severity="P1"):
    return any(f["severity"] <= min_severity for f in findings)
```

P2-level findings остаются **warnings** (видны в отчёте, рассматриваются РОПом), но не **блокируют** live writeback.

Это снижает hard-блок с 14/723 до ~6-8/723 (только true critical: completed_payment_conflict, wrong_person, lost_lead_conflict — все P1).

**Шаг 2.** Документировать изменение в `docs/CRM_TEXT_QUALITY_DETECTOR_DEFAULTS_2026-05-XX.md`:

```markdown
# Default severity for has_blocking_crm_text_quality_risk

С 2026-05-XX default min_severity изменён с P2 на P1.

Причина: P2-level findings (Q4c vague_next_step, Q4b stale_followup_date, Q3a weak_filler, и т.д.) 
включают много фраз, которые **корректны в контексте**, но регекс не отличает контекст 
(«связаться позже» — может быть валидным для ученика, который ещё не зачислен в школу).

Эти findings остаются в отчёте как warnings — РОП видит их, но они не блокируют writeback. 
Только P1-level (completed_payment_conflict, wrong_person, lost_lead_conflict) блокируют.

Если в будущем какой-то P2-класс окажется реально критичным — поднять его severity до P1, 
не менять default обратно.
```

**Шаг 3.** Перед коммитом — пройти **30 ROP-аудит строкам** с действующими P2 findings и убедиться, что они реально не критичны (могут идти в writeback). Это сделать на выборке из `audits/_results/2026-05-12_deal_aware_preview_50_v*/`.

**Тест-якорь:** `tests/test_crm_text_quality_y3_default_severity.py`:

```python
def test_default_min_severity_is_p1():
    """Default изменён с P2 на P1."""
    findings = [
        {"severity": "P1", "class": "completed_payment_conflict"},
        {"severity": "P2", "class": "vague_next_step"},
    ]
    # При default: только P1 блокирует
    assert has_blocking_crm_text_quality_risk([findings[0]]) is True
    assert has_blocking_crm_text_quality_risk([findings[1]]) is False
    # Explicit P2 — блокирует
    assert has_blocking_crm_text_quality_risk([findings[1]], min_severity="P2") is True

def test_p2_findings_still_visible_in_warnings():
    """P2 findings не блокируют, но они НЕ исчезают из общего отчёта."""
    findings = [{"severity": "P2", "class": "vague_next_step"}]
    report = build_text_quality_report(findings)
    assert any(w["severity"] == "P2" for w in report["warnings"])

def test_p1_findings_still_block():
    """Regression: P1 классы всё ещё блокируют."""
    p1_classes = ["completed_payment_conflict", "wrong_person", "lost_lead_conflict", "active_client_loss_reason"]
    for cls in p1_classes:
        findings = [{"severity": "P1", "class": cls}]
        assert has_blocking_crm_text_quality_risk(findings) is True, f"P1 {cls} must still block"
```

**Acceptance:**
1. Default min_severity = P1.
2. Регрессия на P1-классах: все blocking.
3. P2 findings present в warnings (не теряются).
4. На пересборке stage5 quality gate: hard-блок падает с 14/723 до 6-10 (true critical).
5. Документ в docs/ создан и описывает rationale.

---

## 2. Использование субагентов

Codex, можешь использовать до 6 параллельных субагентов:

- **Sub-A**: правка Y.1 (MONEY_AMOUNT_RE + PERCENT_RE) — самая большая по объёму regex.
- **Sub-B**: расширение `tenant_config.yaml` с `money_safe_units` + integration.
- **Sub-C**: правка Y.2 (tenant normalizer brand) — расширение regex + независимый detector.
- **Sub-D**: правка Y.3 (default P2 → P1) + документация + regression test.
- **Sub-E**: regression на 30 ROP-аудит строках с P2 findings перед коммитом Y.3 (проверка что они реально не критичны).
- **Sub-F**: общий sanity check + final integration tests + audit pack.

Распараллеливайся, но координируй: Y.3 ждёт результат Sub-E (проверка ROP rows) перед финальным merge.

---

## 3. Acceptance Criteria (вся правка Y)

### Hard requirements

1. Все 3 правки реализованы и тесты зелёные.
2. Существующие тесты `tests/test_sanitizers*.py`, `tests/test_tenant_normalizer*.py`, `tests/test_crm_text_quality*.py` всё ещё passed.
3. Y.1: 7 цитат с не-денежным контекстом не санитизируются, 3 цитаты с реальным money санитизируются.
4. Y.2: 5 tail-вариантов «УНПК МФТИШ/К/Й/В/НГ» нормализуются, regression на старых вариантах passed.
5. Y.3: default P1, hard-блок падает с 14/723 до 6-10 на stage5 пересборке.

### Soft requirements

6. Over-sanitization candidates падает с ~250 до ≤ 100 после Y.1.
7. tenant_normalizer detect функция использует независимые patterns (grep verification).

---

## 4. Deliverables

1. **Изменённые файлы:**
   - `src/mango_mvp/insights/sanitizers.py` (Y.1)
   - `src/mango_mvp/quality/tenant_text_normalizer.py` (Y.2)
   - `src/mango_mvp/quality/crm_text_quality_detector.py` (Y.3)
   - `tenant_config.yaml` или эквивалент (Y.1 — money_safe_units)

2. **Тесты:**
   - `tests/test_sanitizers_y1_context_whitelist.py`
   - `tests/test_tenant_normalizer_y2_brand_blind_spots.py`
   - `tests/test_crm_text_quality_y3_default_severity.py`

3. **Audit pack:**
   - `audits/_inbox/sanitizer_quality_y_<timestamp>/`:
     - `AUDIT_SCOPE.md`
     - `OVER_SANITIZATION_BEFORE_AFTER.md` — изменение количества candidates до/после Y.1
     - `BRAND_NORMALIZATION_COVERAGE.md` — тесты на 50 brand-вариантов до/после Y.2
     - `STAGE5_BLOCK_RATE_DELTA.md` — изменение hard-блок ratio до/после Y.3
     - `P2_FINDINGS_ROP_REVIEW_30_ROWS.md` — выборка из 30 строк с P2 findings и доказательство что они не должны блокировать

4. **Документация:**
   - `docs/CRM_TEXT_QUALITY_DETECTOR_DEFAULTS_2026-05-XX.md` (новый)
   - Обновление `docs/THREAT_MODEL.md` — пометка про Y.1 (context whitelist) и Y.2 (independent detector patterns)

---

## 5. Граничные условия

— **НЕ трогать** файлы вне `insights/sanitizers.py`, `quality/tenant_text_normalizer.py`, `quality/crm_text_quality_detector.py`.
— **НЕ менять** `bot_safety_detector.py` — это смежный модуль, его трогает ТЗ-Z hygiene.
— **НЕ менять** `services/` модули — это зона ТЗ-X.
— **НЕ менять** structure `findings` / `severity` — только default параметр функции Y.3.
— **НЕ выкидывать** P2 findings из отчёта в Y.3 — они остаются как warnings, просто не блокируют.

---

## 6. Контроль качества

Перед сдачей Codex отвечает сам:

1. Все 3 правки реализованы?
2. Все новые тесты зелёные?
3. Старые тесты всё ещё passed?
4. Y.1: 7 цитат non-money не санитизируются, 3 money санитизируются?
5. Y.2: 5 tail-вариантов нормализуются, регрессия зелёная?
6. Y.3: default P1, на 30 P2-rows ROP review подтверждает что они не критичны?
7. Документ docs/ создан?

---

## 7. Если что-то непонятно

Создай `audits/_inbox/sanitizer_quality_y_clarifications_REQUEST_<timestamp>/QUESTIONS_FOR_CLAUDE.md`.

Вероятные точки уточнения:
- Точный список `money_safe_units` в tenant_config — может быть расширения для конкретного tenant
- Какие именно patterns использовать для независимого detector в Y.2 (есть свобода реализации)
- Какие именно 30 P2-rows выбрать для ROP review — стратификация

---

## 8. Ожидаемый эффект

После реализации ТЗ-Y:
- **Меньше false positives** в санитизации — сохраняется содержательное (числа баллов/часов/учеников, проценты результатов).
- **Brand-нормализация надёжнее** — поймает ASR-варианты с лишними символами.
- **Detector независим** от normalizer — закрывает C8/F8 self-validation loop.
- **Hard-блок live writeback** падает на 30-50%, корректные фразы менеджеров перестают блокироваться.
- Системно: РОП и менеджеры получают меньше «странных» санитизаций в bot_safe_answer и в CRM-полях.

Это **точечный фикс защитного слоя** с измеримым эффектом.
